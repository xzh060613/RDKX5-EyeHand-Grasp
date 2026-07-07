#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alicia-D + Gemini2 eye-in-hand YOLO grasp node, ROS2 version v24 RDK-X5 BPU manual cached-target pick/place.

功能：
1. 订阅 Gemini2 RGB、Depth、CameraInfo
2. 使用 Ultralytics .pt 或 RDK X5 BPU .bin YOLOv8 检测目标
3. 从检测框中心附近取深度中位数
4. 将像素反投影到 camera_color_optical_frame
5. 使用 TF 转换到 base_link
6. 生成 gripper_center 预抓取位姿
7. 调用 /compute_ik -> /plan_kinematic_path -> /execute_trajectory
8. 支持手动两阶段流程：等待时持续显示检测框；按G锁定当前目标并抓取；记录夹爪闭合时关节姿态；按R回到该姿态释放并回初始姿态

推荐先用 execute_motion:=false 只打印坐标，确认稳定后再 execute_motion:=true。
"""

import math
import time
import threading
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from rclpy.executors import MultiThreadedExecutor
from rclpy.duration import Duration
from rclpy.time import Time

from sensor_msgs.msg import Image, CameraInfo, JointState
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, TransformStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.srv import GetPositionIK, GetMotionPlan
from moveit_msgs.msg import Constraints, JointConstraint
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import MoveItErrorCodes

from cv_bridge import CvBridge
# Ultralytics is only needed for .pt fallback. On RDK X5, .bin uses BPU via hobot_dnn.
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

import tf2_ros
from tf2_ros import TransformException
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


def quat_xyzw_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Quaternion xyzw -> 3x3 rotation matrix."""
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-12:
        return np.eye(3)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz

    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)


def transform_to_matrix(tf_msg: TransformStamped) -> np.ndarray:
    t = tf_msg.transform.translation
    q = tf_msg.transform.rotation
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_xyzw_to_rot(q.x, q.y, q.z, q.w)
    T[:3, 3] = [t.x, t.y, t.z]
    return T


COCO80_NAMES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
    'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
    'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
    'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
    'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
    'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch',
    'potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone',
    'microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear',
    'hair drier','toothbrush'
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


class RDKYoloV8BinDetector:
    """RDK X5 BPU YOLOv8 detector for Horizon/D-Robotics .bin models.

    The uploaded model contains strings such as ``yolov8n_640x640_nv12`` and YOLOv8
    detect head outputs ``model.22/cv2.*`` and ``model.22/cv3.*``. This wrapper:
      1. resizes RGB camera frame to 640x640,
      2. converts it to NV12,
      3. runs pyeasy_dnn on BPU,
      4. decodes YOLOv8 DFL outputs in Python,
      5. returns the same tuple as the old Ultralytics path:
         (class_name, confidence, (x1, y1, x2, y2)) in original image coordinates.
    """

    def __init__(self, model_path: str, conf_thres: float = 0.45, iou_thres: float = 0.45,
                 input_size: int = 640, names: Optional[List[str]] = None, logger=None):
        self.model_path = model_path
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.input_size = int(input_size)
        self.names = names or COCO80_NAMES
        self.logger = logger
        try:
            from hobot_dnn import pyeasy_dnn as dnn
        except Exception:
            try:
                from hobot_dnn_rdkx5 import pyeasy_dnn as dnn
            except Exception as e:
                raise RuntimeError(
                    'Cannot import hobot_dnn/pyeasy_dnn. On RDK X5, source /opt/tros/humble/setup.bash '
                    'or install hobot-dnn-rdkx5 first.'
                ) from e
        self.models = dnn.load(model_path)
        self.model = self.models[0]
        if self.logger:
            self.logger.info(f'RDK BPU model loaded: {model_path}')
            try:
                self.logger.info(f'RDK BPU inputs: {[getattr(i.properties, "shape", None) for i in self.model.inputs]}')
                self.logger.info(f'RDK BPU outputs: {[getattr(o.properties, "shape", None) for o in self.model.outputs]}')
            except Exception:
                pass

    @staticmethod
    def rgb_to_nv12_resized(rgb: np.ndarray, size: int) -> np.ndarray:
        # RDK model name says NV12. Convert RGB -> BGR -> resized -> NV12.
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        img = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_LINEAR)
        h, w = img.shape[:2]
        yuv_i420 = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420).reshape(-1)
        y_size = h * w
        uv_size = y_size // 4
        y = yuv_i420[:y_size]
        u = yuv_i420[y_size:y_size + uv_size]
        v = yuv_i420[y_size + uv_size:y_size + uv_size * 2]
        uv = np.empty((uv_size * 2,), dtype=np.uint8)
        uv[0::2] = u
        uv[1::2] = v
        return np.concatenate([y, uv]).astype(np.uint8)

    def _output_array(self, out) -> np.ndarray:
        arr = np.array(out.buffer)
        # In most RDK Python examples, np.array(out.buffer) already has the correct shape.
        # If it is flat, try to reshape from tensor property shape.
        if arr.ndim == 1:
            try:
                shp = tuple(int(x) for x in out.properties.shape)
                if np.prod(shp) == arr.size:
                    arr = arr.reshape(shp)
            except Exception:
                pass
        return arr.astype(np.float32, copy=False)

    @staticmethod
    def _to_hwc(arr: np.ndarray) -> Optional[np.ndarray]:
        arr = np.squeeze(arr)
        if arr.ndim != 3:
            return None
        # NHWC/HWC: H,W,C, where H/W are 80/40/20 and C is 64 or 80.
        if arr.shape[-1] in (64, 80) and arr.shape[0] in (20, 40, 80) and arr.shape[1] in (20, 40, 80):
            return arr
        # CHW: C,H,W.
        if arr.shape[0] in (64, 80) and arr.shape[1] in (20, 40, 80) and arr.shape[2] in (20, 40, 80):
            return np.transpose(arr, (1, 2, 0))
        return None

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
        if boxes.size == 0:
            return []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / np.maximum(areas[i] + areas[order[1:]] - inter, 1e-6)
            order = order[1:][iou <= iou_thres]
        return keep

    def _decode_yolov8_outputs(self, outputs: List[np.ndarray], orig_w: int, orig_h: int):
        hwc = []
        for a in outputs:
            h = self._to_hwc(a)
            if h is not None:
                hwc.append(h)
        reg_maps = {}
        cls_maps = {}
        for a in hwc:
            H, W, C = a.shape
            if H != W:
                continue
            if C == 64:
                reg_maps[H] = a
            elif C == 80:
                cls_maps[H] = a
        detections = []
        proj = np.arange(16, dtype=np.float32)
        for H in sorted(set(reg_maps.keys()) & set(cls_maps.keys()), reverse=True):
            reg = reg_maps[H]  # H,W,64
            cls = cls_maps[H]  # H,W,80
            stride = float(self.input_size) / float(H)
            scores_all = _sigmoid(cls)
            cls_ids = np.argmax(scores_all, axis=-1)
            scores = np.max(scores_all, axis=-1)
            ys, xs = np.where(scores >= self.conf_thres)
            if xs.size == 0:
                continue
            reg_sel = reg[ys, xs].reshape(-1, 4, 16)
            reg_sel = reg_sel - np.max(reg_sel, axis=2, keepdims=True)
            prob = np.exp(reg_sel)
            prob = prob / np.maximum(np.sum(prob, axis=2, keepdims=True), 1e-9)
            dist = np.sum(prob * proj[None, None, :], axis=2) * stride
            cx = (xs.astype(np.float32) + 0.5) * stride
            cy = (ys.astype(np.float32) + 0.5) * stride
            x1 = cx - dist[:, 0]
            y1 = cy - dist[:, 1]
            x2 = cx + dist[:, 2]
            y2 = cy + dist[:, 3]
            # We use direct resize rather than letterbox, so scale x/y independently back to original RGB size.
            sx = float(orig_w) / float(self.input_size)
            sy = float(orig_h) / float(self.input_size)
            boxes = np.stack([x1 * sx, y1 * sy, x2 * sx, y2 * sy], axis=1)
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w - 1)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h - 1)
            for box, sc, cid in zip(boxes, scores[ys, xs], cls_ids[ys, xs]):
                detections.append((int(cid), float(sc), tuple(float(v) for v in box)))
        if not detections and self.logger:
            try:
                shapes = [o.shape for o in outputs]
                self.logger.warn(f'RDK YOLO decode produced no boxes. Output shapes={shapes}')
            except Exception:
                pass
        return detections

    def predict_one(self, rgb: np.ndarray) -> List[Tuple[str, float, Tuple[float, float, float, float]]]:
        h, w = rgb.shape[:2]
        inp = self.rgb_to_nv12_resized(rgb, self.input_size)
        outs = self.model.forward(inp)
        arrays = [self._output_array(o) for o in outs]
        det_raw = self._decode_yolov8_outputs(arrays, w, h)
        if not det_raw:
            return []
        boxes = np.array([d[2] for d in det_raw], dtype=np.float32)
        scores = np.array([d[1] for d in det_raw], dtype=np.float32)
        class_ids = np.array([d[0] for d in det_raw], dtype=np.int32)
        final = []
        # class-wise NMS
        for cid in np.unique(class_ids):
            idx = np.where(class_ids == cid)[0]
            keep_local = self._nms(boxes[idx], scores[idx], self.iou_thres)
            for k in keep_local:
                gi = int(idx[k])
                name = self.names[class_ids[gi]] if 0 <= class_ids[gi] < len(self.names) else str(class_ids[gi])
                final.append((name, float(scores[gi]), tuple(float(v) for v in boxes[gi])))
        final.sort(key=lambda x: x[1], reverse=True)
        return final


class YoloEyeHandPregraspNode(Node):
    def __init__(self):
        super().__init__('yolo_eyehand_pregrasp_node')

        # ---------------- Parameters ----------------
        self.declare_parameter('yolo_model', '/home/xzh/gemini_ws/src/yolov8n.pt')
        # v20: detector_backend=auto detects .bin and uses RDK X5 BPU; .pt uses Ultralytics fallback.
        self.declare_parameter('detector_backend', 'auto')
        self.declare_parameter('rdk_iou_thres', 0.45)
        self.declare_parameter('target_classes', 'suitcase,book,laptop,keyboard,cell phone,bottle,cup')
        self.declare_parameter('fallback_to_any_class', True)
        self.declare_parameter('conf_thres', 0.45)
        self.declare_parameter('infer_imgsz', 640)
        self.declare_parameter('show_window', True)
        self.declare_parameter('publish_debug_image', False)
        self.declare_parameter('debug_image_topic', '/yolo_eyehand/debug_image')
        self.declare_parameter('display_scale', 0.75)
        self.declare_parameter('display_timer_period_sec', 0.033)
        self.declare_parameter('opencv_ui_hz', 30.0)
        self.declare_parameter('keep_display_after_motion', True)
        # run_once only limits motion triggering. Detection can keep running so the OpenCV box follows the live image.
        self.declare_parameter('track_after_motion', True)
        # v14: once a grasp sequence starts, stop running YOLO/depth in the timer thread until motion completes.
        # This prevents target-loss warnings after lift and prevents perception updates from interfering with a locked sequence.
        self.declare_parameter('freeze_detection_during_motion', True)

        # v24 manual pick/place mode for mobile robot workflow.
        # Flow: startup -> move to initial pose -> wait grasp signal -> detect/pick/hold -> wait release signal -> release -> back to initial.
        self.declare_parameter('manual_pick_place_mode', True)
        self.declare_parameter('manual_cmd_topic', '/yolo_eyehand/manual_cmd')
        self.declare_parameter('auto_move_to_initial_on_start', True)
        self.declare_parameter('initial_joint5_deg', -30.0)
        self.declare_parameter('manual_allow_repeat_cycles', True)
        self.declare_parameter('keyboard_manual_control', True)
        # v24: before pressing G, keep detecting and drawing boxes. G locks the current valid target.
        self.declare_parameter('manual_use_cached_target_on_g', True)
        self.declare_parameter('manual_cached_target_max_age_sec', 2.0)
        # v24: after grasp, record arm joints at gripper-close pose; R returns to that pose to release.
        self.declare_parameter('release_to_recorded_grasp_pose', True)
        self.declare_parameter('record_grasp_pose_after_close', True)
        self.declare_parameter('release_recorded_pose_duration_sec', 4.0)

        self.declare_parameter('window_name', 'YOLO Eye-Hand Pregrasp')

        self.declare_parameter('color_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/depth/camera_info')

        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('tcp_frame', 'gripper_center')
        self.declare_parameter('move_group', 'Alicia')

        # v13: 官方 alicia_d_calibration 标定结果。
        # 注意：这里使用 verify_calibration.launch.py apply_optical_correction:=true 后实际发布的
        # gripper_center -> camera_link，而不是 YAML 原始矩阵，也不再使用旧的 Link6 -> camera_link。
        # 如果你已经单独启动官方 verify_calibration.launch.py 发布该 TF，运行本节点时可设置：
        #   -p publish_handeye_tf:=false
        self.declare_parameter('publish_handeye_tf', True)
        self.declare_parameter('handeye_parent_frame', 'gripper_center')
        self.declare_parameter('handeye_child_frame', 'camera_link')
        self.declare_parameter('handeye_xyz', [-0.047930,
                                                0.033591,
                                               -0.075233])
        self.declare_parameter('handeye_quat_xyzw', [-0.025719,
                                                     -0.698251,
                                                     -0.013046,
                                                      0.715272])

        # 深度处理。Gemini2 当前 depth encoding=16UC1，单位按 mm -> m。
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('depth_patch_radius', 8)
        self.declare_parameter('min_depth_m', 0.05)
        self.declare_parameter('max_depth_m', 1.50)

        # 预抓取/抓取参数。
        # execute_full_grasp=false 时：只移动到目标上方 pregrasp。
        # execute_full_grasp=true 时：打开夹爪 -> pregrasp -> grasp -> 闭合夹爪 -> lift。
        self.declare_parameter('pregrasp_z_offset', 0.08)
        self.declare_parameter('grasp_z_offset', 0.015)
        # v24: planar offset for the grasp target. Positive/negative direction can be tuned at runtime.
        # Defaults are 0 so behavior is unchanged unless explicitly configured.
        self.declare_parameter('grasp_x_offset', 0.0)
        self.declare_parameter('grasp_y_offset', 0.0)
        # v15: if the requested grasp height is too low and ExecuteTrajectory fails,
        # automatically retry with slightly higher grasp z offsets before aborting.
        self.declare_parameter('auto_grasp_retry_enabled', True)
        self.declare_parameter('grasp_retry_step_z', 0.01)
        self.declare_parameter('grasp_retry_count', 4)
        self.declare_parameter('grasp_retry_max_offset', 0.06)
        # Before the gripper has closed, abort cleanup should not force the arm back home by default.
        # This avoids the confusing behavior where a failed descend immediately retracts to home.
        self.declare_parameter('abort_return_home_before_close', False)
        self.declare_parameter('lift_z_offset', 0.10)
        self.declare_parameter('execute_motion', False)
        self.declare_parameter('execute_full_grasp', False)
        # If true, only test gripper open/close/open without camera/YOLO/arm motion.
        self.declare_parameter('gripper_test_only', False)
        self.declare_parameter('open_gripper_before_grasp', True)
        self.declare_parameter('run_once', True)
        self.declare_parameter('timer_period_sec', 1.0)

        # Alicia-D gripper controller. URDF/SRDF 中 Gripper: open=0.0, close=0.025。
        # 完整抓取后默认回到机械臂初始 home 关节位；为避免物体从高处掉落，
        # v13: 可在 home 位置垂直下降到接近抓取高度后再松开，然后再回 home。
        self.declare_parameter('return_home_after_lift', True)
        self.declare_parameter('release_at_home', True)
        self.declare_parameter('release_near_grasp_height', True)
        # v14: release before returning home by default: lift -> lower near pickup/grasp height -> open -> home.
        # This avoids dropping the object from the high home pose and avoids difficult vertical lowering at home XY.
        self.declare_parameter('release_before_home', False)
        # v19: after carrying the object back to home, move only Joint5 to a safe release angle
        # while keeping all other joints at the home/initial values, open the gripper, then return home.
        self.declare_parameter('release_by_joint5_after_home', True)
        self.declare_parameter('release_joint5_deg', -40.0)
        self.declare_parameter('release_extra_z_offset', 0.02)
        self.declare_parameter('return_home_after_release', True)
        self.declare_parameter('home_joint_positions', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.declare_parameter('gripper_action_name', '/Gripper_controller/follow_joint_trajectory')
        self.declare_parameter('gripper_joint_name', 'Gripper')
        self.declare_parameter('gripper_open_position', 0.0)
        self.declare_parameter('gripper_close_position', 0.025)
        self.declare_parameter('gripper_duration_sec', 1.5)
        # v11: in full grasp mode repeat the same gripper goal, because some Alicia-D firmwares/controllers
        # occasionally acknowledge a single FollowJointTrajectory goal but do not visibly move under load.
        self.declare_parameter('gripper_command_repeats', 2)
        self.declare_parameter('gripper_repeat_interval_sec', 0.35)
        self.declare_parameter('pause_after_gripper_sec', 0.5)
        # If a later arm step fails after closing, try to return home and open so the object is not kept clamped.
        self.declare_parameter('force_release_on_abort', True)
        self.declare_parameter('pause_after_motion_sec', 0.2)

        # v24: use direct slow joint trajectory for initial/release/home joint-only motions.
        # This is more stable than MoveIt ExecuteTrajectory for simple Joint5 release motions on the real controller.
        self.declare_parameter('arm_action_name', '/Alicia_controller/follow_joint_trajectory')
        self.declare_parameter('use_direct_joint_trajectory_for_home_release', True)
        self.declare_parameter('initial_move_duration_sec', 4.0)
        self.declare_parameter('joint5_release_duration_sec', 4.0)
        self.declare_parameter('return_initial_duration_sec', 4.0)

        # 使用当前 gripper_center 姿态作为目标姿态，第一版最安全。
        self.declare_parameter('use_current_tcp_orientation', True)
        self.declare_parameter('fixed_quat_xyzw', [0.0, 1.0, 0.0, 0.0])

        # 工作空间限制：v24 针对小车场景放宽，允许更大的 x/y 范围和 z 到 -0.40。
        self.declare_parameter('workspace_x_min', -0.65)
        self.declare_parameter('workspace_x_max', 0.15)
        self.declare_parameter('workspace_y_min', -0.45)
        self.declare_parameter('workspace_y_max', 0.45)
        self.declare_parameter('workspace_z_min', -0.40)
        self.declare_parameter('workspace_z_max', 0.35)

        self.declare_parameter('max_velocity_scaling_factor', 0.20)
        self.declare_parameter('max_acceleration_scaling_factor', 0.20)
        self.declare_parameter('allowed_planning_time', 5.0)
        self.declare_parameter('num_planning_attempts', 5)
        self.declare_parameter('ik_request_timeout_sec', 3.0)
        self.declare_parameter('ik_service_timeout_sec', 10.0)
        self.declare_parameter('plan_service_timeout_sec', 15.0)

        # Alicia-D arm joints
        self.arm_joint_names = ['Joint1', 'Joint2', 'Joint3', 'Joint4', 'Joint5', 'Joint6']

        # Read parameters
        self.yolo_model_path = self.get_parameter('yolo_model').value
        self.detector_backend = str(self.get_parameter('detector_backend').value).lower().strip()
        self.rdk_iou_thres = float(self.get_parameter('rdk_iou_thres').value)
        self.target_class_names = self._parse_target_classes(self.get_parameter('target_classes').value)
        self.fallback_to_any_class = bool(self.get_parameter('fallback_to_any_class').value)
        self.conf_thres = float(self.get_parameter('conf_thres').value)
        self.infer_imgsz = int(self.get_parameter('infer_imgsz').value)
        self.show_window = bool(self.get_parameter('show_window').value)
        self.publish_debug_image = bool(self.get_parameter('publish_debug_image').value)
        self.debug_image_topic = self.get_parameter('debug_image_topic').value
        self.display_scale = float(self.get_parameter('display_scale').value)
        self.window_name = self.get_parameter('window_name').value
        self.display_timer_period_sec = float(self.get_parameter('display_timer_period_sec').value)
        self.keep_display_after_motion = bool(self.get_parameter('keep_display_after_motion').value)
        self.track_after_motion = bool(self.get_parameter('track_after_motion').value)
        self.freeze_detection_during_motion = bool(self.get_parameter('freeze_detection_during_motion').value)
        self.opencv_ui_hz = float(self.get_parameter('opencv_ui_hz').value)
        self.manual_pick_place_mode = bool(self.get_parameter('manual_pick_place_mode').value)
        self.manual_cmd_topic = str(self.get_parameter('manual_cmd_topic').value)
        self.auto_move_to_initial_on_start = bool(self.get_parameter('auto_move_to_initial_on_start').value)
        self.initial_joint5_deg = float(self.get_parameter('initial_joint5_deg').value)
        self.manual_allow_repeat_cycles = bool(self.get_parameter('manual_allow_repeat_cycles').value)
        self.keyboard_manual_control = bool(self.get_parameter('keyboard_manual_control').value)
        self.manual_use_cached_target_on_g = bool(self.get_parameter('manual_use_cached_target_on_g').value)
        self.manual_cached_target_max_age_sec = float(self.get_parameter('manual_cached_target_max_age_sec').value)
        self.release_to_recorded_grasp_pose = bool(self.get_parameter('release_to_recorded_grasp_pose').value)
        self.record_grasp_pose_after_close = bool(self.get_parameter('record_grasp_pose_after_close').value)
        self.release_recorded_pose_duration_sec = float(self.get_parameter('release_recorded_pose_duration_sec').value)

        self.color_topic = self.get_parameter('color_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.base_frame = self.get_parameter('base_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.tcp_frame = self.get_parameter('tcp_frame').value
        self.move_group = self.get_parameter('move_group').value

        self.depth_scale = float(self.get_parameter('depth_scale').value)
        self.depth_patch_radius = int(self.get_parameter('depth_patch_radius').value)
        self.min_depth_m = float(self.get_parameter('min_depth_m').value)
        self.max_depth_m = float(self.get_parameter('max_depth_m').value)
        self.pregrasp_z_offset = float(self.get_parameter('pregrasp_z_offset').value)
        self.grasp_z_offset = float(self.get_parameter('grasp_z_offset').value)
        self.grasp_x_offset = float(self.get_parameter('grasp_x_offset').value)
        self.grasp_y_offset = float(self.get_parameter('grasp_y_offset').value)
        self.auto_grasp_retry_enabled = bool(self.get_parameter('auto_grasp_retry_enabled').value)
        self.grasp_retry_step_z = float(self.get_parameter('grasp_retry_step_z').value)
        self.grasp_retry_count = int(self.get_parameter('grasp_retry_count').value)
        self.grasp_retry_max_offset = float(self.get_parameter('grasp_retry_max_offset').value)
        self.abort_return_home_before_close = bool(self.get_parameter('abort_return_home_before_close').value)
        self.lift_z_offset = float(self.get_parameter('lift_z_offset').value)
        self.execute_motion = bool(self.get_parameter('execute_motion').value)
        self.execute_full_grasp = bool(self.get_parameter('execute_full_grasp').value)
        self.gripper_test_only = bool(self.get_parameter('gripper_test_only').value)
        self.open_gripper_before_grasp = bool(self.get_parameter('open_gripper_before_grasp').value)
        self.run_once = bool(self.get_parameter('run_once').value)
        self.return_home_after_lift = bool(self.get_parameter('return_home_after_lift').value)
        self.release_at_home = bool(self.get_parameter('release_at_home').value)
        self.release_near_grasp_height = bool(self.get_parameter('release_near_grasp_height').value)
        self.release_before_home = bool(self.get_parameter('release_before_home').value)
        self.release_by_joint5_after_home = bool(self.get_parameter('release_by_joint5_after_home').value)
        self.release_joint5_deg = float(self.get_parameter('release_joint5_deg').value)
        self.release_extra_z_offset = float(self.get_parameter('release_extra_z_offset').value)
        self.return_home_after_release = bool(self.get_parameter('return_home_after_release').value)
        self.home_joint_positions = [float(x) for x in self.get_parameter('home_joint_positions').value]
        if len(self.home_joint_positions) != 6:
            self.get_logger().warn(f'home_joint_positions length is {len(self.home_joint_positions)}, expected 6. Fallback to all zeros.')
            self.home_joint_positions = [0.0] * 6
        # In mobile-base manual mode, the initial/home pose is the old home with Joint5 lifted up to see near-car objects.
        # Joint order: Joint1, Joint2, Joint3, Joint4, Joint5, Joint6.
        if self.manual_pick_place_mode:
            self.home_joint_positions[4] = math.radians(self.initial_joint5_deg)

        self.gripper_action_name = self.get_parameter('gripper_action_name').value
        self.gripper_joint_name = self.get_parameter('gripper_joint_name').value
        self.gripper_open_position = float(self.get_parameter('gripper_open_position').value)
        self.gripper_close_position = float(self.get_parameter('gripper_close_position').value)
        self.gripper_duration_sec = float(self.get_parameter('gripper_duration_sec').value)
        self.gripper_command_repeats = max(1, int(self.get_parameter('gripper_command_repeats').value))
        self.gripper_repeat_interval_sec = float(self.get_parameter('gripper_repeat_interval_sec').value)
        self.pause_after_gripper_sec = float(self.get_parameter('pause_after_gripper_sec').value)
        self.force_release_on_abort = bool(self.get_parameter('force_release_on_abort').value)
        self.pause_after_motion_sec = float(self.get_parameter('pause_after_motion_sec').value)
        self.arm_action_name = str(self.get_parameter('arm_action_name').value)
        self.use_direct_joint_trajectory_for_home_release = bool(self.get_parameter('use_direct_joint_trajectory_for_home_release').value)
        self.initial_move_duration_sec = float(self.get_parameter('initial_move_duration_sec').value)
        self.joint5_release_duration_sec = float(self.get_parameter('joint5_release_duration_sec').value)
        self.return_initial_duration_sec = float(self.get_parameter('return_initial_duration_sec').value)
        self.use_current_tcp_orientation = bool(self.get_parameter('use_current_tcp_orientation').value)

        self.workspace = {
            'x': (float(self.get_parameter('workspace_x_min').value), float(self.get_parameter('workspace_x_max').value)),
            'y': (float(self.get_parameter('workspace_y_min').value), float(self.get_parameter('workspace_y_max').value)),
            'z': (float(self.get_parameter('workspace_z_min').value), float(self.get_parameter('workspace_z_max').value)),
        }

        self.bridge = CvBridge()
        self.latest_color: Optional[np.ndarray] = None
        self.latest_depth_raw: Optional[np.ndarray] = None
        self.camera_info: Optional[CameraInfo] = None
        self.has_executed = False
        self.motion_in_progress = False
        self.last_detection_for_display = {
            'detection': None,
            'depth_m': None,
            'p_base': None,
            'pregrasp_pose': None,
            'status': 'Waiting for first detection...',
            'workspace_ok': True,
        }
        self.display_lock = threading.Lock()
        self.cached_target_lock = threading.Lock()
        self.cached_target_bundle = None  # dict with detection/depth/p_base/poses/timestamp
        self.latest_joint_lock = threading.Lock()
        self.latest_joint_map = {}
        self.recorded_grasp_joint_positions = None
        self.recorded_grasp_joint_stamp = None

        # v24 manual workflow state.
        self.manual_lock = threading.Lock()
        self.manual_phase = 'WAIT_GRASP' if self.manual_pick_place_mode else 'AUTO'
        self.manual_grasp_requested = False
        self.manual_release_requested = False

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.static_broadcaster = StaticTransformBroadcaster(self)
        if bool(self.get_parameter('publish_handeye_tf').value):
            self._publish_handeye_static_tf()

        # Detector backend
        backend = self.detector_backend
        if backend == 'auto':
            backend = 'rdk_bin' if str(self.yolo_model_path).lower().endswith('.bin') else 'ultralytics'
        self.detector_backend = backend
        self.get_logger().info(f'Loading detector backend={backend}, model={self.yolo_model_path}')
        if backend in ('rdk_bin', 'rdk', 'bpu'):
            self.rdk_detector = RDKYoloV8BinDetector(
                self.yolo_model_path,
                conf_thres=self.conf_thres,
                iou_thres=self.rdk_iou_thres,
                input_size=self.infer_imgsz,
                names=COCO80_NAMES,
                logger=self.get_logger(),
            )
            self.model = None
            self.names: Dict[int, str] = {i: n for i, n in enumerate(COCO80_NAMES)}
            self.get_logger().info(f'RDK BPU YOLO loaded. Target classes: {self.target_class_names}, fallback={self.fallback_to_any_class}')
        elif backend in ('ultralytics', 'pt'):
            if YOLO is None:
                raise RuntimeError('Ultralytics is not available, but detector_backend=ultralytics was requested.')
            self.model = YOLO(self.yolo_model_path)
            self.rdk_detector = None
            self.names: Dict[int, str] = self.model.names
            self.get_logger().info(f'Ultralytics YOLO loaded. Target classes: {self.target_class_names}, fallback={self.fallback_to_any_class}')
        else:
            raise RuntimeError(f'Unknown detector_backend={backend}. Use auto, rdk_bin, or ultralytics.')

        # Subscribers
        self.create_subscription(Image, self.color_topic, self.color_callback, 10)
        self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)
        self.create_subscription(JointState, '/joint_states', self.joint_states_callback, 10)
        if self.manual_pick_place_mode:
            self.create_subscription(String, self.manual_cmd_topic, self.manual_cmd_callback, 10)
            self.get_logger().warn(
                f'v24 manual mode enabled. Command topic: {self.manual_cmd_topic}. '
                f'Publish String data=grasp/start to pick; data=release/place to release. In WAIT_GRASP, boxes are shown continuously; G locks current target.'
            )
        self.debug_image_pub = None
        if self.publish_debug_image:
            self.debug_image_pub = self.create_publisher(Image, self.debug_image_topic, 10)
            self.get_logger().info(f'Publishing debug image topic: {self.debug_image_topic}')

        # MoveIt service/action clients
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.plan_client = self.create_client(GetMotionPlan, '/plan_kinematic_path')
        self.execute_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
        self.gripper_client = ActionClient(self, FollowJointTrajectory, self.gripper_action_name)
        self.arm_traj_client = ActionClient(self, FollowJointTrajectory, self.arm_action_name)

        self.get_logger().info('Waiting for MoveIt services/actions...')
        self.ik_client.wait_for_service()
        self.plan_client.wait_for_service()
        self.execute_client.wait_for_server()
        self.get_logger().info('MoveIt services/actions connected.')
        self.get_logger().warn(
            f'PARAM CHECK: execute_motion={self.execute_motion}, execute_full_grasp={self.execute_full_grasp}, '
            f'gripper_test_only={self.gripper_test_only}, gripper_action={self.gripper_action_name}, '
            f'gripper_joint={self.gripper_joint_name}, open={self.gripper_open_position}, close={self.gripper_close_position}, '
            f'return_home_after_lift={self.return_home_after_lift}, release_at_home={self.release_at_home}, release_before_home={self.release_before_home}, '
            f'release_by_joint5_after_home={self.release_by_joint5_after_home}, release_joint5_deg={self.release_joint5_deg:.1f}, '
            f'home_joint_positions={[round(x, 3) for x in self.home_joint_positions]}'
        )
        if self.execute_full_grasp or self.gripper_test_only:
            self.get_logger().info(f'Waiting for gripper action: {self.gripper_action_name}')
            self.gripper_client.wait_for_server()
            self.get_logger().info('Gripper action connected.')
        if self.use_direct_joint_trajectory_for_home_release:
            self.get_logger().info(f'Waiting for arm joint trajectory action: {self.arm_action_name}')
            self.arm_traj_client.wait_for_server()
            self.get_logger().info('Arm joint trajectory action connected.')

        if self.manual_pick_place_mode and self.auto_move_to_initial_on_start and self.execute_motion and not self.gripper_test_only:
            threading.Thread(target=self._startup_initial_pose_worker, daemon=True).start()

        period = float(self.get_parameter('timer_period_sec').value)
        self.timer = self.create_timer(period, self.timer_callback)

        if self.gripper_test_only:
            self.get_logger().warn('gripper_test_only=true: will run open -> close -> open, no YOLO/arm motion.')
            threading.Thread(target=self._gripper_test_worker, daemon=True).start()

        # 重要：OpenCV 窗口不要放在 ROS2 executor/timer 线程里刷新。
        # v6 中：ROS executor 在后台线程跑，主线程专门运行 OpenCV UI loop。
        # 如果同时需要发布 debug image，则仍然用 ROS timer 发布；但 imshow 只在 main thread 中执行。
        if self.publish_debug_image:
            self.display_timer = self.create_timer(self.display_timer_period_sec, self.display_timer_callback)

        if self.show_window:
            self.get_logger().info('OpenCV imshow enabled. The main thread will be used as the UI loop.')
        elif self.publish_debug_image:
            self.get_logger().info(f'OpenCV imshow disabled. Use rqt_image_view/RViz2 to view: {self.debug_image_topic}')

        if not self.execute_motion:
            self.get_logger().warn('execute_motion=false: only print detection and target pose, robot will NOT move.')
        elif self.execute_full_grasp:
            self.get_logger().warn('execute_motion=true and execute_full_grasp=true: in manual mode robot will wait for grasp/release commands; otherwise it will run automatic full grasp. Keep emergency stop ready.')
        else:
            self.get_logger().warn('execute_motion=true: robot will plan and move to pregrasp pose only. Keep emergency stop ready.')

    @staticmethod
    def _parse_target_classes(s: str) -> List[str]:
        if not s:
            return []
        return [x.strip() for x in s.split(',') if x.strip()]

    def _publish_handeye_static_tf(self):
        xyz = list(self.get_parameter('handeye_xyz').value)
        quat = list(self.get_parameter('handeye_quat_xyzw').value)
        parent = self.get_parameter('handeye_parent_frame').value
        child = self.get_parameter('handeye_child_frame').value

        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = parent
        tf_msg.child_frame_id = child
        tf_msg.transform.translation.x = float(xyz[0])
        tf_msg.transform.translation.y = float(xyz[1])
        tf_msg.transform.translation.z = float(xyz[2])
        tf_msg.transform.rotation.x = float(quat[0])
        tf_msg.transform.rotation.y = float(quat[1])
        tf_msg.transform.rotation.z = float(quat[2])
        tf_msg.transform.rotation.w = float(quat[3])
        self.static_broadcaster.sendTransform(tf_msg)
        self.get_logger().info(f'Published static handeye TF: {parent} -> {child}')
        self.get_logger().info(
            f'Handeye xyz={ [round(float(v), 6) for v in xyz] }, '
            f'quat_xyzw={ [round(float(v), 6) for v in quat] }'
        )

    def color_callback(self, msg: Image):
        try:
            # 当前相机输出 rgb8。Ultralytics 可以直接吃 RGB ndarray。
            self.latest_color = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert color image: {e}')

    def depth_callback(self, msg: Image):
        try:
            # passthrough: 保留 16UC1 原始深度
            self.latest_depth_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Failed to convert depth image: {e}')

    def camera_info_callback(self, msg: CameraInfo):
        self.camera_info = msg

    def joint_states_callback(self, msg: JointState):
        try:
            with self.latest_joint_lock:
                for name, pos in zip(msg.name, msg.position):
                    if name in self.arm_joint_names:
                        self.latest_joint_map[name] = float(pos)
        except Exception as e:
            self.get_logger().warn(f'Failed to update joint_states cache: {e}')

    def get_current_arm_joint_positions(self) -> Optional[List[float]]:
        with self.latest_joint_lock:
            if all(j in self.latest_joint_map for j in self.arm_joint_names):
                return [float(self.latest_joint_map[j]) for j in self.arm_joint_names]
        return None

    def record_current_grasp_joint_positions(self, label: str = 'grasp_close') -> bool:
        joints = self.get_current_arm_joint_positions()
        if joints is None:
            self.get_logger().warn(f'{label}: /joint_states does not contain all arm joints; cannot record grasp release pose.')
            return False
        self.recorded_grasp_joint_positions = list(joints)
        self.recorded_grasp_joint_stamp = time.time()
        self.get_logger().warn(
            f'{label}: recorded arm joints for later release: ' +
            ', '.join([f'{name}={pos:.4f}' for name, pos in zip(self.arm_joint_names, joints)])
        )
        return True

    def manual_cmd_callback(self, msg: String):
        cmd = (msg.data or '').strip().lower()
        if not cmd:
            return
        if cmd in ('grasp', 'pick', 'start', 'g', '抓取', '夹取'):
            self.request_manual_grasp(source=f'topic:{cmd}')
        elif cmd in ('release', 'place', 'drop', 'r', '释放', '放下'):
            self.request_manual_release(source=f'topic:{cmd}')
        else:
            self.get_logger().warn(f'Unknown manual command: {cmd}. Use grasp/start or release/place.')

    def request_manual_grasp(self, source: str = 'manual'):
        if not self.manual_pick_place_mode:
            self.get_logger().warn(f'Ignored grasp command from {source}: manual_pick_place_mode=false.')
            return
        with self.manual_lock:
            if self.motion_in_progress:
                self.get_logger().warn(f'Ignored grasp command from {source}: motion is already in progress.')
                return
            if self.manual_phase not in ('WAIT_GRASP', 'DONE'):
                self.get_logger().warn(f'Ignored grasp command from {source}: current phase={self.manual_phase}.')
                return
            if self.manual_phase == 'DONE' and not self.manual_allow_repeat_cycles:
                self.get_logger().warn('Ignored grasp command: one cycle already completed and manual_allow_repeat_cycles=false.')
                return

        # v24: Prefer using the valid target already shown on screen, so the operator presses G
        # only after confirming the displayed box. If no fresh valid target is cached, fall back to
        # WAIT_TARGET and let the timer detect one.
        if self.manual_use_cached_target_on_g:
            bundle = None
            with self.cached_target_lock:
                if self.cached_target_bundle is not None:
                    age = time.time() - float(self.cached_target_bundle.get('timestamp', 0.0))
                    if age <= self.manual_cached_target_max_age_sec and self.cached_target_bundle.get('workspace_ok', False):
                        bundle = dict(self.cached_target_bundle)
            if bundle is not None:
                with self.manual_lock:
                    self.manual_phase = 'GRASPING'
                    self.manual_grasp_requested = False
                    self.motion_in_progress = True
                    self.has_executed = True
                det = bundle.get('detection')
                self.get_logger().warn(
                    f'MANUAL COMMAND [{source}]: lock displayed target and start pick: '
                    f'{det[0] if det else "unknown"} conf={det[1]:.3f}' if det else
                    f'MANUAL COMMAND [{source}]: lock displayed target and start pick.'
                )
                threading.Thread(
                    target=self._motion_worker,
                    args=(bundle['pregrasp_pose'], bundle['grasp_pose'], bundle['lift_pose']),
                    daemon=True
                ).start()
                return

        with self.manual_lock:
            self.manual_grasp_requested = True
            self.manual_phase = 'WAIT_TARGET'
            self.has_executed = False
        self.get_logger().warn(f'MANUAL COMMAND [{source}]: no fresh cached target; waiting for next valid YOLO target...')

    def request_manual_release(self, source: str = 'manual'):
        if not self.manual_pick_place_mode:
            self.get_logger().warn(f'Ignored release command from {source}: manual_pick_place_mode=false.')
            return
        with self.manual_lock:
            if self.motion_in_progress:
                self.get_logger().warn(f'Ignored release command from {source}: motion is already in progress.')
                return
            if self.manual_phase != 'HOLDING':
                self.get_logger().warn(f'Ignored release command from {source}: current phase={self.manual_phase}, expected HOLDING.')
                return
            self.manual_release_requested = True
            self.manual_phase = 'RELEASING'
            self.motion_in_progress = True
        self.get_logger().warn(f'MANUAL COMMAND [{source}]: release object then return to initial pose.')
        threading.Thread(target=self._manual_release_worker, daemon=True).start()

    def timer_callback(self):
        if getattr(self, 'gripper_test_only', False):
            return
        # v7: run_once 只限制“机械臂运动触发一次”，不再限制 YOLO 继续识别。
        # 这样机械臂移动完成后，OpenCV 窗口里的检测框仍会跟随实时画面刷新。
        if self.run_once and self.has_executed and not self.track_after_motion:
            return
        if self.latest_color is None or self.latest_depth_raw is None or self.camera_info is None:
            self.get_logger().info('Waiting for color/depth/camera_info...')
            self.update_debug_state(status='Waiting for color/depth/camera_info...')
            return

        manual_phase = None
        manual_grasp_requested = False
        if self.manual_pick_place_mode:
            with self.manual_lock:
                manual_phase = self.manual_phase
                manual_grasp_requested = self.manual_grasp_requested
            # v24: WAIT_GRASP does NOT return here. Keep YOLO running so boxes are visible before pressing G.
            if manual_phase == 'HOLDING':
                self.update_debug_state(status='Manual mode: object is clamped. Press r / publish release after car reaches destination.')
                return
            if manual_phase in ('RELEASING', 'GRASPING'):
                self.update_debug_state(status=f'Manual mode: {manual_phase.lower()} in progress...')
                return
            if manual_phase == 'DONE' and not manual_grasp_requested:
                # If repeat is allowed, still keep detecting so the next candidate is visible.
                if not self.manual_allow_repeat_cycles:
                    self.update_debug_state(status='Manual mode: cycle done.')
                    return
            # WAIT_TARGET means first manual signal was received without a cached target; detect until valid.

        # v14: 抓取序列已经锁定目标后，运动期间不再继续 YOLO/depth 计算。
        # 这样抬升后相机看不到目标也不会影响后续动作，也不会持续刷 No valid target/depth。
        if self.motion_in_progress and self.freeze_detection_during_motion:
            self.update_debug_state(status='Motion in progress: target locked; perception frozen')
            return

        # 拷贝一帧做 YOLO，避免相机回调正在写最新图像时被修改。
        rgb = self.latest_color.copy()
        depth_raw = self.latest_depth_raw.copy()
        info = self.camera_info

        all_detections = self.detect_all(rgb)
        detection = self.select_target_from_detections(all_detections)
        if detection is None:
            self.update_debug_state(
                detection=None,
                all_detections=all_detections if 'all_detections' in locals() else [],
                depth_m=None,
                p_base=None,
                pregrasp_pose=None,
                status='No valid YOLO target detected',
                workspace_ok=True,
            )
            with self.cached_target_lock:
                self.cached_target_bundle = None
            self.get_logger().warn('No valid YOLO target detected.')
            return

        cls_name, conf, bbox = detection
        x1, y1, x2, y2 = bbox
        u = int(round((x1 + x2) / 2.0))
        v = int(round((y1 + y2) / 2.0))

        depth_m = self.get_median_depth(u, v, depth_raw)
        if depth_m is None:
            self.update_debug_state(
                detection=detection,
                depth_m=None,
                p_base=None,
                pregrasp_pose=None,
                status=f'No valid depth at ({u},{v})',
                workspace_ok=True,
            )
            with self.cached_target_lock:
                self.cached_target_bundle = None
            self.get_logger().warn(f'No valid depth around bbox center: u={u}, v={v}, class={cls_name}')
            return

        p_cam = self.pixel_to_camera(u, v, depth_m, info)
        p_base = self.transform_point(self.camera_frame, self.base_frame, p_cam)
        if p_base is None:
            self.update_debug_state(
                detection=detection,
                depth_m=depth_m,
                p_base=None,
                pregrasp_pose=None,
                status='TF transform failed',
                workspace_ok=True,
            )
            with self.cached_target_lock:
                self.cached_target_bundle = None
            return

        self.get_logger().info(
            f'DETECT class={cls_name}, conf={conf:.3f}, bbox={[round(x, 1) for x in bbox]}, center=({u},{v}), depth={depth_m:.3f} m')
        self.get_logger().info(
            f'P_{self.camera_frame} = [{p_cam[0]:.4f}, {p_cam[1]:.4f}, {p_cam[2]:.4f}] m')
        self.get_logger().info(
            f'P_{self.base_frame} = [{p_base[0]:.4f}, {p_base[1]:.4f}, {p_base[2]:.4f}] m')

        workspace_ok = self.is_target_in_workspace(p_base)
        pregrasp_pose = self.make_offset_pose(p_base, self.pregrasp_z_offset) if workspace_ok else None
        grasp_pose = self.make_offset_pose(p_base, self.grasp_z_offset) if workspace_ok else None
        lift_pose = self.make_offset_pose(p_base, self.lift_z_offset) if workspace_ok else None

        # v24: cache the valid target that is currently drawn on screen. Pressing G locks this bundle.
        if workspace_ok and pregrasp_pose is not None and grasp_pose is not None and lift_pose is not None:
            with self.cached_target_lock:
                self.cached_target_bundle = {
                    'timestamp': time.time(),
                    'detection': detection,
                    'depth_m': depth_m,
                    'p_base': p_base.copy() if hasattr(p_base, 'copy') else p_base,
                    'pregrasp_pose': pregrasp_pose,
                    'grasp_pose': grasp_pose,
                    'lift_pose': lift_pose,
                    'workspace_ok': True,
                }

        status = ''
        if not workspace_ok:
            status = 'Workspace check failed - motion skipped'
        elif pregrasp_pose is not None:
            status = (f'pregrasp=[{pregrasp_pose.pose.position.x:.3f}, '
                      f'{pregrasp_pose.pose.position.y:.3f}, '
                      f'{pregrasp_pose.pose.position.z:.3f}]')
            if self.manual_pick_place_mode and manual_phase in ('WAIT_GRASP', 'DONE'):
                status = 'READY: press G to pick selected target | ' + status
            elif self.manual_pick_place_mode and manual_phase == 'WAIT_TARGET':
                status = 'G received: locking this valid target...'

        self.update_debug_state(
            detection=detection,
            all_detections=all_detections if 'all_detections' in locals() else [],
            depth_m=depth_m,
            p_base=p_base,
            pregrasp_pose=pregrasp_pose,
            status=status,
            workspace_ok=workspace_ok,
        )

        # v24: in WAIT_GRASP we only display and cache target; do not move until G is pressed.
        if self.manual_pick_place_mode and manual_phase in ('WAIT_GRASP', 'DONE') and not manual_grasp_requested:
            return

        if not workspace_ok:
            self.get_logger().error('Target point outside workspace limit. Motion skipped.')
            return
        if pregrasp_pose is None:
            return

        self.get_logger().info(
            f'Pregrasp {self.tcp_frame} pose in {self.base_frame}: '
            f'pos=[{pregrasp_pose.pose.position.x:.4f}, {pregrasp_pose.pose.position.y:.4f}, {pregrasp_pose.pose.position.z:.4f}]')

        if not self.execute_motion:
            self.get_logger().warn('Motion skipped because execute_motion=false.')
            # execute_motion=false 时也不要置 has_executed=True，否则 run_once 会让检测框停止更新。
            # 需要退出时按 q/ESC 或 Ctrl+C。
            return

        # run_once=True 时，机械臂只允许触发一次；但上面的 YOLO/坐标/显示状态仍继续更新。
        if self.run_once and self.has_executed:
            return

        if self.motion_in_progress:
            self.get_logger().warn('Motion already in progress; update display only, skip new motion command.')
            return

        self.motion_in_progress = True
        # v13: once a valid target has triggered a grasp, lock this run. Later target loss should not affect
        # the remaining cached motion sequence or cause a new target to be selected.
        if self.run_once:
            self.has_executed = True
        if self.manual_pick_place_mode:
            with self.manual_lock:
                self.manual_phase = 'GRASPING'
                self.manual_grasp_requested = False
        threading.Thread(
            target=self._motion_worker,
            args=(pregrasp_pose, grasp_pose, lift_pose),
            daemon=True
        ).start()

    def update_debug_state(self, detection=None, all_detections=None, depth_m=None, p_base=None, pregrasp_pose=None, status='', workspace_ok=True):
        with self.display_lock:
            self.last_detection_for_display = {
                'detection': detection,
                'all_detections': all_detections or [],
                'depth_m': depth_m,
                'p_base': p_base,
                'pregrasp_pose': pregrasp_pose,
                'status': status,
                'workspace_ok': workspace_ok,
                'motion_in_progress': self.motion_in_progress,
                'has_executed': self.has_executed,
            }

    def build_debug_image(self) -> Optional[np.ndarray]:
        """Build one annotated RGB image. Safe to call from the OpenCV UI thread."""
        if self.latest_color is None:
            return None
        if self.run_once and self.has_executed and not self.keep_display_after_motion:
            return None

        img = self.latest_color.copy()
        with self.display_lock:
            dbg = dict(self.last_detection_for_display) if self.last_detection_for_display is not None else {}

        detection = dbg.get('detection')
        all_detections = dbg.get('all_detections', [])
        depth_m = dbg.get('depth_m')
        p_base = dbg.get('p_base')
        status = dbg.get('status', '')
        workspace_ok = dbg.get('workspace_ok', True)
        motion_in_progress = dbg.get('motion_in_progress', self.motion_in_progress)

        # Draw all detections before pressing G; selected target is highlighted green.
        if all_detections:
            for det in all_detections:
                if detection is not None and det == detection:
                    continue
                is_target = (not self.target_class_names) or (det[0] in self.target_class_names)
                self.draw_detection(img, det, None, None, selected=False, target_class=is_target)
        if detection is not None:
            self.draw_detection(img, detection, depth_m, p_base, selected=True, target_class=True)
        if motion_in_progress:
            status = 'Motion in progress... ' + status
        if not workspace_ok:
            status = 'Workspace failed - ' + status
        if status:
            self.draw_status(img, status)
        return img

    def display_timer_callback(self):
        # 只负责发布 debug image；OpenCV imshow 由 main thread 的 opencv_ui_loop 执行。
        if not self.publish_debug_image:
            return
        img = self.build_debug_image()
        if img is not None and self.debug_image_pub is not None:
            self.publish_debug_image_msg(img)

    def opencv_ui_loop(self):
        """Run OpenCV window in the main thread to avoid 'window not responding'."""
        if not self.show_window:
            return
        delay_ms = max(1, int(1000.0 / max(1.0, self.opencv_ui_hz)))
        self.get_logger().info(
            f'OpenCV UI loop started in main thread. Press q or ESC in the image window to quit. hz={self.opencv_ui_hz:.1f}')
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        last_wait_log = 0.0
        try:
            while rclpy.ok():
                img = self.build_debug_image()
                if img is None:
                    # 没有图像时也要持续 waitKey，让窗口事件队列保持响应。
                    now = time.time()
                    if now - last_wait_log > 3.0:
                        self.get_logger().info('OpenCV window waiting for first image...')
                        last_wait_log = now
                    blank = np.zeros((360, 640, 3), dtype=np.uint8)
                    cv2.putText(blank, 'Waiting for camera image...', (30, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                    self.show_debug_window(blank)
                else:
                    self.show_debug_window(img)

                key = cv2.waitKey(delay_ms) & 0xFF
                if key in (27, ord('q')):
                    self.get_logger().info('OpenCV window quit requested.')
                    break
                if self.manual_pick_place_mode and self.keyboard_manual_control:
                    if key == ord('g'):
                        self.request_manual_grasp(source='keyboard:g')
                    elif key == ord('r'):
                        self.request_manual_release(source='keyboard:r')
        finally:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass


    def _gripper_test_worker(self):
        try:
            time.sleep(1.0)
            self.get_logger().warn('GRIPPER TEST: open')
            ok1 = self.move_gripper_to(self.gripper_open_position, 'test_open')
            time.sleep(0.8)
            self.get_logger().warn('GRIPPER TEST: close')
            ok2 = self.move_gripper_to(self.gripper_close_position, 'test_close')
            time.sleep(0.8)
            self.get_logger().warn('GRIPPER TEST: open again')
            ok3 = self.move_gripper_to(self.gripper_open_position, 'test_open_again')
            self.get_logger().warn(f'GRIPPER TEST DONE: open={ok1}, close={ok2}, open_again={ok3}')
        except Exception as e:
            self.get_logger().error(f'GRIPPER TEST exception: {e}')

    def _motion_worker(self, pregrasp_pose: PoseStamped, grasp_pose: Optional[PoseStamped], lift_pose: Optional[PoseStamped]):
        ok = False
        try:
            if self.execute_full_grasp:
                if grasp_pose is None or lift_pose is None:
                    self.get_logger().error('Full grasp requested but grasp/lift pose is None.')
                    self.has_executed = False
                    return
                if self.manual_pick_place_mode:
                    ok = self.execute_pick_and_hold_sequence(pregrasp_pose, grasp_pose, lift_pose)
                else:
                    ok = self.execute_grasp_sequence(pregrasp_pose, grasp_pose, lift_pose)
            else:
                ok = self.move_to_pose(pregrasp_pose)
            if not self.run_once:
                self.has_executed = ok
            elif not ok:
                self.get_logger().warn('Motion sequence failed after target was locked. run_once remains locked to avoid re-selecting a lost/changed target.')
        except Exception as e:
            self.get_logger().error(f'Motion worker exception: {e}')
        finally:
            self.motion_in_progress = False
            if self.manual_pick_place_mode:
                with self.manual_lock:
                    if ok:
                        self.manual_phase = 'HOLDING'
                    else:
                        self.manual_phase = 'WAIT_GRASP'
                        self.manual_grasp_requested = False
                        self.has_executed = False
                if ok:
                    self.get_logger().warn('MANUAL PICK DONE: object is clamped. Move the car, then send release command.')

    def draw_detection(self, img: np.ndarray, detection, depth_m=None, p_base=None, selected=False, target_class=True):
        cls_name, conf, bbox = detection
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        h, w = img.shape[:2]
        x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
        y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))
        if selected:
            color = (0, 255, 0)
            thickness = 3
        else:
            color = (0, 200, 0) if target_class else (255, 0, 0)
            thickness = 2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        u, v = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.circle(img, (u, v), 5, (0, 0, 255), -1)
        label = f'{cls_name} {conf:.2f}'
        if depth_m is not None:
            label += f' z={depth_m:.3f}m'
        cv2.putText(img, label, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if p_base is not None:
            text = f'base: x={p_base[0]:.3f}, y={p_base[1]:.3f}, z={p_base[2]:.3f}'
            cv2.putText(img, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

    def draw_status(self, img: np.ndarray, text: str):
        h, _ = img.shape[:2]
        cv2.putText(img, text, (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

    def publish_debug_image_msg(self, img: np.ndarray):
        try:
            msg = self.bridge.cv2_to_imgmsg(img, encoding='rgb8')
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.camera_frame
            self.debug_image_pub.publish(msg)
        except Exception as e:
            self.get_logger().warn(f'Failed to publish debug image: {e}')

    def show_debug_window(self, img: np.ndarray):
        try:
            show = img
            if self.display_scale > 0 and abs(self.display_scale - 1.0) > 1e-3:
                show = cv2.resize(img, None, fx=self.display_scale, fy=self.display_scale)
            # cv_bridge gives RGB; OpenCV window expects BGR.
            show_bgr = cv2.cvtColor(show, cv2.COLOR_RGB2BGR)
            cv2.imshow(self.window_name, show_bgr)
        except Exception as e:
            self.get_logger().warn(f'OpenCV display failed: {e}')
            self.show_window = False

    def detect_all(self, rgb_image: np.ndarray) -> List[Tuple[str, float, Tuple[float, float, float, float]]]:
        try:
            if self.detector_backend in ('rdk_bin', 'rdk', 'bpu'):
                return list(self.rdk_detector.predict_one(rgb_image))
            results = self.model.predict(rgb_image, conf=self.conf_thres, imgsz=self.infer_imgsz, verbose=False)
            if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                return []
            dets = []
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                name = self.names.get(cls_id, str(cls_id))
                xyxy = boxes.xyxy[i].detach().cpu().numpy().astype(float).tolist()
                dets.append((name, conf, tuple(xyxy)))
            dets.sort(key=lambda x: x[1], reverse=True)
            return dets
        except Exception as e:
            self.get_logger().error(f'Detector predict failed: {e}')
            return []

    def select_target_from_detections(self, dets: List[Tuple[str, float, Tuple[float, float, float, float]]]) -> Optional[Tuple[str, float, Tuple[float, float, float, float]]]:
        if not dets:
            return None
        candidates = []
        for item in dets:
            name = item[0]
            if (not self.target_class_names) or (name in self.target_class_names):
                candidates.append(item)
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0]
        if self.fallback_to_any_class:
            dets = sorted(dets, key=lambda x: x[1], reverse=True)
            self.get_logger().warn(
                f'No target class matched {self.target_class_names}; fallback to highest-confidence class.')
            return dets[0]
        return None

    def detect_target(self, rgb_image: np.ndarray) -> Optional[Tuple[str, float, Tuple[float, float, float, float]]]:
        return self.select_target_from_detections(self.detect_all(rgb_image))

    def get_median_depth(self, u: int, v: int, depth_raw: np.ndarray) -> Optional[float]:
        h, w = depth_raw.shape[:2]
        if u < 0 or u >= w or v < 0 or v >= h:
            return None
        r = self.depth_patch_radius
        x1, x2 = max(0, u - r), min(w, u + r + 1)
        y1, y2 = max(0, v - r), min(h, v + r + 1)
        patch = depth_raw[y1:y2, x1:x2].astype(np.float32) * self.depth_scale
        valid = patch[np.isfinite(patch)]
        valid = valid[(valid > self.min_depth_m) & (valid < self.max_depth_m)]
        if valid.size < 10:
            return None
        return float(np.median(valid))

    @staticmethod
    def pixel_to_camera(u: int, v: int, depth_m: float, info: CameraInfo) -> np.ndarray:
        fx = info.k[0]
        fy = info.k[4]
        cx = info.k[2]
        cy = info.k[5]
        x = (float(u) - cx) * depth_m / fx
        y = (float(v) - cy) * depth_m / fy
        z = depth_m
        return np.array([x, y, z], dtype=np.float64)

    def transform_point(self, source_frame: str, target_frame: str, p_source: np.ndarray) -> Optional[np.ndarray]:
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=1.0)
            )
        except TransformException as e:
            self.get_logger().error(f'TF lookup failed {target_frame} <- {source_frame}: {e}')
            return None
        T = transform_to_matrix(tf_msg)
        p_h = np.array([p_source[0], p_source[1], p_source[2], 1.0], dtype=np.float64)
        return (T @ p_h)[:3]

    def is_target_in_workspace(self, p_base: np.ndarray) -> bool:
        x, y, z = float(p_base[0]), float(p_base[1]), float(p_base[2])
        ok = (self.workspace['x'][0] <= x <= self.workspace['x'][1] and
              self.workspace['y'][0] <= y <= self.workspace['y'][1] and
              self.workspace['z'][0] <= z <= self.workspace['z'][1])
        if not ok:
            self.get_logger().error(
                f'Workspace check failed: p=[{x:.3f},{y:.3f},{z:.3f}], '
                f'x={self.workspace["x"]}, y={self.workspace["y"]}, z={self.workspace["z"]}')
        return ok

    def make_offset_pose(self, p_target_base: np.ndarray, z_offset: float) -> Optional[PoseStamped]:
        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = float(p_target_base[0] + self.grasp_x_offset)
        pose.pose.position.y = float(p_target_base[1] + self.grasp_y_offset)
        pose.pose.position.z = float(p_target_base[2] + z_offset)

        if self.use_current_tcp_orientation:
            try:
                tf_msg = self.tf_buffer.lookup_transform(
                    self.base_frame,
                    self.tcp_frame,
                    Time(),
                    timeout=Duration(seconds=1.0)
                )
                q = tf_msg.transform.rotation
                pose.pose.orientation.x = q.x
                pose.pose.orientation.y = q.y
                pose.pose.orientation.z = q.z
                pose.pose.orientation.w = q.w
            except TransformException as e:
                self.get_logger().error(f'Failed to get current TCP orientation: {e}')
                return None
        else:
            q = list(self.get_parameter('fixed_quat_xyzw').value)
            pose.pose.orientation.x = float(q[0])
            pose.pose.orientation.y = float(q[1])
            pose.pose.orientation.z = float(q[2])
            pose.pose.orientation.w = float(q[3])
        return pose

    def wait_future(self, future, timeout_sec: float, name: str) -> bool:
        """Wait for a ROS future while the main executor keeps spinning in another thread."""
        t0 = time.time()
        while rclpy.ok() and not future.done():
            if time.time() - t0 > timeout_sec:
                self.get_logger().error(f'{name} timeout')
                return False
            time.sleep(0.02)
        return future.done()

    def compute_ik(self, target_pose: PoseStamped) -> Optional[Dict[str, float]]:
        req = GetPositionIK.Request()
        req.ik_request.group_name = self.move_group
        req.ik_request.pose_stamped = target_pose
        req.ik_request.avoid_collisions = True
        req.ik_request.timeout = Duration(seconds=float(self.get_parameter('ik_request_timeout_sec').value)).to_msg()
        # ROS2 Humble's moveit_msgs/PositionIKRequest has no `attempts` field.
        # IK retries are handled by planning attempts later in /plan_kinematic_path.
        if hasattr(req.ik_request, 'ik_link_name'):
            req.ik_request.ik_link_name = self.tcp_frame
        req.ik_request.robot_state.is_diff = True

        future = self.ik_client.call_async(req)
        if not self.wait_future(future, float(self.get_parameter('ik_service_timeout_sec').value), '/compute_ik'):
            return None
        res = future.result()
        if res is None or res.error_code.val != MoveItErrorCodes.SUCCESS:
            code = res.error_code.val if res is not None else 'None'
            self.get_logger().error(f'IK failed, error_code={code}')
            return None

        joint_map = {}
        for name, pos in zip(res.solution.joint_state.name, res.solution.joint_state.position):
            if name in self.arm_joint_names:
                joint_map[name] = float(pos)

        missing = [j for j in self.arm_joint_names if j not in joint_map]
        if missing:
            self.get_logger().error(f'IK solution missing arm joints: {missing}')
            return None

        self.get_logger().info('IK solution: ' + ', '.join([f'{j}={joint_map[j]:.3f}' for j in self.arm_joint_names]))
        return joint_map

    def plan_to_joint_map(self, joint_map: Dict[str, float]):
        req = GetMotionPlan.Request()
        req.motion_plan_request.group_name = self.move_group
        req.motion_plan_request.num_planning_attempts = int(self.get_parameter('num_planning_attempts').value)
        req.motion_plan_request.allowed_planning_time = float(self.get_parameter('allowed_planning_time').value)
        req.motion_plan_request.max_velocity_scaling_factor = float(self.get_parameter('max_velocity_scaling_factor').value)
        req.motion_plan_request.max_acceleration_scaling_factor = float(self.get_parameter('max_acceleration_scaling_factor').value)
        req.motion_plan_request.start_state.is_diff = True

        goal_constraints = Constraints()
        goal_constraints.name = 'ik_joint_goal'
        for joint_name in self.arm_joint_names:
            jc = JointConstraint()
            jc.joint_name = joint_name
            jc.position = float(joint_map[joint_name])
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            goal_constraints.joint_constraints.append(jc)
        req.motion_plan_request.goal_constraints.append(goal_constraints)

        future = self.plan_client.call_async(req)
        if not self.wait_future(future, float(self.get_parameter('plan_service_timeout_sec').value), '/plan_kinematic_path'):
            return None
        res = future.result()
        if res is None or res.motion_plan_response.error_code.val != MoveItErrorCodes.SUCCESS:
            code = res.motion_plan_response.error_code.val if res is not None else 'None'
            self.get_logger().error(f'Planning failed, error_code={code}')
            return None
        traj = res.motion_plan_response.trajectory
        n_points = len(traj.joint_trajectory.points)
        self.get_logger().info(f'Planning success, trajectory points={n_points}')
        return traj

    def execute_trajectory(self, trajectory) -> bool:
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory

        send_future = self.execute_client.send_goal_async(goal)
        if not self.wait_future(send_future, 10.0, 'ExecuteTrajectory send goal'):
            return False
        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().error('ExecuteTrajectory goal rejected')
            return False

        timeout_sec = 30.0
        if trajectory.joint_trajectory.points:
            last = trajectory.joint_trajectory.points[-1].time_from_start
            timeout_sec = float(last.sec) + float(last.nanosec) * 1e-9 + 15.0

        result_future = goal_handle.get_result_async()
        if not self.wait_future(result_future, timeout_sec, 'ExecuteTrajectory result'):
            return False
        result = result_future.result()
        if result.result.error_code.val == MoveItErrorCodes.SUCCESS:
            self.get_logger().info('ExecuteTrajectory success.')
            return True
        self.get_logger().error(f'ExecuteTrajectory failed, error_code={result.result.error_code.val}')
        return False

    def _send_gripper_goal_once(self, position: float, label: str, repeat_idx: int) -> bool:
        goal = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        traj.joint_names = [self.gripper_joint_name]

        point = JointTrajectoryPoint()
        point.positions = [float(position)]
        point.velocities = [0.0]
        point.time_from_start = Duration(seconds=self.gripper_duration_sec).to_msg()
        traj.points.append(point)
        goal.trajectory = traj

        self.get_logger().info(
            f'Move gripper {label} repeat {repeat_idx}/{self.gripper_command_repeats}: '
            f'{self.gripper_joint_name}={position:.4f}, duration={self.gripper_duration_sec:.2f}s'
        )
        send_future = self.gripper_client.send_goal_async(goal)
        if not self.wait_future(send_future, 5.0, f'{label} send goal repeat {repeat_idx}'):
            return False
        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().error(f'{label} repeat {repeat_idx} goal rejected')
            return False

        result_future = goal_handle.get_result_async()
        timeout_sec = self.gripper_duration_sec + 6.0
        if not self.wait_future(result_future, timeout_sec, f'{label} result repeat {repeat_idx}'):
            return False
        result = result_future.result()
        # control_msgs/FollowJointTrajectory: SUCCESSFUL == 0.
        if result.result.error_code == 0:
            self.get_logger().info(f'Gripper {label} repeat {repeat_idx} success.')
            return True
        self.get_logger().error(f'Gripper {label} repeat {repeat_idx} failed, error_code={result.result.error_code}')
        return False

    def move_gripper_to(self, position: float, label: str = 'gripper') -> bool:
        ok_any = False
        for i in range(1, self.gripper_command_repeats + 1):
            ok = self._send_gripper_goal_once(position, label, i)
            ok_any = ok_any or ok
            # Even if first succeeds, repeat once to make the physical gripper visibly settle in full grasp mode.
            if i < self.gripper_command_repeats:
                time.sleep(self.gripper_repeat_interval_sec)
        time.sleep(self.pause_after_gripper_sec)
        if ok_any:
            self.get_logger().info(f'Gripper {label} final accepted/success after {self.gripper_command_repeats} repeat(s).')
            return True
        self.get_logger().error(f'Gripper {label} failed after {self.gripper_command_repeats} repeat(s).')
        return False

    def _safe_home_and_release_after_abort(self, close_done: bool):
        if not self.force_release_on_abort:
            return
        try:
            if close_done:
                self.get_logger().warn('Abort cleanup after gripper closed: return home and open gripper safely.')
                if self.return_home_after_lift:
                    self.get_logger().warn('Abort cleanup: return to home/initial joint position')
                    self.move_to_joint_positions(self.home_joint_positions, label='abort_home')
                    time.sleep(self.pause_after_motion_sec)
                self.get_logger().warn('Abort cleanup: open gripper')
                self.move_gripper_to(self.gripper_open_position, 'abort_release')
            else:
                self.get_logger().warn(
                    'Abort before gripper close: keep arm at current/pregrasp pose by default; open gripper only. '
                    'Set abort_return_home_before_close:=true if you want automatic home retract.'
                )
                self.move_gripper_to(self.gripper_open_position, 'abort_open_before_close')
                if self.abort_return_home_before_close and self.return_home_after_lift:
                    self.get_logger().warn('Abort before close: return home because abort_return_home_before_close=true')
                    self.move_to_joint_positions(self.home_joint_positions, label='abort_home_before_close')
                    time.sleep(self.pause_after_motion_sec)
        except Exception as e:
            self.get_logger().error(f'Abort cleanup exception: {e}')

    def _build_grasp_retry_poses(self, grasp_pose: PoseStamped):
        """Generate grasp poses for v15 retry: requested z first, then slightly higher z values."""
        poses = []
        seen = set()
        base_z = float(grasp_pose.pose.position.z)
        step = max(0.001, float(self.grasp_retry_step_z))
        count = max(0, int(self.grasp_retry_count))
        max_extra = max(0.0, float(self.grasp_retry_max_offset) - float(self.grasp_z_offset))
        extras = [0.0]
        if self.auto_grasp_retry_enabled:
            for i in range(1, count + 1):
                extra = i * step
                if extra <= max_extra + 1e-9:
                    extras.append(extra)
        for extra in extras:
            z = base_z + extra
            key = round(z, 4)
            if key in seen:
                continue
            seen.add(key)
            pose = PoseStamped()
            pose.header.frame_id = grasp_pose.header.frame_id
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = float(grasp_pose.pose.position.x)
            pose.pose.position.y = float(grasp_pose.pose.position.y)
            pose.pose.position.z = z
            pose.pose.orientation = grasp_pose.pose.orientation
            poses.append((pose, extra))
        return poses

    def _descend_to_grasp_with_retries(self, grasp_pose: PoseStamped):
        """Try descending to the requested grasp pose; if execution fails, retry at higher z offsets."""
        retry_poses = self._build_grasp_retry_poses(grasp_pose)
        total = len(retry_poses)
        for idx, (candidate, extra) in enumerate(retry_poses, start=1):
            if extra == 0.0:
                self.get_logger().info(
                    f'Step 2/6: descend to requested grasp pose ' 
                    f'z={candidate.pose.position.z:.4f} (attempt {idx}/{total})'
                )
            else:
                self.get_logger().warn(
                    f'Step 2 retry {idx}/{total}: requested grasp failed, try safer higher grasp pose: '
                    f'z={candidate.pose.position.z:.4f} (+{extra:.3f} m)'
                )
            if self.move_to_pose(candidate):
                if extra > 0.0:
                    self.get_logger().warn(
                        f'Using higher grasp pose after retry: actual_grasp_z={candidate.pose.position.z:.4f}. '
                        f'If this is too high to grip reliably, increase grasp_z_offset slightly or improve approach pose.'
                    )
                return candidate
            time.sleep(self.pause_after_motion_sec)
        self.get_logger().error('All grasp descend attempts failed. Gripper close will NOT be executed.')
        return None

    def execute_pick_and_hold_sequence(self, pregrasp_pose: PoseStamped, grasp_pose: PoseStamped, lift_pose: PoseStamped) -> bool:
        """v24 manual first stage: pick the object and keep it clamped until a later release command."""
        self.get_logger().warn('Start MANUAL PICK sequence v24: open -> pregrasp -> grasp(retry) -> close -> lift -> initial_pose(recorded grasp pose release) -> HOLD')
        close_done = False

        if self.open_gripper_before_grasp:
            self.get_logger().info('Manual pick Step 0/5: open gripper before approach')
            if not self.move_gripper_to(self.gripper_open_position, 'open'):
                self.get_logger().error('Open gripper before grasp failed. Stop sequence.')
                return False

        self.get_logger().info('Manual pick Step 1/5: move to pregrasp pose')
        if not self.move_to_pose(pregrasp_pose):
            self.get_logger().error('Manual pick Step 1 failed: pregrasp motion failed.')
            self._safe_home_and_release_after_abort(close_done=False)
            return False
        time.sleep(self.pause_after_motion_sec)

        actual_grasp_pose = self._descend_to_grasp_with_retries(grasp_pose)
        if actual_grasp_pose is None:
            self.get_logger().error('Manual pick Step 2 failed: descend/grasp motion failed after retries. Gripper close will NOT be executed.')
            self._safe_home_and_release_after_abort(close_done=False)
            return False
        time.sleep(self.pause_after_motion_sec)

        self.get_logger().info('Manual pick Step 3/5: close gripper and keep object clamped')
        if not self.move_gripper_to(self.gripper_close_position, 'close'):
            self.get_logger().error('Manual pick Step 3 failed: close gripper failed.')
            self._safe_home_and_release_after_abort(close_done=False)
            return False
        close_done = True
        if self.record_grasp_pose_after_close:
            self.record_current_grasp_joint_positions(label='Manual pick Step 3')

        self.get_logger().info('Manual pick Step 4/5: lift object')
        if not self.move_to_pose(lift_pose):
            self.get_logger().error('Manual pick Step 4 failed: lift motion failed.')
            self._safe_home_and_release_after_abort(close_done=close_done)
            return False
        time.sleep(self.pause_after_motion_sec)

        if self.return_home_after_lift:
            self.get_logger().info(
                f'Manual pick Step 5/5: return to initial holding pose: '
                f'Joint5={math.degrees(self.home_joint_positions[4]):.1f} deg, gripper remains CLOSED'
            )
            if not self.move_to_initial_joint_positions(label='initial_hold_after_pick', duration_sec=self.return_initial_duration_sec):
                self.get_logger().error('Manual pick Step 5 failed: return to initial holding pose failed. Object remains clamped if possible.')
                return False
            time.sleep(self.pause_after_motion_sec)

        self.get_logger().warn('MANUAL PICK sequence completed: object is clamped; waiting for release signal.')
        return True

    def execute_manual_release_sequence(self) -> bool:
        """v24 manual second stage: return to recorded grasp joint pose, open gripper, then return initial."""
        self.get_logger().warn('Start MANUAL RELEASE sequence v24: recorded grasp pose -> open gripper -> initial pose')

        target_joints = None
        if self.release_to_recorded_grasp_pose and self.recorded_grasp_joint_positions is not None:
            target_joints = list(self.recorded_grasp_joint_positions)
            self.get_logger().info(
                'Manual release Step 1/3: move back to recorded gripper-close/grasp joint pose: ' +
                ', '.join([f'{name}={pos:.4f}' for name, pos in zip(self.arm_joint_names, target_joints)])
            )
        else:
            self.get_logger().warn(
                'No recorded grasp joint pose available; fallback to Joint5 release pose. '
                'This should only happen if no successful pick has been completed.'
            )
            target_joints = list(self.home_joint_positions)
            target_joints[4] = math.radians(self.release_joint5_deg)

        if not self.move_to_joint_positions_home_release(
            target_joints,
            label='manual_recorded_grasp_release_pose',
            duration_sec=self.release_recorded_pose_duration_sec
        ):
            self.get_logger().error('Manual release Step 1 failed: move to recorded grasp/release pose failed. Keep gripper closed for safety.')
            return False
        time.sleep(self.pause_after_motion_sec)

        self.get_logger().info('Manual release Step 2/3: open gripper at recorded grasp pose')
        if not self.move_gripper_to(self.gripper_open_position, 'manual_release_open_at_recorded_grasp_pose'):
            self.get_logger().error('Manual release Step 2 failed: gripper open failed. Will still try to return initial.')

        self.get_logger().info('Manual release Step 3/3: return to initial pose after release')
        if not self.move_to_initial_joint_positions(label='manual_initial_after_release', duration_sec=self.return_initial_duration_sec):
            self.get_logger().error('Manual release Step 3 failed: return initial failed.')
            return False
        time.sleep(self.pause_after_motion_sec)
        self.get_logger().warn('MANUAL RELEASE sequence completed: object released at recorded grasp pose, arm returned to initial pose.')
        self.recorded_grasp_joint_positions = None
        self.recorded_grasp_joint_stamp = None
        return True

    def _manual_release_worker(self):
        ok = False
        try:
            ok = self.execute_manual_release_sequence()
        except Exception as e:
            self.get_logger().error(f'Manual release worker exception: {e}')
        finally:
            self.motion_in_progress = False
            with self.manual_lock:
                if ok:
                    self.manual_phase = 'WAIT_GRASP' if self.manual_allow_repeat_cycles else 'DONE'
                    self.manual_release_requested = False
                    self.has_executed = False
                else:
                    self.manual_phase = 'HOLDING'
                    self.manual_release_requested = False
            if not ok:
                self.get_logger().warn('Manual release failed. Object may still be clamped; fix manually or send release again.')

    def _startup_initial_pose_worker(self):
        try:
            time.sleep(1.0)
            self.get_logger().warn(
                f'v24 startup: move to initial pose with Joint5={math.degrees(self.home_joint_positions[4]):.1f} deg; '
                'then wait for manual grasp signal.'
            )
            self.motion_in_progress = True
            self.move_to_initial_joint_positions(label='startup_initial_pose', duration_sec=self.initial_move_duration_sec)
        except Exception as e:
            self.get_logger().error(f'Startup initial pose exception: {e}')
        finally:
            self.motion_in_progress = False

    def execute_grasp_sequence(self, pregrasp_pose: PoseStamped, grasp_pose: PoseStamped, lift_pose: PoseStamped) -> bool:
        self.get_logger().warn('Start FULL GRASP sequence v19: open -> pregrasp -> grasp(retry if needed) -> close -> lift -> home -> Joint5(-40deg) release -> home')
        close_done = False

        if self.open_gripper_before_grasp:
            self.get_logger().info('Step 0/6: open gripper before approach')
            if not self.move_gripper_to(self.gripper_open_position, 'open'):
                self.get_logger().error('Open gripper before grasp failed. Stop sequence.')
                return False

        self.get_logger().info('Step 1/6: move to pregrasp pose')
        if not self.move_to_pose(pregrasp_pose):
            self.get_logger().error('Step 1 failed: pregrasp motion failed.')
            self._safe_home_and_release_after_abort(close_done=False)
            return False
        time.sleep(self.pause_after_motion_sec)

        actual_grasp_pose = self._descend_to_grasp_with_retries(grasp_pose)
        if actual_grasp_pose is None:
            self.get_logger().error('Step 2 failed: descend/grasp motion failed after retries. Gripper close will NOT be executed.')
            self._safe_home_and_release_after_abort(close_done=False)
            return False
        grasp_pose = actual_grasp_pose
        time.sleep(self.pause_after_motion_sec)

        self.get_logger().info('Step 3/6: close gripper')
        if not self.move_gripper_to(self.gripper_close_position, 'close'):
            self.get_logger().error('Step 3 failed: close gripper failed.')
            self._safe_home_and_release_after_abort(close_done=False)
            return False
        close_done = True

        self.get_logger().info('Step 4/6: lift object')
        if not self.move_to_pose(lift_pose):
            self.get_logger().error('Step 4 failed: lift motion failed.')
            self._safe_home_and_release_after_abort(close_done=close_done)
            return False
        time.sleep(self.pause_after_motion_sec)

        release_low_pose = None

        # v16 optional behavior: release_before_home=true keeps the v14/v15 behavior, releasing low at the pickup XY before home.
        # The new default is release_before_home=false, handled in the legacy-style branch below: lift -> home -> low release -> home.
        if self.release_at_home and self.release_near_grasp_height and self.release_before_home:
            release_z = float(grasp_pose.pose.position.z) + float(self.release_extra_z_offset)
            self.get_logger().info(
                f'Step 5/7: lower near locked grasp height before release. '
                f'grasp_z={grasp_pose.pose.position.z:.4f}, extra={self.release_extra_z_offset:.4f}, release_z={release_z:.4f}'
            )
            release_low_pose = self.make_pose_from_reference_at_z(grasp_pose, release_z, label='release_low_pickup')
            if release_low_pose is None:
                self.get_logger().error('Step 5 failed: cannot build release-low pose. Keep object clamped and try home cleanup if enabled.')
                self._safe_home_and_release_after_abort(close_done=close_done)
                return False
            if not self.move_to_pose(release_low_pose):
                self.get_logger().error('Step 5 failed: lower-to-release-height motion failed. Keep object clamped and try home cleanup if enabled.')
                self._safe_home_and_release_after_abort(close_done=close_done)
                return False
            time.sleep(self.pause_after_motion_sec)

            self.get_logger().info('Step 6/7: open gripper at low release height')
            if not self.move_gripper_to(self.gripper_open_position, 'release_low_pickup'):
                self.get_logger().error('Step 6 failed: release/open gripper failed.')
                return False

            if self.return_home_after_lift or self.return_home_after_release:
                self.get_logger().info('Step 7/7: return home after low release')
                if not self.move_to_initial_joint_positions(label='home_after_release', duration_sec=self.return_initial_duration_sec):
                    self.get_logger().error('Step 7 failed: return home after release failed.')
                    return False
                time.sleep(self.pause_after_motion_sec)
            else:
                self.get_logger().info('Step 7/7: return home disabled; stay at low release pose')

        else:
            # v16 default behavior: lift -> home -> optionally lower near grasp height at current/home XY -> release -> home.
            if self.return_home_after_lift:
                self.get_logger().info('Step 5/8: return to home/initial joint position')
                if not self.move_to_initial_joint_positions(label='home', duration_sec=self.return_initial_duration_sec):
                    self.get_logger().error('Step 5 failed: return home failed. Try release cleanup if enabled.')
                    self._safe_home_and_release_after_abort(close_done=close_done)
                    return False
                time.sleep(self.pause_after_motion_sec)
            else:
                self.get_logger().info('Step 5/8: return_home_after_lift=false, skip home motion')

            if self.release_at_home and self.release_by_joint5_after_home:
                release_joint_positions = list(self.home_joint_positions)
                release_joint_positions[4] = math.radians(self.release_joint5_deg)
                self.get_logger().info(
                    f'Step 6/8: move to Joint5 release pose after home. '
                    f'Joint5={self.release_joint5_deg:.1f} deg ({release_joint_positions[4]:.4f} rad), '
                    f'other joints stay at home={ [round(x, 4) for x in self.home_joint_positions] }'
                )
                if not self.move_to_joint_positions_home_release(release_joint_positions, label='joint5_release_pose', duration_sec=self.joint5_release_duration_sec):
                    self.get_logger().error('Step 6 failed: move to Joint5 release pose failed. Try release at current/home pose.')
                else:
                    time.sleep(self.pause_after_motion_sec)

            elif self.release_at_home and self.release_near_grasp_height:
                release_z = float(grasp_pose.pose.position.z) + float(self.release_extra_z_offset)
                self.get_logger().info(
                    f'Step 6/8: lower near grasp height before release at current/home XY. '
                    f'grasp_z={grasp_pose.pose.position.z:.4f}, extra={self.release_extra_z_offset:.4f}, release_z={release_z:.4f}'
                )
                release_low_pose = self.make_current_tcp_xy_pose_at_z(release_z, label='release_low_home_xy')
                if release_low_pose is None:
                    self.get_logger().error('Step 6 failed: cannot build release-low pose. Try release at current/home pose.')
                elif not self.move_to_pose(release_low_pose):
                    self.get_logger().error('Step 6 failed: lower-to-release-height motion failed. Try release at current/home pose.')
                    release_low_pose = None
                time.sleep(self.pause_after_motion_sec)
            elif self.release_at_home:
                self.get_logger().info('Step 6/8: release_by_joint5_after_home=false and release_near_grasp_height=false; release at current/home pose')
            else:
                self.get_logger().info('Step 6/8: release_at_home=false, skip release motion')

            if self.release_at_home:
                if self.release_by_joint5_after_home:
                    label = 'joint5_release_pose'
                else:
                    label = 'release_low_home_xy' if release_low_pose is not None else 'release_at_home'
                self.get_logger().info(f'Step 7/8: open gripper ({label})')
                if not self.move_gripper_to(self.gripper_open_position, label):
                    self.get_logger().error('Step 7 failed: release/open gripper failed.')
                    return False
            else:
                self.get_logger().info('Step 7/8: release_at_home=false, keep gripper closed')

            if self.release_at_home and self.return_home_after_release:
                self.get_logger().info('Step 8/8: return home after release')
                if not self.move_to_initial_joint_positions(label='home_after_release', duration_sec=self.return_initial_duration_sec):
                    self.get_logger().error('Step 8 failed: return home after release failed.')
                    return False
                time.sleep(self.pause_after_motion_sec)
            else:
                self.get_logger().info('Step 8/8: return_home_after_release=false or release skipped')

        self.get_logger().info('FULL GRASP sequence completed.')
        return True

    def make_pose_from_reference_at_z(self, reference_pose: PoseStamped, target_z: float, label: str = 'release_low_pickup') -> Optional[PoseStamped]:
        """Copy a locked pose's x/y/orientation and replace only z.

        Used for v14 low release: after lift, descend at the same pickup XY using the locked grasp pose,
        so later YOLO target loss cannot change the release position.
        """
        z_min, z_max = self.workspace['z']
        z = float(target_z)
        if z < z_min or z > z_max:
            clamped = min(max(z, z_min), z_max)
            self.get_logger().warn(
                f'{label}: target_z={z:.4f} outside workspace z={self.workspace["z"]}; clamp to {clamped:.4f}'
            )
            z = clamped

        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(reference_pose.pose.position.x)
        pose.pose.position.y = float(reference_pose.pose.position.y)
        pose.pose.position.z = z
        pose.pose.orientation = reference_pose.pose.orientation
        self.get_logger().info(
            f'{label} pose in {self.base_frame}: '
            f'x={pose.pose.position.x:.4f}, y={pose.pose.position.y:.4f}, z={pose.pose.position.z:.4f}'
        )
        return pose

    def make_current_tcp_xy_pose_at_z(self, target_z: float, label: str = 'release_low') -> Optional[PoseStamped]:
        """Build a pose at the current TCP x/y/orientation, but with a specified base-frame z.

        Used after returning to home: keep the home XY and current orientation, then lower only in z
        so the object can be released close to the pickup height instead of being dropped from home height.
        """
        z_min, z_max = self.workspace['z']
        z = float(target_z)
        if z < z_min or z > z_max:
            clamped = min(max(z, z_min), z_max)
            self.get_logger().warn(
                f'{label}: target_z={z:.4f} outside workspace z={self.workspace["z"]}; clamp to {clamped:.4f}'
            )
            z = clamped

        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.tcp_frame,
                Time(),
                timeout=Duration(seconds=1.0)
            )
        except TransformException as e:
            self.get_logger().error(f'{label}: TF lookup failed {self.base_frame} <- {self.tcp_frame}: {e}')
            return None

        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(tf_msg.transform.translation.x)
        pose.pose.position.y = float(tf_msg.transform.translation.y)
        pose.pose.position.z = z
        pose.pose.orientation = tf_msg.transform.rotation

        self.get_logger().info(
            f'{label} pose in {self.base_frame}: '
            f'x={pose.pose.position.x:.4f}, y={pose.pose.position.y:.4f}, z={pose.pose.position.z:.4f}'
        )
        return pose

    def move_to_joint_positions_direct(self, joint_positions: List[float], label: str = 'direct_joint_goal', duration_sec: float = 4.0) -> bool:
        if len(joint_positions) != len(self.arm_joint_names):
            self.get_logger().error(f'{label}: expected {len(self.arm_joint_names)} joint positions, got {len(joint_positions)}')
            return False
        goal = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        traj.joint_names = list(self.arm_joint_names)
        point = JointTrajectoryPoint()
        point.positions = [float(x) for x in joint_positions]
        point.velocities = [0.0] * len(self.arm_joint_names)
        point.time_from_start = Duration(seconds=max(0.5, float(duration_sec))).to_msg()
        traj.points.append(point)
        goal.trajectory = traj
        self.get_logger().info(
            f'Direct arm trajectory to {label}: ' +
            ', '.join([f'{j}={p:.3f}' for j, p in zip(self.arm_joint_names, joint_positions)]) +
            f', duration={duration_sec:.2f}s'
        )
        send_future = self.arm_traj_client.send_goal_async(goal)
        if not self.wait_future(send_future, 5.0, f'{label} direct send goal'):
            return False
        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().error(f'{label} direct goal rejected')
            return False
        result_future = goal_handle.get_result_async()
        if not self.wait_future(result_future, max(8.0, float(duration_sec) + 8.0), f'{label} direct result'):
            return False
        result = result_future.result()
        if result.result.error_code == 0:
            self.get_logger().info(f'Direct arm trajectory {label} success.')
            return True
        self.get_logger().error(f'Direct arm trajectory {label} failed, error_code={result.result.error_code}')
        return False

    def move_to_joint_positions_home_release(self, joint_positions: List[float], label: str = 'home_release', duration_sec: float = 4.0) -> bool:
        if self.use_direct_joint_trajectory_for_home_release:
            ok = self.move_to_joint_positions_direct(joint_positions, label=label, duration_sec=duration_sec)
            if ok:
                return True
            self.get_logger().warn(f'{label}: direct joint trajectory failed; fallback to MoveIt joint plan.')
        return self.move_to_joint_positions(joint_positions, label=label)

    def move_to_initial_joint_positions(self, label: str = 'initial_pose', duration_sec: float = 4.0) -> bool:
        return self.move_to_joint_positions_home_release(self.home_joint_positions, label=label, duration_sec=duration_sec)

    def move_to_joint_positions(self, joint_positions: List[float], label: str = 'joint_goal') -> bool:
        if len(joint_positions) != len(self.arm_joint_names):
            self.get_logger().error(f'{label}: expected {len(self.arm_joint_names)} joint positions, got {len(joint_positions)}')
            return False
        joint_map = {name: float(pos) for name, pos in zip(self.arm_joint_names, joint_positions)}
        self.get_logger().info(
            f'Move arm to {label}: ' + ', '.join([f'{j}={joint_map[j]:.3f}' for j in self.arm_joint_names])
        )
        traj = self.plan_to_joint_map(joint_map)
        if traj is None:
            return False
        return self.execute_trajectory(traj)

    def move_to_pose(self, target_pose: PoseStamped) -> bool:
        joint_map = self.compute_ik(target_pose)
        if joint_map is None:
            return False
        traj = self.plan_to_joint_map(joint_map)
        if traj is None:
            return False
        return self.execute_trajectory(traj)


def main(args=None):
    rclpy.init(args=args)
    node = YoloEyeHandPregraspNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    if node.show_window:
        # OpenCV GUI 需要主线程持续处理 waitKey；ROS2 executor 放到后台线程。
        executor_thread = threading.Thread(target=executor.spin, daemon=True)
        executor_thread.start()
        try:
            node.opencv_ui_loop()
        except KeyboardInterrupt:
            pass
        finally:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            executor.shutdown()
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
            executor_thread.join(timeout=2.0)
    else:
        try:
            executor.spin()
        except KeyboardInterrupt:
            pass
        except RuntimeError as e:
            # Some camera drivers may throw a conversion RuntimeError during Ctrl+C/shutdown.
            # Treat it as a shutdown-time issue rather than a node logic failure.
            if 'Unable to convert call argument to Python object' in str(e):
                node.get_logger().warn(f'Ignored shutdown-time RuntimeError: {e}')
            else:
                raise
        finally:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            executor.shutdown()
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == '__main__':
    main()
