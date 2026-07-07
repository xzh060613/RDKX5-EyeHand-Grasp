# RDKX5-EyeHand-Grasp
本作品基于地瓜机器人RDK X5平台，设计并实现了一款可移动的机械手抓取系统。系统将轻量型机械臂安装于轮式移动底盘上，融合视觉识别与底盘协同控制，能够在指定区域内自主导航并完成目标物体的定点抓取。RDK X5作为主控核心，负责图像采集、目标检测、机械臂逆运动学解算与底盘运动控制。视觉模块采用RGB相机识别物体位置，结合坐标转换引导机械臂完成抓取动作。底盘支持遥控与简易自主移动，实现“移动—定位—抓取”的完整任务流程。该系统结构简洁、成本可控，适用于桌面分拣、物料搬运等基础应用场景，可作为移动操作机器人的入门级验证平台，为更复杂的具身智能任务打下工程基础。
# RDKX5-EyeHand-Grasp

[![ROS2](https://img.shields.io/badge/ROS2-Humble-blue.svg)]()
[![Platform](https://img.shields.io/badge/Platform-RDK%20X5-green.svg)]()
[![Python](https://img.shields.io/badge/Python-3.10-yellow.svg)]()
[![MoveIt2](https://img.shields.io/badge/MoveIt2-Humble-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-red.svg)]()

A ROS2-based **Eye-in-Hand robotic grasping framework** for the **Alicia-D 6-DOF manipulator** equipped with a **Gemini2 RGB-D camera**.

The project provides a complete perception-to-manipulation pipeline including:

- RGB-D perception
- YOLO object detection
- Hand-eye calibration
- 3D target localization
- MoveIt motion planning
- Automatic grasp execution
- Manual pick-and-place workflow

The framework is optimized for the **Horizon Robotics RDK X5** platform and is designed as the manipulation module of a future **mobile manipulation robot**.

---

# Demonstration

A complete demonstration video is available here.

**Video**

Coming Soon...

The video demonstrates:

- Eye-in-Hand calibration
- Real-time object detection
- Manual grasp control
- Automatic grasp execution
- Return-to-home
- Cached grasp pose
- Manual release
- Mobile manipulation workflow

---

# Features

- Eye-in-Hand configuration
- Gemini2 RGB-D camera support
- Alicia-D 6-DOF manipulator
- ROS2 Humble
- MoveIt2 motion planning
- YOLO object detection
- RDK X5 BPU accelerated inference
- Automatic RGB-D target localization
- Hand-eye coordinate transformation
- Manual Pick / Release workflow
- OpenCV real-time visualization
- Cached grasp pose for release
- Configurable workspace limits
- Automatic grasp retry mechanism
- Portable to mobile manipulation robots

---

# Hardware Platform

- Alicia-D 6-DOF Robot Arm
- Gemini2 RGB-D Camera
- Horizon Robotics RDK X5
- Parallel Gripper
- SLLidar S3 (optional)

---

# Software Stack

- Ubuntu 22.04
- ROS2 Humble
- MoveIt2
- OpenCV
- tf2
- cv_bridge
- Ultralytics YOLOv8
- Horizon Robotics DNN Runtime

---

# System Pipeline

```
                 Gemini2 RGB-D Camera
                         │
                         ▼
              RGB-D Image Acquisition
                         │
                         ▼
            YOLO Detection (RDK X5 BPU)
                         │
                         ▼
               Target Selection (User)
                         │
                         ▼
             RGB-D Coordinate Estimation
                         │
                         ▼
             Hand-Eye Transformation
                         │
                         ▼
             Target Pose in base_link
                         │
                         ▼
              MoveIt Motion Planning
                         │
                         ▼
              Alicia-D Manipulator
                         │
                         ▼
              Parallel Gripper Close
                         │
                         ▼
          Cache Grasp Joint Configuration
                         │
                         ▼
            Return to Initial Pose
                         │
                Wait for Release
                         │
                         ▼
        Restore Cached Grasp Configuration
                         │
                         ▼
               Open Parallel Gripper
                         │
                         ▼
            Return to Initial Pose
```

---

# Repository Structure

```
RDKX5-EyeHand-Grasp
│
├── calibration
│   └── hand_eye_calibration_result.yaml
│
├── config
│   └── grasp.yaml
│
├── launch
│   ├── camera.launch.py
│   ├── grasp.launch.py
│   └── README.md
│
├── models
│   ├── yolov8_640x640_nv12.bin
│   └── README.md
│
├── scripts
│   └── yolo_eyehand_pregrasp_node.py
│
├── urdf
│   └── Alicia-D.urdf
│
├── README.md
└── README_cn.md
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/xzh060613/RDKX5-EyeHand-Grasp.git
```

Install ROS2 dependencies

```bash
sudo apt install ros-humble-moveit
sudo apt install ros-humble-tf2-ros
sudo apt install ros-humble-cv-bridge
```

Install Python packages

```bash
pip install ultralytics
pip install opencv-python
```

For RDK X5 deployment, please install the Horizon Robotics DNN Runtime according to the official documentation.

---

# Hand-Eye Calibration

The project uses the official Alicia-D Eye-in-Hand calibration package.

The calibrated transform is published as a static TF:

```
gripper_center
        │
        ▼
camera_link
```

This transform is used to convert target coordinates from the camera frame into the robot base frame.

---

# Usage

Launch the robot

```bash
ros2 launch alicia_d_moveit real_robot.launch.py
```

Run the grasp node

```bash
python3 yolo_eyehand_pregrasp_node.py
```

---

# Manual Workflow

The current version uses a manual pick-and-place workflow.

## Step 1

The robot automatically moves to the observation pose.

Current initial joint configuration:

```
Joint1 = 0°
Joint2 = 0°
Joint3 = 0°
Joint4 = 0°
Joint5 = -30°
Joint6 = 0°
```

---

## Step 2

YOLO continuously detects objects.

Detected objects are displayed in the OpenCV window with bounding boxes.

---

## Step 3

Press **G**

The system will:

- Lock the current target
- Estimate its 3D position
- Transform coordinates into the robot base frame
- Plan the grasp trajectory
- Execute grasp
- Close the gripper
- Record the grasp joint configuration
- Return to the observation pose

---

## Step 4

The mobile robot transports the object.

The robot keeps holding the object until the release command is received.

---

## Step 5

Press **R**

The system will:

- Restore the cached grasp configuration
- Open the gripper
- Return to the observation pose

---

# Configurable Parameters

Typical configurable parameters include:

```
initial_joint5_deg

workspace_x_min
workspace_x_max

workspace_y_min
workspace_y_max

workspace_z_min
workspace_z_max

grasp_x_offset
grasp_y_offset

pregrasp_z_offset

grasp_z_offset

lift_z_offset

depth_patch_radius
```

These parameters can be adjusted according to different robot configurations and application scenarios.

---

# Project Status

Current Version

- ✅ Eye-in-Hand Calibration
- ✅ RGB-D Object Detection
- ✅ Hand-Eye Coordinate Transformation
- ✅ Automatic 3D Target Localization
- ✅ MoveIt Motion Planning
- ✅ Automatic Grasp Execution
- ✅ Parallel Gripper Control
- ✅ OpenCV Real-time Visualization
- ✅ Manual Pick (Key G)
- ✅ Manual Release (Key R)
- ✅ Cached Grasp Pose
- ✅ Return-to-Observation Pose
- ✅ Workspace Limitation
- ✅ Automatic Grasp Retry
- ✅ RDK X5 Deployment
- ✅ RDK X5 BPU Acceleration

---

# Future Work

- Dedicated parcel/box detector
- Multi-object selection
- Automatic grasp orientation estimation
- Navigation2 integration
- Mobile robot autonomous transportation
- Autonomous pick-and-place
- Semantic grasp planning

---

# Citation

If you find this project useful, please consider giving it a ⭐ on GitHub.

---

# License

This project is released under the MIT License.
