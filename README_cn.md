# RDKX5-EyeHand-Grasp

[![ROS2](https://img.shields.io/badge/ROS2-Humble-blue.svg)]()
[![Platform](https://img.shields.io/badge/Platform-RDK%20X5-green.svg)]()
[![Python](https://img.shields.io/badge/Python-3.10-yellow.svg)]()
[![MoveIt2](https://img.shields.io/badge/MoveIt2-Humble-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-red.svg)]()

一个基于 **ROS2** 的 **眼在手上（Eye-in-Hand）机械臂抓取框架**，适用于搭载 **Gemini2 RGB-D 深度相机** 的 **Alicia-D 六自由度机械臂**。

本项目实现了从视觉感知到机械臂抓取的完整流程，包括：

- RGB-D 图像采集
- YOLO 目标检测
- 手眼标定
- 三维目标定位
- MoveIt 运动规划
- 自动抓取执行
- 手动抓取/释放控制流程

整个系统针对 **地平线 RDK X5** 平台进行了优化，并作为未来 **移动操作机器人（Mobile Manipulation Robot）** 的抓取模块进行设计。

---

# 演示视频

完整演示视频：

**Video**

Coming Soon...

演示内容包括：

- 眼在手上手眼标定
- 实时目标检测
- 手动抓取控制
- 自动抓取流程
- 返回初始姿态
- 抓取姿态缓存
- 手动释放物体
- 移动机器人搬运流程

---

# 功能特点

- Eye-in-Hand（眼在手上）配置
- Gemini2 RGB-D 深度相机支持
- Alicia-D 六自由度机械臂
- ROS2 Humble
- MoveIt2 运动规划
- YOLO 目标检测
- RDK X5 BPU 加速推理
- RGB-D 三维目标定位
- 手眼坐标变换
- 手动抓取/释放流程
- OpenCV 实时检测画面
- 抓取姿态缓存
- 可配置工作空间
- 自动抓取重试机制
- 可扩展至移动机器人平台

---

# 硬件平台

- Alicia-D 六自由度机械臂
- Gemini2 RGB-D 深度相机
- Horizon Robotics RDK X5
- 平行夹爪
- 思岚 S3 激光雷达（可选）

---

# 软件环境

- Ubuntu 22.04
- ROS2 Humble
- MoveIt2
- OpenCV
- tf2
- cv_bridge
- Ultralytics YOLOv8
- Horizon Robotics DNN Runtime

---

# 系统流程

```
                 Gemini2 RGB-D 深度相机
                          │
                          ▼
                    RGB-D 图像采集
                          │
                          ▼
                YOLO目标检测（RDK X5 BPU）
                          │
                          ▼
                    用户选择抓取目标
                          │
                          ▼
                    RGB-D 三维定位
                          │
                          ▼
                     手眼坐标变换
                          │
                          ▼
                转换到 base_link 坐标系
                          │
                          ▼
                  MoveIt 运动规划
                          │
                          ▼
                 Alicia-D 六轴机械臂
                          │
                          ▼
                    平行夹爪闭合抓取
                          │
                          ▼
                  缓存抓取时关节姿态
                          │
                          ▼
                    返回观察初始姿态
                          │
                    等待释放指令
                          │
                          ▼
                  恢复抓取时关节姿态
                          │
                          ▼
                     打开夹爪释放物体
                          │
                          ▼
                    返回观察初始姿态
```

---

# 仓库结构

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

# 安装

克隆仓库

```bash
git clone https://github.com/xzh060613/RDKX5-EyeHand-Grasp.git
```

安装 ROS2 依赖

```bash
sudo apt install ros-humble-moveit
sudo apt install ros-humble-tf2-ros
sudo apt install ros-humble-cv-bridge
```

安装 Python 依赖

```bash
pip install ultralytics
pip install opencv-python
```

如果部署到 **RDK X5**，请按照地平线官方文档安装 **Horizon Robotics DNN Runtime**。

---

# 手眼标定

本项目采用 Alicia-D 官方提供的 Eye-in-Hand 手眼标定工具完成标定。

标定完成后会自动发布静态 TF：

```
gripper_center
        │
        ▼
camera_link
```

利用该变换关系将目标坐标从相机坐标系转换到机器人基坐标系。

---

# 使用方法

启动机械臂

```bash
ros2 launch alicia_d_moveit real_robot.launch.py
```

运行抓取节点

```bash
python3 yolo_eyehand_pregrasp_node.py
```

---

# 手动抓取流程

当前版本采用手动控制抓取方式。

## Step 1

机械臂自动运动到观察初始姿态。

当前初始关节角为：

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

YOLO 持续检测目标。

OpenCV 窗口实时显示检测结果，并用目标框标出可抓取物体。

---

## Step 3

按下 **G**

系统执行：

- 锁定当前目标
- 计算目标三维坐标
- 转换至机器人坐标系
- MoveIt 规划运动轨迹
- 机械臂执行抓取
- 夹爪闭合
- 保存抓取时机械臂关节姿态
- 返回观察初始姿态

---

## Step 4

移动机器人运输物体。

机械臂始终保持夹持状态，等待释放命令。

---

## Step 5

按下 **R**

系统执行：

- 恢复抓取时保存的关节姿态
- 打开夹爪释放物体
- 返回观察初始姿态

---

# 可配置参数

常用参数包括：

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

这些参数可根据不同机械臂安装方式及应用场景进行调整。

---

# 当前完成情况

当前版本已实现：

- ✅ Eye-in-Hand 手眼标定
- ✅ RGB-D 目标检测
- ✅ 手眼坐标转换
- ✅ 三维目标定位
- ✅ MoveIt 运动规划
- ✅ 自动抓取执行
- ✅ 平行夹爪控制
- ✅ OpenCV 实时显示
- ✅ G 键抓取
- ✅ R 键释放
- ✅ 抓取姿态缓存
- ✅ 返回观察初始姿态
- ✅ 工作空间限制
- ✅ 自动抓取重试
- ✅ RDK X5 平台部署
- ✅ RDK X5 BPU 推理加速

---

# 后续计划

- 快递盒专用检测模型
- 多目标选择
- 自动抓取姿态估计
- Navigation2 导航集成
- 移动机器人自主搬运
- 自动抓取与放置
- 语义抓取规划

---

# 引用

如果本项目对您的工作有所帮助，欢迎给本仓库点一个 ⭐ Star。

---

# 开源协议

本项目采用 **MIT License** 开源协议。
