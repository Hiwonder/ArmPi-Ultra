# ArmPi Ultra

[English](https://github.com/Hiwonder/ArmPi-Ultra/blob/main/README.md) | 中文

<p align="center">
  <img src="./sources/images/armpi-ultra.png" alt="ArmPi Ultra Logo" width="600"/>
</p>

基于ROS2的桌面级AI三维视觉机械臂，集成3D深度相机、高扭矩智能总线舵机、多模态AI大模型，支持三维空间精准抓取、物品分拣、双臂协作等高阶AI应用。

## 产品介绍

ArmPi Ultra是幻尔科技面向ROS教育场景开发的桌面级AI三维视觉机械臂，它搭载了3D深度相机，将3D视觉技术与机械臂控制相结合，并配备了6个大扭矩智能总线舵机、树莓派5主控、多功能扩展板、AI语音交互盒等高性能硬件，通过内置的高阶运动学算法，不仅可以实现机器人三维运动控制，还能在三维空间内精准识别、追踪、抓取目标物品。

ArmPi Ultra部署了多模态AI大模型，结合3D视觉、语音或文本，使其可以深度理解环境、规划行动并灵活执行任务，能实现更多高阶具身智能应用。通过树莓派主板的CPU，可以进行机器学习模型训练，能够轻松实现垃圾、物品、人脸、人体特征识别等各种AI创意项目。

ArmPi Ultra采用精巧的模块化设计，使其支持多模态扩展，结合麦克纳姆轮可变身为移动搬运机器人，从而实现移动抓取、智能搬运等功能。ArmPi Ultra还支持电动滑轨及电动传送带拓展，可以自由搭建双臂协作、物品分拣等自动化项目，可适用于多样化应用场景!

## 官方资源

### Hiwonder官方
- **官方网站**: [https://www.hiwonder.net/](https://www.hiwonder.net/)
- **产品页面**: [https://www.hiwonder.com/products/armpi-ultra](https://www.hiwonder.com/products/armpi-ultra)
- **官方文档**: [https://docs.hiwonder.com/projects/ArmPi-Ultra/en/latest/](https://docs.hiwonder.com/projects/ArmPi-Ultra/en/latest/)
- **技术支持**: support@hiwonder.com

## 主要功能

### AI视觉与识别
- **3D深度视觉** - 先进的三维物体检测和空间感知
- **物品识别** - 基于机器学习的物体分类
- **人脸识别** - 全面的人脸检测和识别
- **垃圾分类** - AI驱动的垃圾分类和分拣
- **颜色追踪** - 实时基于颜色的目标追踪
- **形状识别** - 几何形状检测和分析

### 高级操控
- **三维空间抓取** - 三维空间内精准物体抓取
- **物品分拣** - 自动化物品分类和分拣
- **目标追踪** - 实时目标追踪和操作
- **标签堆叠** - 基于AprilTag的堆叠操作
- **手指追踪** - 手势追踪和跟随
- **标定系统** - 先进的相机和机械臂标定工具

### 智能控制
- **逆运动学** - 先进的运动规划算法
- **ROS2集成** - 完整的机器人操作系统2支持
- **舵机控制** - 高精度总线舵机管理
- **多模态扩展** - 支持麦克纳姆轮、滑轨和传送带
- **语音交互** - AI驱动的自然语言命令
- **仿真支持** - 完整的开发和测试环境

### 编程接口
- **Python编程** - 全面的Python SDK
- **ROS2功能包** - 完整的ROS2包生态系统
- **多模态AI模型** - 先进的具身AI能力
- **YOLOv8检测** - 最先进的目标检测框架
- **开源平台** - 完整的开源平台支持定制化

## 硬件配置
- **处理器**: 树莓派5
- **操作系统**: ROS2兼容Linux系统
- **视觉系统**: 高清3D深度相机
- **舵机**: 6个大扭矩智能总线舵机
- **扩展板**: 多功能扩展板
- **AI集成**: 多模态AI大模型配AI语音交互盒
- **扩展支持**: 麦克纳姆轮、电动滑轨、电动传送带

## 项目结构

```
armpi_ultra/
├── app/                    # 应用模块
│   ├── calibration.py      # 相机和机械臂标定
│   ├── color_tracker.py    # 颜色追踪应用
│   ├── face_tracker.py     # 人脸追踪应用
│   ├── finger_trace.py     # 手指追踪应用
│   ├── grasp.py            # 抓取控制
│   ├── object_sorting.py   # 物品分拣应用
│   ├── object_tracking.py  # 目标追踪应用
│   ├── shape_recognition.py # 形状识别应用
│   ├── tag_stackup.py      # 基于标签的堆叠应用
│   └── waste_classification.py # 垃圾分类应用
├── bringup/               # 系统启动和配置
├── driver/                # 硬件驱动
│   ├── controller/        # 机械臂控制器
│   ├── kinematics/        # 运动学算法
│   ├── servo_controller/  # 舵机控制驱动
│   └── sdk/               # 硬件SDK
├── example/               # 示例应用和演示
├── interfaces/            # ROS2消息定义
├── large_models/          # AI大模型集成
├── large_models_msgs/     # 大模型消息定义
├── peripherals/           # 外设支持
├── simulations/           # 仿真环境
└── yolov8_detect/        # YOLOv8目标检测集成
```

## 版本信息
- **当前版本**: ArmPi Ultra v1.0.0
- **支持平台**: 树莓派5

### 相关技术
- [ROS2](https://ros.org/) - 机器人操作系统2
- [OpenCV](https://opencv.org/) - 计算机视觉库
- [YOLOv8](https://github.com/ultralytics/ultralytics) - 目标检测框架
- [MoveIt](https://moveit.ros.org/) - 运动规划框架

---

**注**: 所有程序已预装在ArmPi Ultra机器人系统中，可直接运行。详细使用教程请参考[官方文档](https://docs.hiwonder.com/projects/ArmPi-Ultra/en/latest/)。
