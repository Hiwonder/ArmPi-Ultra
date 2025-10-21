#!/usr/bin/env python3
# encoding: utf-8
# 物体位姿计算
import os
import cv2
import yaml
import time
import rclpy
import numpy as np
from sdk import common
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image as RosImage, CameraInfo

class CoordinateTransformation(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(
            name,
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True
        )
        self.bridge = CvBridge()  # ROS Image和OpenCV图像转换桥梁
        self.K = None  # 相机内参矩阵
        self.white_area_center = None  # 白色区域中心点（世界坐标系）
        self.camera_type = os.environ.get('CAMERA_TYPE', '')  # 读取CAMERA_TYPE环境变量，若无则空字符串
        self.config_file = 'transform.yaml'  # 位姿计算配置文件名
        self.calibration_file = 'calibration.yaml'  # 校准文件名
        self.config_path = "/home/ubuntu/ros2_ws/src/app/config/"  # 配置文件目录

        # 订阅图像结果话题和相机内参话题
        self.image_sub = self.create_subscription(RosImage, '/color_detection/result_image', self.image_callback, 1)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/depth_cam/rgb/camera_info', self.camera_info_callback, 1)

    def get_yaml(self):
        """读取配置文件，获取相机外参和白色区域世界坐标"""
        with open(os.path.join(self.config_path, self.config_file), 'r') as f:
            config = yaml.safe_load(f)
            extristric = np.array(config['extristric'])
            corners = np.array(config['corners']).reshape(-1, 3)  # 角点，目前没有用到
            self.white_area_center = np.array(config['white_area_pose_world'])

        tvec = extristric[:1]  # 第一行平移向量
        rmat = extristric[1:]  # 剩余三行旋转矩阵

        # 调用外参平移偏移计算函数，要求传入np.matrix类型
        tvec, rmat = common.extristric_plane_shift(
            np.matrix(tvec).reshape((3, 1)),
            np.matrix(rmat),
            0.03
        )

        self.extristric = tvec, rmat

    def camera_info_callback(self, msg):
        """收到相机内参消息时触发，转换成3x3矩阵"""
        # 这里用np.matrix，确保调用common时不会报.I错
        self.K = np.matrix(msg.k).reshape(3, 3)

    def image_callback(self, result_image):
        """收到图像消息时触发，识别目标，计算实际位置"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(result_image, "mono8")  # 转为灰度图

            if cv_image is not None:
                contours = cv2.findContours(cv_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]

                if contours:
                    # 找最大轮廓
                    c = max(contours, key=cv2.contourArea)
                    rect = cv2.minAreaRect(c)
                    corners = np.int0(cv2.boxPoints(rect))
                    x, y = rect[0][0], rect[0][1]

                    if self.camera_type == 'usb_cam':
                        # 这里可填usb_cam专用校正逻辑，先留空避免报错
                        pass

                    # 读取外参配置
                    self.get_yaml()

                    # 组装投影矩阵，记得转成np.matrix
                    projection_matrix = np.row_stack((
                        np.column_stack((self.extristric[1], self.extristric[0])),
                        np.array([[0, 0, 0, 1]])
                    ))
                    projection_matrix = np.matrix(projection_matrix)

                    # 将像素坐标转换到世界坐标系，传入np.matrix类型参数
                    world_pose = common.pixels_to_world([[x, y]], self.K, projection_matrix)[0]

                    # 做坐标系调整
                    world_pose[0] = -world_pose[0]
                    world_pose[1] = -world_pose[1]

                    # 加上白区中心坐标，计算真实坐标
                    position = self.white_area_center[:3, 3] + world_pose
                    position[2] = 0.03  # 固定高度

                    # 读取标定参数做缩放和位移
                    config_data = common.get_yaml_data(os.path.join(self.config_path, self.calibration_file))
                    offset = tuple(config_data['pixel']['offset'])
                    scale = tuple(config_data['pixel']['scale'])
                    for i in range(3):
                        position[i] = position[i] * scale[i] + offset[i]

                    # 打印像素及实际坐标
                    self.get_logger().info(f"像素坐标为: x: {x:.2f}, y: {y:.2f}")
                    self.get_logger().info(f"实际坐标为： x={position[0]:.4f}, y={position[1]:.4f}, z={position[2]:.4f}")
                    time.sleep(1)
                else:
                    self.get_logger().info("未检测到所需识别的颜色，请将色块放置到相机视野内。")

        except Exception as e:
            self.get_logger().error(f"image_callback异常: {e}")

def main():
    node = CoordinateTransformation('coordinate_transformation')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
