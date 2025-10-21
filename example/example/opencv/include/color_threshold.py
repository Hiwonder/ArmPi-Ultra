#!/usr/bin/env python3
# encoding: utf-8
# 颜色阈值
import cv2
import yaml
import rclpy
import queue
import threading
import numpy as np
from rclpy.node import Node
import sdk.common as common
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class Camera(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name)
        self.name = name
        self.image = None
        self.running = True
        self.image_sub = None
        self.thread_started = False

        self.image_queue = queue.Queue(maxsize=2)
        self.data = common.get_yaml_data("/home/ubuntu/ros2_ws/src/app/config/lab_config.yaml")
        self.lab_data = self.data['/**']['ros__parameters']
        # 创建 CvBridge 对象，用于将 ROS Image 转为 OpenCV 格式
        self.bridge = CvBridge()
        self.target_color = "red"  # 设置目标颜色

        # 图像订阅者
        self.image_sub = self.create_subscription(Image, '/depth_cam/rgb/image_raw', self.image_callback, 1)

    # 处理 ROS Image 消息并转化为 OpenCV 格式
    def image_callback(self, ros_image):
        # 将 ROS Image 消息转换为 OpenCV 图像
        bgr_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        if self.image_queue.full():
            # 如果队列已满，丢弃最旧的图像
            try:
                self.image_queue.get_nowait() # 使用 get_nowait 避免阻塞
            except queue.Empty:
                pass # 如果队列在检查时已经为空，则忽略异常
        # 将图像放入队列
        try:
            self.image_queue.put_nowait(bgr_image)  # 使用 put_nowait 避免阻塞
        except queue.Full:
            pass # 如果队列在检查时已经为满，则忽略异常
        if not self.thread_started:
            # 只在第一次调用时启动线程
            threading.Thread(target=self.image_process, daemon=True).start()
            self.thread_started = True

    def image_process(self):
        while self.running:
            try:
                bgr_image = self.image_queue.get(block=True, timeout=1) # 使用 get 时添加 try-except
            except queue.Empty:
                #print(self.get_name(), "image queue is empty!")
                continue 

            # 将图像的颜色空间转换成LAB
            lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)

            # 设置颜色阈值并进行二值化,默认为红色
            result_image = cv2.inRange(lab_image, np.array(self.lab_data['color_range_list'][self.target_color]['min']), np.array(self.lab_data['color_range_list'][self.target_color]['max']))

            # 发布处理后的图像
            try:
                cv2.imshow('bgr_image', bgr_image)
                cv2.imshow('lab_image', lab_image)
                cv2.imshow('result_image', result_image)
                cv2.waitKey(1)
            except Exception as e:
                self.get_logger().warn(f"图像显示发生错误: {e}")
                continue
        print("image_process thread end")


    def destroy_node(self):
        self.running = False
        print("destroy node")
        super().destroy_node()
        #cv2.destroyAllWindows()

def main():
    camera_node = Camera('color_threshold')
    rclpy.spin(camera_node)
    camera_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
