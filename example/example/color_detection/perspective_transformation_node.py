#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/02/21
# @author:aiden
# 订阅摄像头图像，检测白色识别区域进行透视变换处理，发布处理后的图像

import cv2
import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import Empty
import threading
import numpy as np
import pandas as pd
from sdk import common

CONFIG_NAME = '/config'

class PerspectiveTransformationNode(Node):
    def __init__(self):
        super().__init__('perspective_transformation')
        self.lock = threading.RLock()

        self.config = self.get_parameter(CONFIG_NAME).get_parameter_value().string_value
        self.lab_data = self.config['lab']

        # roi区域，先缩放再截取
        self.roi = [self.config['roi']['height'][0],
                    self.config['roi']['height'][1],
                    self.config['roi']['width'][0],
                    self.config['roi']['width'][1]]

        # 图像缩放比例系数(0, 1)，加速识别
        self.size = self.config['image_proc_size']
        
        self.src_points = None
        scale = self.config['white_area_world_size']['height']/self.config['white_area_world_size']['width']
        width = self.config['white_area_pixel_size']['width']
        camera_resolution = self.config['camera_resolution']
        camera_width = camera_resolution['width']
        camera_height = camera_resolution['height']
        self.dst_points = np.float32(
            [[int((camera_width - width)/2), int((camera_height + width*scale)/2)],
             [int((camera_width - width)/2), int((camera_height - width*scale)/2)],
             [int((camera_width + width)/2), int((camera_height - width*scale)/2)],
             [int((camera_width + width)/2), int((camera_height + width*scale)/2)]])

        self.left_top_x = []
        self.left_top_y = []
        self.right_top_x = []
        self.right_top_y = []
        self.left_down_x = []
        self.left_down_y = []
        self.right_down_x = []
        self.right_down_y = []
        
        self.yaw = 0
        self.count = 0
        self.save = True
        self.bgr_image = None
        self.running = True
        self.image_sub = None
        self.tag_sub = None
        self.start_calibration = False

        self.declare_parameter('debug', False)

        # 检测图像发布
        self.image_pub = self.create_publisher(Image, '~image_result', 10)

        self.create_service(Empty, '~enter', self.enter_func)
        self.create_service(Empty, '~exit', self.exit_func)
        self.create_service(Empty, '~start', self.start_func)
        self.create_service(Empty, '~stop', self.stop_func)
        self.create_service(Empty, '~save', self.save_func)
        self.create_service(Empty, '~update_param', self.update_param)

        self.get_logger().info("PerspectiveTransformationNode init finish")
        self.image_proc()

    def shutdown(self):
        self.running = False

    def update_param(self, request, response):
        self.config = self.get_parameter(CONFIG_NAME).get_parameter_value().string_value
        self.lab_data = self.config['lab']

        # roi区域，先缩放再截取
        self.roi = [self.config['roi']['height'][0],
                    self.config['roi']['height'][1],
                    self.config['roi']['width'][0],
                    self.config['roi']['width'][1]]

        # 图像缩放比例系数(0, 1)，加速识别
        self.size = self.config['image_proc_size']
        
        scale = self.config['white_area_world_size']['height']/self.config['white_area_world_size']['width']
        width = self.config['white_area_pixel_size']['width']
        camera_resolution = self.config['camera_resolution']
        camera_width = camera_resolution['width']
        camera_height = camera_resolution['height']
        self.dst_points = np.float32(
            [[int((camera_width - width)/2), int((camera_height + width*scale)/2)],
             [int((camera_width - width)/2), int((camera_height - width*scale)/2)],
             [int((camera_width + width)/2), int((camera_height - width*scale)/2)],
             [int((camera_width + width)/2), int((camera_height + width*scale)/2)]])
        self.get_logger().info('update param')
        return response

    def enter_func(self, request, response):
        if self.image_sub is None:
            camera_name = self.get_parameter('/camera/camera_name', 'camera')  # 获取参数
            image_topic = self.get_parameter('/camera/image_topic', 'image_rect_color')  # 获取参数
            self.image_sub = self.create_subscription(Image, f'/{camera_name}/{image_topic}', self.image_callback, 10)  # 订阅原始图像
        self.get_logger().info("enter")
        return response

    def exit_func(self, request, response):
        if self.image_sub is not None:
            self.image_sub.destroy()
            self.image_sub = None
        self.get_logger().info('exit')
        return response

    def start_func(self, request, response):
        with self.lock:
            self.count = 0
            self.save = True
            self.src_points = None
            self.start_calibration = True
        self.get_logger().info("start")
        return response

    def stop_func(self, request, response):
        with self.lock:
            self.start_calibration = False
            self.src_points = None
        self.get_logger().info("stop")
        return response
    
    def save_func(self, request, response):
        with self.lock:
            self.save = True
        
        return response

    def image_proc(self):
        while self.running:
            if self.bgr_image is not None:
                warped_image = self.bgr_image.copy()
                if self.start_calibration:  # 如果开启检测
                    bgr_image = self.bgr_image.copy()
                    img_h, img_w = bgr_image.shape[:2]  # 获取原图大小
                    image_resize = cv2.resize(bgr_image, (self.size['width'], self.size['height']), interpolation=cv2.INTER_NEAREST)  # 图像缩放, 加快图像处理速度, 不能太小，否则会丢失大量细节
                    # 计算缩放后的roi
                    roi = [int(self.roi[0]*self.size['height']),
                           int(self.roi[1]*self.size['height']),
                           int(self.roi[2]*self.size['width']),
                           int(self.roi[3]*self.size['width'])]

                    roi_image = image_resize[roi[0]:roi[1], roi[2]:roi[3]]  # 截取roi

                    image_lab = cv2.cvtColor(roi_image, cv2.COLOR_BGR2LAB)
                    lower = self.lab_data['black']['min']
                    upper = self.lab_data['black']['max']
                    binary = cv2.inRange(image_lab, tuple(lower), tuple(upper))
                    
                    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    eroded = cv2.erode(binary, element)  # 腐蚀
                    dilated = 255 - cv2.dilate(eroded, element)  # 膨胀
                    
                    contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]  # 找出所有轮廓
                    max_contour = common.get_area_max_contour(contours, 10)[0]
                    approx = cv2.approxPolyDP(max_contour, 75, True)
                    for j in range(4):
                        approx[:, 0][j, 0], approx[:, 0][j, 1] = common.point_remapped([approx[:, 0][j, 0] + roi[2], approx[:, 0][j, 1] + roi[0]],
                                                                      [self.size['width'], self.size['height']], [img_w, img_h])  # 点映射到原图大小
                    
                    cv2.polylines(bgr_image, [approx], True, (255, 0, 255), 2)

                    corners = np.argpartition(approx[..., 0][:, 0], -2)
                    left = approx[corners[:2]][:, 0]
                    right = approx[corners[-2:]][:, 0]
                    if left[0, 1] < left[1, 1]:
                        left_top = left[0, :]
                        left_down = left[1, :]
                    else:
                        left_top = left[1, :]
                        left_down = left[0, :]

                    if right[0, 1] < right[1, 1]:
                        right_top = right[0, :]
                        right_down = right[1, :]
                    else:
                        right_top = right[1, :]
                        right_down = right[0, :]
                    
                    if len(approx) == 4:
                        self.count += 1
                    if self.count > 10:
                        self.right_top_x.append(right_top[0])
                        self.right_top_y.append(right_top[1])
                        self.left_top_x.append(left_top[0])
                        self.left_top_y.append(left_top[1])
                        self.left_down_x.append(left_down[0])
                        self.left_down_y.append(left_down[1])
                        self.right_down_x.append(right_down[0])
                        self.right_down_y.append(right_down[1])
                    if self.count > 15:
                        right_top_point = [self.average_process(self.right_top_x), self.average_process(self.right_top_y)]
                        right_down_point = [self.average_process(self.right_down_x), self.average_process(self.right_down_y)]
                        left_top_point = [self.average_process(self.left_top_x), self.average_process(self.left_top_y)]
                        left_down_point = [self.average_process(self.left_down_x), self.average_process(self.left_down_y)]
                        # left_down
                        # left_up
                        # right_up
                        # right_down
                        self.src_points = np.float32(
                            [left_down_point,
                            left_top_point,
                            right_top_point,
                            right_down_point])
                        
                        self.count = 0
                        self.save = True
                        self.start_calibration = False
                        self.get_logger().info('finish calibration')
                    if self.debug:
                        cv2.imshow('bgr_image', bgr_image)
                else:
                    if self.src_points is not None:
                        warped_image, matrix, matrix_inv = common.perspective_transform(self.bgr_image, self.src_points, self.dst_points, True)
                        cv2.rectangle(warped_image, tuple(self.dst_points[0, :]), tuple(self.dst_points[2, :]), (255, 255, 0), 2)
                        if self.save:
                            self.config['perspective_transformation_matrix'] = matrix.tolist()
                            self.config['perspective_transformation_matrix_inv'] = matrix_inv.tolist()
                            self.save_yaml_data(self.config, '/home/hiwonder/ros_ws/src/hiwonder_imgproc/color_detection/config/config.yaml') 
                            self.save = False
                            self.get_logger().info('save config')
                    rclpy.sleep(0.01)
                ros_image = common.cv2_image2ros(warped_image, self.name)
                self.image_pub.publish(ros_image)  # 发布图像
                
                if self.debug:
                    cv2.imshow('warped_image', warped_image)
                    cv2.waitKey(1)
            else:
                rclpy.sleep(0.01)

    def image_callback(self, ros_image):
        rgb_image = np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=np.uint8,
                           buffer=ros_image.data)  # 将ros格式图像消息转化为opencv格式
        self.bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)


def main(args=None):
    rclpy.init(args=args)
    perspective_transformation_node = PerspectiveTransformationNode()
    rclpy.spin(perspective_transformation_node)
    perspective_transformation_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
