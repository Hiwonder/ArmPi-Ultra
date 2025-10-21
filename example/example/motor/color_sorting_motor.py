#!/usr/bin/env python3
# coding: utf8

import os
import cv2
import yaml
import time
import math
import queue
import rclpy
import threading
import numpy as np
from rclpy.node import Node
from sdk import common, fps
from cv_bridge import CvBridge
from std_srvs.srv import SetBool
from sensor_msgs.msg import Image
from rclpy.executors import MultiThreadedExecutor
from servo_controller_msgs.msg import ServosPosition
from rclpy.callback_groups import ReentrantCallbackGroup
from kinematics.kinematics_control import set_pose_target
from kinematics_msgs.srv import GetRobotPose, SetRobotPose
from ros_robot_controller_msgs.msg import ConveyorSpeed
from servo_controller.bus_servo_control import set_servo_position

PLACE_POSITION = {
    'red': [0.15, -0.08, 0.01],
    'green': [0.15, 0.0, 0.01],
    'blue': [0.15, 0.08, 0.01],
}

class ColorSortingNode(Node):

    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)

        self.running = True
        self.start_count = False
        self.lock = threading.RLock()
        self.image_queue = queue.Queue(maxsize=2)
        self.fps = fps.FPS()
        self.bridge = CvBridge()
        self.center_imgpts = None
        self.offset = 0.005
        self.pick_pitch = 60
        self.count = 0
        self.data = common.get_yaml_data("/home/ubuntu/ros2_ws/src/app/config/lab_config.yaml")
        self.lab_data = self.data['/**']['ros__parameters']
        self.camera_type = os.environ['CAMERA_TYPE']
        self.min_area = 500
        self.max_area = 15000
        self.target = None
        self.image_sub = None

        self.conveyor_pub = self.create_publisher(ConveyorSpeed, '/ros_robot_controller/set_conveyor_speed', 1)
        self.joints_pub = self.create_publisher(ServosPosition, '/servo_controller', 1)
        self.image_sub = self.create_subscription(Image, '/depth_cam/rgb/image_raw', self.image_callback, 1)

        timer_cb_group = ReentrantCallbackGroup()
        self.get_current_pose_client = self.create_client(GetRobotPose, '/kinematics/get_current_pose', callback_group=timer_cb_group)
        self.get_current_pose_client.wait_for_service()
        self.kinematics_client = self.create_client(SetRobotPose, '/kinematics/set_pose_target')
        self.kinematics_client.wait_for_service()

        set_servo_position(self.joints_pub, 1.5, ((1, 210), (2, 500), (3, 110), (4, 825), (5, 600), (6, 880)))
        threading.Thread(target=self.main, daemon=True).start()

    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()

    def adaptive_threshold(self, gray_image):
        binary = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 7)
        return binary

    def canny_proc(self, bgr_image):
        mask = cv2.Canny(bgr_image, 9, 41, 9, L2gradient=True)
        mask = 255 - cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))
        return mask

    def get_top_surface(self, rgb_image):
        image_scale = cv2.convertScaleAbs(rgb_image, alpha=2.5, beta=0)
        image_gray = cv2.cvtColor(image_scale, cv2.COLOR_RGB2GRAY)
        image_mb = cv2.medianBlur(image_gray, 3)
        image_gs = cv2.GaussianBlur(image_mb, (5, 5), 5)
        binary = self.adaptive_threshold(image_gs)
        mask = self.canny_proc(image_gs)
        mask1 = cv2.bitwise_and(binary, mask)
        roi_image_mask = cv2.bitwise_and(rgb_image, rgb_image, mask=mask1)
        return roi_image_mask

    def image_callback(self, ros_image):
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
        rgb_image = np.array(cv_image, dtype=np.uint8)
        if self.image_queue.full():
            self.image_queue.get()
        self.image_queue.put(rgb_image)

    def image_processing(self):
        rgb_image = self.image_queue.get()
        result_image = np.copy(rgb_image)
        target_list = []
        index = 0
        if self.start_count:
            img_h, img_w = rgb_image.shape[:2]
            rgb_image = self.get_top_surface(rgb_image)
            image_lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
            for i in ['red', 'green', 'blue']:
                mask = cv2.inRange(image_lab, tuple(self.lab_data['color_range_list'][i]['min']), tuple(self.lab_data['color_range_list'][i]['max']))
                eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
                dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
                contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
                contours_area = map(lambda c: (math.fabs(cv2.contourArea(c)), c), contours)
                contours = map(lambda a_c: a_c[1], filter(lambda a: self.min_area <= a[0] <= self.max_area, contours_area))
                for c in contours:
                    area = math.fabs(cv2.contourArea(c))
                    rect = cv2.minAreaRect(c)
                    (center_x, center_y), _ = cv2.minEnclosingCircle(c)
                    cv2.circle(result_image, (int(center_x), int(center_y)), 8, (0, 0, 0), -1)
                    corners = list(map(lambda p: (p[0], p[1]), cv2.boxPoints(rect)))
                    cv2.drawContours(result_image, [np.intp(corners)], -1, (0, 255, 255), 2, cv2.LINE_AA)
                    index += 1
                    angle = int(round(rect[2]))
                    target_list.append([i, index, (center_x, center_y), angle])
        return target_list, result_image

    def move(self, center_y):
        if self.running:
            y = ((center_y - 235) / (335 - 235)) * (0.11 - 0.16) + 0.16
            y = y + self.offset
            position = [0, y, 0.08]

            if self.running:
                position[2] += 0.01
                self.get_logger().info('0') 
                msg = set_pose_target(position, self.pick_pitch, [-180.0, 180.0], 1.0)
                res = self.send_request(self.kinematics_client, msg)
                servo_data = res.pulse
                set_servo_position(self.joints_pub, 1.0, ((6, servo_data[0]),))
                time.sleep(1)
                set_servo_position(self.joints_pub, 1.0, ((5, servo_data[1]), (3, servo_data[2]), (3, servo_data[3])))
                time.sleep(1)
                self.get_logger().info('0.0') 

            if self.running:
                self.get_logger().info('1') 
                # position[2] -= 0.03
                msg = set_pose_target(position, self.pick_pitch, [-90.0, 90.0], 1.0)
                res = self.send_request(self.kinematics_client, msg)
                servo_data = res.pulse
                set_servo_position(self.joints_pub, 1.0, ((6, servo_data[0]), (5, servo_data[1]), (4, servo_data[2]), (3, servo_data[3])))
                time.sleep(1)
                set_servo_position(self.joints_pub, 1.0, ((1, 600),))
                time.sleep(1)

            if self.running:
                self.get_logger().info('2') 
                # position[2] += 0.04
                msg = set_pose_target(position, self.pick_pitch, [-180.0, 180.0], 1.0)
                res = self.send_request(self.kinematics_client, msg)
                servo_data = res.pulse
                self.get_logger().info('3') 
                set_servo_position(self.joints_pub, 1.0, ((5, servo_data[1]), (4, servo_data[2]), (3, servo_data[3])))
                time.sleep(1)
                self.get_logger().info('4') 
                set_servo_position(self.joints_pub, 1.0, ((6, 850), (5, 630), (4, 810), (3, 85), (2, 480)))
                time.sleep(1)
                position = PLACE_POSITION[self.target[0]]

            if self.running:
                self.get_logger().info('5') 
                position[2] += 0.03
                msg = set_pose_target(position, self.pick_pitch, [-90.0, 90.0], 1.0)
                res = self.send_request(self.kinematics_client, msg)
                servo_data = res.pulse
                set_servo_position(self.joints_pub, 2.0, ((6, servo_data[0]),))
                time.sleep(2)
                set_servo_position(self.joints_pub, 1.0, ((5, servo_data[1]), (4, servo_data[2]), (3, servo_data[3])))
                time.sleep(1)
                self.get_logger().info('6') 
            if self.running:
                position[2] -= 0.03
                msg = set_pose_target(position, self.pick_pitch, [-90.0, 90.0], 1.0)
                res = self.send_request(self.kinematics_client, msg)
                servo_data = res.pulse
                set_servo_position(self.joints_pub, 1.0, ((6, servo_data[0]), (5, servo_data[1]), (4, servo_data[2]), (3, servo_data[3])))
                time.sleep(1)
                set_servo_position(self.joints_pub, 1.0, ((1, 200),))
                time.sleep(1)

            if self.running:
                set_servo_position(self.joints_pub, 1.0, ((5, 630), (4, 810), (3, 90), (2, 480), (1, 200)))
                time.sleep(1)
                set_servo_position(self.joints_pub, 2.0, ((6, 850),))
                time.sleep(2)
                self.target = None
                self.start_count = True

            if not self.running:
                set_servo_position(self.joints_pub, 1.0, ((5, 630), (4, 810), (3, 90), (2, 480), (1, 200)))
                time.sleep(1)
                set_servo_position(self.joints_pub, 2.0, ((6, 850),))
                time.sleep(2)
                self.target = None

    def main(self):
        while rclpy.ok():
            try:
                target_list, result_image = self.image_processing()
                if target_list and self.running:
                    if self.target is not None:
                        if self.target[0] == target_list[0][0]:
                            self.count += 1
                        else:
                            self.target = target_list[0]
                            self.count = 0
                    else:
                        self.target = target_list[0]
                if self.count > 50:
                    self.count = 0
                    self.start_count = False
                    threading.Thread(target=self.move, args=(target_list[0][2][0],), daemon=True).start()

                if result_image is not None:
                    self.fps.update()
                    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                    cv2.imshow('result_image', result_image)
                    key = cv2.waitKey(1)
                    if key != -1:
                        if key == 97:  # a
                            self.running = True
                            self.start_count = True
                            conveyor_msg = ConveyorSpeed()
                            conveyor_msg.speed = 100
                            self.conveyor_pub.publish(conveyor_msg)
                        if key == 115:  # s
                            self.running = False
                            self.count = 0
                            conveyor_msg = ConveyorSpeed()
                            conveyor_msg.speed = 0
                            self.conveyor_pub.publish(conveyor_msg)
            except queue.Empty:
                continue

def main():
    try:
        node = ColorSortingNode('color_sorting')
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if 'node' in locals() and rclpy.ok():
             node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
