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
from sdk import common
from rclpy.node import Node
from cv_bridge import CvBridge
from interfaces.srv import SetStringBool
from std_srvs.srv import Trigger, SetBool
from sensor_msgs.msg import Image, CameraInfo
from rclpy.executors import MultiThreadedExecutor
from servo_controller_msgs.msg import ServosPosition
from rclpy.callback_groups import ReentrantCallbackGroup
from kinematics.kinematics_control import set_pose_target
from kinematics_msgs.srv import GetRobotPose, SetRobotPose
from servo_controller.bus_servo_control import set_servo_position
from ros_robot_controller_msgs.srv import MoveStepper
from app.utils import calculate_grasp_yaw, position_change_detect, pick_and_place, image_process, distortion_inverse_map

TARGET_POSITION = {
    'red': (2600),
    'green': (4000),
    'blue': (5200),
}

class ColorSortingNode(Node):

    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)

        self.K = None
        self.D = None
        self.lock = threading.RLock()
        self.image_queue = queue.Queue(maxsize=2)
        self.bridge = CvBridge()
        self.imgpts = None
        self.tag_size = 0.025
        self.center_imgpts = None
        self.roi = None
        self.count_move = 0
        self.count_still = 0
        self.target_miss_count = 0
        self.white_area_width = 0.167
        self.white_area_height = 0.108
        self.calibration_file = 'calibration.yaml'
        self.config_file = 'transform.yaml'
        self.config_path = "/home/ubuntu/ros2_ws/src/example/example/stepper/config/"
        self.data = common.get_yaml_data("/home/ubuntu/ros2_ws/src/app/config/lab_config.yaml")  
        self.lab_data = self.data['/**']['ros__parameters'] 
        self.camera_type = os.environ['CAMERA_TYPE']
        self.min_area = 500
        self.max_area = 7000 
        self.target = None
        self.last_position = None
        self.start_transport = False

        self.image_sub = None
        self.camera_info_sub = None
        self.joints_pub = self.create_publisher(ServosPosition, '/servo_controller', 1)
        self.image_sub = self.create_subscription(Image, '/depth_cam/rgb/image_raw', self.image_callback, 1)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/depth_cam/rgb/camera_info', self.camera_info_callback, 1)

        self.timer_cb_group = ReentrantCallbackGroup()
        self.client = self.create_client(Trigger, '/controller_manager/init_finish', callback_group=self.timer_cb_group)
        self.client.wait_for_service()
        self.get_current_pose_client = self.create_client(GetRobotPose, '/kinematics/get_current_pose', callback_group=self.timer_cb_group)
        self.get_current_pose_client.wait_for_service()
        self.kinematics_client = self.create_client(SetRobotPose, '/kinematics/set_pose_target', callback_group=self.timer_cb_group)
        self.kinematics_client.wait_for_service()
        self.move_stepper_client = self.create_client(MoveStepper, '/ros_robot_controller/move_stepper', callback_group=self.timer_cb_group)
        self.move_stepper_client.wait_for_service()
        
        self.timer = self.create_timer(0.0, self.init_process, callback_group=self.timer_cb_group)

    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()

    def camera_info_callback(self, msg):
        K = np.matrix(msg.k).reshape(1, -1, 3)
        D = np.array(msg.d)
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (640, 400), 0, (640, 400))
        self.K, self.D = np.matrix(new_K), np.zeros((5, 1))

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

    def init_process(self):
        self.timer.cancel()
        req = MoveStepper.Request()
        req.steps = 0
        self.send_request(self.move_stepper_client, req)
        set_servo_position(self.joints_pub, 1.5, ((1, 210), (2, 500), (3, 110), (4, 825), (5, 600), (6, 500)))
        time.sleep(1.5)
        threading.Thread(target=self.main, daemon=True).start()
        threading.Thread(target=self.transport_thread, daemon=True).start()
        with open(self.config_path + self.config_file, 'r') as f:
            config = yaml.safe_load(f)
            corners = np.array(config['corners']).reshape(-1, 3)
            extristric = np.array(config['extristric'])
            white_area_pose_world = np.array(config['white_area_pose_world'])

        self.white_area_center = white_area_pose_world.reshape(4, 4)
        tvec = extristric[:1]
        rmat = extristric[1:]
        while self.K is None or self.D is None:
            time.sleep(0.5)

        center_imgpts, jac = cv2.projectPoints(corners[-1:], np.array(rmat), np.array(tvec), self.K, self.D)
        self.center_imgpts = np.int32(center_imgpts).reshape(2)

        tvec, rmat = common.extristric_plane_shift(np.array(tvec).reshape((3, 1)), np.array(rmat), 0.03)
        self.extristric = tvec, rmat
        tvec, rmat = common.extristric_plane_shift(np.array(tvec).reshape((3, 1)), np.array(rmat), 0.0)
        imgpts, jac = cv2.projectPoints(corners[:-1], np.array(rmat), np.array(tvec), self.K, self.D)
        self.imgpts = np.int32(imgpts).reshape(-1, 2)
        x_min = min(self.imgpts, key=lambda p: p[0])[0]
        x_max = max(self.imgpts, key=lambda p: p[0])[0]
        y_min = min(self.imgpts, key=lambda p: p[1])[1]
        y_max = max(self.imgpts, key=lambda p: p[1])[1]
        roi = np.maximum(np.array([y_min, y_max, x_min, x_max]), 0)
        self.roi = roi
 
    def image_processing(self):
        rgb_image = self.image_queue.get()
        result_image = np.copy(rgb_image)
        if self.center_imgpts is not None:
            cv2.line(result_image, (self.center_imgpts[0]-10, self.center_imgpts[1]), (self.center_imgpts[0]+10, self.center_imgpts[1]), (255, 255, 0), 2)
            cv2.line(result_image, (self.center_imgpts[0], self.center_imgpts[1]-10), (self.center_imgpts[0], self.center_imgpts[1]+10), (255, 255, 0), 2)
        target_list = []
        index = 0
        if self.roi is not None  and self.imgpts is not None and not self.start_transport :
            roi_area_mask = np.zeros(shape=(rgb_image.shape[0], rgb_image.shape[1], 1), dtype=np.uint8)
            roi_area_mask = cv2.drawContours(roi_area_mask, [self.imgpts], -1, 255, cv2.FILLED)
            rgb_image = cv2.bitwise_and(rgb_image, rgb_image, mask=roi_area_mask)
            roi_img = rgb_image[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
            roi_img = self.get_top_surface(roi_img)
            image_lab = cv2.cvtColor(roi_img, cv2.COLOR_RGB2LAB)
            img_h, img_w = rgb_image.shape[:2]

            for i in ['red', 'green', 'blue']:
                mask = cv2.inRange(image_lab, tuple(self.lab_data['color_range_list'][i]['min']), tuple(self.lab_data['color_range_list'][i]['max']))
                eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
                dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
                contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
                contours_area = map(lambda c: (math.fabs(cv2.contourArea(c)), c), contours)
                contours = map(lambda a_c: a_c[1], filter(lambda a: self.min_area <= a[0] <= self.max_area, contours_area))
                for c in contours:
                    rect = cv2.minAreaRect(c)
                    (center_x, center_y), _ = cv2.minEnclosingCircle(c)
                    center_x, center_y = self.roi[2] + center_x, self.roi[0] + center_y
                    cv2.circle(result_image, (int(center_x), int(center_y)), 8, (0, 0, 0), -1)
                    corners = list(map(lambda p: (self.roi[2] + p[0], self.roi[0] + p[1]), cv2.boxPoints(rect)))
                    cv2.drawContours(result_image, [np.intp(corners)], -1, (0, 255, 255), 2, cv2.LINE_AA)
                    index += 1
                    angle = int(round(rect[2]))
                    target_list.append([i, index, (center_x, center_y), (int(rect[1][0]), int(rect[1][1])), angle])
                cv2.drawContours(result_image, [self.imgpts], -1, (255, 255, 0), 2, cv2.LINE_AA) 
                for p in self.imgpts:
                    cv2.circle(result_image, tuple(p), 8, (255, 0, 0), -1)
                pass

        return target_list, result_image

    def get_object_world_position(self, position, intrinsic, extristric, white_area_center, height=0.03):
        self.get_logger().info(f"原始位置: {position}")  # 打印物体的初始位置
        projection_matrix = np.row_stack((np.column_stack((extristric[1], extristric[0])), np.array([[0, 0, 0, 1]])))
        world_pose = common.pixels_to_world([position], intrinsic, projection_matrix)[0]
        self.get_logger().info(f"转换后的世界坐标: {world_pose}")  # 打印转换后的世界坐标
        world_pose[0] = -world_pose[0]
        world_pose[1] = -world_pose[1]
        position = white_area_center[:3, 3] + world_pose
        position[2] = height
        
        config_data = common.get_yaml_data(os.path.join(self.config_path, self.calibration_file))
        offset = tuple(config_data['pixel']['offset'])
        scale = tuple(config_data['pixel']['scale'])
        for i in range(3):
            position[i] = position[i] * scale[i]
            position[i] = position[i] + offset[i]
        return position, projection_matrix

    def calculate_pick_grasp_yaw(self, position, target, target_info, intrinsic, projection_matrix):
        yaw = math.degrees(math.atan2(position[1], position[0]))
        self.get_logger().info(f"计算的角度 (yaw): {yaw}")  # 打印计算的角度
        if position[0] < 0 and position[1] < 0:
            yaw = yaw + 180
            self.get_logger().info(f"调整后的角度 (yaw): {yaw}")  # 打印调整后的角度
        elif position[0] < 0 and position[1] > 0:
            yaw = yaw - 180
        gripper_size = [common.calculate_pixel_length(0.09, intrinsic, projection_matrix),
                        common.calculate_pixel_length(0.02, intrinsic, projection_matrix)]
        return calculate_grasp_yaw.calculate_gripper_yaw_angle(target, target_info, gripper_size, yaw)

    def calculate_place_grasp_yaw(self, position, angle=0):
        yaw = math.degrees(math.atan2(position[1], position[0]))
        if position[0] < 0 and position[1] < 0:
            yaw = yaw + 180
        elif position[0] < 0 and position[1] > 0:
            yaw = yaw - 180
        yaw1 = yaw + angle
        if yaw < 0:
            yaw2 = yaw1 + 90
        else:
            yaw2 = yaw1 - 90
        yaw = yaw2
        if abs(yaw1) < abs(yaw2):
            yaw = yaw1
        yaw = 500 + int(yaw / 240 * 1000)
        return yaw

    def transport_thread(self):
        while True:
            if self.start_transport:
                position, yaw, target = self.transport_info
                if position[0] > 0.22:
                    position[2] += 0.01
                config_data = common.get_yaml_data(os.path.join(self.config_path, self.calibration_file))
                offset = tuple(config_data['kinematics']['offset'])
                scale = tuple(config_data['kinematics']['scale'])
                for i in range(3):
                    position[i] = position[i] * scale[i]
                    position[i] = position[i] + offset[i]
                finish = pick_and_place.pick(position, 80, yaw, 550, 0.02, self.joints_pub, self.kinematics_client)
                if finish:
                    stepper_position = TARGET_POSITION[self.target[0]]
                    req = MoveStepper.Request()  # 正确创建 MoveStepper 请求对象
                    req.steps = stepper_position  # 设置步进电机的步数
                    
                    # 发送请求给 MoveStepper 服务
                    self.send_request(self.move_stepper_client, req)
                    stepper_time = stepper_position / 1000  # 计算所需时间
                    time.sleep(stepper_time)
                    
                    # 继续后续操作
                    position = [0.25, 0, 0.015]
                    yaw = self.calculate_place_grasp_yaw(position, 0)
                    config_data = common.get_yaml_data(os.path.join(self.config_path, self.calibration_file))
                    offset = tuple(config_data['kinematics']['offset'])
                    scale = tuple(config_data['kinematics']['scale'])
                    angle = math.degrees(math.atan2(position[1], position[0]))
                    if angle > 45:
                        position = [position[0] * scale[1], position[1] * scale[0], position[2] * scale[2]]
                        position = [position[0] + offset[1], position[1] + offset[0], position[2] + offset[2]]
                    elif angle < -45:
                        position = [position[0] * scale[1], position[1] * scale[0], position[2] * scale[2]]
                        position = [position[0] + offset[1], position[1] - offset[0], position[2] + offset[2]]
                    else:
                        position = [position[0] * scale[0], position[1] * scale[1], position[2] * scale[2]]
                        position = [position[0] + offset[0], position[1] + offset[1], position[2] + offset[2]]
                    finish = pick_and_place.place(position, 80, 500, 200, self.joints_pub, self.kinematics_client)
                    if finish:
                        set_servo_position(self.joints_pub, 1.5, ((1, 210), (2, 500), (3, 110), (4, 825), (5, 600), (6, 500)))
                        time.sleep(1.5)
                        # 创建新的请求并发送
                        req = MoveStepper.Request()  # 正确创建 MoveStepper 请求对象
                        req.steps = -stepper_position  # 让步进电机回到原始位置
                        self.send_request(self.move_stepper_client, req)
                        stepper_time = stepper_position / 1000  # 计算所需时间
                        time.sleep(stepper_time)
                        self.target = None
                        self.start_transport = False
            else:
                time.sleep(0.1)


    def main(self):
        while True:
            target_list, result_image = self.image_processing()
            if target_list:
                target_miss = True 
                for target in target_list:
                    if self.target is not None :
                        if self.target[0] != target[0] or self.target[1] != target[1]:
                            continue
                        else:
                            target_miss = False
                            self.target = target
                    position, projection_matrix = self.get_object_world_position(target[2], self.K, self.extristric, self.white_area_center, 0.03)
                    result = self.calculate_pick_grasp_yaw(position, target, target_list, self.K, projection_matrix)
                    if result is not None and self.target is None:
                            self.target = target
                            break
                    if self.last_position is not None and self.target is not None :
                        e_distance = round(math.sqrt(pow(self.last_position[0] - position[0], 2)) + math.sqrt(pow(self.last_position[1] - position[1], 2)), 5)
                        if e_distance <= 0.003:
                            cv2.line(result_image, result[1][0], result[1][1], (255, 255, 0), 2, cv2.LINE_AA)
                            self.count_move = 0
                            self.count_still += 1
                        else:
                            self.count_move += 1
                            self.count_still = 0
                        if self.count_move > 5:
                            self.target = None
                        if self.count_still > 15:
                            self.count_still = 0
                            self.count_move = 0
                            self.target = target
                            yaw = 500 + int(result[0] / 240 * 1000)
                            self.transport_info = [position, yaw, target]
                            self.start_transport = True
                    self.last_position = position
                if target_miss:
                    self.target_miss_count += 1
                if self.target_miss_count > 5:
                    self.target_miss_count = 0
                    self.target = None
            if result_image is not None :
                result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                cv2.imshow('result_image', result_image)
                key = cv2.waitKey(1)

def main():
    node = ColorSortingNode('color_sorting')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
