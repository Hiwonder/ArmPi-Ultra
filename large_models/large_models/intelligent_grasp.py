#!/usr/bin/env python3
# encoding: utf-8
# @Author: Aiden
# @Date: 2024/11/18
import os
import cv2
import math
import yaml
import time
import torch
import queue
import rclpy
import threading
import numpy as np
import sdk.fps as fps
import message_filters
from sdk import common
from rclpy.node import Node
from std_msgs.msg import String, Float32
from std_srvs.srv import Trigger, SetBool, Empty
from sensor_msgs.msg import Image, CameraInfo
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from ultralytics.models.fastsam import FastSAMPredictor
from tf2_ros import Buffer, TransformListener, TransformException
from ultralytics.utils.ops import scale_masks


from speech import speech
from app.common import Heart
from large_models_msgs.srv import SetString, SetModel, SetBox
from kinematics_msgs.srv import SetRobotPose, SetJointValue
from kinematics.kinematics_control import set_pose_target, set_joint_value_target
from servo_controller.bus_servo_control import set_servo_position
from servo_controller_msgs.msg import ServosPosition, ServoPosition
from app.utils import utils, image_process, calculate_grasp_yaw_by_depth, pick_and_place

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def prompt(results, bboxes=None, points=None, labels=None, texts=None):
    if bboxes is None and points is None and texts is None:
        return results
    prompt_results = []
    if not isinstance(results, list):
        results = [results]
    for result in results:
        if len(result) == 0:
            prompt_results.append(result)
            continue
        masks = result.masks.data
        if masks.shape[1:] != result.orig_shape:
            masks = scale_masks(masks[None], result.orig_shape)[0]
        idx = torch.zeros(len(result), dtype=torch.bool, device=device)
        if bboxes is not None:
            bboxes = torch.as_tensor(bboxes, dtype=torch.int32, device=device)
            bboxes = bboxes[None] if bboxes.ndim == 1 else bboxes
            bbox_areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
            mask_areas = torch.stack([masks[:, b[1] : b[3], b[0] : b[2]].sum(dim=(1, 2)) for b in bboxes])
            full_mask_areas = torch.sum(masks, dim=(1, 2))
 
            u = mask_areas / full_mask_areas  
            u = torch.nan_to_num(u, nan=0.0) 
            indices = (u >= (torch.max(u) - 0.1)).nonzero(as_tuple=True)[1] 
            u1 = full_mask_areas / bbox_areas
            max_index = indices[torch.argmax(u1[indices])]
            idx[max_index] = True

        prompt_results.append(result[idx])

    return prompt_results

class IntelligentGrasp(Node):
    hand2cam_tf_matrix = [
        [0.0, 0.0, 1.0, -0.101],  # x
        [-1.0, 0.0, 0.0, 0.01],  # y
        [0.0, -1.0, 0.0, 0.05],  # z
        [0.0, 0.0, 0.0, 1.0]
    ]
    def __init__(self, name):
        rclpy.init()
        super().__init__(name)
        self.fps = fps.FPS() # frame rate counter 帧率统计器
        self.image_queue = queue.Queue(maxsize=2)
        self._init_parameters()
        
        self.set_above = False
        self.lock = threading.RLock()
        self.base_gripper_height = utils.get_gripper_size(500)[1]
        self.joints_pub = self.create_publisher(ServosPosition, 'servo_controller', 1)

        timer_cb_group = ReentrantCallbackGroup()
        self.set_joint_value_target_client = self.create_client(SetJointValue, 'kinematics/set_joint_value_target', callback_group=timer_cb_group)
        self.set_joint_value_target_client.wait_for_service()
        self.kinematics_client = self.create_client(SetRobotPose, 'kinematics/set_pose_target')
        self.kinematics_client.wait_for_service()

        self.enter_srv = self.create_service(Trigger, '~/enter', self.enter_srv_callback)
        self.exit_srv = self.create_service(Trigger, '~/exit', self.exit_srv_callback)
        self.enable_sorting_srv = self.create_service(SetBool, '~/enable_transport', self.enable_transport_srv_callback)
        self.set_target_srv = self.create_service(SetBox, '~/set_target', self.set_target_srv_callback)
        
        code_path = os.path.abspath(os.path.split(os.path.realpath(__file__))[0])
        overrides = dict(conf=0.4, task="segment", mode="predict", model=os.path.join(code_path, 'resources/models', "FastSAM-x.pt"), save=False, imgsz=640)
        self.predictor = FastSAMPredictor(overrides=overrides)
        self.predictor(np.zeros((640, 480, 3), dtype=np.uint8))
        self.config_file = 'transform.yaml'
        self.calibration_file = 'calibration.yaml'
        self.config_path = "/home/ubuntu/ros2_ws/src/app/config/"
        self.data = common.get_yaml_data("/home/ubuntu/ros2_ws/src/app/config/lab_config.yaml")  
        self.lab_data = self.data['/**']['ros__parameters']
        tf_buffer = Buffer()
        self.tf_listener = TransformListener(tf_buffer, self)
        tf_future = tf_buffer.wait_for_transform_async(
            target_frame='depth_cam_depth_optical_frame',
            source_frame='depth_cam_color_frame',
            time=rclpy.time.Time()
        )

        rclpy.spin_until_future_complete(self, tf_future)
        try:
            transform = tf_buffer.lookup_transform(
                'depth_cam_color_frame', 'rgb_camera_link', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=5.0) )
            self.static_transform = transform  # 保存变换数据
            self.get_logger().info(f'Static transform: {self.static_transform}')
        except TransformException as e:
            self.get_logger().error(f'Failed to get static transform: {e}')

        # Extract translation and rotation. 提取平移和旋转
        translation = transform.transform.translation
        rotation = transform.transform.rotation

        self.transform_matrix = common.xyz_quat_to_mat([translation.x, translation.y, translation.z], [rotation.w, rotation.x, rotation.y, rotation.z])
        self.hand2cam_tf_matrix = np.matmul(self.transform_matrix, self.hand2cam_tf_matrix)

        self.timer = self.create_timer(0.0, self.init_process, callback_group=timer_cb_group)

    def get_node_state(self, request, response):
        return response

    def _init_parameters(self):
        self.heart = None
        self.enter = False
        self.start_transport = False
        self.enable_transport = False
        self.sync = None
        self.start_get_roi = False
        self.rgb_sub = None
        self.depth_sub = None
        self.info_sub = None
        self.depth_info_sub = None
        self.white_area_center = None
        self.roi = None
        self.plane = None
        self.extristric = None
        self.corners = None
        self.intrinsic = None
        self.distortion = None
        self.box_info = []
        self.target = []
        self.start_stamp = time.time()

    def init_process(self):
        self.timer.cancel()

        threading.Thread(target=self.main, daemon=True).start()
        threading.Thread(target=self.transport_thread, daemon=True).start()
        self.create_service(Empty, '~/init_finish', self.get_node_state)
        self.go_home()
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')

    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()

    def go_home(self, interrupt=True, back=True):
        if interrupt:
            set_servo_position(self.joints_pub, 0.5, ((1, 200), ))
            time.sleep(0.5)

        joint_angle = [500, 600, 825, 110, 500]
                
        msg = set_joint_value_target(joint_angle)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        endpoint = self.send_request(self.set_joint_value_target_client, msg)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'bbbbbbbbbbbbbbbbbbbbbbbbbbb')


        pose_t, pose_r = endpoint.pose.position, endpoint.pose.orientation
        self.endpoint = common.xyz_quat_to_mat([pose_t.x, pose_t.y, pose_t.z], [pose_r.w, pose_r.x, pose_r.y, pose_r.z])
        set_servo_position(self.joints_pub, 1, ((5, joint_angle[1]), (4, joint_angle[2]), (3, joint_angle[3]), (2, 500)))
        time.sleep(1)        
        
        if back:
            set_servo_position(self.joints_pub, 1, ((6, joint_angle[0]), ))
            time.sleep(1)

    def enter_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "enter object transport")
        with self.lock:
            self._init_parameters()
            self.heart = Heart(self, '~/heartbeat', 5, lambda _: self.exit_srv_callback(request=Trigger.Request(), response=Trigger.Response()))  # Heartbeat package 心跳包
            if self.sync is None:
                self.rgb_sub = message_filters.Subscriber(self, Image, 'depth_cam/rgb/image_raw')
                self.depth_sub = message_filters.Subscriber(self, Image, 'depth_cam/depth/image_raw')
                self.depth_info_sub = message_filters.Subscriber(self, CameraInfo, 'depth_cam/depth/camera_info')
                self.info_sub = message_filters.Subscriber(self, CameraInfo, 'depth_cam/rgb/camera_info')
                # Synchronize the timestamp. The time is allowed to have an error of 0.03 seconds. 同步时间戳, 时间允许有误差在0.03s
                self.sync = message_filters.ApproximateTimeSynchronizer(
                    [self.rgb_sub, self.depth_sub, self.info_sub, self.depth_info_sub], 3, 0.2)
                self.sync.registerCallback(self.multi_callback)
           
            self.enter = True
            self.start_get_roi = True
        
        # self.go_home()
        
        response.success = True
        response.message = "start"
        return response

    def exit_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "exit  object transport")
        with self.lock:
            self._init_parameters()
            if self.sync is not None:
                self.sync.disconnect(self.multi_callback)
                self.sync = None
            pick_and_place.interrupt()
        response.success = True
        response.message = "start"
        return response

    def enable_transport_srv_callback(self, request, response):
        with self.lock:
            if request.data:
                self.get_logger().info('\033[1;32m%s\033[0m' % 'enable  object transport')
                self.enable_transport = True
            else:
                self.get_logger().info('\033[1;32m%s\033[0m' % 'exit  object transport')
                pick_and_place.interrupt()
                self.enable_transport = False
        response.success = True
        response.message = "start"
        return response

    def set_target_srv_callback(self, request, response):
        with self.lock:
            self.get_logger().info('\033[1;32m%s\033[0m' % 'set target')
            self.box_info = [request.label, request.box] 
            # self.get_logger().info(f'11111 {self.box_info}')
        response.success = True
        response.message = "start"
        return response

    def get_roi(self):
        with open(self.config_path + self.config_file, 'r') as f:
            config = yaml.safe_load(f)

            # Convert to numpy array. 转换为 numpy 数组
            corners = np.array(config['corners']).reshape(-1, 3)
            with self.lock:
                self.extristric = np.array(config['extristric'])
                self.white_area_center = np.array(config['white_area_pose_world'])
                self.plane = config['plane']
                self.corners = np.array(config['corners'])

        while self.intrinsic is None or self.distortion is None:
            self.get_logger().info("waiting for camera info")
            time.sleep(0.1)

        with self.lock:
            tvec = self.extristric[:1]  # Take the first row. 取第一行
            rmat = self.extristric[1:]  # Take the last three rows. 取后面三行

            tvec, rmat = common.extristric_plane_shift(np.array(tvec).reshape((3, 1)), np.array(rmat), 0.03)
            # self.get_logger().info(f'corners: {corners}')
            imgpts, jac = cv2.projectPoints(corners[:-1], np.array(rmat), np.array(tvec), self.intrinsic, self.distortion)
            imgpts = np.int32(imgpts).reshape(-1, 2)

            # Crop RIO region 裁切出ROI区域
            x_min = min(imgpts, key=lambda p: p[0])[0] # The minimum value of the X-axis. x轴最小值
            x_max = max(imgpts, key=lambda p: p[0])[0] # The maximum value of the X-axis. x轴最大值
            y_min = min(imgpts, key=lambda p: p[1])[1] # The minimum value of the Y-axis. y轴最小值
            y_max = max(imgpts, key=lambda p: p[1])[1] # The maximum value of the Y-axis. y轴最大值
            roi = np.maximum(np.array([y_min, y_max, x_min, x_max]), 0)
            self.roi = roi

    def cal_grap_point(self, mask, x, y, box, edge_index, name):
        # Calculate the direction vector of the edge. 计算边的方向向量
        edge_vector = box[(edge_index + 1) % 4] - box[edge_index]
        edge_vector = edge_vector / np.linalg.norm(edge_vector)

        # Calculate direction vector perpendicular to the edge. 计算垂直于边的方向向量
        perpendicular_vector = np.array([-edge_vector[1], edge_vector[0]])

        # Define a long line starting from the centroid. 定义一条从质心出发的长线
        line_length = 1000  # Length of the line. 线的长度
        line_start = (x - int(perpendicular_vector[0] * line_length), y - int(perpendicular_vector[1] * line_length))
        line_end = (x + int(perpendicular_vector[0] * line_length), y + int(perpendicular_vector[1] * line_length))

        # Draw the line on the ROI image. 在ROI图像上绘制这条线
        line_image = np.zeros_like(mask)
        cv2.line(line_image, line_start, line_end, 255, 5, cv2.LINE_AA)
        # Find the intersection between the line and the ROI. 找到线与ROI的交点
        intersection_image = cv2.bitwise_and(mask, line_image)
        # cv2.imshow(name, intersection_image)
        intersection_contours, _ = cv2.findContours(intersection_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areaMaxContour, area_max = common.get_area_max_contour(intersection_contours)
        if areaMaxContour is not None:
            rect = cv2.minAreaRect(areaMaxContour)  # Obtain the minimum bounding rectangle. 获取最小外接矩形
            center = [rect[0][0], rect[0][1]]
            object_width = max(rect[1])
            return [center, object_width]
        else:
            return False

    def get_object_pixel_position(self, image, roi):
        # roi[x, y, x, y]
        everything_results = self.predictor(image)
        # self.get_logger().info(f'everything_results: {everything_results}')
        # Prompt inference
        results = prompt(everything_results, bboxes=[roi])
        mask = results[0].masks

        mask = mask.data  # Usually a torch.Tensor. 通常是 torch.Tensor
        if not isinstance(mask, np.ndarray):
            mask = mask.cpu().numpy()
        # self.get_logger().info(f'results: {results[0].boxes}')
        if mask.ndim == 3 and mask.shape[0] == 1:  # Might be (1, H, W), need to remove the first dimension. 可能是 (1, H, W) 需要去掉第一维
            mask = mask[0]

        mask = (mask * 255).astype(np.uint8)
        
        # cv2.imshow('mask', mask)
        M = cv2.moments(mask)
        annotated_frame = results[0].plot()
        # cv2.imshow("YOLO Inference", annotated_frame)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            grasp_point = (cx, cy)
            cv2.circle(image, grasp_point, 5, (255, 0, 0), -1)
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
            areaMaxContour, area_max = common.get_area_max_contour(contours)
            if areaMaxContour is not None:
                rect = cv2.minAreaRect(areaMaxContour)  # Obtain the minimum bounding rectangle 获取最小外接矩形
                # In version 4.5, w is defined as the edge that first aligns with the x-axis under clockwise rotation; angle is the clockwise rotation from the x-axis, ranging in (0, 90].
                # 4.5版本定义为，x轴顺时针旋转最先重合的边为w，angle为x轴顺时针旋转的角度，angle取值为(0,90]
                _, (width, height), angle = rect
                # self.get_logger().info(f'angle: {angle} {rect}')
                bbox = np.intp(cv2.boxPoints(rect))
                cv2.drawContours(image, [bbox], -1, (0, 255, 255), 2, cv2.LINE_AA)  # Draw rectangle contour. 绘制矩形轮廓
                
                # Calculate the long and short sides of the rectangle. 计算矩形的长边和短边
                edge_lengths = [np.linalg.norm(bbox[i] - bbox[(i + 1) % 4]) for i in range(4)]
                long_edge_index = np.argmax(edge_lengths)
                short_edge_index = np.argmin(edge_lengths)
                
                grasp_point1 = self.cal_grap_point(mask, cx, cy, bbox, long_edge_index, 'long')
                grasp_point2 = self.cal_grap_point(mask, cx, cy, bbox, short_edge_index, 'short')

                if not grasp_point1 and not grasp_point2:
                    # cv2.imshow('img', image)
                    return False
                if grasp_point1:
                    if width > height:
                        grasp_point1.append(angle - 90 )
                    else:
                        grasp_point1.append(angle)
                    cv2.circle(image, (int(grasp_point1[0][0]), int(grasp_point1[0][1])), 5, (0, 0, 255), -1)
                    # self.get_logger().info(f'grasp_point1: {grasp_point1}')
                if grasp_point2:
                    if width > height:
                        grasp_point2.append(angle)
                    else:
                        grasp_point2.append(angle - 90)
                    cv2.circle(image, (int(grasp_point2[0][0]), int(grasp_point2[0][1])), 5, (255, 0, 0), -1)
                    # self.get_logger().info(f'grasp_point2: {grasp_point2}')
                # cv2.imshow('img', image)
                return [grasp_point1, grasp_point2]
        return False

    def get_grap_angle(self, bgr_image, depth_image, object_info, max_dist, depth_intrinsic_matrix):
        image_height, image_width = bgr_image.shape[:2]
        # self.get_logger().info(f'111111111111 {object_info}')
        x, y, object_width = object_info[0][0], object_info[0][1], object_info[1]
        w = 50
        if x + w > image_width:
            w = image_width - x
        elif x - w < 0:
            w = x
        h = 50
        if y + h > image_height:
            h = image_height - y
        elif y - h < 0:
            h = y

        angle = 0
        box = np.intp(cv2.boxPoints(((x + 25, y - 40), (w, h), angle)))
        # Create mask. 创建掩码
        mask = np.zeros(depth_image.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [box], 255)
        # Apply mask to depth image. 应用掩码到深度图像
        masked_depth = cv2.bitwise_and(depth_image, depth_image, mask=mask)
        min_dist = utils.find_depth_range(masked_depth, max_dist)
        fx, fy = depth_intrinsic_matrix[0], depth_intrinsic_matrix[4]
        # self.get_logger().info(f'{min_dist}')

        object_width = object_width / (fx / (min_dist / 1000.0))

        gripper_angle = utils.set_gripper_size(object_width)

        return gripper_angle, min_dist

    def get_position(self, object_info, min_dist, angle, gripper_angle, depth_intrinsic_matrix):
        gripper_info = utils.get_gripper_size(gripper_angle)
        d = gripper_info[1] - self.base_gripper_height
        # self.get_logger().info(f'gripper_info: {gripper_info} {d}')
    
        x, y = object_info[0][0], object_info[0][1]
        cx, cy = x + 25, y - 40
        position = utils.calculate_world_position(cx, cy, min_dist, self.plane, self.endpoint, self.hand2cam_tf_matrix, depth_intrinsic_matrix)
        position[-1] += d

        config_data = common.get_yaml_data(os.path.join(self.config_path, self.calibration_file))
        offset = tuple(config_data['depth']['offset'])
        scale = tuple(config_data['depth']['scale'])
        for i in range(3):
            position[i] = position[i] * scale[i]
            position[i] = position[i] + offset[i]

        # Compute base yaw angle. 计算基础偏航角度
        yaw = math.degrees(math.atan2(position[1], position[0]))

        # Adjust yaw angle based on quadrant. 根据象限调整偏航角度
        if position[0] < 0:
            if position[1] < 0:
                yaw += 180
            else:
                yaw -= 180

        # Apply the first angle in the angle list. 应用角度列表中的第一个角度
        yaw += angle

        # Map the angle to the control range. 将角度映射到控制范围
        yaw = 500 + int(yaw / 240 * 1000)
        
        object_info = [position, [x, y], yaw]
        return object_info 

    def get_object_world_position(self, object_name, bgr_image, depth_image, object_info, max_dist, depth_intrinsic_matrix, plane_values):
        object_info_ = []
        gripper_angle = []
        # self.get_logger().info(f'object_info {object_info}')
        if object_info[0]:
            gripper_angle, min_dist = self.get_grap_angle(bgr_image, depth_image, object_info[0], max_dist, depth_intrinsic_matrix)
            if 200 < gripper_angle < 700:  
                object_info_ = self.get_position(object_info[0], min_dist, object_info[0][-1], gripper_angle, depth_intrinsic_matrix)
                # self.get_logger().info(f'object_info_1 {object_info_}')
        if object_info[1]:
            gripper_angle_, min_dist = self.get_grap_angle(bgr_image, depth_image, object_info[1], max_dist, depth_intrinsic_matrix)
            if 200 < gripper_angle_ < 700:  
                info = self.get_position(object_info[1], min_dist, object_info[1][-1], gripper_angle_, depth_intrinsic_matrix)
                if object_info_:
                    if abs(object_info_[-1] - 500) > abs(info[-1] - 500):
                        object_info_ = info
                        gripper_angle = gripper_angle_
                        # self.get_logger().info(f'object_info_2 {object_info_}')
                else:
                    object_info_ = info
                    gripper_angle = gripper_angle_
                    # self.get_logger().info(f'object_info_2 {object_info_}')
        if object_info_:
            object_info_.append(gripper_angle)
            object_info_.append(object_name)
        return object_info_

    def main(self):
        while True:
            if self.enter:
                try:
                    bgr_image, depth_image, camera_info, depth_camera_info = self.image_queue.get(block=True, timeout=1)
                except queue.Empty:
                    continue
                with self.lock:
                    self.intrinsic = np.matrix(camera_info.k).reshape(1, -1, 3)
                    self.distortion = np.array(camera_info.d)
                    if self.start_get_roi:
                        self.get_roi()
                        self.start_get_roi = False
                max_dist = 350
                depth_image = utils.create_roi_mask(depth_image, bgr_image, self.corners, camera_info, self.extristric, max_dist, 0.08)
                sim_depth_image = (1 - np.clip(depth_image, 0, max_dist).astype(np.float64) / max_dist) * 255
                depth_color_map = cv2.applyColorMap(sim_depth_image.astype(np.uint8), cv2.COLORMAP_JET)
                if self.enable_transport:
                    with self.lock:
                        if not self.target: 
                            depth_intrinsic_matrix = depth_camera_info.k
                            plane_values = utils.get_plane_values(depth_image, self.plane, depth_intrinsic_matrix)
                            object_info = self.get_object_pixel_position(bgr_image, self.box_info[1]) # xyxy
                            
                            # self.get_logger().info(str(object_info))
                            if object_info:
                                self.target = self.get_object_world_position(self.box_info[0], bgr_image, depth_image, object_info, max_dist, depth_intrinsic_matrix, plane_values)
                                # self.get_logger().info(f'{self.target}')
                                self.start_stamp = time.time()
                                if not self.target:
                                    self.get_logger().info('object too big')
                        else:
                            if time.time() - self.start_stamp > 2:
                                self.start_transport = True        
                            cv2.circle(bgr_image, (int(self.target[1][0]), int(self.target[1][1])), 10, (255, 0, 0), -1)
                        # self.get_logger().info(f'{self.box_info}')
                        cv2.rectangle(bgr_image, (self.box_info[1][0], self.box_info[1][1]), (self.box_info[1][2], self.box_info[1][3]), (0, 255, 0), 2, 1)
                self.fps.update()
                self.fps.show_fps(bgr_image)
                #result_image = np.concatenate([bgr_image[40:440, ], depth_color_map], axis=1)
                result_image = np.concatenate([bgr_image, depth_color_map], axis=1)
                cv2.imshow('image', result_image)
                cv2.waitKey(1)
                if not self.set_above:
                    cv2.moveWindow('image', 1920 - 640*2, 0)
                    os.system("wmctrl -r image -b add,above")
                    self.set_above = True
            else:
                time.sleep(0.1)
        cv2.destroyAllWindows()

    def transport_thread(self):
        while True:
            if self.start_transport:
                self.enable_transport = False
                self.start_transport = False
                pose = []
                config_data = common.get_yaml_data(os.path.join(self.config_path, self.calibration_file))
                offset = tuple(config_data['kinematics']['offset'])
                scale = tuple(config_data['kinematics']['scale'])
                p = self.target
                for i in range(3):
                    p[0][i] = p[0][i] * scale[i]
                    p[0][i] = p[0][i] + offset[i]
                # self.get_logger().info(f'pick_and_place: {p}')
                if p[3]:
                    finish = pick_and_place.pick_without_back(p[0], 80, p[2], p[3], 0.015, self.joints_pub, self.kinematics_client)
                    if not finish:
                        self.go_home()
                    else:
                        self.go_home(False, False)
                    finish = pick_and_place.place([0.25, 0.2, 0.18], 60, 500, 200, self.joints_pub, self.kinematics_client)
                    time.sleep(1)
                    if finish:
                        self.go_home(False)
                    else:
                        self.go_home()
                else:
                    self.get_logger().info('object too big')
                with self.lock:
                    self.target = []
            else:
                time.sleep(0.1)

    def multi_callback(self, ros_rgb_image, ros_depth_image, camera_info, depth_camera_info):
        rgb_image = np.ndarray(shape=(ros_rgb_image.height, ros_rgb_image.width, 3), dtype=np.uint8,
                               buffer=ros_rgb_image.data)  # Original RGB image 原始 RGB 画面
        #bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        depth_image = np.ndarray(shape=(ros_depth_image.height, ros_depth_image.width), dtype=np.uint16,
                                 buffer=ros_depth_image.data)
        if self.image_queue.full():
            # # If the queue is full, discard the oldest image. 如果队列已满，丢弃最旧的图像
            self.image_queue.get()
            # # Put the image into the queue. 将图像放入队列
        self.image_queue.put((rgb_image, depth_image, camera_info, depth_camera_info))

def main():
    node = IntelligentGrasp('intelligent_grasp')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()

if __name__ == "__main__":
    main()
