#!/usr/bin/python3
#coding=utf8

import cv2
import time
import rclpy
import queue
import threading
import numpy as np
import message_filters
from rclpy.node import Node
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image, CameraInfo
from rclpy.executors import MultiThreadedExecutor
from servo_controller_msgs.msg import ServosPosition
from ros_robot_controller_msgs.msg import BuzzerState
from rclpy.callback_groups import ReentrantCallbackGroup
from kinematics.kinematics_control import set_pose_target
from kinematics_msgs.srv import GetRobotPose, SetRobotPose
from servo_controller.bus_servo_control import set_servo_position
from typing import Tuple, List
from cv_bridge import CvBridge  
from sdk import common  

def depth_pixel_to_camera(pixel_coords, depth, intrinsics):
    fx, fy, cx, cy = intrinsics
    px, py = pixel_coords
    x = (px - cx) * depth / fx
    y = (py - cy) * depth / fy
    z = depth
    return np.array([x, y, z])

class RemoveTooHighObjectNode(Node):

    def __init__(self, name):
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.bridge = CvBridge()  # CvBridge 实例
        self.endpoint = None
        self.moving = False
        self.running = True
        self.stamp = time.time()
        self.start_process = False
        self.image_queue = queue.Queue(maxsize=2)

        # 加载手眼标定矩阵
        try:
            camera_info_data = common.get_yaml_data("/home/ubuntu/ros2_ws/src/peripherals/config/camera_info.yaml")
            self.hand2cam_tf_matrix = np.array(camera_info_data['hand2cam_tf_matrix'])
            self.get_logger().info('手眼变换矩阵加载成功.')
        except Exception as e:
            self.get_logger().warn(f'加载手眼变换矩阵失败，使用默认值: {e}')
            self.hand2cam_tf_matrix = np.array([
                [0.0, 0.0, 1.0, -0.101],
                [-1.0, 0.0, 0.0, 0.01],
                [0.0, -1.0, 0.0, 0.05],
                [0.0, 0.0, 0.0, 1.0]
            ])

        self.joints_pub = self.create_publisher(ServosPosition, '/servo_controller', 1)
        self.buzzer_pub = self.create_publisher(BuzzerState, '/ros_robot_controller/set_buzzer', 1)

        cb_group = ReentrantCallbackGroup()
        self.create_service(Trigger, f'~/{self.get_name()}/start', self.start_srv_callback, callback_group=cb_group)
        self.create_service(Trigger, f'~/{self.get_name()}/stop', self.stop_srv_callback, callback_group=cb_group)

        self.get_current_pose_client = self.create_client(GetRobotPose, '/kinematics/get_current_pose')
        self.set_pose_target_client = self.create_client(SetRobotPose, '/kinematics/set_pose_target')
        self.get_logger().info('Waiting for services...')
        self.get_current_pose_client.wait_for_service()
        self.set_pose_target_client.wait_for_service()
        self.get_logger().info('Services are ready.')

        rgb_sub = message_filters.Subscriber(self, Image, '/depth_cam/rgb/image_raw')
        depth_sub = message_filters.Subscriber(self, Image, '/depth_cam/depth/image_raw')
        info_sub = message_filters.Subscriber(self, CameraInfo, '/depth_cam/depth/camera_info')
        sync = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, info_sub], 5, 0.1)
        sync.registerCallback(self.multi_callback)

        self.timer = self.create_timer(0.1, self.init_process, callback_group=cb_group)

        self.pick_offset = [-0.02, 0.01, 0.0, -0.02, -0.01]

    def init_process(self):
        self.timer.cancel()
        set_servo_position(self.joints_pub, 1, ((6, 500), (5, 600), (4, 825), (3, 110), (2, 500), (1, 210)))
        time.sleep(2)
        threading.Thread(target=self.main_loop, daemon=True).start()
        self.get_logger().info(f'\033[1;32mNode {self.get_name()} is ready.\033[0m')
        self.get_logger().info(f"\033[1;33mPress 's' or call 'start' service to begin processing.\033[0m")

    def start_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32mStart service called. Starting processing.\033[0m')
        self.start_process = True
        response.success = True
        return response

    def stop_srv_callback(self, request, response):
        self.get_logger().info('\033[1;31mStop service called. Stopping processing.\033[0m')
        self.start_process = False
        self.moving = False
        set_servo_position(self.joints_pub, 1, ((6, 500), (5, 600), (4, 815), (3, 110), (2, 500), (1, 210)))
        response.success = True
        return response

    def send_request(self, client, msg):
        future = client.call_async(msg)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def multi_callback(self, ros_rgb_image, ros_depth_image, depth_camera_info):
        if not self.image_queue.full():
            self.image_queue.put((ros_rgb_image, ros_depth_image, depth_camera_info))

    def get_endpoint(self):
        req = GetRobotPose.Request()
        res = self.send_request(self.get_current_pose_client, req)
        if res is not None:
            endpoint = res.pose
            self.endpoint = common.xyz_quat_to_mat([endpoint.position.x, endpoint.position.y, endpoint.position.z],
                                                   [endpoint.orientation.w, endpoint.orientation.x, endpoint.orientation.y, endpoint.orientation.z])
            return self.endpoint
        return None

    def get_plane_values(self, depth_image: np.ndarray, plane: Tuple[float, float, float, float], intrinsic_matrix: List[float]) -> np.ndarray:
        a, b, c, d = plane
        fx, fy, cx, cy = intrinsic_matrix[0], intrinsic_matrix[4], intrinsic_matrix[2], intrinsic_matrix[5]

        H, W = depth_image.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))

        z = depth_image.astype(np.float32) / 1000.0

        valid_mask = (z > 0.1) & (z < 2.0)
        x = np.zeros_like(z)
        y = np.zeros_like(z)
        x[valid_mask] = (u[valid_mask] - cx) * z[valid_mask] / fx
        y[valid_mask] = (v[valid_mask] - cy) * z[valid_mask] / fy

        plane_distances = np.full_like(z, np.inf)
        normal_magnitude = np.sqrt(a**2 + b**2 + c**2)
        if normal_magnitude > 1e-6:
            numerator = a * x + b * y + c * z + d
            plane_distances[valid_mask] = numerator[valid_mask] / normal_magnitude
        return plane_distances

    def robust_plane_detection(self, depth_image: np.ndarray, intrinsic_matrix: List[float], sample_ratio: float = 0.2) -> Tuple[float, float, float, float]:
        fx, fy, cx, cy = intrinsic_matrix[0], intrinsic_matrix[4], intrinsic_matrix[2], intrinsic_matrix[5]

        H, W = depth_image.shape
        z = depth_image.astype(np.float32) / 1000.0

        valid_mask = (z > 0.2) & (z < 2.0)
        valid_indices = np.where(valid_mask)
        if len(valid_indices[0]) < 1000:
            return (0, 0, 1, 0)

        num_points = len(valid_indices[0])
        num_samples = min(num_points, max(int(num_points * sample_ratio), 1000))
        sample_indices = np.random.choice(num_points, num_samples, replace=False)

        v_s, u_s = valid_indices[0][sample_indices], valid_indices[1][sample_indices]
        z_s = z[v_s, u_s]
        x_s = (u_s - cx) * z_s / fx
        y_s = (v_s - cy) * z_s / fy
        points = np.column_stack([x_s, y_s, z_s])

        try:
            A_matrix = np.c_[points[:, :2], np.ones(len(points))]
            C, _, _, _ = np.linalg.lstsq(A_matrix, points[:, 2], rcond=None)
            a, b, c, d = C[0], C[1], -1.0, C[2]
            norm = np.sqrt(a**2 + b**2 + c**2)
            if norm == 0:
                return (0, 0, 1, 0)
            return (a / norm, b / norm, c / norm, d / norm)
        except np.linalg.LinAlgError:
            return (0, 0, 1, 0)

    def pick(self, position, angle):
        self.moving = True
        angle = angle % 90
        angle = angle - 90 if angle > 45 else (angle + 90 if angle < -45 else angle)

        set_servo_position(self.joints_pub, 0.5, ((1, 210),))
        time.sleep(0.5)

        if position[0] > 0.21:
            offset_x = self.pick_offset[0]
        else:
            offset_x = self.pick_offset[1]

        if position[1] > 0:
            offset_y = self.pick_offset[2]
        else:
            offset_y = self.pick_offset[3]

        offset_z = self.pick_offset[-1]

        target_pos = [position[0] + offset_x, position[1] + offset_y, position[2] + offset_z + 0.05]
        res = self.send_request(self.set_pose_target_client, set_pose_target(target_pos, 80, [-180.0, 180.0], 1.5))
        if res and res.pulse:
            set_servo_position(self.joints_pub, 1.5, ((6, res.pulse[0]), (5, res.pulse[1]), (4, res.pulse[2]), (3, res.pulse[3])))
        else:
            self.get_logger().warn('Pick: Failed to get pose for approach')
            self.moving = False
            return
        time.sleep(1.5)

        final_angle = 500 + int(1000 * (angle + res.rpy[-1]) / 240)
        set_servo_position(self.joints_pub, 0.5, ((2, final_angle),))
        time.sleep(0.5)

        target_pos[2] -= 0.065
        res = self.send_request(self.set_pose_target_client, set_pose_target(target_pos, 80, [-180.0, 180.0], 1.5))
        if res and res.pulse:
            set_servo_position(self.joints_pub, 1.5, ((6, res.pulse[0]), (5, res.pulse[1]), (4, res.pulse[2]), (3, res.pulse[3])))
        time.sleep(1.5)

        set_servo_position(self.joints_pub, 1.0, ((1, 600),))
        time.sleep(1.0)

        target_pos[2] += 0.08
        res = self.send_request(self.set_pose_target_client, set_pose_target(target_pos, 80, [-180.0, 180.0], 1.0))
        if res and res.pulse:
            set_servo_position(self.joints_pub, 1.0, ((6, res.pulse[0]), (5, res.pulse[1]), (4, res.pulse[2]), (3, res.pulse[3])))
        time.sleep(1.0)

        set_servo_position(self.joints_pub, 1.0, ((6, 150), (5, 635), (4, 900), (3, 260), (2, 500)))
        time.sleep(1.5)

        set_servo_position(self.joints_pub, 1.0, ((1, 200),))
        time.sleep(1.0)

        set_servo_position(self.joints_pub, 1.0, ((6, 500), (5, 600), (4, 825), (3, 110), (2, 500), (1, 210)))
        time.sleep(1.5)
        self.moving = False

    def main_loop(self):
        while self.running:
            try:
                ros_rgb, ros_depth, cam_info = self.image_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            try:
                # 使用 CvBridge 安全转换
                rgb_image = self.bridge.imgmsg_to_cv2(ros_rgb, desired_encoding='rgb8')
                depth_image = self.bridge.imgmsg_to_cv2(ros_depth, desired_encoding='passthrough')

                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

                K = cam_info.k

                plane_coeffs = self.robust_plane_detection(depth_image, K)
                height_map = self.get_plane_values(depth_image, plane_coeffs, K)

                # 方向修正（保证高度为正）
                MIN_OBJECT_HEIGHT_FOR_CHECK = 0.01
                potential_object_heights = height_map[np.abs(height_map) > MIN_OBJECT_HEIGHT_FOR_CHECK]
                potential_object_heights = potential_object_heights[np.isfinite(potential_object_heights)]
                if potential_object_heights.size > 20 and np.median(potential_object_heights) < 0:
                    height_map *= -1.0

                MIN_OBJECT_HEIGHT = 0.015
                object_mask = np.where((height_map > MIN_OBJECT_HEIGHT) & np.isfinite(height_map), 255, 0).astype(np.uint8)
                object_mask = cv2.erode(object_mask, np.ones((3, 3), np.uint8), iterations=1)
                object_mask = cv2.dilate(object_mask, np.ones((5, 5), np.uint8), iterations=1)

                contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                display_height_map = height_map.copy()
                display_height_map[~np.isfinite(display_height_map)] = 0
                display_height_map = np.clip(display_height_map, 0, 0.1)
                display_height_map = (display_height_map * 2550).astype(np.uint8)
                height_color_map = cv2.applyColorMap(display_height_map, cv2.COLORMAP_JET)

                tallest_object = None
                max_height = 0.0

                if len(contours) > 0:
                    for contour in contours:
                        if cv2.contourArea(contour) < 400 or self.moving:
                            continue

                        mask = np.zeros(height_map.shape, dtype=np.uint8)
                        cv2.drawContours(mask, [contour], -1, 255, -1)

                        heights_in_contour = height_map[mask == 255]
                        heights_in_contour = heights_in_contour[np.isfinite(heights_in_contour)]

                        if heights_in_contour.size > 0:
                            current_max_height = np.max(heights_in_contour)
                            if current_max_height > max_height:
                                max_height = current_max_height
                                peak_coords = np.where((height_map == current_max_height) & (mask == 255))
                                peak_y, peak_x = peak_coords[0][0], peak_coords[1][0]
                                tallest_object = {"height": current_max_height, "contour": contour, "peak_xy": (peak_x, peak_y)}

                if tallest_object and not self.moving:
                    contour = tallest_object["contour"]
                    cx, cy = tallest_object["peak_xy"]

                    # 只绘制轮廓和小圆点，不显示高度文本
                    cv2.drawContours(bgr_image, [contour], -1, (0, 255, 0), 2)
                    cv2.circle(bgr_image, (cx, cy), 5, (0, 0, 255), -1)

                    PICK_HEIGHT_THRESHOLD = 0.0015
                    if tallest_object["height"] > PICK_HEIGHT_THRESHOLD and self.start_process:
                        if time.time() - self.stamp > 1.0:
                            self.stamp = time.time()
                            depth_mm = depth_image[cy, cx]
                            if depth_mm > 0:
                                self.get_endpoint()
                                if self.endpoint is not None:
                                    cam_coords = depth_pixel_to_camera((cx, cy), depth_mm / 1000.0, (K[0], K[4], K[2], K[5]))
                                    pose_end = np.matmul(self.hand2cam_tf_matrix, common.xyz_euler_to_mat(cam_coords, (0, 0, 0)))
                                    world_pose = np.matmul(self.endpoint, pose_end)
                                    pose_t, _ = common.mat_to_xyz_euler(world_pose)
                                    rect = cv2.minAreaRect(contour)
                                    threading.Thread(target=self.pick, args=(list(pose_t), rect[2]), daemon=True).start()
                    else:
                        self.stamp = time.time()

                result_image = np.concatenate([bgr_image, height_color_map], axis=1)
                cv2.imshow("Remove Too High Object", result_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    self.start_process = not self.start_process
                    if self.start_process:
                        self.get_logger().info("\033[1;32mProcessing started by keyboard.\033[0m")
                    else:
                        self.get_logger().info("\033[1;31mProcessing stopped by keyboard.\033[0m")
                elif key == ord('q'):
                    self.running = False
                    break

            except Exception as e:
                self.get_logger().error(f'Main loop error: {e}', throttle_duration_sec=1.0)

        cv2.destroyAllWindows()
        self.get_logger().info("Shutting down...")

def main(args=None):
    rclpy.init(args=args)
    try:
        node = RemoveTooHighObjectNode('remove_too_high_object')
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals() and rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == "__main__":
    main()
