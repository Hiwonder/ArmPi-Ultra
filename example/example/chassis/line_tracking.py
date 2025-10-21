#!/usr/bin/env python3
# encoding: utf-8
# 取色巡线

import os
import cv2
import math
import time
import rclpy
import queue
import threading
import numpy as np

import sdk.pid as pid
import sdk.misc as misc
import sdk.common as common

from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, Point
from std_srvs.srv import SetBool
from interfaces.srv import SetPoint
from servo_controller_msgs.msg import ServosPosition
from servo_controller.bus_servo_control import set_servo_position
from app.common import ColorPicker  


class LineFollower:
    def __init__(self, color, node, threshold=0.3):
        self.node = node
        self.chassis_type = os.environ.get('CHASSIS_TYPE', 'Mecanum')
        self.camera_type = os.environ.get('CAMERA_TYPE', 'astra')
        self.pid_yaw = pid.PID(2.0, 0.0, 0.005)   
        self.pid_dist = pid.PID(2.0, 0.0, 0.005)


        self.last_color_circle = None
        self.lost_target_count = 0

        self.target_lab, self.target_rgb = color
        self.weight_sum = 1.0
        self.threshold = threshold

        self.x_stop = 320
        if self.camera_type == 'aurora':
            self.y_stop = 200
            self.pro_size = (640, 400)
        else:
            self.y_stop = 240
            self.pro_size = (640, 480)

        self.rois = [
            (0.9, 0.95, 0.0, 1.0, 0.7),
            (0.8, 0.85, 0.0, 1.0, 0.2),
            (0.7, 0.75, 0.0, 1.0, 0.1),
        ]
        self.weight_sum = sum([r[4] for r in self.rois])

    def get_area_max_contour(self, contours, threshold=30):
        contour_area = [(c, abs(cv2.contourArea(c))) for c in contours]
        filtered = [c for c in contour_area if c[1] > threshold]
        if not filtered:
            return None
        return max(filtered, key=lambda x: x[1])

    def __call__(self, image, result_image, threshold):
        h, w = image.shape[:2]
        centroid_sum = 0.0

        # 颜色阈值
        min_color = np.array([
            max(0, int(self.target_lab[0] - 50 * threshold * 2)),
            max(0, int(self.target_lab[1] - 50 * threshold)),
            max(0, int(self.target_lab[2] - 50 * threshold))
        ], dtype=np.uint8)

        max_color = np.array([
            min(255, int(self.target_lab[0] + 50 * threshold * 2)),
            min(255, int(self.target_lab[1] + 50 * threshold)),
            min(255, int(self.target_lab[2] + 50 * threshold))
        ], dtype=np.uint8)

        for roi in self.rois:
            top = int(h * roi[0])
            bottom = int(h * roi[1])
            left = int(w * roi[2])
            right = int(w * roi[3])

            roi_img = image[top:bottom, left:right]
            lab_roi = cv2.cvtColor(roi_img, cv2.COLOR_RGB2LAB)
            blur_roi = cv2.GaussianBlur(lab_roi, (3, 3), 3)

            mask = cv2.inRange(blur_roi, min_color, max_color)
            mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
            max_contour_area = self.get_area_max_contour(contours, threshold=30)

            if max_contour_area is not None:
                contour, area = max_contour_area
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                box[:, 1] += top  # 调整回原图坐标
                box[:, 0] += left

                cv2.drawContours(result_image, [box], 0, (0, 255, 255), 2)

                pt1 = box[0]
                pt3 = box[2]
                center_x = (pt1[0] + pt3[0]) / 2
                center_y = (pt1[1] + pt3[1]) / 2
                cv2.circle(result_image, (int(center_x), int(center_y)), 5, (self.target_rgb[0], self.target_rgb[1], self.target_rgb[2]), -1)

                centroid_sum += center_x * roi[4]

        if centroid_sum == 0:
            # 未检测到线
            self.lost_target_count += 1
            if self.lost_target_count > 50:
                self.node.get_logger().warn("Line lost for too long.")
            return result_image, None

        self.lost_target_count = 0

        center_pos = centroid_sum / self.weight_sum
        deflection_angle = -math.atan((center_pos - w / 2) / (h / 2))

        return result_image, deflection_angle


class LineFollowingNode(Node):
    def __init__(self, name):
        super().__init__(name)
        self.name = name
        self.set_callback = False
        self.is_running = False
        self.color_picker = None
        self.follower = None
        self.threshold = 0.5
        self.lock = threading.RLock()
        self.image_height = None
        self.image_width = None
        self.image_queue = queue.Queue(2)

        self.bridge = CvBridge()

        self.mecanum_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        self.image_sub = self.create_subscription(Image, '/depth_cam/rgb/image_raw', self.image_callback, 1)

        self.create_service(SetBool, '~/set_running', self.set_running_srv_callback)
        self.create_service(SetPoint, '~/set_target_color', self.set_target_color_srv_callback)

        threading.Thread(target=self.visualization_loop, daemon=True).start()

    def visualization_loop(self):
        cv2.namedWindow("Line Following", cv2.WINDOW_AUTOSIZE)
        while rclpy.ok():
            try:
                img = self.image_queue.get(timeout=1)
            except queue.Empty:
                continue
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow("Line Following", bgr)
            if not self.set_callback:
                self.set_callback = True
                cv2.setMouseCallback("Line Following", self.mouse_callback)
            if cv2.waitKey(1) == 27:
                break
        cv2.destroyAllWindows()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.image_width and self.image_height:
            with self.lock:
                norm_x = x / self.image_width
                norm_y = y / self.image_height
                self.color_picker = ColorPicker(Point(x=norm_x, y=norm_y), 20)
                self.follower = None
                self.get_logger().info(f"Picked color point norm coords: ({norm_x:.3f}, {norm_y:.3f})")

    def set_target_color_srv_callback(self, request, response):
        with self.lock:
            self.color_picker = ColorPicker(request.data, 20)
            self.follower = None
            response.success = True
            response.message = "Color picker initialized."
        return response

    def set_running_srv_callback(self, request, response):
        with self.lock:
            self.is_running = request.data
            if not self.is_running:
                self.mecanum_pub.publish(Twist())  # 停车
            response.success = True
            response.message = f"Set running: {self.is_running}"
        return response

    def image_callback(self, ros_img):
        cv_image = self.bridge.imgmsg_to_cv2(ros_img, "rgb8")
        self.image_height, self.image_width = cv_image.shape[:2]
        result_image = cv_image.copy()

        with self.lock:
            if self.color_picker:
                try:
                    target_color, result_image = self.color_picker(cv_image, result_image)
                    if target_color:
                        self.follower = LineFollower(target_color, self)
                        self.color_picker = None
                        self.get_logger().info(f"Color picked LAB: {target_color[0]}")
                except Exception as e:
                    self.get_logger().error(f"ColorPicker error: {e}")

            elif self.follower:
                try:
                    result_image, deflect_angle = self.follower(cv_image, result_image, self.threshold)
                    twist = Twist()
                    twist.linear.x = 5.0
                    if deflect_angle is not None and self.is_running:
                        # PID控制转角速度
                        self.follower.pid_yaw.update(deflect_angle)
                        angular_speed = misc.map(common.set_range(self.follower.pid_yaw.output, -6, 6), -6, 6, -10.0, 10.0)
                        twist.angular.z = angular_speed
                        self.mecanum_pub.publish(twist)
                    else:
                        # 无线、停止或未运行
                        self.follower.pid_yaw.clear()
                        self.mecanum_pub.publish(Twist())
                except Exception as e:
                    self.get_logger().error(f"Follower error: {e}")

        if self.image_queue.full():
            self.image_queue.get()
        self.image_queue.put(result_image)


def main(args=None):
    rclpy.init(args=args)
    node = LineFollowingNode("line_following")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.mecanum_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
