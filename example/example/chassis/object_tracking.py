#!/usr/bin/env python3
# encoding: utf-8
# Color Tracking 颜色跟踪
import os
import cv2
import math
import time
import queue
import rclpy
import threading
import numpy as np
import sdk.pid as pid
import sdk.misc as misc
import sdk.common as common
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from app.common import ColorPicker
from std_srvs.srv import SetBool, Trigger
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from interfaces.srv import SetPoint, SetFloat64
from servo_controller_msgs.msg import ServosPosition
from rclpy.callback_groups import ReentrantCallbackGroup
from servo_controller.bus_servo_control import set_servo_position

class ObjectTracker:
    def __init__(self, color, node):
        self.node = node
        self.chassis_type = os.environ.get('CHASSIS_TYPE', 'Mecanum')
        self.camera_type = os.environ.get('CAMERA_TYPE', 'astra')
        self.pid_yaw = pid.PID(5.0, 0.0, 0.005)  
        self.pid_dist = pid.PID(2.0, 0.0, 0.005)

        self.last_color_circle = None
        self.lost_target_count = 0
        self.target_lab, self.target_rgb = color
        self.weight_sum = 1.0
        self.x_stop = 320
        if self.camera_type == 'aurora':
            self.y_stop = 200
            self.pro_size = (640, 400)
        else:
            self.y_stop = 240
            self.pro_size = (640, 480)

    def __call__(self, image, result_image, threshold):
        twist_msg = Twist()
        
        h, w = image.shape[:2]
        image_resized = cv2.resize(image, self.pro_size)
        image_lab = cv2.cvtColor(image_resized, cv2.COLOR_RGB2LAB)
        image_blur = cv2.GaussianBlur(image_lab, (5, 5), 5)

        min_color = [int(self.target_lab[0] - 50 * threshold * 2),
                     int(self.target_lab[1] - 50 * threshold),
                     int(self.target_lab[2] - 50 * threshold)]
        max_color = [int(self.target_lab[0] + 50 * threshold * 2),
                     int(self.target_lab[1] + 50 * threshold),
                     int(self.target_lab[2] + 50 * threshold)]
        
        mask = cv2.inRange(image_blur, tuple(min_color), tuple(max_color))
        eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
        contour_area = map(lambda c: (c, math.fabs(cv2.contourArea(c))), contours)
        contour_area = list(filter(lambda c: c[1] > 40, contour_area))
        
        circle = None
        if len(contour_area) > 0:
            if self.last_color_circle is None:
                contour, area = max(contour_area, key=lambda c_a: c_a[1])
                circle = cv2.minEnclosingCircle(contour)
            else:
                (last_x, last_y), last_r = self.last_color_circle
                circles = map(lambda c: cv2.minEnclosingCircle(c[0]), contour_area)
                circle_dist = list(map(lambda c: (c, math.sqrt(((c[0][0] - last_x) ** 2) + ((c[0][1] - last_y) ** 2))), circles))
                circle, dist = min(circle_dist, key=lambda c: c[1])
                if dist > 150:
                    contour, area = max(contour_area, key=lambda c_a: c_a[1])
                    circle = cv2.minEnclosingCircle(contour)

        if circle is not None:
            self.lost_target_count = 0
            (x, y), r = circle
            x = x / self.pro_size[0] * w
            y = y / self.pro_size[1] * h
            r = r / self.pro_size[0] * w
            self.last_color_circle = circle

            cv2.circle(result_image, (self.x_stop, self.y_stop), 5, (0, 255, 255), -1)
            cv2.circle(result_image, (int(x), int(y)), int(r), (self.target_rgb[0], self.target_rgb[1], self.target_rgb[2]), 2)
            
            if abs(y - self.y_stop) > 20:
                self.pid_dist.update(y - self.y_stop)
                self.node.get_logger().info(f'pid_dist.output: {self.pid_dist.output}')
                twist_msg.linear.x = misc.map(common.set_range(self.pid_dist.output, -6.0, 6.0), -6.0, 6.0, -6.0, 6.0)
            else:
                self.pid_dist.clear()

            if abs(x - self.x_stop) > 20:
                self.pid_yaw.update(x - self.x_stop)
                self.node.get_logger().info(f'pid_yaw.output: {self.pid_yaw.output}')
                twist_msg.angular.z = misc.map(common.set_range(self.pid_yaw.output, -6.0, 6.0), -6.0, 6.0, -6.0, 6.0)
            else:
                self.pid_yaw.clear()
        else:
            self.lost_target_count += 1
            if self.lost_target_count > 10:
                self.last_color_circle = None

        return result_image, twist_msg

class OjbectTrackingNode(Node):
    def __init__(self, name):
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.name = name
        self.set_callback = False
        self.color_picker = None
        self.tracker = None
        self.is_running = False
        self.threshold = 0.5
        self.lock = threading.RLock()
        self.image_height = None
        self.image_width = None
        self.bridge = CvBridge()
        self.image_queue = queue.Queue(2)
        
        self.mecanum_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        self.image_sub = self.create_subscription(Image, '/depth_cam/rgb/image_raw', self.image_callback, 1)

        self.set_running_srv = self.create_service(SetBool, '~/set_running', self.set_running_srv_callback)
        self.set_target_srv = self.create_service(SetPoint, '~/set_target_color', self.set_target_color_srv_callback)
        self.joints_pub = self.create_publisher(ServosPosition, 'servo_controller/set_servos_position', 1)
        
        self.timer_cb_group = ReentrantCallbackGroup()
        self.timer = self.create_timer(0.1, self.init_process, callback_group=self.timer_cb_group)

    def init_process(self):
        self.timer.cancel()
        set_servo_position(self.joints_pub, 1, ((6, 500), (5, 600), (4, 825), (3, 110), (2, 500), (1, 210)))
        time.sleep(1.0)
        threading.Thread(target=self.main_loop, daemon=True).start()

    def main_loop(self):
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        while rclpy.ok():
            try:
                image = self.image_queue.get(block=True, timeout=1.0)
            except queue.Empty:
                continue

            result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("result", result)
            if not self.set_callback:
                self.set_callback = True
                cv2.setMouseCallback("result", self.mouse_callback)
            
            key = cv2.waitKey(1)
            if key == 27:
                break
        
        self.mecanum_pub.publish(Twist())
        cv2.destroyAllWindows()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            with self.lock:
                if self.image_width is not None and self.image_height is not None:
                    # 【修改 2】: 创建一个Point对象，并用归一化的坐标填充
                    click_point = Point()
                    click_point.x = float(x) / self.image_width
                    click_point.y = float(y) / self.image_height
                    
                    self.tracker = None
                    # 【修改 3】: 将Point对象传递给ColorPicker
                    self.color_picker = ColorPicker(click_point, 10)
                    self.get_logger().info(f'Start picking color at normalized coords: ({click_point.x:.2f}, {click_point.y:.2f})')


    def set_target_color_srv_callback(self, request, response):
        self.get_logger().info('Setting target color via service')
        with self.lock:
            self.tracker = None
            self.color_picker = ColorPicker(request.data, 10)
            response.success = True
            response.message = f"Set target color at normalized coords: ({request.data.x:.2f}, {request.data.y:.2f})"
        return response


    def set_running_srv_callback(self, request, response):
        with self.lock:
            self.is_running = request.data
            if self.is_running:
                self.get_logger().info('Start running')
            else:
                self.mecanum_pub.publish(Twist())
                self.get_logger().info('Stop running')
        response.success = True
        response.message = "set_running"
        return response

    def image_callback(self, ros_image):
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
        result_image = np.copy(cv_image)
        
        with self.lock:
            if self.image_width is None or self.image_height is None:
                self.image_height, self.image_width = cv_image.shape[:2]
            
            if self.color_picker is not None:
                target_color, result_image = self.color_picker(cv_image, result_image)
                if target_color is not None:
                    self.tracker = ObjectTracker(target_color, self)
                    self.color_picker = None
                    self.get_logger().info(f'Color picked. Start tracking.')
            elif self.tracker is not None:
                try:
                    result_image, twist_to_publish = self.tracker(cv_image, result_image, self.threshold)
                    if self.is_running:
                        self.mecanum_pub.publish(twist_to_publish)
                    else:
                        if self.tracker is not None: # 检查tracker是否存在
                            self.mecanum_pub.publish(Twist()) 
                            self.tracker.pid_dist.clear()
                            self.tracker.pid_yaw.clear()
                except Exception as e:
                    self.get_logger().error(f'Error in tracker: {str(e)}')
        
        if self.image_queue.full():
            self.image_queue.get_nowait()
        self.image_queue.put(result_image)


def main(args=None):
    rclpy.init(args=args)
    node = OjbectTrackingNode('object_tracking')
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
