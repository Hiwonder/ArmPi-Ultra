#!/usr/bin/python3
#coding=utf8
#距离测量

import os
import cv2
import time
import rclpy
import queue
import threading
import numpy as np
import sdk.fps as fps
import mediapipe as mp
import message_filters
from rclpy.node import Node
from cv_bridge import CvBridge
from std_srvs.srv import SetBool
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image, CameraInfo
from rclpy.executors import MultiThreadedExecutor
from servo_controller_msgs.msg import ServosPosition
from rclpy.callback_groups import ReentrantCallbackGroup
from servo_controller.bus_servo_control import set_servo_position

class DistanceMeasureNode(Node):

    def __init__(self, name):
        rclpy.init()
        super().__init__(name)
        self.running = True
        self.fps = fps.FPS()

        self.image_queue = queue.Queue(maxsize=2)
        self.joints_pub = self.create_publisher(ServosPosition, '/servo_controller', 1) # 舵机控制
        
        self.client = self.create_client(Trigger, '/controller_manager/init_finish')
        self.client.wait_for_service()
        # self.client = self.create_client(SetBool, '/depth_cam/set_ldp_enable')
        # self.client.wait_for_service()

        rgb_sub = message_filters.Subscriber(self, Image, '/depth_cam/rgb/image_raw')
        depth_sub = message_filters.Subscriber(self, Image, '/depth_cam/depth/image_raw')

        # 同步时间戳, 时间允许有误差在0.02s
        sync = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 4, 0.5)
        sync.registerCallback(self.multi_callback) #执行反馈函数
        
        timer_cb_group = ReentrantCallbackGroup()
        self.timer = self.create_timer(0.0, self.init_process, callback_group=timer_cb_group)
        self.target_point = None
        self.last_event = 0
        cv2.namedWindow("depth")
        cv2.setMouseCallback('depth', self.click_callback)
        threading.Thread(target=self.main, daemon=True).start()
    def init_process(self):
        self.timer.cancel()

        msg = SetBool.Request()
        msg.data = False
        #self.send_request(self.client, msg)

        #set_servo_position(self.joints_pub, 1.5, ((1, 500), (2, 500), (3, 330), (4, 900), (5, 700), (6, 500)))
        set_servo_position(self.joints_pub, 1.5, ((6, 500), (5, 600), (4, 785), (3, 110), (2, 500), (1, 210)))
        time.sleep(1)

        # threading.Thread(target=self.main, daemon=True).start()
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')
    
    def multi_callback(self, ros_rgb_image, ros_depth_image):
        if self.image_queue.full():

            # 如果队列已满，丢弃最旧的图像
            self.image_queue.get()
        # 将图像放入队列
        self.image_queue.put((ros_rgb_image, ros_depth_image))

    def click_callback(self, event, x, y, flags, params):
        if event == cv2.EVENT_RBUTTONDOWN or event == cv2.EVENT_MBUTTONDOWN or event == cv2.EVENT_LBUTTONDBLCLK:
            self.target_point = None
        if event == cv2.EVENT_LBUTTONDOWN and self.last_event != cv2.EVENT_LBUTTONDBLCLK:
            if x >= 640:
                self.target_point = (x - 640, y)
            else:
                self.target_point = (x, y)
        self.last_event = event

    # def send_request(self, client, msg):
    #     future = client.call_async(msg)
    #     while rclpy.ok():
    #         if future.done() and future.result():
    #             return future.result()

    def main(self):
        while self.running:
            try:
                ros_rgb_image, ros_depth_image = self.image_queue.get(block=True, timeout=1)
            except queue.Empty:
                if not self.running:
                    break
                else:
                    self.get_logger().info('\033[1;31m%s\033[0m' % 'sdsdfsd')
                    continue
            try:
                rgb_image = np.ndarray(shape=(ros_rgb_image.height, ros_rgb_image.width, 3), dtype=np.uint8, buffer=ros_rgb_image.data)
                depth_image = np.ndarray(shape=(ros_depth_image.height, ros_depth_image.width), dtype=np.uint16, buffer=ros_depth_image.data)

                h, w = depth_image.shape[:2]

                depth = np.copy(depth_image).reshape((-1, ))
                depth[depth<=0] = 55555
                min_index = np.argmin(depth)
                min_y = min_index // w
                min_x = min_index - min_y * w
                if self.target_point is not None:
                    min_x, min_y = self.target_point

                sim_depth_image = np.clip(depth_image, 0, 2000).astype(np.float64) / 2000 * 255
                depth_color_map = cv2.applyColorMap(sim_depth_image.astype(np.uint8), cv2.COLORMAP_JET)

                txt = 'Dist: {}mm'.format(depth_image[min_y, min_x])
                cv2.circle(depth_color_map, (int(min_x), int(min_y)), 8, (32, 32, 32), -1)
                cv2.circle(depth_color_map, (int(min_x), int(min_y)), 6, (255, 255, 255), -1)
                cv2.putText(depth_color_map, txt, (11, 380), cv2.FONT_HERSHEY_PLAIN, 2.0, (32, 32, 32), 6, cv2.LINE_AA)
                cv2.putText(depth_color_map, txt, (10, 380), cv2.FONT_HERSHEY_PLAIN, 2.0, (240, 240, 240), 2, cv2.LINE_AA)
                
                # 颜色空间转换
                bgr_image = rgb_image  #  再进行转换
                # bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)  

                # 确保尺寸一致
                depth_color_map = cv2.resize(depth_color_map, (rgb_image.shape[1], rgb_image.shape[0]))

                # 确保 depth_color_map 是 BGR 格式
                if len(depth_color_map.shape) == 2:
                    depth_color_map = cv2.cvtColor(depth_color_map, cv2.COLOR_GRAY2BGR)

                cv2.circle(bgr_image, (int(min_x), int(min_y)), 8, (32, 32, 32), -1)
                cv2.circle(bgr_image, (int(min_x), int(min_y)), 6, (255, 255, 255), -1)
                cv2.putText(bgr_image, txt, (11, h - 20), cv2.FONT_HERSHEY_PLAIN, 2.0, (32, 32, 32), 6, cv2.LINE_AA)
                cv2.putText(bgr_image, txt, (10, h - 20), cv2.FONT_HERSHEY_PLAIN, 2.0, (240, 240, 240), 2, cv2.LINE_AA)

                self.fps.update()
                # bgr_image = self.fps.show_fps(bgr_image)
                result_image = np.concatenate([bgr_image, depth_color_map], axis=1)
                result_image = self.fps.show_fps(result_image)
                cv2.imshow("depth", result_image)
                key = cv2.waitKey(1)

            except Exception as e:
                self.get_logger().info('error: ' + str(e))
        rclpy.shutdown()



def main():
    node = DistanceMeasureNode('distance_measure')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
 
if __name__ == "__main__":
    main()
