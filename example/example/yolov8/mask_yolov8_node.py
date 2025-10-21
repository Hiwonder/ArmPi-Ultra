#!/usr/bin/python3
#coding=utf8
# YOLOv8 口罩识别识别节点

import cv2
import os
import time
import math
import queue
import rclpy
import ctypes
import signal
import threading
import numpy as np
import sdk.fps as fps
from sdk import common
from rclpy.node import Node
from ultralytics import YOLO
from cv_bridge import CvBridge
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image
from servo_controller_msgs.msg import ServosPosition
from servo_controller.bus_servo_control import set_servo_position

MODE_PATH = os.path.split(os.path.realpath(__file__))[0]

# class MaskYolov8Node:
class MaskYolov8Node(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.name = name
        self.running = True

        self.start_time = time.time()
        self.frame_count = 0
        self.start = True
        self.fps = fps.FPS()  # fps计算器


        self.bridge = CvBridge()
        self.image_queue = queue.Queue(maxsize=2)

        signal.signal(signal.SIGINT, self.shutdown)
        
        self.joints_pub = self.create_publisher(ServosPosition, 'servo_controller', 1) # 舵机控制
        #等待服务启动
        self.client = self.create_client(Trigger, 'controller_manager/init_finish')
        self.client.wait_for_service()
        set_servo_position(self.joints_pub, 1.5, ((1, 500), (2, 500), (3, 330), (4, 900), (5, 700), (6, 500)))


        model_path = MODE_PATH + "/" + "mask_classification_640s.pt"
        #self.yolov8 = YOLO("/home/ubuntu/ros2_ws/src/example/example/yolov8/mask_classification_640s.pt")
        self.yolov8 = YOLO(model_path)
        self.classes = self.get_parameter('classes').value
        #self.classes = ['no-mask', 'mask']        
                
        self.image_sub = self.create_subscription(Image, 'depth_cam/rgb/image_raw', self.image_callback, 1)
        threading.Thread(target=self.image_proc, daemon=True).start()
        self.create_service(Trigger, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')

    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()

    def get_node_state(self, request, response):
        response.success = True
        return response

    def image_callback(self, ros_image):
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        bgr_image = np.array(cv_image, dtype=np.uint8)
        if self.image_queue.full():
            # 如果队列已满，丢弃最旧的图像
            self.image_queue.get()
            # 将图像放入队列
        self.image_queue.put(bgr_image)
   
    def shutdown(self, signum, frame):
        self.running = False
        self.get_logger().info('\033[1;32m%s\033[0m' % "shutdown")

    def image_proc(self):
        while self.running:
            try:
                result_image = self.image_queue.get(block=True, timeout=1)
            except queue.Empty:
                if not self.running:
                    break
                else:
                    continue

            try:
                if self.start:
                    objects_info = []
                    h, w = result_image.shape[:2]
                    
                    detect_result = self.yolov8(result_image) 
                    for result in detect_result:               
                        boxes = result.boxes
                        if boxes is not None:
                            boxs = boxes.xyxy.numpy()
                            scores = boxes.conf.numpy()
                            classid = boxes.cls.numpy()   
                            
                            for i in range(len(boxs)):
                                 if scores[i] > 0.5: 
                                     x1, y1, x2, y2 = boxs[i] 
                                     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                     cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                     label = f'{self.classes[int(classid[i])]}: {scores[i]:.2f}'
                                     cv2.putText(result_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  
            except BaseException as e:
                print(e)
            
            cv2.imshow("result",result_image)
            cv2.waitKey(1)   
        else:
            time.sleep(0.01)        
        rclpy.shutdown()


def main():
    node = MaskYolov8Node('mask_yolov8')
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
