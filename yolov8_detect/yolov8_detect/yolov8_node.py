#!/usr/bin/python3
#coding=utf8
# yolo pt识别节点

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
from cv_bridge import CvBridge
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image
from interfaces.msg import ObjectInfo, ObjectsInfo

from ultralytics import YOLO
from openvino import Core

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLo11 project.
    param:
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return

    """
    tl = (
            line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )



class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
colors = Colors()  # create instance for 'from utils.plots import colors'

# class yoloNode:
class yoloNode(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.name = name
        # self.image_sub = None
        # self.declare_parameter('start', False)
        self.start = self.get_parameter('start').get_parameter_value().bool_value

        # self.bgr_image = None
        self.running = True

        self.start_time = time.time()
        self.frame_count = 0
        self.fps = fps.FPS()  # fps计算器

        self.bridge = CvBridge()
        self.image_queue = queue.Queue(maxsize=2)

        signal.signal(signal.SIGINT, self.shutdown)

        self.core = Core()
        # self.declare_parameter('model', 'yolov8s')
        model_name = self.get_parameter('model').get_parameter_value().string_value
        MODEL_PATH = f'/home/ubuntu/ros2_ws/src/yolov8_detect/models/{model_name}.xml'
        self.net = self.core.compile_model(MODEL_PATH, device_name="AUTO")
        self.output_node = self.net.outputs[0]
        self.ir = self.net.create_infer_request()
        self.prev_time = time.time()

        self.classes = self.get_parameter('classes').value

        self.create_service(Trigger, '~/start', self.start_srv_callback)  # 进入玩法
        self.create_service(Trigger, '~/stop', self.stop_srv_callback)  # 退出玩法

        self.image_sub = self.create_subscription(Image, '/ascamera/camera_publisher/rgb0/image', self.image_callback, 1)

        self.object_pub = self.create_publisher(ObjectsInfo, '~/object_detect', 1)
        self.result_image_pub = self.create_publisher(Image, '~/object_image', 1)
        threading.Thread(target=self.image_proc, daemon=True).start()
        self.create_service(Trigger, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')

        if self.get_parameter('start').value:
            self.start_srv_callback(Trigger.Request(), Trigger.Response())


    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()

    def get_node_state(self, request, response):
        response.success = True
        return response

    def start_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "start yolo detect")

        self.start = True
        response.success = True
        response.message = "start"
        return response


    def stop_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "stop yolo detect")

        self.start = False
        response.success = True
        response.message = "start"
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
                result_image = self.image_queue.get(timeout=1)
            except queue.Empty:
                continue

            try:
                if self.start:
                    objects_info = []
                    h, w = result_image.shape[:2]

                    # ---------- yolo infer----------
                    length = max(h, w)
                    input_image = np.zeros((length, length, 3), dtype=np.uint8)
                    input_image[0:h, 0:w] = result_image
                    scale = length / 640

                    blob = cv2.dnn.blobFromImage(input_image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
                    outputs = self.ir.infer(blob)[self.output_node]
                    outputs = np.array([cv2.transpose(outputs[0])])

                    boxes, scores, class_ids = [], [], []
                    for i in range(outputs.shape[1]):
                        cls_scores = outputs[0][i][4:]
                        _, maxScore, _, (x, cls_id) = cv2.minMaxLoc(cls_scores)
                        if maxScore >= 0.25:
                            box = [
                                outputs[0][i][0] - 0.5 * outputs[0][i][2],
                                outputs[0][i][1] - 0.5 * outputs[0][i][3],
                                outputs[0][i][2],
                                outputs[0][i][3],
                            ]
                            boxes.append(box)
                            scores.append(maxScore)
                            class_ids.append(cls_id)

                    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45)

                    for i in indices:
                        index = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
                        box = boxes[index]
                        x1 = int(box[0] * scale)
                        y1 = int(box[1] * scale)
                        x2 = int((box[0] + box[2]) * scale)
                        y2 = int((box[1] + box[3]) * scale)

                        cls_id = int(class_ids[index])
                        cls_name = self.classes[cls_id]
                        cls_conf = scores[index]

                        color = colors(cls_id, True)
                        # color = [int(c) for c in self.colors[cls_id]]

                        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(result_image, f"{cls_name}:{cls_conf:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        object_info = ObjectInfo()
                        object_info.class_name = cls_name
                        object_info.box = [x1, y1, x2, y2]
                        object_info.width = w
                        object_info.height = h
                        object_info.score = float(cls_conf)
                        objects_info.append(object_info)

                    object_msg = ObjectsInfo()
                    object_msg.objects = objects_info
                    self.object_pub.publish(object_msg)

            except BaseException as e:
                self.get_logger().error(f"Error in detection: {e}")

            # ---------- FPS & 发布 ----------
            now = time.time()
            fps = 1.0 / (now - self.prev_time)
            self.prev_time = now
            cv2.putText(result_image, f"FPS: {fps:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            self.result_image_pub.publish(self.bridge.cv2_to_imgmsg(result_image, "bgr8"))
        
        rclpy.shutdown()


def main():
    node = yoloNode('yolo_node')
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
