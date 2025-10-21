#!/usr/bin/python3
# coding=utf8

import cv2
import time
import queue
import rclpy
import signal
import threading
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image
from interfaces.msg import ObjectInfo, ObjectsInfo
from ultralytics import YOLO


class Colors:
    def __init__(self):
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()

MODEL_DEFAULT_NAME = 'yolov8n'
MODAL_PATH = '/home/ubuntu/third_party_ros2/yolo/yolov8/'


CLASSES_NAMES_DEFAULT = []

CLASSES_NAMES_YOLOV8 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

CLASSES_NAMES_GARBAGE_CLASSIFICATION = [
    'BananaPeel',
    'BrokenBones',
    'CigaretteEnd',
    'DisposableChopsticks',
    'Ketchup','Marker','OralLiquidBottle',
    'Plate','PlasticBottle','StorageBattery',
    'Toothbrush', 'Umbrella'
]

CLASSES_NAMES_TRAFFIC = [
    'go', 'right', 'park', 'red', 'green', 'crosswalk'
]

class yoloNode(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)

        # Declare parameters
        # model_name

        # start
        self.start = self.get_parameter('start').get_parameter_value().bool_value if self.has_parameter('start') else False

        # image_topic
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value if self.has_parameter('image_topic') else '/ascamera/camera_publisher/rgb0/image'


        if MODEL_DEFAULT_NAME == 'yolov8n':
            CLASSES_NAMES = CLASSES_NAMES_YOLOV8
        elif MODEL_DEFAULT_NAME == 'garbage_classification':
            CLASSES_NAMES = CLASSES_NAMES_GARBAGE_CLASSIFICATION
        elif MODEL_DEFAULT_NAME == 'best_traffic':
            CLASSES_NAMES = CLASSES_NAMES_TRAFFIC
        else:
            CLASSES_NAMES = CLASSES_NAMES_DEFAULT
        self.classes = CLASSES_NAMES

        # Get parameters
        self.start = self.get_parameter('start').get_parameter_value().bool_value
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.conf_threshold = 0.5
        self.nms_threshold = 0.5

        model_path = MODAL_PATH + f'{MODEL_DEFAULT_NAME}.pt'
        self.model = YOLO(model_path)
        self.get_logger().info(f"Using YOLO model: {model_path}")

        self.bridge = CvBridge()
        self.image_queue = queue.Queue(maxsize=2)
        self.running = True
        self.prev_time = time.time()

        signal.signal(signal.SIGINT, self.shutdown)

        self.create_service(Trigger, '~/start', self.start_srv_callback)
        self.create_service(Trigger, '~/stop', self.stop_srv_callback)
        self.create_service(Trigger, '~/init_finish', self.get_node_state)

        self.image_sub = self.create_subscription(Image, image_topic, self.image_callback, 1)
        self.object_pub = self.create_publisher(ObjectsInfo, '~/object_detect', 1)
        self.result_image_pub = self.create_publisher(Image, '~/object_image', 1)

        threading.Thread(target=self.image_proc, daemon=True).start()

        if self.start:
            self.start_srv_callback(Trigger.Request(), Trigger.Response())

    def get_node_state(self, request, response):
        response.success = True
        return response

    def start_srv_callback(self, request, response):
        self.get_logger().info("Start YOLO detection")
        self.start = True
        response.success = True
        response.message = "start"
        return response

    def stop_srv_callback(self, request, response):
        self.get_logger().info("Stop YOLO detection")
        self.start = False
        response.success = True
        response.message = "stop"
        return response

    def image_callback(self, ros_image):
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        if self.image_queue.full():
            self.image_queue.get()
        self.image_queue.put(cv_image)

    def shutdown(self, signum, frame):
        self.running = False
        self.get_logger().info("Shutting down YOLO node")

    def image_proc(self):
        while self.running:
            try:
                result_image = self.image_queue.get(timeout=1)
            except queue.Empty:
                continue

            if self.start:
                try:
                    objects_info = []
                    h, w = result_image.shape[:2]

                    results = self.model(result_image, imgsz=640, conf=self.conf_threshold, iou=self.nms_threshold)[0]

                    for box in results.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_id = int(box.cls[0])
                        score = float(box.conf[0])
                        cls_name = self.classes[cls_id] if cls_id < len(self.classes) else f"id_{cls_id}"

                        color = colors(cls_id, True)
                        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(result_image, f"{cls_name}:{score:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        object_info = ObjectInfo()
                        object_info.class_name = cls_name
                        object_info.box = [x1, y1, x2, y2]
                        object_info.width = w
                        object_info.height = h
                        object_info.score = score
                        objects_info.append(object_info)

                    object_msg = ObjectsInfo()
                    object_msg.objects = objects_info
                    self.object_pub.publish(object_msg)

                except Exception as e:
                    self.get_logger().error(f"Detection error: {e}")

            now = time.time()
            fps_val = 1.0 / (now - self.prev_time)
            self.prev_time = now
            cv2.putText(result_image, f"FPS: {fps_val:.2f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('result_img',result_image)
            if cv2.waitKey(1) == 'q':
                break
            # self.result_image_pub.publish(self.bridge.cv2_to_imgmsg(result_image, "bgr8"))

        rclpy.shutdown()


def main():
    node = yoloNode('yolo_node')
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
