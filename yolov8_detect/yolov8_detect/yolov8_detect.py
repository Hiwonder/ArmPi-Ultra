#!/usr/bin/env python3
# encoding: utf-8
"""
An example that uses TensorRT's Python api to make inferences.
"""
import os
import sys
import time
import ctypes
import shutil
import random
import threading
import numpy as np

import cv2
import numpy as np
from openvino.runtime import Core


def get_img_path_batches(batch_size, img_dir):
    ret = []
    batch = []
    for root, dirs, files in os.walk(img_dir):
        for name in files:
            if len(batch) == batch_size:
                ret.append(batch)
                batch = []
            batch.append(os.path.join(root, name))
    if len(batch) > 0:
        ret.append(batch)
    return ret



class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('DC143C', '7FFF00', 'FF1493', '7CFC00', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
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




YOLOV8_CLASSES = [
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


class Yolov8Detector:
    def __init__(self, model_path, class_names, conf_threshold=0.25, iou_threshold=0.45):
        """
        :param model_path: OpenVINO IR (.xml) 模型路径
        :param class_names: 类别名称列表，例如 ["person", "car", ...]
        :param conf_threshold: 最小置信度阈值，低于该值的结果将被过滤
        :param iou_threshold: 非极大值抑制用的 IOU 阈值（可拓展）
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = class_names

        # 初始化 OpenVINO 模型
        self.core = Core()
        self.model = self.core.read_model(model=model_path)
        self.compiled_model = self.core.compile_model(self.model, "CPU")
        self.input_layer = self.model.inputs[0]
        self.output_layer = self.model.outputs[0]

        # 获取输入尺寸
        _, self.input_c, self.input_h, self.input_w = self.input_layer.shape

    def preprocess(self, image):
        image_resized = cv2.resize(image, (self.input_w, self.input_h))
        image_transposed = image_resized.transpose(2, 0, 1)[np.newaxis, :]
        return image_transposed.astype(np.float32) / 255.0

    def infer_video(self, image):
        orig_h, orig_w = image.shape[:2]
        input_tensor = self.preprocess(image)
        outputs = self.compiled_model([input_tensor])[self.output_layer]
        return self.postprocess(outputs, orig_w, orig_h)

    def postprocess(self, preds, orig_w, orig_h):
        preds = np.squeeze(preds)  # shape: (num_detections, 84)
        boxes = []
        scores = []
        class_ids = []

        for pred in preds:
            obj_conf = pred[4]
            if obj_conf < self.conf_threshold:
                continue

            cls_scores = pred[5:]
            cls_id = np.argmax(cls_scores)
            cls_conf = cls_scores[cls_id]
            score = obj_conf * cls_conf  # YOLOv8 置信度计算方式

            if score < self.conf_threshold:
                continue

            cx, cy, w, h = pred[:4]
            x1 = int((cx - w / 2) * orig_w / self.input_w)
            y1 = int((cy - h / 2) * orig_h / self.input_h)
            x2 = int((cx + w / 2) * orig_w / self.input_w)
            y2 = int((cy + h / 2) * orig_h / self.input_h)

            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            class_ids.append(cls_id)

        return boxes, scores, class_ids

    def colors(self, class_id, use_bgr=False):
        np.random.seed(class_id)
        color = np.random.randint(0, 255, size=3)
        return tuple(int(c) for c in (color if use_bgr else color[::-1]))



if __name__ == '__main__':
    classes =  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
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

    yolo = Yolov8Detector("yolov8s.xml",classes,0.45)

    # 在回调中
    boxes, scores, classid = yolo.infer_video(image)

    # for box, cls_conf, cls_id in zip(boxes, scores, classid):
    #     color = self.yolo.colors(cls_id, True)
    #     object_info = ObjectInfo()
    #     object_info.class_name = self.yolo.classes[cls_id]
    #     object_info.box = box
    #     object_info.score = cls_conf
    #     object_info.width = w
    #     object_info.height = h
    #     objects_info.append(object_info)

    #     plot_one_box(box, image, color=color,
    #         label="{} {:.2f}".format(self.yolo.classes[cls_id], cls_conf)
    #     )

    # object_msg = ObjectsInfo()
    # object_msg.objects = objects_info
    # object_pub.publish(object_msg)