#!/usr/bin/python3
#coding=utf8
# YOLOv8 OBB识别节点
"""
An example that uses TensorRT's Python api to make inferences.
"""
import ctypes
import os
import math
import rclpy
import shutil
import sys
import threading
import time
import cv2
import numpy as np
#import tensorrt as trt
import sdk.fps as fps

CONF_THRESH = 0.5
IOU_THRESHOLD = 0.4
POSE_NUM = 17 * 3
DET_NUM = 6
SEG_NUM = 32
OBB_NUM = 1

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
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


def xywhr2xyxyxyxy(x, origin_h=None, origin_w=None):
    """
    Convert batched Oriented Bounding Boxes (OBB) from [xywh, rotation] to [xy1, xy2, xy3, xy4]. Rotation values should
    be in degrees from 0 to 90.

    Args:
        rboxes (numpy.ndarray | torch.Tensor): Boxes in [cx, cy, w, h, rotation] format of shape (n, 5) or (b, n, 5).

    Returns:
        (numpy.ndarray | torch.Tensor): Converted corner points of shape (n, 4, 2) or (b, n, 4, 2).
    """
    #print("x",x)
    cos, sin = np.cos, np.sin
    # if isinstance(x, tuple):
        # x = np.array(x)
    ctr = x[..., :2]
    w, h, angle = (x[..., i: i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = np.concatenate(vec1, axis=-1)
    vec2 = np.concatenate(vec2, axis=-1)
    #print("ctr + vec1 + vec2",ctr + vec1 + vec2)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return np.stack([pt1, pt2, pt3, pt4], axis=-2)


def plot_one_box(x, img, color=None, label=None, line_thickness=None, rotated=False):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param: 
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return

    """
    tl = (line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) ) + 1)  # line/font thickness
    if rotated:
        box = xywhr2xyxyxyxy(x).reshape(-1, 4, 2).squeeze()
        # box[0][1] -= size
        # box[1][1] -= size
        # box[2][1] -= size
        # box[3][1] -= size

        p1 = [int(b) for b in box[0]]
  
        #print("p1,p2",p1,p2)
        #cv2.circle(img,(int(p1[1]),int(p1[1])),5,(0,255,0),-1)
 
        # NOTE: cv2-version polylines needs np.asarray type.
        cv2.polylines(img, [np.asarray(box, dtype=int)], True, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h >= 3
            p2 = [p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3]
            cv2.rectangle(img, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(
                img,
                label,
                (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                tl / 3,
                [225, 255, 255],
                thickness=tf,
                lineType=cv2.LINE_AA,
            )
        return box
    else:
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






class warmUpThread(threading.Thread):
    def __init__(self, yolov8_wrapper):
        threading.Thread.__init__(self)
        self.yolov8_wrapper = yolov8_wrapper

    def run(self):
        batch_image_raw, use_time = self.yolov8_wrapper.infer(self.yolov8_wrapper.get_raw_image_zeros())
        # print('warm_up->{}, time->{:.2f}ms'.format(batch_image_raw[0].shape, use_time * 1000))


if __name__ == "__main__":
    # load custom plugin and engine
    test = True
    running = True
    if test:
        PLUGIN_LIBRARY = "/home/ubuntu/ros2_ws/src/example/example/yolov8/libmyplugins.so"
        engine_file_path = "/home/ubuntu/ros2_ws/src/example/example/yolov8/garbage_classification_640s.engine"

        ctypes.CDLL(PLUGIN_LIBRARY)

        # load DOTAV 1.5 labels

        categories = ["BananaPeel","BrokenBones","CigaretteEnd","DisposableChopsticks","Ketchup","Marker","OralLiquidBottle","PlasticBottle","Plate","StorageBattery","Toothbrush","umbrella"]

        # a YoLov8TRT instance
        yolov8_wrapper = YoLov8TRT(engine_file_path)
        cap = cv2.VideoCapture(-1)
        # if not cap.isOpened():
            # print("Error: Could not open video capture.")
            # exit()

        fps = fps.FPS()
        while running:
            time.sleep(0.01)
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            if not ret or frame is None:
                print("Failed to read frame from video capture")
                continue
            if frame is not None:

                boxes, scores, classid  = yolov8_wrapper.infer(frame)
                for box, cls_conf, cls_id in zip(boxes, scores, classid):
                    color = colors(cls_id, True)
           
                    box =  np.array(box)
                    angle_in_degrees = math.degrees(box[4])
                    box =  box
                    # print("box",box)
                    # print("angle_in_degrees",angle_in_degrees)
                    plot_one_box(
                    box,
                    frame,
                    label="{}:{:.2f}".format(categories[cls_id], cls_conf),
                    color=color,
                    line_thickness=1,
                    rotated=True)
                fps.update()
                frame = fps.show_fps(frame)
                cv2.imshow('frame', frame)
            else:
                time.sleep(0.01)

            key = cv2.waitKey(1)
            if key != -1:
                break

        cap.release()
        cv2.destroyAllWindows()

        yolov8_wrapper.destroy()
