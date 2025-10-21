# #!/usr/bin/env python3
# # encoding: utf-8
# # @Author: Aiden
# # @Date: 2022/10/22
# import cv2
# import math
# import numpy as np

# range_rgb = {
#     'red': (255, 0, 50),
#     'blue': (0, 50, 255),
#     'green': (0, 255, 50),
#     'black': (0, 0, 0),
#     'white': (255, 255, 255)
# }

# class CropLayer(object):
#     def __init__(self, params, blobs):
#         self.startX = 0
#         self.startY = 0
#         self.endX = 0
#         self.endY = 0

#     def getMemoryShapes(self, inputs):
#         (inputShape, targetShape) = (inputs[0], inputs[1])
#         (batchSize, numChannels) = (inputShape[0], inputShape[1])
#         (H, W) = (targetShape[2], targetShape[3])

#         self.startX = int((inputShape[3] - targetShape[3]) / 2)
#         self.startY = int((inputShape[2] - targetShape[2]) / 2)
#         self.endX = self.startX + W
#         self.endY = self.startY + H

#         return [[batchSize, numChannels, H, W]]

#     def forward(self, inputs):
#         return [inputs[0][:, :, self.startY:self.endY,
#                                 self.startX:self.endX]]

# class Colors:
#     # Ultralytics color palette https://ultralytics.com/
#     def __init__(self):
#         # hex = matplotlib.colors.TABLEAU_COLORS.values()
#         hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
#                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
#         self.palette = [self.hex2rgb('#' + c) for c in hex]
#         self.n = len(self.palette)

#     def __call__(self, i, bgr=False):
#         c = self.palette[int(i) % self.n]
#         return (c[2], c[1], c[0]) if bgr else c

#     @staticmethod
#     def hex2rgb(h):  # rgb order (PIL)
#         return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

# colors = Colors()  # create instance for 'from utils.plots import colors'

# def plot_one_box(x, img, color=None, label=None, line_thickness=None):
#     """
#     description: Plots one bounding box on image img,
#                  this function comes from YoLov5 project.
#     param:
#         x:      a box likes [x1,y1,x2,y2]
#         img:    a opencv image object
#         color:  color to draw rectangle, such as (0,255,0)
#         label:  str
#         line_thickness: int
#     return:
#         no return

#     """
#     tl = (
#             line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
#     )  # line/font thickness
#     color = color or [random.randint(0, 255) for _ in range(3)]
#     c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
#     cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
#     if label:
#         tf = max(tl - 1, 1)  # font thickness
#         t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
#         c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
#         cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
#         cv2.putText(
#             img,
#             label,
#             (c1[0], c1[1] - 2),
#             0,
#             tl / 3,
#             [225, 255, 255],
#             thickness=tf,
#             lineType=cv2.LINE_AA,
#         )

# def show_faces(detect_img, result_img, boxes, landmarks, bbox_color=(0, 255, 0), ll_color=(0, 0, 255)):
#     """Draw bounding boxes and face landmarks on image."""
#     detect_size = detect_img.shape[:2]
#     show_size = result_img.shape[:2]
#     for bb, ll in zip(boxes, landmarks):
#         p1 = point_remapped(bb[:2], detect_size, show_size, data_type=int)
#         p2 = point_remapped(bb[2:4], detect_size, show_size, data_type=int)
#         cv2.rectangle(result_img, p1, p2, bbox_color, 2)
#         for i, p in enumerate(ll):
#             x, y = point_remapped(p, detect_size, show_size, data_type=int)
#             cv2.circle(result_img, (x, y), 2, ll_color, 2)
#     return result_img


# def mp_face_location(results, img):
#     h, w, c, = img.shape
#     boxes = []
#     keypoints = []
#     if results.detections:
#         for detection in results.detections:
#             x_min = detection.location_data.relative_bounding_box.xmin
#             y_min = detection.location_data.relative_bounding_box.ymin
#             width = detection.location_data.relative_bounding_box.width
#             height = detection.location_data.relative_bounding_box.height
#             x_min, y_min = max(x_min * w, 0), max(y_min * h, 0)
#             x_max, y_max = min(x_min + width * w, w), min(y_min + height * h, h)
#             boxes.append((x_min, y_min, x_max, y_max))
#             relative_keypoints = detection.location_data.relative_keypoints
#             keypoints.append([(point.x * w, point.y * h) for point in relative_keypoints])
#     return boxes, keypoints

# def draw_tags(image, tags, corners_color=(0, 125, 255), center_color=(0, 255, 0)):
#     for tag in tags:
#         corners = tag.corners.astype(int)
#         center = tag.center.astype(int)
#         cv2.putText(image, "%d"%tag.tag_id, (int(center[0] - (7 * len("%d"%tag.tag_id))), int(center[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
#         if corners_color is not None:
#             for p in corners:
#                 cv2.circle(image, tuple(p.tolist()), 5, corners_color, -1)
#         if center_color is not None:
#             cv2.circle(image, tuple(center.tolist()), 8, center_color, -1)
#     return image

# def get_area_max_contour(contours, threshold=50):
#     """
#     获取轮廓中面积最重大的一个, 过滤掉面积过小的情况(get the contour whose area is the largest. Filter out those whose area is too small)
#     :param contours: 轮廓列表(contour list)
#     :param threshold: 面积阈值, 小于这个面积的轮廓会被过滤(area threshold. Contour whose area is less than this value will be filtered out)
#     :return: 如果最大的轮廓面积大于阈值则返回最大的轮廓, 否则返回None(if the maximum contour area is greater than this threshold, return the
#     largest contour, otherwise return None)
#     """
#     contour_area_max = 0
#     area_max_contour = None

#     for c in contours:  # 历遍所有轮廓
#         contour_area_temp = math.fabs(cv2.contourArea(c))  # 计算轮廓面积
#         if contour_area_temp > contour_area_max:
#             contour_area_max = contour_area_temp
#             if contour_area_temp > threshold:  # 过滤干扰
#                 area_max_contour = c

#     return area_max_contour,  contour_area_max  # 返回最大的轮廓

# def bgr8_to_jpeg(value, quality=75):
#     """
#     将cv bgr8格式数据转换为jpg格式(convert data in the format of cv bgr8 into jpg)
#     :param value: 原始数据(original data)
#     :param quality:  jpg质量 最大值100(jpg quality. Maximum value is 100)
#     :return:
#     """
#     return bytes(cv2.imencode('.jpg', value)[1])

# class GetObjectSurface:
#     def __init__(self, canny_threshold1=100, canny_threshold2=200):
#         # self.net = cv2.dnn.readNetFromCaffe(proto_path, model_path)  # Removed Caffe net
#         # cv2.dnn_registerLayer("Crop", CropLayer) # Removed Caffe layer registration
#         # self.net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)  # 使用 CUDA 作为后端  # Removed Caffe net settings
#         # self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) # Removed Caffe net settings
#         # self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)  # 目标为 CUDA # Removed Caffe net settings
#         self.canny_threshold1 = canny_threshold1
#         self.canny_threshold2 = canny_threshold2
        
#     def adaptive_threshold(self, gray_image):
#         # 用自适应阈值先进行分割, 过滤掉侧面(Segment using adaptive threshold, and filter out the side view)
#         # cv2.ADAPTIVE_THRESH_MEAN_C： 邻域所有像素点的权重值是一致的(all neighboring pixel values have equal weights)
#         # cv2.ADAPTIVE_THRESH_GAUSSIAN _C ： 与邻域各个像素点到中心点的距离有关，通过高斯方程得到各个点的权重值(the wights if each point are related to the distance between each neighboring pixel and the center pixel, and are calculated using the Gaussian function)
#         binary = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
#         return binary

#     def edge_detect(self, bgr_image):
#         # (H, W) = bgr_image.shape[:2]  # Removed HED-specific code
#         # blob = cv2.dnn.blobFromImage(bgr_image, scalefactor=1.0, size=(W, H),
#         #                              mean=(104.00698794, 116.66876762, 122.67891434),
#         #                              swapRB=False, crop=False)
#         #
#         # self.net.setInput(blob)
#         # hed = self.net.forward()
#         # hed = cv2.resize(hed[0, 0], (W, H))
#         # mask = (255 * hed).astype("uint8")
#         # _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
#         # mask = 255 - cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))

#         gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY) # convert to grayscale
#         mask = cv2.Canny(gray, self.canny_threshold1, self.canny_threshold2) # Apply Canny edge detection
#         mask = 255 - cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))  # Optional: Dilate edges for better coverage
#         return mask

#     def get_top_surface(self, bgr_image):
#         # 为了只提取物体最上层表面(to extract only the top surface of the object)
#         # kernel = np.array([[0, -1, 0],
#                        # [-1, 5, -1],
#                        # [0, -1, 0]])

#         # sharpened = cv2.filter2D(rgb_image, -1, kernel)
#         # image_gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
#         # image_mb = cv2.medianBlur(image_gray, 3)  # 中值滤波(median filtering)

#         # binary = self.adaptive_threshold(image_mb)  # 阈值自适应(adaptive thresholding)
#         # image_gs = cv2.GaussianBlur(rgb_image, (5, 5), 5)  # 高斯模糊去噪(Gaussian blur for noise reduction)
#         mask = self.edge_detect(bgr_image)
#         # mask1 = cv2.bitwise_and(binary,
#                                 # mask)  # 合并两个提取出来的图像，保留他们共有的地方(merge two extracted images and retain their common areas)
#         roi_image_mask = cv2.bitwise_and(bgr_image, bgr_image,
#                                          mask=mask)  # 和原图做遮罩，保留需要识别的区域(mask the original image to retain the area to be recognized)
#         return roi_image_mask


#!/usr/bin/env python3
# encoding: utf-8
# @Author: Aiden
# @Date: 2022/10/22
import cv2
import math
import numpy as np

range_rgb = {
    'red': (255, 0, 50),
    'blue': (0, 50, 255),
    'green': (0, 255, 50),
    'black': (0, 0, 0),
    'white': (255, 255, 255)
}

class CropLayer(object):
    def __init__(self, params, blobs):
        self.startX = 0
        self.startY = 0
        self.endX = 0
        self.endY = 0

    def getMemoryShapes(self, inputs):
        (inputShape, targetShape) = (inputs[0], inputs[1])
        (batchSize, numChannels) = (inputShape[0], inputShape[1])
        (H, W) = (targetShape[2], targetShape[3])

        self.startX = int((inputShape[3] - targetShape[3]) / 2)
        self.startY = int((inputShape[2] - targetShape[2]) / 2)
        self.endX = self.startX + W
        self.endY = self.startY + H

        return [[batchSize, numChannels, H, W]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.startY:self.endY,
                                self.startX:self.endX]]

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

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    tl = (
            line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
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

def show_faces(detect_img, result_img, boxes, landmarks, bbox_color=(0, 255, 0), ll_color=(0, 0, 255)):
    detect_size = detect_img.shape[:2]
    show_size = result_img.shape[:2]
    for bb, ll in zip(boxes, landmarks):
        p1 = point_remapped(bb[:2], detect_size, show_size, data_type=int)
        p2 = point_remapped(bb[2:4], detect_size, show_size, data_type=int)
        cv2.rectangle(result_img, p1, p2, bbox_color, 2)
        for i, p in enumerate(ll):
            x, y = point_remapped(p, detect_size, show_size, data_type=int)
            cv2.circle(result_img, (x, y), 2, ll_color, 2)
    return result_img

def mp_face_location(results, img):
    h, w, c, = img.shape
    boxes = []
    keypoints = []
    if results.detections:
        for detection in results.detections:
            x_min = detection.location_data.relative_bounding_box.xmin
            y_min = detection.location_data.relative_bounding_box.ymin
            width = detection.location_data.relative_bounding_box.width
            height = detection.location_data.relative_bounding_box.height
            x_min, y_min = max(x_min * w, 0), max(y_min * h, 0)
            x_max, y_max = min(x_min + width * w, w), min(y_min + height * h, h)
            boxes.append((x_min, y_min, x_max, y_max))
            relative_keypoints = detection.location_data.relative_keypoints
            keypoints.append([(point.x * w, point.y * h) for point in relative_keypoints])
    return boxes, keypoints

def draw_tags(image, tags, corners_color=(0, 125, 255), center_color=(0, 255, 0)):
    for tag in tags:
        corners = tag.corners.astype(int)
        center = tag.center.astype(int)
        cv2.putText(image, "%d"%tag.tag_id, (int(center[0] - (7 * len("%d"%tag.tag_id))), int(center[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        if corners_color is not None:
            for p in corners:
                cv2.circle(image, tuple(p.tolist()), 5, corners_color, -1)
        if center_color is not None:
            cv2.circle(image, tuple(center.tolist()), 8, center_color, -1)
    return image

def get_area_max_contour(contours, threshold=50):
    contour_area_max = 0
    area_max_contour = None
    for c in contours:
        contour_area_temp = math.fabs(cv2.contourArea(c))
        if contour_area_temp > contour_area_max:
            contour_area_max = contour_area_temp
            if contour_area_temp > threshold:
                area_max_contour = c
    return area_max_contour,  contour_area_max

def bgr8_to_jpeg(value, quality=75):
    return bytes(cv2.imencode('.jpg', value)[1])

class GetObjectSurface:
    def __init__(self, canny_threshold1=100, canny_threshold2=200):
        self.canny_threshold1 = canny_threshold1
        self.canny_threshold2 = canny_threshold2
        
    def adaptive_threshold(self, gray_image):
        binary = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 5)
        return binary

    def canny_proc(self, bgr_image):
        # 边缘提取，用来进一步分割(当两个相同颜色物体靠在一起时，只能靠边缘去区分)(Edge detection is used for further segmentation (when two objects of the same color are adjacent, only edges can be used to distinguish them))
        mask = cv2.Canny(bgr_image, 9, 41, 9, L2gradient=True)
        mask = 255 - cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))  # 加粗边界，黑白反转(thicken the edge and invert black and white)
        return mask

    def get_top_surface(self, rgb_image):
        image_scale = cv2.convertScaleAbs(rgb_image, alpha=1.2, beta=0)
        image_gray = cv2.cvtColor(image_scale, cv2.COLOR_RGB2GRAY)
        image_mb = cv2.medianBlur(image_gray, 3)
        image_gs = cv2.GaussianBlur(image_mb, (3, 3), 3)
        binary = self.adaptive_threshold(image_gs)
        mask = self.canny_proc(image_gs)
        mask1 = cv2.bitwise_and(binary, mask)
        roi_image_mask = cv2.bitwise_and(rgb_image, rgb_image, mask=mask1)
        return roi_image_mask
#!/usr/bin/env python3
# encoding: utf-8
# @Author: Aiden
# @Date: 2022/10/22
#!/usr/bin/env python3
# encoding: utf-8
# @Author: Aiden
# @Date: 2022/10/22
# import cv2
# import math
# import numpy as np
# import random

# range_rgb = {
#     'red': (255, 0, 50),
#     'blue': (0, 50, 255),
#     'green': (0, 255, 50),
#     'black': (0, 0, 0),
#     'white': (255, 255, 255)
# }

# class CropLayer(object):
#     def __init__(self, params, blobs):
#         self.startX = 0
#         self.startY = 0
#         self.endX = 0
#         self.endY = 0

#     def getMemoryShapes(self, inputs):
#         (inputShape, targetShape) = (inputs[0], inputs[1])
#         (batchSize, numChannels) = (inputShape[0], inputShape[1])
#         (H, W) = (targetShape[2], targetShape[3])

#         self.startX = int((inputShape[3] - targetShape[3]) / 2)
#         self.startY = int((inputShape[2] - targetShape[2]) / 2)
#         self.endX = self.startX + W
#         self.endY = self.startY + H

#         return [[batchSize, numChannels, H, W]]

#     def forward(self, inputs):
#         return [inputs[0][:, :, self.startY:self.endY,
#                                 self.startX:self.endX]]

# class Colors:
#     def __init__(self):
#         hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
#                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
#         self.palette = [self.hex2rgb('#' + c) for c in hex]
#         self.n = len(self.palette)

#     def __call__(self, i, bgr=False):
#         c = self.palette[int(i) % self.n]
#         return (c[2], c[1], c[0]) if bgr else c

#     @staticmethod
#     def hex2rgb(h):
#         return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

# colors = Colors()

# def plot_one_box(x, img, color=None, label=None, line_thickness=None):
#     tl = (
#             line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
#     )
#     color = color or [random.randint(0, 255) for _ in range(3)]
#     c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
#     cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
#     if label:
#         tf = max(tl - 1, 1)
#         t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
#         c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
#         cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
#         cv2.putText(
#             img,
#             label,
#             (c1[0], c1[1] - 2),
#             0,
#             tl / 3,
#             [225, 255, 255],
#             thickness=tf,
#             lineType=cv2.LINE_AA,
#         )


# def show_faces(detect_img, result_img, boxes, landmarks, bbox_color=(0, 255, 0), ll_color=(0, 0, 255)):
#     detect_size = detect_img.shape[:2]
#     show_size = result_img.shape[:2]
#     for bb, ll in zip(boxes, landmarks):
#         p1 = point_remapped(bb[:2], detect_size, show_size, data_type=int)
#         p2 = point_remapped(bb[2:4], detect_size, show_size, data_type=int)
#         cv2.rectangle(result_img, p1, p2, bbox_color, 2)
#         for i, p in enumerate(ll):
#             x, y = point_remapped(p, detect_size, show_size, data_type=int)
#             cv2.circle(result_img, (x, y), 2, ll_color, 2)
#     return result_img

# def mp_face_location(results, img):
#     h, w, c, = img.shape
#     boxes = []
#     keypoints = []
#     if results.detections:
#         for detection in results.detections:
#             x_min = detection.location_data.relative_bounding_box.xmin
#             y_min = detection.location_data.relative_bounding_box.ymin
#             width = detection.location_data.relative_bounding_box.width
#             height = detection.location_data.relative_bounding_box.height
#             x_min, y_min = max(x_min * w, 0), max(y_min * h, 0)
#             x_max, y_max = min(x_min + width * w, w), min(y_min + height * h, h)
#             boxes.append((x_min, y_min, x_max, y_max))
#             relative_keypoints = detection.location_data.relative_keypoints
#             keypoints.append([(point.x * w, point.y * h) for point in relative_keypoints])
#     return boxes, keypoints

# def draw_tags(image, tags, corners_color=(0, 125, 255), center_color=(0, 255, 0)):
#     for tag in tags:
#         corners = tag.corners.astype(int)
#         center = tag.center.astype(int)
#         cv2.putText(image, "%d"%tag.tag_id, (int(center[0] - (7 * len("%d"%tag.tag_id))), int(center[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
#         if corners_color is not None:
#             for p in corners:
#                 cv2.circle(image, tuple(p.tolist()), 5, corners_color, -1)
#         if center_color is not None:
#             cv2.circle(image, tuple(center.tolist()), 8, center_color, -1)
#     return image

# def get_area_max_contour(contours, threshold=50):
#     contour_area_max = 0
#     area_max_contour = None
#     for c in contours:
#         contour_area_temp = math.fabs(cv2.contourArea(c))
#         if contour_area_temp > contour_area_max:
#             contour_area_max = contour_area_temp
#             if contour_area_temp > threshold:
#                 area_max_contour = c
#     return area_max_contour,  contour_area_max

# def bgr8_to_jpeg(value, quality=75):
#     return bytes(cv2.imencode('.jpg', value)[1])

# # ###################### MODIFICATION START ######################

# class GetObjectSurface:
#     def __init__(self):
#         """
#         此类使用“漫水填充”算法，根据用户点击来分割物体。
#         """
#         pass

#     def get_top_surface(self, rgb_image, point):
#         """
#         使用“漫水填充”精确分割给定点的物体表面。
#         针对光照不均问题进行了优化。

#         :param rgb_image: 源RGB图像
#         :param point: 用户的点击坐标 (x, y)，作为种子点
#         :return: 仅包含分割后物体的图像，其余部分为黑色
#         """
#         # 1. 验证输入点
#         if point is None or not (0 <= point[0] < rgb_image.shape[1] and 0 <= point[1] < rgb_image.shape[0]):
#             return np.zeros_like(rgb_image)  # 如果点无效，返回黑色图像

#         # 2. 【优化】对图像进行高斯模糊，以平滑颜色渐变和噪点
#         # (5, 5) 是模糊核的大小，可以根据实际效果调整
#         blurred_image = cv2.GaussianBlur(rgb_image, (5, 5), 0)

#         # 3. 设置漫水填充的掩码（Mask）。它必须比图像大2个像素
#         h, w = blurred_image.shape[:2]
#         mask = np.zeros((h + 2, w + 2), np.uint8)

#         # 4. 【优化】定义一个更宽松的颜色容忍度
#         # 这是关键参数。值越低，颜色匹配越严格。
#         # 我们将其从(20, 20, 20)增加到(35, 35, 35)以适应光照变化
#         tolerance = (35, 35, 35)

#         # 5. 执行漫水填充操作
#         # 我们在模糊后的图像副本上操作，以避免修改原始图像
#         # 标志位解释:
#         # - cv2.FLOODFILL_MASK_ONLY: 我们不想改变原图颜色，只想获取掩码
#         # - cv2.FLOODFILL_FIXED_RANGE: 将每个像素颜色与种子点的颜色进行比较，而不是其邻居
#         # - (255 << 8): 一个技巧，告诉floodFill用值255来填充掩码
#         flags = (255 << 8) | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY
#         cv2.floodFill(blurred_image.copy(), mask, point, (255, 255, 255), tolerance, tolerance, flags)

#         # 6. 掩码比图像大2个像素，因此我们将其裁剪以匹配原始图像大小
#         final_mask = mask[1:-1, 1:-1]

#         # 7. 使用生成的掩码从【原始未模糊】的图像中提取分割出的物体
#         # 按位与操作仅保留掩码为白色(255)的像素
#         segmented_object = cv2.bitwise_and(rgb_image, rgb_image, mask=final_mask)
        
#         return segmented_object
