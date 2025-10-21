import cv2
import math
import numpy as np
from position_change_detect import position_reorder

class ColorDetection:
    def __init__(self, config, color_list, distance, get_surface=True):
        self.get_surface = get_surface
        self.lab_data = config['lab']
        self.color_list = color_list
        self.distance = distance
        self.last_object_info_list = []
        self.roi = [config['roi']['height'][0], config['roi']['height'][1],
                    config['roi']['width'][0], config['roi']['width'][1]]
        self.min_area = config['area']['min_area']
        self.max_area = config['area']['max_area']
        self.size = config['image_proc_size']
        self.perspective_transformation_matrix = np.array(config['perspective_transformation_matrix']).reshape((3, 3))

    def update_config(self, config):
        self.lab_data = config['lab']
        self.roi = [config['roi']['height'][0], config['roi']['height'][1],
                    config['roi']['width'][0], config['roi']['width'][1]]
        self.min_area = config['area']['min_area']
        self.max_area = config['area']['max_area']
        self.size = config['image_proc_size']

    def update_color(self, color_list):
        self.color_list = color_list

    def adaptive_threshold(self, gray_image):
        return cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
    
    def canny_proc(self, bgr_image):
        mask = cv2.Canny(bgr_image, 20, 60, 3, L2gradient=True)
        mask = 255 - cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))  # 加粗边界，黑白反转
        return mask
    
    def erode_and_dilate(self, binary, kernel=3):
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
        eroded = cv2.erode(binary, element)  # 腐蚀
        dilated = cv2.dilate(eroded, element)  # 膨胀
        return dilated

    def point_remapped(self, point, now, new, data_type=float):
        x, y = point
        now_w, now_h = now
        new_w, new_h = new
        new_x = x * new_w / now_w
        new_y = y * new_h / now_h
        return data_type(new_x), data_type(new_y)

    def get_top_surface(self, rgb_image):
        image_gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        image_mb = cv2.medianBlur(image_gray, 3)  # 中值滤波
        binary = self.adaptive_threshold(image_mb)  # 阈值自适应
        image_gs = cv2.GaussianBlur(rgb_image, (5, 5), 5)  # 高斯模糊去噪
        mask = self.canny_proc(image_gs)  # 边缘检测
        mask1 = cv2.bitwise_and(binary, mask)  # 合并两个提取出来的图像，保留它们共有的地方
        roi_image_mask = cv2.bitwise_and(rgb_image, rgb_image, mask=mask1)  # 和原图做遮罩，保留需要识别的区域
        return roi_image_mask

    def detect(self, bgr_image):
        try:
            img_h, img_w = bgr_image.shape[:2]
            image_resize = cv2.resize(bgr_image, (self.size['width'], self.size['height']), interpolation=cv2.INTER_NEAREST)
            roi = [int(self.roi[0]*self.size['height']),
                   int(self.roi[1]*self.size['height']),
                   int(self.roi[2]*self.size['width']),
                   int(self.roi[3]*self.size['width'])]
            roi_image = image_resize[roi[0]:roi[1], roi[2]:roi[3]]

            if self.get_surface:
                roi_image_mask = self.get_top_surface(roi_image)
            else:
                roi_image_mask = roi_image

            image_gb = cv2.GaussianBlur(roi_image_mask, (3, 3), 3)
            image_lab = cv2.cvtColor(image_gb, cv2.COLOR_BGR2LAB)

            object_info_list = []
            for color in self.color_list:
                if color in self.lab_data:
                    lower = tuple(self.lab_data[color]['min'])
                    upper = tuple(self.lab_data[color]['max'])
                    binary = cv2.inRange(image_lab, lower, upper)
                    dilated = self.erode_and_dilate(binary)
                    contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
                    for c in contours:
                        area = math.fabs(cv2.contourArea(c))
                        if self.min_area <= area <= self.max_area:
                            rect = cv2.minAreaRect(c)
                            corners = np.int0(cv2.boxPoints(rect))
                            for j in range(4):
                                corners[j, 0], corners[j, 1] = self.point_remapped([corners[j, 0] + roi[2], corners[j, 1] + roi[0]],
                                                                               [self.size['width'], self.size['height']], [img_w, img_h], data_type=int)
                            x, y = self.point_remapped([rect[0][0] + roi[2], rect[0][1] + roi[0]],
                                                                               [self.size['width'], self.size['height']], [img_w, img_h], data_type=int)
                            width, height = self.point_remapped([rect[1][0], rect[1][1]],
                                                                 [self.size['width'], self.size['height']], [img_w, img_h], data_type=int)
                            color_index = color + str(index)
                            position = [x, y]
                            size = [width, height]
                            angle = int(round(rect[2]))
                            object_info_list.append([color_index, position, size, angle])

            reorder_object_info_list = position_reorder(object_info_list, self.last_object_info_list, self.distance)
            self.last_object_info_list = reorder_object_info_list

            return bgr_image, reorder_object_info_list
        except BaseException as e:
            print('color detect error:', e)
            return bgr_image, []
