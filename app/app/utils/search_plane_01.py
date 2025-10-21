# #!/usr/bin/env python3
# # encoding: utf-8
# # 通过深度相机的内参和三点来计算平面方程
# import time
# import random
# import numpy as np

# class SearchPlane:
#     def __init__(self, depth_image_width, depth_image_height, depth_camera_intrinsics,
#                  camera_roll_offset=-10.0, camera_pitch_offset=0.0, camera_yaw_offset=0.0,
#                  roi_rect=None):
#         """
#         初始化平面搜索类
#         Args:
#             depth_image_width: 深度图像宽度
#             depth_image_height: 深度图像高度
#             depth_camera_intrinsics: 深度相机内参 [fx, fy, cx, cy]
#             camera_roll_offset: 相机 Roll 轴的倾斜角度 (度)
#             camera_pitch_offset: 相机 Pitch 轴的倾斜角度 (度)
#             camera_yaw_offset: 相机 Yaw 轴的倾斜角度 (度)
#             roi_rect: 感兴趣区域矩形 [x, y, width, height]
#         """
#         # 相机内参
#         self.fx = depth_camera_intrinsics[0]  # 焦距x
#         self.fy = depth_camera_intrinsics[1]  # 焦距y
#         self.cx = depth_camera_intrinsics[2]  # 光心x
#         self.cy = depth_camera_intrinsics[3]  # 光心y

#         # 图像尺寸
#         self.width = depth_image_width
#         self.height = depth_image_height

#         # 相机倾斜角度
#         self.camera_roll_offset = camera_roll_offset
#         self.camera_pitch_offset = camera_pitch_offset
#         self.camera_yaw_offset = camera_yaw_offset

#         # ROI区域设置
#         self.roi_rect = roi_rect
#         self.roi_mask = np.zeros(shape=[self.height, self.width], dtype=np.uint8)
#         if self.roi_rect is not None:
#             self.roi_mask[roi_rect[1]:roi_rect[1] + roi_rect[3],
#             roi_rect[0]:roi_rect[0] + roi_rect[2]] = 1
#         else:
#             self.roi_mask[:, :] = 1

#         # 构建相机旋转矩阵
#         self.camera_rotation_matrix = self._build_camera_rotation_matrix(
#             self.camera_roll_offset, self.camera_pitch_offset, self.camera_yaw_offset)

#     def _build_camera_rotation_matrix(self, roll, pitch, yaw):
#         """构建相机旋转矩阵"""
#         roll_rad = np.radians(roll)
#         pitch_rad = np.radians(pitch)
#         yaw_rad = np.radians(yaw)

#         # 绕 X 轴旋转 (Roll)
#         rotation_x = np.array([
#             [1, 0, 0],
#             [0, np.cos(roll_rad), -np.sin(roll_rad)],
#             [0, np.sin(roll_rad), np.cos(roll_rad)]
#         ])

#         # 绕 Y 轴旋转 (Pitch)
#         rotation_y = np.array([
#             [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
#             [0, 1, 0],
#             [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
#         ])

#         # 绕 Z 轴旋转 (Yaw)
#         rotation_z = np.array([
#             [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
#             [np.sin(yaw_rad), np.cos(yaw_rad), 0],
#             [0, 0, 1]
#         ])

#         # 组合旋转矩阵 (ZYX 顺序)
#         rotation_matrix = np.dot(rotation_z, np.dot(rotation_y, rotation_x))
#         return rotation_matrix

#     def calculate_distance(self, point1, point2):
#         """计算两点之间的欧氏距离"""
#         return np.sqrt(np.sum((point1 - point2) ** 2))

#     def estimate_plane(self, points, normalize=True):
#         """
#         使用三个点估计平面参数
#         Args:
#             points: 3x3的点坐标数组
#             normalize: 是否归一化法向量
#         Returns:
#             平面参数 [a, b, c, d] 其中 ax + by + cz + d = 0
#         """
#         vector1 = points[1, :] - points[0, :]
#         vector2 = points[2, :] - points[0, :]

#         # 检查点是否共线
#         if not np.all(vector1):
#             return None

#         # 检查两个向量是否平行
#         direction_ratio = vector2 / vector1
#         if not ((direction_ratio[0] != direction_ratio[1]) or
#                 (direction_ratio[2] != direction_ratio[1])):
#             return None

#         # 计算平面法向量
#         normal = np.cross(vector1, vector2)
#         if normalize:
#             normal = normal / np.linalg.norm(normal)

#         # 确保法向量朝向摄像头
#         if normal[2] > 0:  # 假设摄像头视线方向为 [0, 0, 1]
#             normal = -normal

#         # 计算平面方程的d参数
#         d = -np.dot(normal, points[0, :])
#         return np.append(normal, d)

#     def ransac_plane_fit(self, points,
#                          distance_threshold=0.05,
#                          confidence=0.99,
#                          sample_size=3,
#                          max_iterations=1000,
#                          min_point_distance=0.1):
#         """
#         使用RANSAC算法拟合平面
#         Args:
#             points: 点云数据 (已考虑相机倾斜)
#             distance_threshold: 内点判定阈值
#             confidence: 置信度
#             sample_size: 采样点数
#             max_iterations: 最大迭代次数
#             min_point_distance: 采样点之间的最小距离
#         """
#         random.seed(time.time())
#         best_inlier_count = -999
#         iteration_count = 0
#         required_iterations = 10
#         total_points = len(points)
#         point_indices = range(total_points)

#         while iteration_count < required_iterations:
#             # 随机采样三个点
#             sampled_indices = random.sample(point_indices, sample_size)

#             # 检查采样点之间的距离是否满足最小距离要求
#             if (self.calculate_distance(points[sampled_indices[0]], points[sampled_indices[1]]) < min_point_distance or
#                     self.calculate_distance(points[sampled_indices[0]],
#                                             points[sampled_indices[2]]) < min_point_distance or
#                     self.calculate_distance(points[sampled_indices[1]],
#                                             points[sampled_indices[2]]) < min_point_distance):
#                 continue

#             # 估计平面参数
#             plane_params = self.estimate_plane(points[sampled_indices, :])
#             if plane_params is None:
#                 continue

#             # 计算所有点到平面的距离
#             distances = np.abs(np.dot(points, plane_params[:3]) + plane_params[3])
#             inlier_mask = distances < distance_threshold
#             inlier_count = np.sum(inlier_mask)

#             # 更新最佳模型
#             if inlier_count > best_inlier_count:
#                 best_inlier_count = inlier_count
#                 best_plane_params = plane_params
#                 best_inlier_mask = inlier_mask
#                 best_sample = sampled_indices

#                 # 更新所需迭代次数
#                 inlier_ratio = inlier_count / total_points
#                 required_iterations = int(np.log(1 - confidence) /
#                                           np.log(1 - inlier_ratio ** sample_size))

#             iteration_count += 1
#             if iteration_count > max_iterations:
#                 return None, None, None

#         return np.where(best_inlier_mask)[0], best_plane_params, best_sample

#     def find_plane(self, depth_image):
#         """
#         在深度图像中寻找平面
#         Args:
#             depth_image: 深度图像（单位：毫米）
#         Returns:
#             inlier_indices: 平面内点的索引
#             plane_params: 平面参数
#             sampled_indices: 用于估计平面的采样点索引
#         """
#         # 应用ROI掩码
#         if self.roi_rect is not None:
#             depth_image = depth_image * self.roi_mask

#         # 提取有效深度点
#         valid_mask = depth_image > 10
#         rows, cols = np.where(valid_mask)
#         depths = depth_image[valid_mask] / 1000.0  # 转换为米

#         # 计算三维坐标
#         points_3d = np.vstack([
#             (cols - self.cx) * depths / self.fx,  # X坐标
#             (rows - self.cy) * depths / self.fy,  # Y坐标
#             depths  # Z坐标
#         ]).T

#         # 转换到齐次坐标
#         points_3d_homogeneous = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])

#         # 应用相机旋转
#         # 注意：因为 self.camera_rotation_matrix 是用于将 *未倾斜* 的坐标系旋转到 *倾斜* 的坐标系，
#         #       而我们这里要将 *倾斜* 的坐标系转换回 *未倾斜* 的坐标系，所以需要取逆
#         points_3d_corrected = np.dot(points_3d_homogeneous, np.vstack([np.hstack([self.camera_rotation_matrix, np.zeros((3,1))]), [0,0,0,1]]))[:, :3]


#         # 使用RANSAC拟合平面
#         return self.ransac_plane_fit(points_3d_corrected, distance_threshold=0.01)
#!/usr/bin/env python3
# encoding: utf-8
# 通过深度相机的内参和三点来计算平面方程
import time
import random
import numpy as np

class SearchPlane:
    def __init__(self, depth_image_width, depth_image_height, depth_camera_intrinsics, roi_rect=None):
        """
        初始化平面搜索类
        Args:
            depth_image_width: 深度图像宽度
            depth_image_height: 深度图像高度
            depth_camera_intrinsics: 深度相机内参 [fx, fy, cx, cy]
            roi_rect: 感兴趣区域矩形 [x, y, width, height]
        """
        # 相机内参
        self.fx = depth_camera_intrinsics[0]  # 焦距x
        self.fy = depth_camera_intrinsics[1]  # 焦距y
        self.cx = depth_camera_intrinsics[2]  # 光心x
        self.cy = depth_camera_intrinsics[3]  # 光心y

        # 图像尺寸
        self.width = depth_image_width
        self.height = depth_image_height

        # ROI区域设置
        self.roi_rect = roi_rect
        self.roi_mask = np.zeros(shape=[self.height, self.width], dtype=np.uint8)
        if self.roi_rect is not None:
            self.roi_mask[roi_rect[1]:roi_rect[1] + roi_rect[3],
            roi_rect[0]:roi_rect[0] + roi_rect[2]] = 1
        else:
            self.roi_mask[:, :] = 1

    def calculate_distance(self, point1, point2):
        """计算两点之间的欧氏距离"""
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def estimate_plane(self, points, normalize=True):
        """
        使用三个点估计平面参数
        Args:
            points: 3x3的点坐标数组
            normalize: 是否归一化法向量
        Returns:
            平面参数 [a, b, c, d] 其中 ax + by + cz + d = 0
        """
        vector1 = points[1, :] - points[0, :]
        vector2 = points[2, :] - points[0, :]

        # 检查点是否共线
        if not np.all(vector1):
            return None

        # 检查两个向量是否平行
        direction_ratio = vector2 / vector1
        if not ((direction_ratio[0] != direction_ratio[1]) or
                (direction_ratio[2] != direction_ratio[1])):
            return None

        # 计算平面法向量
        normal = np.cross(vector1, vector2)
        if normalize:
            normal = normal / np.linalg.norm(normal)
        
        # 确保法向量朝向摄像头
        if normal[2] > 0:  # 假设摄像头视线方向为 [0, 0, 1]
            normal = -normal

        # 计算平面方程的d参数
        d = -np.dot(normal, points[0, :])
        return np.append(normal, d)

    def ransac_plane_fit(self, points,
                         distance_threshold=0.05,
                         confidence=0.99,
                         sample_size=3,
                         max_iterations=1000,
                         min_point_distance=0.1):
        """
        使用RANSAC算法拟合平面
        Args:
            points: 点云数据
            distance_threshold: 内点判定阈值
            confidence: 置信度
            sample_size: 采样点数
            max_iterations: 最大迭代次数
            min_point_distance: 采样点之间的最小距离
        """
        random.seed(time.time())
        best_inlier_count = -999
        iteration_count = 0
        required_iterations = 10
        total_points = len(points)
        point_indices = range(total_points)

        while iteration_count < required_iterations:
            # 随机采样三个点
            sampled_indices = random.sample(point_indices, sample_size)

            # 检查采样点之间的距离是否满足最小距离要求
            if (self.calculate_distance(points[sampled_indices[0]], points[sampled_indices[1]]) < min_point_distance or
                    self.calculate_distance(points[sampled_indices[0]],
                                            points[sampled_indices[2]]) < min_point_distance or
                    self.calculate_distance(points[sampled_indices[1]],
                                            points[sampled_indices[2]]) < min_point_distance):
                continue

            # 估计平面参数
            plane_params = self.estimate_plane(points[sampled_indices, :])
            if plane_params is None:
                continue

            # 计算所有点到平面的距离
            distances = np.abs(np.dot(points, plane_params[:3]) + plane_params[3])
            inlier_mask = distances < distance_threshold
            inlier_count = np.sum(inlier_mask)

            # 更新最佳模型
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_plane_params = plane_params
                best_inlier_mask = inlier_mask
                best_sample = sampled_indices

                # 更新所需迭代次数
                inlier_ratio = inlier_count / total_points
                required_iterations = int(np.log(1 - confidence) /
                                          np.log(1 - inlier_ratio ** sample_size))

            iteration_count += 1
            if iteration_count > max_iterations:
                return None, None, None

        return np.where(best_inlier_mask)[0], best_plane_params, best_sample

    def find_plane(self, depth_image):
        """
        在深度图像中寻找平面
        Args:
            depth_image: 深度图像（单位：毫米）
        Returns:
            inlier_indices: 平面内点的索引
            plane_params: 平面参数
            sampled_indices: 用于估计平面的采样点索引
        """
        # 应用ROI掩码
        if self.roi_rect is not None:
            depth_image = depth_image * self.roi_mask

        # 提取有效深度点
        valid_mask = depth_image > 10
        rows, cols = np.where(valid_mask)
        depths = depth_image[valid_mask] / 1000.0  # 转换为米

        # 计算三维坐标
        points_3d = np.vstack([
            (cols - self.cx) * depths / self.fx,  # X坐标
            (rows - self.cy) * depths / self.fy,  # Y坐标
            depths  # Z坐标
        ]).T

        # 使用RANSAC拟合平面
        return self.ransac_plane_fit(points_3d, distance_threshold=0.01)
