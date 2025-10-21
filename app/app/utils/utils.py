# #!/usr/bin/python3
# # coding=utf8
# # @Author: Aiden
# # @Date: 2024/12/31
# import cv2
# import copy
# import math
# import numpy as np
# from sdk import common
# from typing import Tuple, List, Optional

# # Physical parameters of the robotic gripper with units in meters. 机械臂夹持器的物理参数(单位:米)
# GRIPPER_HB = 0.014  
# GRIPPER_BC = 0.03 
# GRIPPER_ED = 0.037
# GRIPPER_DC = 0.022
# EDC = math.radians(180 - 21)
# GRIPPER_IH = 0.02
# GRIPPER_IG = 0.005
# LCD = math.acos((GRIPPER_HB - GRIPPER_IG) / GRIPPER_IH)
# GRIPPER_EC = (GRIPPER_ED**2 + GRIPPER_DC**2 - 2 * GRIPPER_ED * GRIPPER_DC * math.cos(EDC)) ** 0.5
# ECD = math.acos((GRIPPER_DC**2 + GRIPPER_EC**2 - GRIPPER_ED**2) / (2 * GRIPPER_DC * GRIPPER_EC))

# def get_gripper_size(angle: float, angle_zero=200) -> Tuple[float, float]:
#     """
#     Compute the gripper's width and height based on its opening angle. 根据夹持器角度计算夹持器的宽度和高度
    
#     Args:
#         angle: Gripper opening angle with a range from 0 to 1000. 夹持器角度(0-1000)
#         angle_zero: Zero reference angle for the gripper, set to 200 by default). 夹持器角度零点(默认200)
#     Returns:
#         width: Gripper width 夹持器宽度
#         height: Gripper height 夹持器高度
#     """
#     angle = math.radians((angle - angle_zero) / 1000 * 180)
     
#     GRIPPER_BJ = math.cos(angle) * GRIPPER_BC
#     GRIPPER_HJ = GRIPPER_HB + GRIPPER_BJ
#     GRIPPER_LC = math.cos(LCD + ECD) * GRIPPER_EC
#     GRIPPER_KE = GRIPPER_HJ - GRIPPER_LC

#     GRIPPER_KE = GRIPPER_HB + math.cos(angle) * GRIPPER_BC - math.cos(LCD + ECD) * GRIPPER_EC

#     GRIPPER_JC = (GRIPPER_BC**2 - GRIPPER_BJ**2)**0.5
#     GRIPPER_LE = (GRIPPER_EC**2 - GRIPPER_LC**2)**0.5
#     gripper_depth = GRIPPER_JC + GRIPPER_LE
#     gripper_width = 2*GRIPPER_KE
    
#     return gripper_width, gripper_depth

# def set_gripper_size(width: float) -> int:
#     """
#     Compute the gripper angle required for a target width. 根据目标宽度计算夹持器需要的角度
    
#     Args:
#         width: Target width in meters. 目标宽度(米)
        
#     Returns:
#         angle: Gripper angle (range: 0–1000). 夹持器角度(0-1000)
#     """
#     width = width / 2
#     a = (width - GRIPPER_HB + math.cos(LCD + ECD) * GRIPPER_EC) / GRIPPER_BC
#     a = max(-1.0, min(1.0, a))
#     return int(math.degrees(math.acos(a)) / 180 * 1000 + 200)

# def world_to_pixels(world_points, K, T):
#     """
#     Convert world coordinates to pixel coordinates. 将世界坐标点转换为像素坐标
#     Args:
#         world_points: List of world coordinates. 世界坐标点列表
#         K: Camera intrinsic matrix 相机内参矩阵
#         T: Extrinsic transformation matrix [R|t] 外参矩阵 [R|t]
#     Returns:
#         pixel_points: List of corresponding pixel coordinates 像素坐标点列表
#     """
#     pixel_points = []
#     for wp in world_points:
#         # Convert world coordinates to homogeneous coordinates. 将世界坐标转换为齐次坐标
#         world_homo = np.append(wp, 1).reshape(4, 1)
#         # Transform to camera coordinates using the extrinsic matrix. 通过外参矩阵转换到相机坐标系
#         camera_point = np.dot(T, world_homo)
#         # Project onto the image plane using the intrinsic matrix. 投影到像素平面
#         pixel_homo = np.dot(K, camera_point[:3])
#         # Normalization 归一化
#         pixel = (pixel_homo / pixel_homo[2])[:2].reshape(-1)
#         pixel_points.append(pixel)
#     return pixel_points

# def calculate_pixel_length(world_length, K, T):
#     """
#     Compute the corresponding pixel length for a given length in world coordinates. 计算世界坐标中的长度在像素坐标中的对应长度
#     Args:
#         world_length: Length in world space 世界坐标中的长度
#         K: Camera intrinsic matrix 相机内参矩阵
#         T: Extrinsic transformation matrix 外参矩阵
#     Returns:
#         pixel_length: Corresponding length in pixel space 像素坐标中的长度
#     """
#     # Define a starting point and direction. 定义起始点和方向
#     start_point = np.array([0, 0, 0])  # starting point 起始点
#     direction = np.array([0, 1, 0])  # y-direction y方向

#     # Compute the endpoint. 计算终点坐标
#     end_point = start_point + direction * world_length
#     # Transform both endpoints to pixel coordinates. 转换两个端点到像素坐标
#     pixels = world_to_pixels([start_point, end_point], K, T)
#     # Calculate Euclidean distance in pixel space. 计算像素距离
#     pixel_length = np.linalg.norm(pixels[1] - pixels[0])

#     return int(pixel_length)

# def get_plane_values(depth_image: np.ndarray, 
#                     plane: Tuple[float, float, float, float],
#                     intrinsic_matrix: np.ndarray) -> np.ndarray:
#     """
#     计算深度图像中每个点到平面的距离 - 自动适应倾斜相机
    
#     Args:
#         depth_image: Depth image 深度图像
#         plane: Plane equation parameters 平面方程参数(a,b,c,d)
#         intrinsic_matrix: Camera intrinsic matrix 相机内参矩阵
        
#     Returns:
#         plane_values: Distance from each pixel to the plane 每个点到平面的距离
#     """
#     a, b, c, d = plane
    
#     # 提取相机内参，支持多种格式
#     if len(intrinsic_matrix) >= 9:  # 3x3矩阵展平或更长
#         fx = intrinsic_matrix[0]
#         fy = intrinsic_matrix[4] 
#         cx = intrinsic_matrix[2]
#         cy = intrinsic_matrix[5]
#     else:  # 直接提供的参数列表 [fx, fy, cx, cy]
#         fx, fy, cx, cy = intrinsic_matrix[:4]
    
#     # 图像尺寸
#     H, W = depth_image.shape
    
#     # 生成像素坐标网格
#     u, v = np.meshgrid(np.arange(W), np.arange(H))
    
#     # 处理深度值，自动判断单位
#     z = depth_image.astype(np.float32)
    
#     # 自动判断深度单位并转换为米
#     if np.max(z[z > 0]) > 50:  # 深度值大于50，可能是毫米
#         z = z / 1000.0
    
#     # 过滤无效深度值
#     valid_mask = (z > 0) & (z < 10.0)  # 有效深度范围：0到10米
    
#     # 计算相机坐标系下的3D坐标
#     x = np.zeros_like(z)
#     y = np.zeros_like(z)
    
#     # 只对有效深度值进行计算
#     x[valid_mask] = (u[valid_mask] - cx) * z[valid_mask] / fx
#     y[valid_mask] = (v[valid_mask] - cy) * z[valid_mask] / fy
    
#     # 计算点到平面的有向距离
#     # 平面方程：ax + by + cz + d = 0
#     plane_distances = np.zeros_like(z)
    
#     # 计算平面法向量的模长
#     normal_magnitude = np.sqrt(a*a + b*b + c*c)
    
#     if normal_magnitude > 1e-6:  # 确保平面方程有效
#         # 计算有向距离
#         plane_distances[valid_mask] = (
#             a * x[valid_mask] + 
#             b * y[valid_mask] + 
#             c * z[valid_mask] + d
#         ) / normal_magnitude
#     else:
#         # 如果平面方程无效，返回深度值作为备选
#         plane_distances = z.copy()
    
#     # 对于无效深度值，设置为无穷大
#     plane_distances[~valid_mask] = np.inf
    
#     return plane_distances


# def robust_plane_detection(depth_image: np.ndarray, 
#                           intrinsic_matrix: np.ndarray,
#                           sample_ratio: float = 0.1) -> Tuple[float, float, float, float]:
#     """
#     鲁棒的平面检测 - 自动从深度图像中检测主平面
    
#     Args:
#         depth_image: 深度图像
#         intrinsic_matrix: 相机内参矩阵
#         sample_ratio: 采样比例（0-1）
#     Returns:
#         plane: 平面参数 (a, b, c, d)
#     """
#     # 提取相机内参
#     if len(intrinsic_matrix) >= 9:
#         fx = intrinsic_matrix[0]
#         fy = intrinsic_matrix[4]
#         cx = intrinsic_matrix[2] 
#         cy = intrinsic_matrix[5]
#     else:
#         fx, fy, cx, cy = intrinsic_matrix[:4]
    
#     H, W = depth_image.shape
    
#     # 转换深度值到米
#     z = depth_image.astype(np.float32)
#     if np.max(z[z > 0]) > 50:
#         z = z / 1000.0
    
#     # 找到有效深度点
#     valid_mask = (z > 0.1) & (z < 5.0)  # 有效深度范围
#     valid_indices = np.where(valid_mask)
    
#     if len(valid_indices[0]) < 100:  # 需要足够的有效点
#         return (0, 0, 1, 0)  # 返回水平平面作为默认值
    
#     # 采样点以提高效率
#     num_points = len(valid_indices[0])
#     num_samples = max(int(num_points * sample_ratio), 1000)
#     sample_indices = np.random.choice(num_points, 
#                                     min(num_samples, num_points), 
#                                     replace=False)
    
#     # 获取采样点的像素坐标
#     v_sample = valid_indices[0][sample_indices]
#     u_sample = valid_indices[1][sample_indices]
#     z_sample = z[v_sample, u_sample]
    
#     # 转换为相机坐标系
#     x_sample = (u_sample - cx) * z_sample / fx
#     y_sample = (v_sample - cy) * z_sample / fy
    
#     # 构建点云
#     points = np.column_stack([x_sample, y_sample, z_sample])
    
#     # 使用RANSAC拟合平面
#     try:
#         from sklearn.linear_model import RANSACRegressor
#         from sklearn.preprocessing import StandardScaler
        
#         # 标准化数据
#         scaler = StandardScaler()
#         X = scaler.fit_transform(points[:, :2])  # x, y坐标
#         y = points[:, 2]  # z坐标
        
#         # RANSAC拟合
#         ransac = RANSACRegressor(
#             max_trials=1000,
#             min_samples=3,
#             residual_threshold=0.01,
#             random_state=42
#         )
#         ransac.fit(X, y)
        
#         # 获取平面参数
#         coef = ransac.estimator_.coef_
#         intercept = ransac.estimator_.intercept_
        
#         # 转换回原始尺度
#         a = coef[0] / scaler.scale_[0]
#         b = coef[1] / scaler.scale_[1]
#         c = -1.0  # z = ax + by + c 形式转换为 ax + by + cz + d = 0
#         d = intercept
        
#         # 标准化平面参数
#         norm = np.sqrt(a*a + b*b + c*c)
#         return (a/norm, b/norm, c/norm, d/norm)
        
#     except ImportError:
#         # 如果没有sklearn，使用简单的最小二乘法
#         A = np.column_stack([points[:, 0], points[:, 1], np.ones(len(points))])
#         b = points[:, 2]
        
#         # 解最小二乘问题：A * [a, b, d] = b，其中平面方程为 z = ax + by + d
#         try:
#             plane_params = np.linalg.lstsq(A, b, rcond=None)[0]
#             a, b, d = plane_params
#             c = -1.0
            
#             # 标准化
#             norm = np.sqrt(a*a + b*b + c*c)
#             return (a/norm, b/norm, c/norm, d/norm)
#         except:
#             return (0, 0, 1, 0)  # 默认水平平面

# def create_roi_mask(
#     depth_image: np.ndarray,
#     bgr_image: np.ndarray,
#     corners: np.ndarray,
#     camera_info: object,
#     extrinsic: np.ndarray,
#     max_height: float,
#     max_obj_height: float,
# ) -> np.ndarray:
#     """
#     Create a region of interest (ROI) mask. 创建感兴趣区域(ROI)的遮罩
#     Args:
#         depth_image: Depth image 深度图像
#         bgr_image: BGR image BGR图像
#         corners: corner coordinates 角点坐标
#         camera_info: Camera intrinsic parameters 相机参数
#         extrinsic: Camera extrinsic matrix 外参矩阵
#         max_height: Maximum threshold height 最大高度
#         max_obj_height: Maximum object height 物体最大高度
#     Returns:
#         mask: ROI mask ROI遮罩
#     """
#     image_height, image_width = depth_image.shape[:2]
    
#     # Decompose the extrinsic matrix 分解外参矩阵
#     translation_vec = extrinsic[:1]
#     rotation_mat = extrinsic[1:]
#     corners_copy = copy.deepcopy(corners)
    
#     # Project the central point 投影中心点
#     center_points, _ = cv2.projectPoints(
#         corners_copy[-1:],
#         np.array(rotation_mat),
#         np.array(translation_vec),
#         np.matrix(camera_info.k).reshape(1, -1, 3),
#         np.array(camera_info.d)
#     )
#     center_points = np.int32(center_points).reshape(2)

#     # Compute new extrinsic matrix after applying plane offset 计算平面偏移后的外参
#     shifted_tvec, shifted_rmat = common.extristric_plane_shift(
#         np.array(translation_vec).reshape((3, 1)),
#         np.array(rotation_mat),
#         max_obj_height
#     )
    
#     # Project other ROI corner points 投影其他角点
#     projected_points, _ = cv2.projectPoints(
#         corners_copy[:-1],
#         np.array(shifted_rmat),
#         np.array(shifted_tvec),
#         np.matrix(camera_info.k).reshape(1, -1, 3),
#         np.array(camera_info.d)
#     )
#     projected_points = np.int32(projected_points).reshape(-1, 2)
    
#     # Calculate the bounding box for the ROI 计算ROI边界
#     x_min = max(0, min(projected_points[:, 0]))
#     x_max = min(image_width, max(projected_points[:, 0]))
#     y_min = max(0, min(projected_points[:, 1]))
#     y_max = min(image_height, max(projected_points[:, 1]))
   
#     # Draw the ROI box on the BGR image 在BGR图像上绘制ROI框
#     # cv2.rectangle(bgr_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

#     # Create ROI are 创建ROI区域
#     x, y = x_min + 10, y_min - 40
#     w, h = x_max - x_min, y_max - y_min
    
#     # create mask 创建遮罩
#     mask = np.zeros_like(depth_image)
#     x2 = min(x + w, image_width)
#     y2 = max(y, 0)
#     mask[y2:y+h, x:x2] = depth_image[y2:y+h, x:x2]

#     # Zero out all regions outside the ROI in the depth image 将深度图像中对应的区域外设置为0
#     depth_image[mask == 0] = max_height

#     return depth_image

# def find_depth_range(depth_image: np.ndarray, max_distance: float) -> Tuple[float, float]:
#     """
#     Find the minimum distance in a depth image. 查找深度图像中的最小
#     Args:
#         depth_image: Depth image 深度图像
#     Returns:
#         min_distance: Minimum distance in millimeters. 最小距离(mm)
#     """
#     height, width = depth_image.shape[:2]
    
#     # Process depth data 处理深度数据
#     depth = np.copy(depth_image).reshape(-1)
#     depth[depth <= 0] = max_distance  # Set invalid values to max_distance 将无效值设为max_distance
    
#     # Find the closest point 找到最近点
#     min_idx = np.argmin(depth)
#     min_y, min_x = min_idx // width, min_idx % width
#     min_distance = depth_image[min_y, min_x] 
    
#     return min_distance

# def extract_contours(
#     plane_values: np.ndarray,
#     filter_height: float
# ) -> List[np.ndarray]:
#     """
#     Extract contours from a depth image. 提取深度图像中的轮廓
#     Args:
#         plane_values: Plane value 平面值
#         filter_height: Height threshold for filtering 过滤高度
#     Returns:
#         contours: List of extracted contours 轮廓列表
#     """
#     # Apply height threshold 过滤高度
#     filtered_image = np.where(plane_values <= filter_height, 0, 255).astype(np.uint8)
    
#     # Perform binarization and contour extraction 二值化和轮廓提取
#     _, binary = cv2.threshold(filtered_image, 1, 255, cv2.THRESH_BINARY)
#     # cv2.imshow(color, binary)
#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
#     return contours

# def convert_depth_to_camera_coords(
#     pixel_coords: Tuple[float, float, float],
#     intrinsic_matrix: np.ndarray
# ) -> np.ndarray:
#     """
#     Convert depth pixel coordinates to camera coordinate system. 将深度像素坐标转换为相机坐标系
#     Args:
#         pixel_coords: Pixel coordinates 像素坐标 (x, y, z)
#         intrinsic_matrix: Camera intrinsic matrix 相机内参矩阵
#     Returns:
#         camera_coords: Coordinates in the camera coordinate system 相机坐标系下的坐标
#     """
#     fx, fy = intrinsic_matrix[0], intrinsic_matrix[4]
#     cx, cy = intrinsic_matrix[2], intrinsic_matrix[5]
#     px, py, pz = pixel_coords
    
#     x = (px - cx) * pz / fx
#     y = (py - cy) * pz / fy
#     z = pz
    
#     return np.array([x, y, z])

# def calculate_world_position(
#     pixel_x: float,
#     pixel_y: float,
#     depth: float,
#     plane: Tuple[float, float, float, float],
#     endpoint: np.ndarray,
#     hand2cam_tf_matrix: np.ndarray,
#     intrinsic_matrix: np.ndarray
# ) -> np.ndarray:
#     """
#     计算世界坐标系中的位置 - 自动适应倾斜相机
#     不依赖固定角度，通过深度信息和变换矩阵自动处理倾斜
    
#     Args:
#         pixel_x: Pixel x-coordinate 像素x坐标
#         pixel_y: Pixel y-coordinate 像素y坐标
#         depth: Depth value 深度值
#         plane: Plane parameters 平面参数 (a, b, c, d)
#         endpoint: End-effector pose 末端执行器位姿
#         hand2cam_tf_matrix: Hand–eye transformation matrix 手眼变换矩阵
#         intrinsic_matrix: Camera intrinsic matrix 相机内参矩阵
#     Returns:
#         world_position: Position in the world coordinate system 世界坐标系中的位置
#     """
#     # 将深度值转换为米（如果是毫米）
#     depth_meters = depth / 1000.0 if depth > 50 else depth
    
#     # 将像素坐标转换为相机坐标系下的3D点
#     camera_position = convert_depth_to_camera_coords(
#         [pixel_x, pixel_y, depth_meters],
#         intrinsic_matrix
#     )
    
#     # 创建齐次坐标点
#     camera_point_homo = np.array([
#         camera_position[0], 
#         camera_position[1], 
#         camera_position[2], 
#         1.0
#     ])
    
#     # 通过手眼标定矩阵转换到末端执行器坐标系
#     # 这里包含了相机倾斜的补偿信息
#     end_effector_point = np.dot(hand2cam_tf_matrix, camera_point_homo)
    
#     # 通过末端执行器位姿转换到世界坐标系
#     world_point_homo = np.dot(endpoint, end_effector_point)
    
#     # 提取世界坐标（前3个分量）
#     world_position = world_point_homo[:3]
    
#     return world_position


# def calculate_world_position_with_plane_constraint(
#     pixel_x: float,
#     pixel_y: float,
#     depth: float,
#     plane: Tuple[float, float, float, float],
#     endpoint: np.ndarray,
#     hand2cam_tf_matrix: np.ndarray,
#     intrinsic_matrix: np.ndarray,
#     use_plane_constraint: bool = False
# ) -> np.ndarray:
#     """
#     带平面约束的世界坐标计算（备用方法）
    
#     Args:
#         pixel_x: 像素x坐标
#         pixel_y: 像素y坐标  
#         depth: 深度值
#         plane: 平面参数 (a, b, c, d)
#         endpoint: 末端执行器位姿
#         hand2cam_tf_matrix: 手眼变换矩阵
#         intrinsic_matrix: 相机内参矩阵
#         use_plane_constraint: 是否使用平面约束
#     Returns:
#         world_position: 世界坐标系中的位置
#     """
#     # 先使用深度信息计算精确位置
#     world_position = calculate_world_position(
#         pixel_x, pixel_y, depth, plane, endpoint, 
#         hand2cam_tf_matrix, intrinsic_matrix
#     )
    
#     # 如果需要平面约束且平面参数有效，则应用平面约束
#     if use_plane_constraint and plane is not None and len(plane) == 4:
#         a, b, c, d = plane
#         # 检查平面方程是否有效（非垂直平面）
#         if abs(c) > 1e-6:
#             # 使用平面方程约束z坐标：ax + by + cz + d = 0
#             z_constrained = -(a * world_position[0] + b * world_position[1] + d) / c
#             # 只有当约束结果合理时才应用
#             if abs(z_constrained - world_position[2]) < 0.1:  # 差异小于10cm
#                 world_position[2] = z_constrained
    
#     return world_position


# def adaptive_depth_to_world_transform(
#     pixel_coords: List[Tuple[float, float]],
#     depth_values: List[float], 
#     intrinsic_matrix: np.ndarray,
#     hand2cam_tf_matrix: np.ndarray,
#     endpoint: np.ndarray
# ) -> List[np.ndarray]:
#     """
#     自适应深度到世界坐标变换
#     处理多个点，自动适应相机倾斜
    
#     Args:
#         pixel_coords: 像素坐标列表 [(x1,y1), (x2,y2), ...]
#         depth_values: 对应的深度值列表
#         intrinsic_matrix: 相机内参矩阵
#         hand2cam_tf_matrix: 手眼变换矩阵
#         endpoint: 末端执行器位姿
#     Returns:
#         world_positions: 世界坐标列表
#     """
#     world_positions = []
    
#     for (px, py), depth in zip(pixel_coords, depth_values):
#         if depth <= 0:  # 跳过无效深度
#             continue
            
#         world_pos = calculate_world_position(
#             px, py, depth, None, endpoint, 
#             hand2cam_tf_matrix, intrinsic_matrix
#         )
#         world_positions.append(world_pos)
    
#     return world_positions
#!/usr/bin/python3
# coding=utf8
# @Author: Aiden
# @Date: 2024/12/31
import cv2
import copy
import math
import numpy as np
from sdk import common
from typing import Tuple, List, Optional

# Physical parameters of the robotic gripper with units in meters. 机械臂夹持器的物理参数(单位:米)
GRIPPER_HB = 0.014  
GRIPPER_BC = 0.03 
GRIPPER_ED = 0.037
GRIPPER_DC = 0.022
EDC = math.radians(180 - 21)
GRIPPER_IH = 0.02
GRIPPER_IG = 0.005
LCD = math.acos((GRIPPER_HB - GRIPPER_IG) / GRIPPER_IH)
GRIPPER_EC = (GRIPPER_ED**2 + GRIPPER_DC**2 - 2 * GRIPPER_ED * GRIPPER_DC * math.cos(EDC)) ** 0.5
ECD = math.acos((GRIPPER_DC**2 + GRIPPER_EC**2 - GRIPPER_ED**2) / (2 * GRIPPER_DC * GRIPPER_EC))

def get_gripper_size(angle: float, angle_zero=200) -> Tuple[float, float]:
    """
    Compute the gripper's width and height based on its opening angle. 根据夹持器角度​计算夹持器的宽度和高度
    
    Args:
        angle: Gripper opening angle with a range from 0 to 1000. 夹持器角度(0-1000)
        angle_zero: Zero reference angle for the gripper, set to 200 by default). 夹持器角度零点(默认200)
    Returns:
        width: Gripper width 夹持器宽度
        height: Gripper height 夹持器高度
    """
    angle = math.radians((angle - angle_zero) / 1000 * 180)
     
    GRIPPER_BJ = math.cos(angle) * GRIPPER_BC
    GRIPPER_HJ = GRIPPER_HB + GRIPPER_BJ
    GRIPPER_LC = math.cos(LCD + ECD) * GRIPPER_EC
    GRIPPER_KE = GRIPPER_HJ - GRIPPER_LC

    GRIPPER_KE = GRIPPER_HB + math.cos(angle) * GRIPPER_BC - math.cos(LCD + ECD) * GRIPPER_EC

    GRIPPER_JC = (GRIPPER_BC**2 - GRIPPER_BJ**2)**0.5
    GRIPPER_LE = (GRIPPER_EC**2 - GRIPPER_LC**2)**0.5
    gripper_depth = GRIPPER_JC + GRIPPER_LE
    gripper_width = 2*GRIPPER_KE
    
    return gripper_width, gripper_depth

def set_gripper_size(width: float) -> int:
    """
    Compute the gripper angle required for a target width. 根据目标宽度计算夹持器需要的角度
    
    Args:
        width: Target width in meters. 目标宽度(米)
        
    Returns:
        angle: Gripper angle (range: 0–1000). 夹持器角度(0-1000)
    """
    width = width / 2
    a = (width - GRIPPER_HB + math.cos(LCD + ECD) * GRIPPER_EC) / GRIPPER_BC
    a = max(-1.0, min(1.0, a))
    return int(math.degrees(math.acos(a)) / 180 * 1000 + 200)

def world_to_pixels(world_points, K, T):
    """
    Convert world coordinates to pixel coordinates. 将世界坐标点转换为像素坐标
    Args:
        world_points: List of world coordinates. 世界坐标点列表
        K: Camera intrinsic matrix 相机内参矩阵
        T: Extrinsic transformation matrix [R|t] 外参矩阵 [R|t]
    Returns​:
        pixel_points: List of corresponding pixel coordinates 像素坐标点列表
    """
    pixel_points = []
    for wp in world_points:
        # Convert world coordinates to homogeneous coordinates. 将世界坐标转换为齐次坐标
        world_homo = np.append(wp, 1).reshape(4, 1)
        # Transform to camera coordinates using the extrinsic matrix. 通过外参矩阵转换到相机坐标系
        camera_point = np.dot(T, world_homo)
        # Project onto the image plane using the intrinsic matrix. 投影到像素平面
        pixel_homo = np.dot(K, camera_point[:3])
        # Normalization 归一化
        pixel = (pixel_homo / pixel_homo[2])[:2].reshape(-1)
        pixel_points.append(pixel)
    return pixel_points

def calculate_pixel_length(world_length, K, T):
    """
    Compute the corresponding pixel length for a given length in world coordinates. 计算世界坐标中的长度在像素坐标中的对应长度
    Args:
        world_length: Length in world space 世界坐标中的长度
        K: Camera intrinsic matrix 相机内参矩阵
        T: Extrinsic transformation matrix 外参矩阵
    Returns:
        pixel_length: Corresponding length in pixel space 像素坐标中的长度
    """
    # Define a starting point and direction. 定义起始点和方向
    start_point = np.array([0, 0, 0])  # starting point 起始点
    direction = np.array([0, 1, 0])  # y-direction y方向

    # Compute the endpoint. 计算终点坐标
    end_point = start_point + direction * world_length
    # Transform both endpoints to pixel coordinates. 转换两个端点到像素坐标
    pixels = world_to_pixels([start_point, end_point], K, T)
    # Calculate Euclidean distance in pixel space. 计算像素距离
    pixel_length = np.linalg.norm(pixels[1] - pixels[0])

    return int(pixel_length)

def get_plane_values(depth_image: np.ndarray, 
                    plane: Tuple[float, float, float, float],
                    intrinsic_matrix: np.ndarray) -> np.ndarray:
    """
    计算深度图像中每个点到平面的距离 - 自动适应倾斜相机
    
    Args:
        depth_image: Depth image 深度图像
        plane: Plane equation parameters 平面方程参数(a,b,c,d)
        intrinsic_matrix: Camera intrinsic matrix 相机内参矩阵
        
    Returns:
        plane_values: Distance from each pixel to the plane 每个点到平面的距离
    """
    a, b, c, d = plane
    

    if len(intrinsic_matrix) >= 9:  # 3x3矩阵展平或更长
        fx = intrinsic_matrix[0]
        fy = intrinsic_matrix[4] 
        cx = intrinsic_matrix[2]
        cy = intrinsic_matrix[5]
    else:  # 直接提供的参数列表 [fx, fy, cx, cy]
        fx, fy, cx, cy = intrinsic_matrix[:4]
    
    # 图像尺寸
    H, W = depth_image.shape
    
    # 生成像素坐标网格
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # 处理深度值，自动判断单位
    z = depth_image.astype(np.float32)
    
    # 自动判断深度单位并转换为米
    if np.max(z[z > 0]) > 50:  # 深度值大于50，可能是毫米
        z = z / 1000.0
    
    # 过滤无效深度值
    valid_mask = (z > 0) & (z < 10.0)  # 有效深度范围：0到10米
    
    # 计算相机坐标系下的3D坐标
    x = np.zeros_like(z)
    y = np.zeros_like(z)
    
    # 只对有效深度值进行计算
    x[valid_mask] = (u[valid_mask] - cx) * z[valid_mask] / fx
    y[valid_mask] = (v[valid_mask] - cy) * z[valid_mask] / fy
    
    # 计算点到平面的有向距离
    # 平面方程：ax + by + cz + d = 0
    plane_distances = np.zeros_like(z)
    
    # 计算平面法向量的模长
    normal_magnitude = np.sqrt(a*a + b*b + c*c)
    
    if normal_magnitude > 1e-6:  # 确保平面方程有效
        # 计算有向距离
        plane_distances[valid_mask] = (
            a * x[valid_mask] + 
            b * y[valid_mask] + 
            c * z[valid_mask] + d
        ) / normal_magnitude
    else:
        # 如果平面方程无效，返回深度值作为备选​
        plane_distances = z.copy()
    
    # 对于无效深度值，设置为无穷大
    plane_distances[~valid_mask] = np.inf
    
    return plane_distances


def robust_plane_detection(depth_image: np.ndarray, 
                          intrinsic_matrix: np.ndarray,
                          sample_ratio: float = 0.1) -> Tuple[float, float, float, float]:
    """
    平面检测 - 自动从深度图像中检测主平面
    
    Args:
        depth_image: 深度图像
        intrinsic_matrix: 相机内参矩阵
        sample_ratio: 采样比例（0-1）
    Returns:
        plane: 平面参数 (a, b, c, d)
    """
    # 提取相机内参
    if len(intrinsic_matrix) >= 9:
        fx = intrinsic_matrix[0]
        fy = intrinsic_matrix[4]
        cx = intrinsic_matrix[2] 
        cy = intrinsic_matrix[5]
    else:
        fx, fy, cx, cy = intrinsic_matrix[:4]
    
    H, W = depth_image.shape
    
    # 转换深度值到米
    z = depth_image.astype(np.float32)
    if np.max(z[z > 0]) > 50:
        z = z / 1000.0
    
    # 找到有效深度点
    valid_mask = (z > 0.1) & (z < 5.0)  # 有效深度范围
    valid_indices = np.where(valid_mask)
    
    if len(valid_indices[0]) < 100:  # 需要足够的有效点
        return (0, 0, 1, 0)  # 返回水平平面作为默认值
    
    # 采样点以提高效率
    num_points = len(valid_indices[0])
    num_samples = max(int(num_points * sample_ratio), 1000)
    sample_indices = np.random.choice(num_points, 
                                    min(num_samples, num_points), 
                                    replace=False)
    
    # 获取采样点的像素坐标
    v_sample = valid_indices[0][sample_indices]
    u_sample = valid_indices[1][sample_indices]
    z_sample = z[v_sample, u_sample]
    
    # 转换为相机坐标系
    x_sample = (u_sample - cx) * z_sample / fx
    y_sample = (v_sample - cy) * z_sample / fy
    
    # 构建点云
    points = np.column_stack([x_sample, y_sample, z_sample])
    
    # 使用RANSAC拟合平面
    try:
        from sklearn.linear_model import RANSACRegressor
        from sklearn.preprocessing import StandardScaler
        
        # 标准化数据
        scaler = StandardScaler()
        X = scaler.fit_transform(points[:, :2])  # x, y坐标
        y = points[:, 2]  # z坐标
        
        # RANSAC拟合
        ransac = RANSACRegressor(
            max_trials=1000,
            min_samples=3,
            residual_threshold=0.01,
            random_state=42
        )
        ransac.fit(X, y)
        
        # 获取平面参数
        coef = ransac.estimator_.coef_
        intercept = ransac.estimator_.intercept_
        
        # 转换回原始尺度
        a = coef[0] / scaler.scale_[0]
        b = coef[1] / scaler.scale_[1]
        c = -1.0  # z = ax + by + c 形式转换为 ax + by + cz + d = 0
        d = intercept
        
        # 标准化平面参数
        norm = np.sqrt(a*a + b*b + c*c)
        return (a/norm, b/norm, c/norm, d/norm)
        
    except ImportError:
        # 如果没有sklearn，使用简单的最小二乘法
        A = np.column_stack([points[:, 0], points[:, 1], np.ones(len(points))])
        b = points[:, 2]
        
        # 解最小二乘问题：A * [a, b, d] = b，其中平面方程为 z = ax + by + d
        try:
            plane_params = np.linalg.lstsq(A, b, rcond=None)[0]
            a, b, d = plane_params
            c = -1.0
            
            # 标准化
            norm = np.sqrt(a*a + b*b + c*c)
            return (a/norm, b/norm, c/norm, d/norm)
        except:
            return (0, 0, 1, 0)  # 默认水平平面

def create_roi_mask(
    depth_image: np.ndarray,
    bgr_image: np.ndarray,
    corners: np.ndarray,
    camera_info: object,
    extrinsic: np.ndarray,
    max_height: float,
    max_obj_height: float,
) -> np.ndarray:
    """
    Create a region of interest (ROI) mask. 创建感兴趣区域(ROI)的遮罩​
    Args:
        depth_image: Depth image 深度图像
        bgr_image: BGR image BGR图像
        corners: corner coordinates 角点坐标
        camera_info: Camera intrinsic parameters 相机参数
        extrinsic: Camera extrinsic matrix 外参矩阵
        max_height: Maximum threshold height 最大高度
        max_obj_height: Maximum object height 物体最大高度
    Returns:
        mask: ROI mask ROI遮罩
    """
    image_height, image_width = depth_image.shape[:2]
    
    # Decompose the extrinsic matrix 分解外参矩阵
    translation_vec = extrinsic[:1]
    rotation_mat = extrinsic[1:]
    corners_copy = copy.deepcopy(corners)
    
    # Project the central point 投影中心点
    center_points, _ = cv2.projectPoints(
        corners_copy[-1:],
        np.array(rotation_mat),
        np.array(translation_vec),
        np.matrix(camera_info.k).reshape(1, -1, 3),
        np.array(camera_info.d)
    )
    center_points = np.int32(center_points).reshape(2)

    # Compute new extrinsic matrix after applying plane offset 计算平面偏移后的外参
    shifted_tvec, shifted_rmat = common.extristric_plane_shift(
        np.array(translation_vec).reshape((3, 1)),
        np.array(rotation_mat),
        max_obj_height
    )
    
    # Project other ROI corner points 投影其他角点
    projected_points, _ = cv2.projectPoints(
        corners_copy[:-1],
        np.array(shifted_rmat),
        np.array(shifted_tvec),
        np.matrix(camera_info.k).reshape(1, -1, 3),
        np.array(camera_info.d)
    )
    projected_points = np.int32(projected_points).reshape(-1, 2)
    
    # Calculate the bounding box for the ROI 计算ROI边界
    x_min = max(0, min(projected_points[:, 0]))
    x_max = min(image_width, max(projected_points[:, 0]))
    y_min = max(0, min(projected_points[:, 1]))
    y_max = min(image_height, max(projected_points[:, 1]))
   
    # 将ROI区域在图像中向下扩展约5cm。
    # 这个像素值是一个估算值，您可以根据实际效果调整。
    # 如果扩展得不够，可以把80改大一点（比如100）；如果太多了，可以改小一点（比如60）。
    offset_pixels_for_5cm = 80  
    y_max = min(y_max + offset_pixels_for_5cm, image_height) #
    # Draw the ROI box on the BGR image 在BGR图像上绘制ROI框
    # cv2.rectangle(bgr_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Create ROI are 创建ROI区域
    x, y = x_min + 10, y_min - 40
    w, h = x_max - x_min, y_max - y_min
    
    # create mask 创建遮罩
    mask = np.zeros_like(depth_image)
    x2 = min(x + w, image_width)
    y2 = max(y, 0)
    mask[y2:y+h, x:x2] = depth_image[y2:y+h, x:x2]

    # Zero out all regions outside the ROI in the depth image 将深度图像中对应的区域外设置为0
    depth_image[mask == 0] = max_height

    return depth_image

def find_depth_range(depth_image: np.ndarray, max_distance: float) -> Tuple[float, float]:
    """
    Find the minimum distance in a depth image. 查找深度图像中的最小
    Args:
        depth_image: Depth image 深度图像
    Returns:
        min_distance: Minimum distance in millimeters. 最小距离(mm)
    """
    height, width = depth_image.shape[:2]
    
    # Process depth data 处理深度数据
    depth = np.copy(depth_image).reshape(-1)
    depth[depth <= 0] = max_distance  # Set invalid values to max_distance 将无效值设为max_distance
    
    # Find the closest point 找到最近点
    min_idx = np.argmin(depth)
    min_y, min_x = min_idx // width, min_idx % width
    min_distance = depth_image[min_y, min_x] 
    
    return min_distance

def extract_contours(
    plane_values: np.ndarray,
    filter_height: float
) -> List[np.ndarray]:
    """
    Extract contours from a depth image. 提取深度图像中的轮廓
    Args:
        plane_values: Plane value 平面值
        filter_height: Height threshold for filtering 过滤高度
    Returns:
        contours: List of extracted contours 轮廓列表
    """
    # Apply height threshold 过滤高度
    filtered_image = np.where(plane_values <= filter_height, 0, 255).astype(np.uint8)
    
    # Perform binarization and contour extraction 二值化和轮廓提取
    _, binary = cv2.threshold(filtered_image, 1, 255, cv2.THRESH_BINARY)
    # cv2.imshow(color, binary)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    return contours

def convert_depth_to_camera_coords(
    pixel_coords: Tuple[float, float, float],
    intrinsic_matrix: np.ndarray
) -> np.ndarray:
    """
    Convert depth pixel coordinates to camera coordinate system. 将深度像素坐标转换为相机坐标系
    Args:
        pixel_coords: Pixel coordinates 像素坐标 (x, y, z)
        intrinsic_matrix: Camera intrinsic matrix 相机内参矩阵
    Returns:
        camera_coords: Coordinates in the camera coordinate system 相机坐标系下的坐标
    """
    fx, fy = intrinsic_matrix[0], intrinsic_matrix[4]
    cx, cy = intrinsic_matrix[2], intrinsic_matrix[5]
    px, py, pz = pixel_coords
    
    x = (px - cx) * pz / fx
    y = (py - cy) * pz / fy
    z = pz
    
    return np.array([x, y, z])

def calculate_world_position(
    pixel_x: float,
    pixel_y: float,
    depth: float,
    plane: Tuple[float, float, float, float],
    endpoint: np.ndarray,
    hand2cam_tf_matrix: np.ndarray,
    intrinsic_matrix: np.ndarray
) -> np.ndarray:
    """
    计算世界坐标系中的位置 - 自动适应倾斜相机
    不依赖固定角度，通过深度信息和变换矩阵自动处理倾斜
    
    Args:
        pixel_x: Pixel x-coordinate 像素x坐标
        pixel_y: Pixel y-coordinate 像素y坐标
        depth: Depth value 深度值
        plane: Plane parameters 平面参数 (a, b, c, d)
        endpoint: End-effector pose 末端执行器位姿
        hand2cam_tf_matrix: Hand–eye transformation matrix 手眼变换矩阵
        intrinsic_matrix​: Camera intrinsic matrix 相机内参矩阵
    Returns:
        world_position: Position in the world coordinate system 世界坐标系中的位置
    """
    # 将深度值转换为米（如果是毫米）
    depth_meters = depth / 1000.0 if depth > 50 else depth
    
    # 将像素坐标转换为相机坐标系下的3D点
    camera_position = convert_depth_to_camera_coords(
        [pixel_x, pixel_y, depth_meters],
        intrinsic_matrix
    )
    
    # 创建齐次坐标点
    camera_point_homo = np.array([
        camera_position[0], 
        camera_position[1], 
        camera_position[2], 
        1.0
    ])
    
    # 通过手眼标定矩阵转换到末端执行器坐标系
    # 这里包含了相机倾斜的补偿信息
    end_effector_point = np.dot(hand2cam_tf_matrix, camera_point_homo)
    
    # 通过末端执行器位姿转换到世界坐标系
    world_point_homo = np.dot(endpoint, end_effector_point)
    
    # 提取世界坐标（前3个分量）
    world_position = world_point_homo[:3]
    
    return world_position


def calculate_world_position_with_plane_constraint(
    pixel_x: float,
    pixel_y: float,
    depth: float,
    plane: Tuple[float, float, float, float],
    endpoint: np.ndarray,
    hand2cam_tf_matrix: np.ndarray,
    intrinsic_matrix: np.ndarray,
    use_plane_constraint: bool = False
) -> np.ndarray:
    """
    带平面约束的世界坐标计算（备用方法）
    
    Args:
        pixel_x: 像素x坐标
        pixel_y: 像素y坐标  
        depth: 深度值
        plane: 平面参数 (a, b, c, d)
        endpoint: 末端执行器位姿
        hand2cam_tf_matrix: 手眼变换矩阵
        intrinsic_matrix: 相机内参矩阵
        use_plane_constraint: 是否使用平面约束
    Returns:
        world_position: 世界坐标系中的位置
    """
    # 先使用深度信息计算精确位置
    world_position = calculate_world_position(
        pixel_x, pixel_y, depth, plane, endpoint, 
        hand2cam_tf_matrix, intrinsic_matrix
    )
    
    # 如果需要平面约束且平面参数有效，则应用平面约束
    if use_plane_constraint and plane is not None and len(plane) == 4:
        a, b, c, d = plane
        # 检查平面方程是否有效（非垂直平面）
        if abs(c) > 1e-6:
            # 使用平面方程约束z坐标：ax + by + cz + d = 0
            z_constrained = -(a * world_position[0] + b * world_position[1] + d) / c
            # 只有当约束结果合理时才应用
            if abs(z_constrained - world_position[2]) < 0.1:  # 差异小于10cm
                world_position[2] = z_constrained
    
    return world_position


def adaptive_depth_to_world_transform(
    pixel_coords: List[Tuple[float, float]],
    depth_values: List[float], 
    intrinsic_matrix: np.ndarray,
    hand2cam_tf_matrix: np.ndarray,
    endpoint: np.ndarray
) -> List[np.ndarray]:
    """
    自适应深度到世界坐标变换
    处理多个点，自动适应相机倾斜
    
    Args:
        pixel_coords: 像素坐标列表 [(x1,y1), (x2,y2), ...]
        depth_values: 对应的深度值列表
        intrinsic_matrix: 相机内参矩阵
        hand2cam_tf_matrix: 手眼变换矩阵
        endpoint: 末端执行器位姿
    Returns:
        world_positions: 世界坐标列表
    """
    world_positions = []
    
    for (px, py), depth in zip(pixel_coords, depth_values):
        if depth <= 0:  # 跳过无效深度
            continue
            
        world_pos = calculate_world_position(
            px, py, depth, None, endpoint, 
            hand2cam_tf_matrix, intrinsic_matrix
        )
        world_positions.append(world_pos)
    
    return world_positions


