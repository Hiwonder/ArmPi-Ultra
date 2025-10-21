#!/usr/bin/env python3
# encoding: utf-8
# 深度图转换
import os
import cv2
import time
import rclpy
import queue
import signal
import threading
import numpy as np
from sdk import pid
import open3d as o3d
import message_filters
from rclpy.node import Node
from std_srvs.srv import Trigger
from std_srvs.srv import SetBool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, CameraInfo
from rclpy.executors import MultiThreadedExecutor
from servo_controller_msgs.msg import ServosPosition
from rclpy.callback_groups import ReentrantCallbackGroup
from servo_controller.bus_servo_control import set_servo_position

class TrackObjectNode(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        signal.signal(signal.SIGINT, self.shutdown)
        self.scale = 4
        self.proc_size = [int(640/self.scale), int(480/self.scale)]
        self.haved_add = False
        self.get_point = False
        self.display = True
        self.running = True
        self.pc_queue = queue.Queue(maxsize=1)
        self.target_cloud = o3d.geometry.PointCloud() # 要显示的点云

        self.t0 = time.time()
        self.joints_pub = self.create_publisher(ServosPosition, '/servo_controller', 0) # 舵机控制

        timer_cb_group = ReentrantCallbackGroup()

        self.client = self.create_client(Trigger, '/controller_manager/init_finish')
        self.client.wait_for_service()

        # self.client = self.create_client(SetBool, '/depth_cam/set_ldp_enable')
        # self.client.wait_for_service()

        camera_name = 'depth_cam'
        rgb_sub = message_filters.Subscriber(self, Image, '/%s/rgb/image_raw' % camera_name)
        depth_sub = message_filters.Subscriber(self, Image, '/%s/depth/image_raw' % camera_name)
        info_sub = message_filters.Subscriber(self, CameraInfo, '/%s/depth/camera_info' % camera_name)

        # 同步时间戳, 时间允许有误差在0.02s
        sync = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, info_sub], 3, 2)
        sync.registerCallback(self.multi_callback) #执行反馈函数

        self.timer = self.create_timer(0.0, self.init_process, callback_group=timer_cb_group)

    def init_process(self):
        self.timer.cancel()

        set_servo_position(self.joints_pub, 1, ((6, 500), (5, 765), (4, 915), (3, 150), (2, 500), (1, 200)))

        msg = SetBool.Request()
        msg.data = False
        #self.send_request(self.client, msg)

        threading.Thread(target=self.main, daemon=True).start()
        self.create_service(Trigger, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')

    def get_node_state(self, request, response):
        response.success = True
        return response

    # def send_request(self, client, msg):
    #     future = client.call_async(msg)
    #     while rclpy.ok():
    #         if future.done() and future.result():
    #             return future.result()

    def multi_callback(self, ros_rgb_image, ros_depth_image, depth_camera_info):
        print("multi_callback called") # 确认回调函数被调用
        try:
            # ros格式转为numpy
            rgb_image = np.ndarray(shape=(ros_rgb_image.height, ros_rgb_image.width, 3), dtype=np.uint8, buffer=ros_rgb_image.data)
            depth_image = np.ndarray(shape=(ros_depth_image.height, ros_depth_image.width), dtype=np.uint16, buffer=ros_depth_image.data)

            print(f"RGB Image Shape: {rgb_image.shape}, Depth Image Shape: {depth_image.shape}") # 打印图像尺寸
            print(f"RGB Image dtype: {rgb_image.dtype}, Depth Image dtype: {depth_image.dtype}") # 打印图像数据类型

            rgb_image = cv2.resize(rgb_image, tuple(self.proc_size), interpolation=cv2.INTER_NEAREST)
            depth_image = cv2.resize(depth_image, tuple(self.proc_size), interpolation=cv2.INTER_NEAREST)
            intrinsic = o3d.camera.PinholeCameraIntrinsic(int(depth_camera_info.width / self.scale),
                                                               int(depth_camera_info.height / self.scale),
                                                               float(depth_camera_info.k[0] / self.scale), float(depth_camera_info.k[4] / self.scale),
                                                               float(depth_camera_info.k[2] / self.scale), float(depth_camera_info.k[5] / self.scale))

            print(f"Intrinsic Matrix: {intrinsic.intrinsic_matrix}")  # 打印相机内参

            o3d_image_rgb = o3d.geometry.Image(rgb_image)
            o3d_image_depth = o3d.geometry.Image(np.ascontiguousarray(depth_image))

            # rgbd_function --> point_cloud
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_image_rgb, o3d_image_depth, convert_rgb_to_intensity=False)
            # cpu占用大
            pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)#, extrinsic=extrinsic)ic)

            print(f"Point Cloud Size: {len(pc.points)}") # 添加打印语句

            # 去除最大平面，即地面, 距离阈4mm，邻点数，迭代次数
            plane_model, inliers = pc.segment_plane(distance_threshold=0.05,
                     ransac_n=10,
                     num_iterations=50)

            # 保留内点
            inlier_cloud = pc.select_by_index(inliers, invert=True)
            self.target_cloud.points = inlier_cloud.points
            self.target_cloud.colors = inlier_cloud.colors

            # 转180度方便查看
            self.target_cloud.transform(np.asarray([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]))
            try:
                self.pc_queue.put_nowait(self.target_cloud)
                print("Point cloud enqueued")  # 确认点云已入队
            except queue.Full:
                pass

        except BaseException as e:
            print('callback error:', e)
        self.t0 = time.time()

    def shutdown(self, signum, frame):
        self.running = False

    def main(self):
        if self.display:
            # 创建可视化窗口
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name='point cloud', width=640, height=400, visible=1)
        while self.running:
            if not self.haved_add:
                if self.display:
                    try:
                        point_cloud = self.pc_queue.get(block=True, timeout=2)
                        print("Point cloud dequeued") # 确认点云已出队
                    except queue.Empty:
                        continue
                    vis.add_geometry(point_cloud)
                self.haved_add = True
            if self.haved_add:
                try:
                    point_cloud = self.pc_queue.get(block=True, timeout=2)
                    print("Point cloud dequeued") # 确认点云已出队
                except queue.Empty:
                    continue
                # 刷新
                points = np.asarray(point_cloud.points)
                print(f"Points array shape: {points.shape}")  # 确认 points 数组的形状

                if len(points) > 0:
                    twist = Twist()
                    min_index = np.argmax(points[:, 2])
                    min_point = points[min_index]
                    if len(point_cloud.colors) < min_index:
                        continue
                    point_cloud.colors[min_index] = [255, 255, 0]

                    if self.display:
                        vis.update_geometry(point_cloud)
                        #o3d.io.write_point_cloud("output.ply", point_cloud)

                        vis.poll_events()
                        vis.update_renderer()

            else:
                time.sleep(0.01)
        # 销毁所有显示的几何图形
        vis.destroy_window()
        self.get_logger().info('\033[1;32m%s\033[0m' % 'shutdown')
        rclpy.shutdown()

def main():
    node = TrackObjectNode('track_object')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()

if __name__ == "__main__":
    main()
