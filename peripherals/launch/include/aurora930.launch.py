from launch_ros.actions import Node
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
import time

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('transform_publish_delay', default_value='1.0', description="Delay before publishing static transforms"),


        Node(
            package="deptrum-ros-driver-aurora930",
            executable="aurora930_node",
            namespace="aurora",
            parameters=[
                {"rgb_enable": LaunchConfiguration('rgb_enable', default=True),
                 "ir_enable": LaunchConfiguration('ir_enable', default=True),
                 "depth_enable": LaunchConfiguration('depth_enable', default=True),
                 "rgbd_enable": LaunchConfiguration('rgbd_enable', default=False),
                 "point_cloud_enable": LaunchConfiguration('point_cloud_enable', default=True),
                 "boot_order": LaunchConfiguration('boot_order', default=1),
                 "ir_fps": LaunchConfiguration('ir_fps', default=12),
                 "rgb_fps": LaunchConfiguration('rgb_fps', default=12),
                 "exposure_enable": LaunchConfiguration('exposure_enable', default=False),
                 "exposure_time": LaunchConfiguration('exposure_time', default=10),
                 "gain_enable": LaunchConfiguration('gain_enable', default=True),
                 "gain_value": LaunchConfiguration('gain_value', default=10),
                 "usb_port_number": LaunchConfiguration('usb_port_number', default=""),
                 "threshold_size": LaunchConfiguration('threshold_size', default=110),
                 "depth_correction": LaunchConfiguration('depth_correction', default=True),
                 "align_mode": LaunchConfiguration('align_mode', default=True),
                 "laser_power": LaunchConfiguration('laser_power', default=1.0),
                 "minimum_filter_depth_value": LaunchConfiguration('minimum_filter_depth_value', default=150),
                 "maximum_filter_depth_value": LaunchConfiguration('maximum_filter_depth_value', default=4000),
                 "resolution_mode_index": LaunchConfiguration('resolution_mode_index', default=2),
                 "log_dir": LaunchConfiguration('log_dir', default="/tmp/"),
                 "stream_sdk_log_enable": LaunchConfiguration('stream_sdk_log_enable', default=False),
                 "heart_enable": LaunchConfiguration('heart_enable', default=False),
                 "update_file_path": LaunchConfiguration('update_file_path', default=""),
                 }
            ],
            remappings=[
                ("/aurora/depth/image_raw", "/depth_cam/depth/image_raw"),
                ("/aurora/ir/camera_info", "/depth_cam/depth/camera_info"),
                ("/aurora/ir/image_raw", "/depth_cam/ir/image_raw"),
                ("/aurora/points2", "/depth_cam/depth_registered/points"),
                ("/aurora/rgb/camera_info", "/depth_cam/rgb/camera_info"),
                ("/aurora/rgb/image_raw", "/depth_cam/rgb/image_raw"),
            ],
            output="screen",
        ),
    

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='depth_camera_to_rgb_camera_link',
            arguments=[
                '--x', '0',
                '--y', '0',
                '--z', '0',
                '--qx', '0',
                '--qy', '0',
                '--qz', '0',
                '--qw', '1',
                '--frame-id', 'depth_camera_link',
                '--child-frame-id', 'rgb_camera_link'
            ],
            parameters=[{"transform_publish_delay": LaunchConfiguration('transform_publish_delay')}]
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='rgb_camera_to_color_frame', 
            arguments=[
                '--x', '0',
                '--y', '0',
                '--z', '0',
                '--qx', '0',
                '--qy', '0',
                '--qz', '0',
                '--qw', '1',
                '--frame-id', 'rgb_camera_link',
                '--child-frame-id', 'depth_cam_color_frame'
            ],
            parameters=[{"transform_publish_delay": LaunchConfiguration('transform_publish_delay')}]
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='rgb_camera_to_depth_optical_frame', 
            arguments=[
                '--x', '0',
                '--y', '0',
                '--z', '0',
                '--qx', '0',
                '--qy', '0',
                '--qz', '0',
                '--qw', '1',
                '--frame-id', 'rgb_camera_link',
                '--child-frame-id', 'depth_cam_depth_optical_frame'
            ],
            parameters=[{"transform_publish_delay": LaunchConfiguration('transform_publish_delay')}]
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='color_frame_to_color_optical_frame', #
            arguments=[
                '--x', '0',
                '--y', '0',
                '--z', '0',
                '--qx', '0',
                '--qy', '0',
                '--qz', '0',
                '--qw', '1',
                '--frame-id', 'depth_cam_color_frame',
                '--child-frame-id', 'depth_cam_color_optical_frame'
            ],
            parameters=[{"transform_publish_delay": LaunchConfiguration('transform_publish_delay')}]
        ),

    ])
