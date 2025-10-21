import os

from launch_ros.actions import Node
from launch.conditions import IfCondition
from launch import LaunchDescription, LaunchService
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription, OpaqueFunction, GroupAction, DeclareLaunchArgument

def launch_setup(context):
    compiled = os.environ['need_compile']
    if compiled == 'True':
        sdk_package_path = get_package_share_directory('sdk')
        chassis_package_path = get_package_share_directory('controller')
        peripherals_package_path = get_package_share_directory('peripherals')
    else:
        sdk_package_path = '/home/ubuntu/ros2_ws/src/driver/sdk'
        chassis_package_path = '/home/ubuntu/ros2_ws/src/driver/controller'
        peripherals_package_path = '/home/ubuntu/ros2_ws/src/peripherals'


    object_tracking_node = GroupAction([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(peripherals_package_path, 'launch/depth_camera.launch.py')),
            ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(sdk_package_path, 'launch/armpi_ultra.launch.py')),
            ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(chassis_package_path, 'launch/controller.launch.py')),
            ),

        Node(
            package='example',
            executable='object_tracking',
            output='screen',
            ),
    ])

    return [
            object_tracking_node,
            ]

def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function = launch_setup)
    ])

if __name__ == '__main__':
    # Create a LaunchDescription object. 创建一个LaunchDescription对象
    ld = generate_launch_description()

    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()
