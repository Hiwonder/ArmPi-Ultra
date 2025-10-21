import os
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch import LaunchDescription, LaunchService
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, OpaqueFunction, TimerAction

def launch_setup(context):
    mode = LaunchConfiguration('mode', default=1)
    mode_arg = DeclareLaunchArgument('mode', default_value=mode)

    peripherals_package_path = get_package_share_directory('peripherals')
    sdk_package_path = get_package_share_directory('sdk')
    large_models_package_path = get_package_share_directory('large_models')

    # 启动 depth_camera_launch
    depth_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(peripherals_package_path, 'launch/depth_camera.launch.py')),
    )

    # 在5秒后启动 sdk_launch 和 large_models_launch
    delayed_launch = TimerAction(
        period=5.0,  # 延迟时间为5秒
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(sdk_package_path, 'launch/armpi_ultra.launch.py')),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(large_models_package_path, 'launch/start.launch.py')),
                launch_arguments={'mode': mode}.items(),
            ),
        ]
    )

    vllm_with_camera_node = Node(
        package='large_models',
        executable='vllm_with_camera',
        output='screen',
    )

    return [depth_camera_launch,  # 立即启动 depth_camera_launch
            delayed_launch,       # 启动延迟后的 sdk_launch 和 large_models_launch
            vllm_with_camera_node,
            ]

def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=launch_setup)
    ])

if __name__ == '__main__':
    # 创建一个LaunchDescription对象
    ld = generate_launch_description()

    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()