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

    # 获取包路径
    large_models_package_path = get_package_share_directory('large_models')

    object_sorting_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(large_models_package_path, 'launch/3d_object_sorting.launch.py')),
    )

    # 在5秒后启动 large_models_launch 和 llm_3d_object_sorting_node
    delayed_launch = TimerAction(
        period=5.0,  # 延迟时间为5秒
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(large_models_package_path, 'launch/start.launch.py')),
                launch_arguments={'mode': mode}.items(),
            ),
            Node(
                package='large_models',
                executable='llm_3d_object_sorting',
                output='screen',
            ),
        ]
    )

    return [mode_arg,
            object_sorting_launch,  # 立即启动 object_sorting_launch
            delayed_launch,          # 启动后延迟的部分
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