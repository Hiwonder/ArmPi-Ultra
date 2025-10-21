from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

def generate_launch_description():

    # ros_robot_controller node
    ros_robot_controller_node = Node(
        package='ros_robot_controller',
        executable='ros_robot_controller',
        output='screen',
    )

    # mecanum_chassis node
    mecanum_chassis_node = Node(
        package='controller',
        executable='mecanum',
        name='mecanum_chassis_node',
        output='screen',
        parameters=[],
        remappings=[]
    )

    return LaunchDescription([
        #ros_robot_controller_node,
        mecanum_chassis_node
    ])
