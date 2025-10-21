import os
from ament_index_python.packages import get_package_share_directory

from launch_ros.actions import Node
from launch import LaunchDescription, LaunchService
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, OpaqueFunction

def launch_setup(context):
    compiled = os.environ['need_compile']

    if compiled == 'True':
        peripherals_package_path = get_package_share_directory('peripherals')
    else:
        peripherals_package_path = '/home/ubuntu/ros2_ws/src/peripherals'

    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(peripherals_package_path, 'launch/depth_camera.launch.py')),
    )


    yolov8_node = Node(
        package='yolov8_detect',
        executable='yolov8_node',
        output='screen',
        # parameters=[
        #             {'classes': [
        #                 "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        #                 "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
        #                 "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
        #                 "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
        #                 "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
        #                 "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        #                 "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        #                 "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
        #                 "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
        #                 "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        #                 "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        #                 "scissors", "teddy bear", "hair drier", "toothbrush"
        #             ]},
        #             {'model': 'yolov8s','conf': 0.52,'start': True}
        #             ]
        parameters=[{'classes': ['BananaPeel','BrokenBones','CigaretteEnd','DisposableChopsticks','Ketchup','Marker','OralLiquidBottle','PlasticBottle','Plate','StorageBattery','Toothbrush', 'Umbrella']},
                    { "model": "garbage_classification_640s", 'conf': 0.8,'start': True},]
        )
    


    return [
            camera_launch,
            yolov8_node,
            ]

def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function = launch_setup)
    ])

if __name__ == '__main__':
    # 创建一个LaunchDescription对象(create a LaunchDescription object)
    ld = generate_launch_description()

    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()

