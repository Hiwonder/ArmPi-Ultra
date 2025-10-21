from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'yolov8_detect'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'models'), glob('models/**')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='2436210442@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "yolov8_node   = yolov8_detect.yolov8_node:main",
            "yolov8_detect = yolov8_detect.yolov8_detect:main",
            "yolov8_detect_demo = yolov8_detect.yolov8_detect_demo:main"
        ],
    },
)
