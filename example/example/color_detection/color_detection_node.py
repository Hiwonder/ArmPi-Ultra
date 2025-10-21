import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import Empty
from hiwonder_interfaces.msg import ObjectInfo, ObjectsInfo
from color_detection import ColorDetection
from sdk import common

CONFIG_NAME = '/config'

class ColorDetectionNode(Node):
    def __init__(self):
        super().__init__('color_detection')
        
        # 初始化参数
        self.image = None
        self.running = True
        self.start_detect = False
        
        # 获取参数
        config = self.get_parameter(CONFIG_NAME).get_parameter_value().string_value
        color_list = self.get_parameter('~color_list').get_parameter_value().string_value
        distance = self.get_parameter('~distance').get_parameter_value().double_value
        self.debug = self.get_parameter('~debug').get_parameter_value().bool_value
        self.camera = self.get_parameter('/camera').get_parameter_value().string_value
        
        # 初始化颜色检测类
        self.detect = ColorDetection(config, color_list, distance)
        
        # 创建图像发布器
        self.image_pub = self.create_publisher(Image, '~image_result', 10)
        
        # 物体位姿发布器
        self.color_info_pub = self.create_publisher(ObjectsInfo, '/object/pixel_coords', 10)

        # 创建服务
        self.create_service(Empty, '~enter', self.enter_func)
        self.create_service(Empty, '~exit', self.exit_func)
        self.create_service(Empty, '~start', self.start_func)
        self.create_service(Empty, '~stop', self.stop_func)
        self.create_service(Empty, '~update_color', self.update_color)
        self.create_service(Empty, '~update_param', self.update_param)
        
        self.enable_display = self.get_parameter('~enable_display').get_parameter_value().bool_value
        
        self.get_logger().info("%s init finish" % self.get_name())
        
        if self.debug:
            self.enter_func(None)
            self.start_func(None)
        
        # 启动图像处理
        self.image_proc()

    def enter_func(self, request, response):
        self.image = None
        if self.image_sub is None:
            self.image_sub = self.create_subscription(Image, '/%s/image_rect_color' % self.camera, self.image_callback, 10)
        self.get_logger().info("%s enter" % self.get_name())
        return response

    def exit_func(self, request, response):
        if self.image_sub is not None:
            self.image_sub.destroy()
            self.image_sub = None
        self.image = None
        self.start_detect = False
        self.get_logger().info('%s exit' % self.get_name())
        return response

    def start_func(self, request, response):
        self.start_detect = True
        self.get_logger().info("%s start" % self.get_name())
        return response

    def stop_func(self, request, response):
        self.start_detect = False
        self.get_logger().info("%s stop" % self.get_name())
        return response

    def update_color(self, request, response):
        color_list = self.get_parameter('~color_list').get_parameter_value().string_value
        self.detect.update_color(color_list)
        self.get_logger().info('update color list')
        return response

    def update_param(self, request, response):
        config = self.get_parameter(CONFIG_NAME).get_parameter_value().string_value
        self.detect.update_config(config)
        self.get_logger().info('%s update param' % self.get_name())
        return response

    def image_proc(self):
        while self.running:
            if self.image is not None:
                image = self.image.copy()
                if self.start_detect:  # 开始检测
                    frame_result, poses = self.detect.detect(image)
                    if poses:
                        colors_info = []
                        for p in poses:
                            color_info = ObjectInfo()
                            color_info.label = p[0]
                            color_info.center.x = p[1][0]
                            color_info.center.y = p[1][1]
                            color_info.size.width = p[2][0]
                            color_info.size.height = p[2][1]
                            color_info.yaw = p[3]
                            color_info.height = 0.03
                            colors_info.append(color_info)
                        self.color_info_pub.publish(colors_info)  # 发布位姿
                else:
                    frame_result = image
                    self.get_logger().info("no detection in progress")
                
                # 显示图像
                if self.enable_display:
                    cv2.imshow('color_detection', frame_result)
                    cv2.waitKey(1)

                ros_image = common.cv2_image2ros(frame_result, self.get_name())  # 转为 ROS 图像消息
                self.image_pub.publish(ros_image)  # 发布图像
            else:
                rclpy.sleep(0.01)

    def image_callback(self, ros_image):
        rgb_image = np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=np.uint8,
                               buffer=ros_image.data)  # 转换为 OpenCV 格式
        self.image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

def main(args=None):
    rclpy.init(args=args)
    node = ColorDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
