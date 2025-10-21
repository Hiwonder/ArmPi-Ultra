#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
import kinematics.transform as transform
from kinematics.forward_kinematics import ForwardKinematics
from kinematics.inverse_kinematics import get_ik
from servo_controller_msgs.msg import ServoPosition, ServosPosition

class IkDemo(Node):
    def __init__(self):
        super().__init__('ik_demo_node')
        self.servos_pub = self.create_publisher(ServosPosition, '/servo_controller', 1)
        self.client = self.create_client(Trigger, '/controller_manager/init_finish')
        self.client.wait_for_service()
        self.client = self.create_client(Trigger, '/kinematics/init_finish')
        self.client.wait_for_service()

        coordinates = [
            [-0.15, 0.16, 0.25],
            #[-0.15, 0.16, 0.09],
            #[0.21033, 0.02384, 0.020203]
        ]
        for coord in coordinates:
            self.coordinate = coord
            res = get_ik(self.coordinate, 90, [-180, 180])
            if res:
                pulse = transform.angle2pulse(res[0][0])
                self.set_servo_position(self.servos_pub, 1.0, pulse[0])

    def set_servo_position(self, pub, duration, positions):
        msg = ServosPosition()
        msg.duration = float(duration)
        position_list = []
        for i in range(1, 6):
            position = ServoPosition()
            position.id = i
            position.position = float(positions[i-1])
            position_list.append(position)
        msg.position = position_list
        msg.position_unit = "pulse"
        pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    ik_demo_node = IkDemo()
    rclpy.spin(ik_demo_node)
    ik_demo_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
