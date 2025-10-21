#!/usr/bin/env python3
# coding=utf8
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from ros_robot_controller_msgs.msg import MotorSpeedControl, MotorsSpeedControl
import math

class MecanumChassis:
    def __init__(self, wheelbase=0.195, track_width=0.22, wheel_diameter=0.0965):
        self.wheelbase = wheelbase
        self.track_width = track_width
        self.wheel_diameter = wheel_diameter
        
        self.max_motor_speed_rps = 127.0
        self.min_motor_speed_rps = -126.0 

    def speed_covert(self, speed):
        return speed / (math.pi * self.wheel_diameter)

    def set_velocity(self, linear_x, linear_y, angular_z):
        K = (self.wheelbase + self.track_width) / 2
        motor1_linear = (linear_x + linear_y + angular_z * K)
        motor2_linear = (linear_x - linear_y + angular_z * K)
        motor3_linear = (linear_x - linear_y - angular_z * K)
        motor4_linear = (linear_x + linear_y - angular_z * K)
        # 对应 Motor ID 1, 2, 3, 4
        motor1_rps = self.speed_covert(motor1_linear)
        motor2_rps = self.speed_covert(-motor2_linear) 
        motor3_rps = self.speed_covert(-motor3_linear)
        motor4_rps = self.speed_covert(motor4_linear)

        target_rps_values = [motor1_rps, motor2_rps, motor3_rps, motor4_rps]


        max_abs_target_rps = 0.0
        if target_rps_values:
            max_abs_target_rps = max(abs(v) for v in target_rps_values)

        scale_factor = 1.0
        if max_abs_target_rps > self.max_motor_speed_rps and max_abs_target_rps > 0:
            scale_factor = self.max_motor_speed_rps / max_abs_target_rps

        scaled_rps_values = [v * scale_factor for v in target_rps_values]

        data = []
        for i in range(len(scaled_rps_values)):
            msg = MotorSpeedControl()
            msg.id = i + 1  # 电机ID从1开始
            
            msg.speed = float(max(min(scaled_rps_values[i], self.max_motor_speed_rps), self.min_motor_speed_rps))
            data.append(msg)

        motors_speed_msg = MotorsSpeedControl()
        motors_speed_msg.data = data
        return motors_speed_msg

class MecanumChassisNode(Node):
    def __init__(self):
        super().__init__('mecanum_chassis_node')
        self.chassis = MecanumChassis()

        self.sub_cmd_vel = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10)

        self.pub_motors_speed = self.create_publisher(
            MotorsSpeedControl,
            '/ros_robot_controller/set_motor_speeds', 
            10)

        # self.get_logger().info('Mecanum chassis node started, waiting for /cmd_vel messages...')

    def cmd_vel_callback(self, msg):
        linear_x = msg.linear.x
        linear_y = msg.linear.y
        angular_z = msg.angular.z

        # self.get_logger().info(
        #     f"Received /cmd_vel: linear_x={linear_x:.3f}, linear_y={linear_y:.3f}, angular_z={angular_z:.3f}")

        motors_msg = self.chassis.set_velocity(linear_x, linear_y, angular_z)
        self.pub_motors_speed.publish(motors_msg)
        # self.get_logger().info("Published MotorsSpeedControl message with scaled and clamped speeds.")

def main(args=None):
    rclpy.init(args=args)
    node = MecanumChassisNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
