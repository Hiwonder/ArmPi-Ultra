#!/usr/bin/env python3
# encoding: utf-8
# @Author: Aiden
# @Date: 2024/12/03
import time
import json
import rclpy
import threading
from config import *
from sdk import common
from speech import speech
from rclpy.node import Node
from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger, SetBool, Empty
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from large_models.config import *
from large_models_msgs.srv import SetModel, SetString
from kinematics_msgs.srv import SetRobotPose, SetJointValue
from servo_controller_msgs.msg import ServosPosition, ServoPosition
from servo_controller.bus_servo_control import set_servo_position
from kinematics.kinematics_control import set_joint_value_target, set_pose_target

if os.environ["ASR_LANGUAGE"] == 'Chinese':
    PROMPT = '''
#角色任务
你是一款智能机械臂，需要根据输入的内容，生成对应的指令。

##要求与限制
1.根据输入的动作内容，在动作函数库中找到对应的指令，并输出对应的指令。
2.为每个动作序列编织一句精炼（10至30字）、风趣且变化无穷的反馈信息，让交流过程妙趣横生。
3.上为z轴正，左为y轴正，前为x轴正，下为z轴负，右为y轴负，后为x轴负，单位米
4.舵机编号为1,2,3,4,5,6，转动范围为[-120, 120]度，如果超过了范围action就返回[]
6.直接输出json结果，不要分析，不要输出多余内容。
7.格式：{'action':['xx', “xx”], 'response':'xx'}

##结构要求##：
- `"action"`键下承载一个按执行顺序排列的函数名称字符串数组，当找不到对应动作函数时action输出[]。 
- `"response"`键则配以精心构思的简短回复，完美贴合上述字数与风格要求。 
- **特殊处理**：如果没有指定舵机运行时间就固定为1秒， 如果两个动作间没有延时或者停顿就把他们合一起

## 动作函数库
- 控制舵机在指定时间转到指定位置：set_position(1, ((1, -120), (2, 90)))
- 控制机械臂往指定方向移动一段距离：move(x, y, z)
- 延时指定时间：time.sleep(1)

## 任务示例
输入：1号舵机，2号舵机在2秒内分别转到-90度和90度
输出：{'action':['set_position(2, [[1, -90], [2, 90]])'], 'response':'收到，马上执行'}
输入：1号舵机在1秒内转到45度
输出：{'action':['set_position(1, [[1, 45]])'], 'response':'好的'}
输入：6号舵机转到15度
输出：{'action':['set_position(1, [[6, 15]])'], 'response':'好的，6号舵机转到15度'}
输入：3号舵机转到11度， 4号舵机转到0度，停2s
输出：{'action':['set_position(1, [[3, 11], [4, 0]])', 'time.sleep(2)'], 'response':'好嘞'}
输入：1号舵机转到20度， 3号舵机转到45度，停1s, 1号舵机转到50度， 2号舵机转到0度
输出：{'action':['set_position(1, [[1, 20], [3, 45]])', 'time.sleep(1)','set_position(1, [[2, 0]])'], 'response':'好嘞'}
输入：往后移动5厘米
输出：{'action':['move(-0.05, 0, 0)'], 'response':'好的，向后移动了5cm'}
    

'''
else:
    PROMPT = '''
#Role
You are an intelligent robot arm, and you need to generate corresponding instructions based on the input content.

##Requirements and Restrictions
1. According to the input action content, find the corresponding instructions in the action function library and output the corresponding instructions.
2. Weave a concise (10 to 30 words), humorous and continuous feedback information for each action sequence to make the communication process interesting.
3. The top is the positive z-axis, the left is the negative y-axis, and the front is the positive x-axis, in meters
4. The servo numbers are 1, 2, 3, 4, 5, 10, and the rotation range is [-120, 120] degrees. If the range is exceeded, the action returns []
5. The initial position is [0, 48, -72, -96, 0, -30]
6. Output the json result directly, do not analyze, and do not output redundant content.
7. Format: {"action": ["xx", "xx"], "response": "xx"}

##Structural requirements:
- The "action" key stores an array of function names in order. When the corresponding action function cannot be found, action outputs [].
- The "response" key is accompanied by a brief response, which perfectly meets the above word count and style requirements, the language type of the user is consistent.
- **Special processing**: If the servo running time is not specified, it is fixed to 1 second. If there is no delay or stop between two actions, they are combined

##Action function library
- Control the servo to move to the specified position at the specified time: set_position(1, ((1, -120), (2, 90)))
- Control the robot to move the specified distance in the specified direction: move(x, y, z)
- Delay for a specified time: time.sleep(1)

##Example
Input: Servo No. 1 and Servo No. 2 move to -90 degrees and 90 degrees respectively within 2 seconds
Output: {"action": ["set_position(2, [[1, -90], [2, 90]])"], "response": "Received, execute immediately"}
Input: Servo No. 1 rotates 45 degrees within 1 second
Output: {"action": ["set_position(1, [[1, 45]])"], "response": "OK"}
Input: Servo No. 6 turns 15 degrees
Output: {"action": ["set_position(1, [[6, 15]])"], "response": "OK, Servo No. 6 turns to 15 degrees"}
Input: Servo No. 3 turns to 11 degrees, Servo No. 4 turns to 0 degrees, stop for 2 seconds
Output: {"action": ["set_position(1, [[3, 11], [4, 0]])", "time.sleep(2)"], "response": "OK"}
Input: Servo No. 1 turns 20 degrees, Servo No. 3 turns 45 degrees, stop for 1 second, Servo No. 1 turns 50 degrees, Servo No. 2 turns 0 degrees
Output: {"action":["set_position(1, [[1, 20], [3, 45]])", "time.sleep(1)", "set_position(1, [[2, 0]])"], "response": "OK"}
Input: Move down 2cm, stop for 1s, Servo 4 turns to 30 degrees, stop for 1s, move forward 3cm
Output: {"action":["move(0, 0, -0.02)", "time.sleep(1)", "set_position(1, [[4, 30]])", "time.sleep(1)", "move(0.03, 0, 0)"], "response": "No problem"}
    '''


class LLMControlServo(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name)
        
        self.action = []
        self.interrupt = False
        self.llm_result = ''
        self.action_finish = False
        self.play_audio_finish = False
        # self.llm_result = '{\'action\':[\'set_position(1, [[10, -20]])\'], \'response\':\'好嘞\'}'
        # self.llm_result = '{\"action\": [\"move(0, 0, -0.05)\"], \"response\": \"坐稳了，准备下移5cm！\"}'
        self.running = True
        self.id_dict = {'1': 5, "2": 4, "3": 3, "4": 2, "5": 1, "6": 0}
        self.current_pulse =  [500, 600, 825, 110, 500, 210]

       
        self.current_position = [0.1, 0, 0.226]

        timer_cb_group = ReentrantCallbackGroup()
        self.tts_text_pub = self.create_publisher(String, 'tts_node/tts_text', 1)
        self.create_subscription(String, 'agent_process/result', self.llm_result_callback, 1)
        self.create_subscription(Bool, 'tts_node/play_finish', self.play_audio_finish_callback, 1, callback_group=timer_cb_group)
        self.awake_client = self.create_client(SetBool, 'vocal_detect/enable_wakeup')
        self.awake_client.wait_for_service()
        self.create_subscription(Bool, 'vocal_detect/wakeup', self.wakeup_callback, 1)

        self.set_model_client = self.create_client(SetModel, 'agent_process/set_model')
        self.set_model_client.wait_for_service()
        self.set_prompt_client = self.create_client(SetString, 'agent_process/set_prompt')
        self.set_prompt_client.wait_for_service()
        self.set_pose_target_client = self.create_client(SetRobotPose, 'kinematics/set_pose_target')
        self.set_pose_target_client.wait_for_service()
        self.set_joint_value_target_client = self.create_client(SetJointValue, 'kinematics/set_joint_value_target')
        self.set_joint_value_target_client.wait_for_service()
        self.joints_pub = self.create_publisher(ServosPosition, 'servo_controller', 1) # Servo control 舵机控制

        self.timer = self.create_timer(0.0, self.init_process, callback_group=timer_cb_group)

    def get_node_state(self, request, response):
        return response

    def init_process(self):
        self.timer.cancel()

        msg = SetModel.Request()
        msg.model = llm_model
        msg.model_type = 'llm'
        msg.api_key = api_key 
        msg.base_url = base_url
        self.send_request(self.set_model_client, msg)

        msg = SetString.Request()
        msg.data = PROMPT
        self.send_request(self.set_prompt_client, msg)
        # -92 
        #self.set_position(1.5, [[6, 0], [5, 3], [4, 78], [3, -115], [2, 0], [1, -30]])
        #self.set_position(1.5, [[6, 0], [5, 48], [4, -72], [3, -96], [2, 0], [1, -30]])
        msg = set_joint_value_target(self.current_pulse)        
        endpoint = self.send_request(self.set_joint_value_target_client, msg)
        pose_t, pose_r = endpoint.pose.position, endpoint.pose.orientation
        mat = common.xyz_quat_to_mat([pose_t.x, pose_t.y, pose_t.z], [pose_r.w, pose_r.x, pose_r.y, pose_r.z])
        position, rpy = common.mat_to_xyz_euler(mat)

        self.current_position = [position, rpy]

        speech.play_audio(start_audio_path) 
        threading.Thread(target=self.process, daemon=True).start()
        self.create_service(Empty, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')
        self.get_logger().info('\033[1;32m%s\033[0m' % PROMPT)

    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()

    def wakeup_callback(self, msg):
        if self.llm_result:
            self.get_logger().info('wakeup interrupt')
            self.interrupt = msg.data

    def llm_result_callback(self, msg):
        self.llm_result = msg.data

    def set_position(self, duration, positions):
        msg = ServosPosition()
        msg.duration = float(duration)
        position_list = []
        for i in positions:
            position = ServoPosition()
            position.id = i[0]
            position.position = float(i[1])
            position_list.append(position)
            if i[0] != 1: 
                self.current_pulse[self.id_dict[str(i[0])]] = 500 + float(i[1])/240*1000
            else:
                self.current_pulse[self.id_dict[str(i[0])]] = 700 + float(i[1])/240*1000
        msg.position = position_list
        msg.position_unit = "deg"
        self.joints_pub.publish(msg)
        # self.get_logger().info(f'{self.current_pulse}') 
        msg = set_joint_value_target(self.current_pulse[:-1])
        endpoint = self.send_request(self.set_joint_value_target_client, msg)
        pose_t, pose_r = endpoint.pose.position, endpoint.pose.orientation
        mat = common.xyz_quat_to_mat([pose_t.x, pose_t.y, pose_t.z], [pose_r.w, pose_r.x, pose_r.y, pose_r.z])
        position, rpy = common.mat_to_xyz_euler(mat)
        self.current_position = [position, rpy]

    def move(self, x, y, z):
        self.current_position[0][0] += x
        self.current_position[0][1] += y
        self.current_position[0][2] += z
        self.get_logger().info(f'{self.current_position}')
        msg = set_pose_target(self.current_position[0], self.current_position[1][1], [-180.0, 180.0], 1.0)
        res = self.send_request(self.set_pose_target_client, msg)
        if res.pulse:
            servo_data = res.pulse
            self.get_logger().info(f'{servo_data}')
            #set_servo_position(self.joints_pub, 1, ((4, servo_data[3]), (3, servo_data[2]), (2, servo_data[1]), (1, servo_data[0])))
            set_servo_position(self.joints_pub, 1, ((3, servo_data[3]), (4, servo_data[2]), (5, servo_data[1]), (6, servo_data[0])))
            self.current_pulse[:-1] = servo_data

    def play_audio_finish_callback(self, msg):
        self.play_audio_finish = msg.data

    def process(self):
        while self.running:
            if self.llm_result:
                msg = String()
                if 'action' in self.llm_result:  # If a corresponding action is returned, extract it and proceed with processing. 如果有对应的行为返回那么就提取处理
                    result = eval(self.llm_result[self.llm_result.find('{'):self.llm_result.find('}') + 1])
                    # self.get_logger().info(str(result))
                    action_list = []
                    if 'action' in result:
                        action_list = result['action']
                    if 'response' in result:
                        response = result['response']
                    msg.data = response
                    self.tts_text_pub.publish(msg)
                    for i in action_list:
                        if self.interrupt:
                            self.get_logger().info('interrupt')
                            break
                        if 'set_position' in i:
                            eval('self.' + i)
                        elif 'move' in i:
                            eval('self.' + i)
                        else:
                            eval(i)
                else:  # No corresponding action, just respond. 没有对应的行为，只回答
                    response = self.llm_result
                    msg.data = response
                    self.tts_text_pub.publish(msg)
                self.action_finish = True 
                self.llm_result = ''
            else:
                time.sleep(0.01)
            if self.play_audio_finish and self.action_finish:
                    self.play_audio_finish = False
                    self.action_finish = False
                    msg = SetBool.Request()
                    msg.data = True
                    self.send_request(self.awake_client, msg)
        rclpy.shutdown()

def main():
    node = LLMControlServo('llm_control_servo')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
 
if __name__ == "__main__":
    main()
