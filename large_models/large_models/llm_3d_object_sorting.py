#!/usr/bin/env python3
# encoding: utf-8
# @Author: Aiden
# @Date: 2024/11/18
import re
import time
import json
import rclpy
import threading
from speech import speech
from rclpy.node import Node
from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger, SetBool, Empty

from large_models.config import *
from large_models_msgs.srv import SetModel, SetString

from interfaces.srv import SetStringList
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

if os.environ["ASR_LANGUAGE"] == 'Chinese':
    PROMPT = '''
#角色任务
你是一款智能机械臂，需要根据输入的内容，生成对应的指令。

 ##要求与限制
1.根据输入的动作内容，在动作函数库中找到对应的指令，并输出对应的指令。
2.为动作序列编织一句精炼（10至30字）、风趣且变化无穷的反馈信息，让交流过程妙趣横生。
3 物体分为三类：圆柱体（cylinder），球体（sphere），长方体（cuboid）。
4.直接输出json结果，不要分析，不要输出多余内容。
5.格式：{'action':'xx', 'response':'xx'}

##结构要求##：
 - `"action"`键下承载一个函数名称字符串数组，当找不到对应动作函数时action输出“”。 
- `"response"`键则配以精心构思的简短回复，完美贴合上述字数与风格要求。 

## 实施原则 - 在最终输出前，实施全面校验，确保回复不仅格式合规，而且内容充满意趣，无一遗漏上述规范。

 ## 动作函数库
- 分拣物体：object_sorting('cylinder')

## 任务示例
输入：分拣圆柱体
输出：{'action': "object_sorting('cylinder')", 'response':'好的，开始分拣圆柱体'}
输入：将长方体拿走
输出：{'action': "object_sorting('cuboid')", 'response':'长方体走你'}
输入：只留下球体，其他拿走
输出：{'action': "object_sorting('cylinder'，'cuboid')", 'response':'好的，马上执行'}
输入：分拣物体
输出：{'action': "object_sorting('cylinder', 'cuboid'，'sphere')", 'response':'没问题'}
'''
else:
    PROMPT = '''
#Role
You are an intelligent robot arm that needs to generate corresponding instructions based on the input content.

##Requirements and Restrictions
1. According to the input action content, find the corresponding instructions in the action function library and output the corresponding instructions.
2. Weave a concise (10 to 30 words), humorous and ever-changing feedback information for the action sequence to make the communication process interesting.
3. Objects are divided into three categories: cylinder, sphere, and cuboid.
4. Directly output the json result, do not analyze it, and do not output redundant content.
5. Format: {"action": "xx", "response": "xx"}

##Structural Requirements:
- The "action" key carries a function name string array. When the corresponding action function cannot be found, action outputs "".
- The "response" key is accompanied by a carefully conceived short reply, which perfectly meets the above word count and style requirements.

##Implementation Principles 
- Before final output, implement comprehensive verification to ensure that the response is not only formatted in compliance, but also interesting and without missing any of the above specifications.

##Action function library
- Sorting objects: object_sorting('cylinder')

##Example
Input: Sorting cylinders
Output: {"action": "object_sorting('cylinder')", "response": "OK, start sorting cylinders"}
Input: Take the cuboid away
Output: {"action": "object_sorting('cuboid')", "response": "The cuboid is gone"}
Input: Only keep the sphere and take the others away
Output: {"action": "object_sorting('cylinder'，'cuboid')", "response": "OK, execute now"}
Input: Sorting objects
Output: {"action": "object_sorting('cylinder', 'cuboid'，'sphere')", "response": "No problem"}
    '''

class LLMObjectSorting(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name)

        self.action = []
        self.llm_result = ''
        self.action_finish = False
        self.play_audio_finish = False
        # self.llm_result = '{"action": "object_sorting(\'cylinder\', \'cuboid\', \'sphere\')", "response": "收到，我马上收拾"}'
        self.running = True

        timer_cb_group = ReentrantCallbackGroup()
        self.tts_text_pub = self.create_publisher(String, 'tts_node/tts_text', 1)
        self.create_subscription(Bool, 'tts_node/play_finish', self.play_audio_finish_callback, 1, callback_group=timer_cb_group)
        self.create_subscription(String, 'agent_process/result', self.llm_result_callback, 1)
        self.create_subscription(Bool, 'vocal_detect/wakeup', self.wakeup_callback, 1)

        self.awake_client = self.create_client(SetBool, 'vocal_detect/enable_wakeup')
        self.awake_client.wait_for_service()

        self.set_model_client = self.create_client(SetModel, 'agent_process/set_model')
        self.set_model_client.wait_for_service()

        self.set_prompt_client = self.create_client(SetString, 'agent_process/set_prompt')
        self.set_prompt_client.wait_for_service()

        self.enter_client = self.create_client(Trigger, 'shape_recognition/enter')
        self.enter_client.wait_for_service()

        self.start_client = self.create_client(SetBool, 'shape_recognition/set_running')
        self.start_client.wait_for_service()

        self.set_target_client = self.create_client(SetStringList, 'shape_recognition/set_shape')
        self.set_target_client.wait_for_service()

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

        self.send_request(self.enter_client, Trigger.Request())       
        

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
        if msg.data and self.llm_result:
            self.send_request(self.enter_client, Trigger.Request())

    def llm_result_callback(self, msg):
        self.llm_result = msg.data

    def play_audio_finish_callback(self, msg):
        self.play_audio_finish = msg.data

    def process(self):
        while self.running:
            if self.llm_result:
                msg = String()
                if 'action' in self.llm_result:  # If there is a corresponding action returned, extract and handle it. 如果有对应的行为返回那么就提取处理
                    result = eval(self.llm_result[self.llm_result.find('{'):self.llm_result.find('}') + 1])
                    if 'action' in result:
                        text = result['action']
                        # Use regular expressions to extract all strings inside. 使用正则表达式提取括号中的所有字符串
                        pattern = r"object_sorting\((['\"][^'\"]+['\"](?:\s*,\s*['\"][^'\"]+['\"])*)\)"
                        # Use re.search to find the matching result. 使用re.search找到匹配的结果
                        match = re.search(pattern, text)
                        # Extract the result. 提取结果
                        if match:
                            # Retrieve all argument parts (the content inside the parentheses). 获取所有的参数部分（括号内的内容）
                            params = match.group(1)
                            # Use regular expressions to split the string arguments. 使用正则拆分出各个字符串参数
                            objects = re.findall(r"['\"]([^'\"]+)['\"]", params)
                            # self.get_logger().info(str(objects))
                            # Enable recognition. 开启识别
                            msgs = SetStringList.Request()
                            msgs.data = objects                            
                            res = self.send_request(self.set_target_client, msgs)
                            if res.success:
                                self.get_logger().info('\033[1;32m%s\033[0m' % 'set object success')
                            else:
                                self.get_logger().info('\033[1;32m%s\033[0m' % 'set object fail')
                            start_msg = SetBool.Request()
                            start_msg.data = True
                            # Enable sorting. 开启分拣
                            self.send_request(self.start_client, start_msg)
                    if 'response' in result:
                        msg.data = result['response']
                else:  # No corresponding action, just respond. 没有对应的行为，只回答
                    msg.data = self.llm_result
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
    node = LLMObjectSorting('llm_object_sorting')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()


if __name__ == "__main__":
    main()
