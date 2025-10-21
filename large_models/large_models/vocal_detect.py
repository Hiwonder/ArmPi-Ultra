#!/usr/bin/env python3
# encoding: utf-8
# @Author: Aiden
# @Date: 2024/11/18
import time
import rclpy
import threading
from rclpy.node import Node
from std_msgs.msg import Int32, String, Bool
from std_srvs.srv import SetBool, Trigger, Empty

from speech import awake
from speech import speech
from large_models.config import *
from large_models_msgs.srv import SetInt32

class VocalDetect(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name)

        self.running = True

        # Declaration parameter (声明参数)
        self.declare_parameter('awake_method', 'xf')
        self.declare_parameter('mic_type', 'mic6_circle')
        self.declare_parameter('port', '/dev/wonderecho')
        self.declare_parameter('enable_wakeup', True)
        self.declare_parameter('enable_setting', False)
        self.declare_parameter('awake_word', 'hello hi wonder')
        self.declare_parameter('mode', 1)

        self.awake_method = self.get_parameter('awake_method').value
        mic_type = self.get_parameter('mic_type').value
        port = self.get_parameter('port').value
        awake_word = self.get_parameter('awake_word').value
        enable_setting = self.get_parameter('enable_setting').value 
        self.enable_wakeup = self.get_parameter('enable_wakeup').value
        self.mode = int(self.get_parameter('mode').value)

        if self.awake_method == 'xf':
            self.kws = awake.CircleMic(port, awake_word, mic_type, enable_setting)
        else:
            self.kws = awake.WonderEchoPro(port) 
        
        self.language = os.environ["ASR_LANGUAGE"] 
        if self.awake_method == 'xf':
            if self.language == 'Chinese':
                self.asr = speech.RealTimeASR(log=self.get_logger())
            else:
                self.asr = speech.RealTimeOpenAIASR(log=self.get_logger())
                self.asr.update_session(model=asr_model, language='en')
        else:
            if self.language == 'Chinese':
                self.asr = speech.RealTimeASR(log=self.get_logger())
            else:
                self.asr = speech.RealTimeOpenAIASR(log=self.get_logger())
                self.asr.update_session(model=asr_model, language='en')
        
        self.asr_pub = self.create_publisher(String, '~/asr_result', 1)
        self.wakeup_pub = self.create_publisher(Bool, '~/wakeup', 1)
        self.awake_angle_pub = self.create_publisher(Int32, '~/angle', 1)
        self.create_service(SetInt32, '~/set_mode', self.set_mode_srv)
        self.create_service(SetBool, '~/enable_wakeup', self.enable_wakeup_srv)

        threading.Thread(target=self.pub_callback, daemon=True).start()
        self.create_service(Empty, '~/init_finish', self.get_node_state)
        
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')

    def get_node_state(self, request, response):
        return response

    def record(self, mode, angle=None):
        self.get_logger().info('\033[1;32m%s\033[0m' % 'asr...')
        if self.language == 'Chinese':
            asr_result = self.asr.asr(model=asr_model)  # Start the recording and recognition (开启录音并识别)
        else:
            asr_result = self.asr.asr()
        if asr_result: 
            speech.play_audio(dong_audio_path)
            if self.awake_method == 'xf' and self.mode == 1: 
                msg = Int32()
                msg.data = int(angle)
                self.awake_angle_pub.publish(msg)
            asr_msg = String()
            asr_msg.data = asr_result
            self.asr_pub.publish(asr_msg)
            self.enable_wakeup = False
            self.get_logger().info('\033[1;32m%s\033[0m' % 'publish asr result:' + asr_result)
        else:
            self.get_logger().info('\033[1;32m%s\033[0m' % 'no voice detect')
            speech.play_audio(dong_audio_path)
            if mode == 1:
                speech.play_audio(no_voice_audio_path)

    def pub_callback(self):
        self.kws.start()
        while self.running:
            if self.enable_wakeup:
                if self.mode == 1:
                    result = self.kws.wakeup()
                    if result:
                        self.wakeup_pub.publish(Bool(data=True))
                        speech.play_audio(wakeup_audio_path)  # Wake up playback(唤醒播放)
                        self.record(self.mode, result)
                    else:
                        time.sleep(0.02)
                elif self.mode == 2:
                    self.record(self.mode)
                    self.mode = 1
                elif self.mode == 3:
                    self.record(self.mode)
                else:
                    time.sleep(0.02)
            else:
                time.sleep(0.02)
        rclpy.shutdown()

    def enable_wakeup_srv(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % ('enable_wakeup'))
        self.kws.start()
        self.enable_wakeup = request.data
        response.success = True
        return response 

    def set_mode_srv(self, request, response):
        self.get_logger().info(f'\033[1;32mset mode: {request.data}\033[0m')
        self.kws.start()
        self.mode = int(request.data)
        if self.mode != 1:
            self.enable_wakeup = True
        response.success = True
        return response 

def main():
    node = VocalDetect('vocal_detect')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('shutdown')
    finally:
        rclpy.shutdown() 

if __name__ == "__main__":
    main()
