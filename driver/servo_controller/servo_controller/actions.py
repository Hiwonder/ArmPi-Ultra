import rclpy
import time
from servo_controller import bus_servo_control 

def goto_left(pub, duration=1.5):
    bus_servo_control.set_servo_position(pub, int(duration * 1000), ((6, 875), (5, 610), (4, 930), (3, 110), (2, 500), (1, 200)))
    time.sleep(duration)

def goto_right(pub, duration=1.5):
    bus_servo_control.set_servo_position(pub, int(duration * 1000), ((6, 125), (5, 610), (4, 930), (3, 110), (2, 500), (1, 200)))
    time.sleep(duration)
   

def go_home(pub, duration=0.8):
    bus_servo_control.set_servo_position(pub, int(duration), ( (5, 600), (4, 825), (3, 110), (2, 500), (1, 210)))
    time.sleep(duration+0.2)
    bus_servo_control.set_servo_position(pub, int(duration), ((6, 500),))
    time.sleep(duration)
def goto_home(pub, duration=0.8):
    bus_servo_control.set_servo_position(pub, int(duration), ((5, 500),))
    time.sleep(duration+0.2)
    bus_servo_control.set_servo_position(pub, int(duration), ( (4, 560), (4, 770), (3, 115), (2, 500), (1, 200)))
    time.sleep(duration)
def goto_default(pub, duration=1.5):
    go_home(pub, duration)

def place(pub, duration=1.5):
    bus_servo_control.set_servo_position(pub, int(duration), ((6, 500), (5, 610), (4, 930), (3, 140), (2, 500), (1, 200)))

def go_back(pub, duration=0.8):
  
    bus_servo_control.set_servo_position(pub, int(duration), ((6, 500), (5, 560), (4, 870), (3, 115), (2, 500), (1, 550)))
    time.sleep(duration)
