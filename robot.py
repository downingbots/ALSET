import time
import traitlets
from traitlets.config.configurable import SingletonConfigurable
from .mcp23017 import *
from .motor import *
from .sir_joystick import *
from .gather_data import *
from .nn import *

class Robot(SingletonConfigurable):
    # config
    def __init__(self, *args, **kwargs):
        # super(Robot, self).__init__(*args, **kwargs)
        self.gather_data = GatherData()
        self.NN = SIRNN(self)
        self.sir_robot = SIR_control(self)
        self.left_motor = Motor(self.sir_robot, "LEFT")
        self.right_motor = Motor(self.sir_robot, "RIGHT")
        sir_joystick_daemon(self)
        
    def set_motors(self, left_speed, right_speed):
        print("set_motors:", left_speed, right_speed)
        if self.gather_data.is_on():
            if left_speed == None:
                left_speed = self.left_motor.get_speed()
                if left_speed == None:
                  left_speed = 0
            if right_speed == None:
                right_speed = self.right_motor.get_speed()
                if right_speed == None:
                  right_speed = 0
            if abs(left_speed) <= 0.1 or abs(right_speed) <= 0.1:
                print("stop")
                self.stop()
            elif left_speed <= 0 and right_speed >= 0:
                print("forward")
                self.forward()
            elif left_speed >= 0 and right_speed <= 0:
                print("backward")
                self.backward()
            elif left_speed >= 0 and right_speed >= 0:
                print("left")
                self.left()
            elif left_speed <= 0 and right_speed <= 0:
                print("right")
                self.right()
            else:
                print("stop")
                self.stop()
        else:
            self.sir_robot.set_motors(left_speed, right_speed)
        
    def forward(self, speed=1.0, duration=None):
        if self.gather_data.is_on():
            self.gather_data.set_function("FORWARD")
            self.sir_robot.drive_forward(speed)
        else:
            self.sir_robot.drive_forward(speed)

    def backward(self, speed=1.0):
        if self.gather_data.is_on():
            self.gather_data.set_function("REVERSE")
            self.sir_robot.drive_reverse(speed)
        else:
            self.sir_robot.drive_reverse(speed)

    def left(self, speed=1.0):
        if self.gather_data.is_on():
            self.gather_data.set_function("LEFT")
            self.sir_robot.drive_rotate_left(speed)
        else:
            self.sir_robot.drive_rotate_left(speed)

    def right(self, speed=1.0):
        if self.gather_data.is_on():
            self.gather_data.set_function("RIGHT")
            self.sir_robot.drive_rotate_right(speed)
        else:
            self.sir_robot.drive_rotate_right(speed)

    def stop(self):
        self.gather_data.set_function(None)
        self.sir_robot.drive_stop()

    def upper_arm(self,direction):
        if direction == "STOP":
          self.gather_data.set_function(None)
        else:
          self.gather_data.set_function("UPPER_ARM_" + direction)
        self.sir_robot.upper_arm(direction)

    def lower_arm(self,direction):
        if direction == "STOP":
          self.gather_data.set_function(None)
        else:
          self.gather_data.set_function("LOWER_ARM_" + direction)
        self.sir_robot.lower_arm(direction)

    def wrist(self,direction):
        self.sir_robot.wrist(direction)

    def gripper(self,direction):
        if direction == "STOP":
          self.gather_data.set_function(None)
        else:
          self.gather_data.set_function("GRIPPER_" + direction)
        self.sir_robot.gripper(direction)

    # gather data / teleop mode
    def set_gather_data_mode(self, mode=True): 
        self.gather_data.turn_on(mode)

    def get_gather_data_mode(self): 
        return self.gather_data.is_on()

    # run pytorch NN to determine next move
    def set_NN_mode(self, mode=True): 
        self.NN.turn_on(mode)

    def get_NN_mode(self):
        return self.NN.is_on()

    # webcam integration
    def ready_for_capture(self):
        return self.NN.ready_for_capture()

    def capture_frame(self, img):
        return self.NN.capture_frame(img)

    def capture_frame_location(self):
        return self.gather_data.capture_frame_location()

    def capture_frame_completed(self):
        self.gather_data.capture_frame_completed()


