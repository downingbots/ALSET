import os
import cv2
from uuid import uuid1
import time

class GatherData():

    def mkdirs(self):
        for dir_name in self.robot_dir:
          try:
              os.makedirs(dir_name)
          except FileExistsError:
            print('Directory already exists')

    def save_snapshot(self):
      directory = self.robot_dir[self.function_name]
      self.frame_location = os.path.join(directory, str(uuid1()) + '.jpg')
      i = 0
      while self.frame_location != None:
          i = i+1
          time.sleep(.01)
          # ARD: to debug when camera errors occur and camera not needed
          # if i >= 1:
          if i > 100:
              print("save snapshot race condition; break loop")
              break
      print("save snapshot wait time: %f" % (i*.01))
      return self.frame_location

    def is_on(self):
        return self.mode

    def turn_on(self, mode):
        self.mode = mode

    def get_speed(self):
        return self.speed

    def set_function(self, func):
        try:
          if func != None:
            func_dir = self.robot_dir[func]
          self.function_name = func
        except:
            print("bad function name: %s" % func)

    # Video capture is done in __main__()
    def capture_frame_location(self):
        return self.frame_location

    def capture_frame_completed(self):
        self.frame_location = None


    def __init__(self):
        self.mode = False
        self.function_name = None
        self.frame_location = None
        self.speed = .5
        self.robot_dir = {}
        self.robot_dir["FORWARD"] = "dataset/forward"
        self.robot_dir["REVERSE"] = "dataset/reverse"
        self.robot_dir["LEFT"] = "dataset/left"
        self.robot_dir["RIGHT"] = "dataset/right"
        self.robot_dir["UPPER_ARM_UP"] = "dataset/upper_arm_up"
        self.robot_dir["UPPER_ARM_DOWN"] = "dataset/upper_arm_down"
        self.robot_dir["LOWER_ARM_UP"] = "dataset/lower_arm_up"
        self.robot_dir["LOWER_ARM_DOWN"] = "dataset/lower_arm_down"
        self.robot_dir["GRIPPER_OPEN"] = "dataset/gripper_open"
        self.robot_dir["GRIPPER_CLOSE"] = "dataset/gripper_close"
        # Video capture is done in __main__()
