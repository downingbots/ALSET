import os
import cv2
from uuid import uuid1
import time

class GatherData():

    def save_snapshot(self):
      directory = os.path.join(self.nn_app.get_snapshot_dir(), self.function_name)
      # directory = self.robot_dir[self.function_name]
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
        if self.nn_app.set_function(func):
          self.function_name = func

    # Video capture is done in __main__()
    def capture_frame_location(self):
        return self.frame_location

    def capture_frame_completed(self):
        self.frame_location = None


    def __init__(self, app):
        self.mode = False
        self.nn_app = app
        self.function_name = None
        self.frame_location = None
        self.speed = .5
        # Video capture is done in __main__()
