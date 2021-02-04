import os
import cv2
from uuid import uuid1
import time
from dataset_utils import *

class GatherData():

    def save_snapshot(self, process_image=False):
      if (process_image):
        self.set_process_image(True)
        self.frame_location = "dummy"
      else:
        directory = os.path.join(self.nn_app.get_snapshot_dir(), self.function_name)
        # directory = self.robot_dir[self.function_name]
        self.frame_location = os.path.join(directory, str(uuid1()) + '.jpg')
        frm_loc = self.frame_location
      i = 0
      while self.frame_location != None:
          i = i+1
          time.sleep(.01)
          # ARD: to debug when camera errors occur and camera not needed
          # if i >= 1:
          if i > 100:
              print("save snapshot race condition; break loop")
              break
          if self.frame_location == "DONE":
              exit()
      if i <= 100:
        ds_line = self.ds_util.dataset_line(frm_loc)
        with open(self.current_ds_idx, 'a') as f:
          f.write(ds_line)

      print("save snapshot wait time: %f" % (i*.01), self.frame_location)
      return self.process_image_action

    def set_process_image(self, value):
        self.process_image_value = value

    def do_process_image(self):
        return self.process_image_value

    def process_image(self):
        self.process_image_action = self.nn_app.nn_process_image()
        directory = os.path.join(self.nn_app.get_snapshot_dir(), self.process_image_action)
        frame_location = os.path.join(directory, str(uuid1()) + '.jpg')
        self.set_process_image(False)
        if self.process_image_action == "DONE":
            return "DONE"
        return frame_location

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
        if self.frame_location == "DONE":
            return
        self.frame_location = None


    def __init__(self, app):
        self.mode = False
        # wrong. nn_app changes based on NN function
        self.nn_app = app
        self.function_name = None
        self.frame_location = None
        self.speed = .5
        self.process_image_value = False
        self.process_image_action = None
        # Video capture is done in __main__()
        self.ds_util =  DatasetUtils(self.nn_app.app_num)
        print(self.nn_app.app_name)
        self.current_ds_idx = self.ds_util.new_dataset_idx_name(self.nn_app.app_name)

