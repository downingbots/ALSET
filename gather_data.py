import os
import cv2
from uuid import uuid1
import time
from dataset_utils import *

class GatherData():

    def save_snapshot(self, process_image=False):
      frm_loc = None
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
          # if i > 100:
          # Increasing wait time in case first time initializing NN.
          if i > 1500:
              print("save snapshot race condition; break loop")
              break
          if self.frame_location == "DONE":
              exit()
      if i <= 100 and not process_image:
        if frm_loc is not None:
          ds_line = self.ds_util.dataset_line(frm_loc)
          with open(self.current_ds_idx, 'a') as f:
            f.write(ds_line)
          print("save to idx:", self.current_ds_idx, ds_line)

      print("save snapshot wait time: %f" % (i*.01), frm_loc, process_image)
      return self.process_image_action

    def set_process_image(self, value):
        self.process_image_value = value

    def do_process_image(self):
        return self.process_image_value

    def process_image(self):
        if self.is_on() and self.nn_app.nn_automatic_mode():
          self.process_image_action = self.nn_app.nn_automatic_action(self.function_name)
        else:
          self.process_image_action = self.nn_app.nn_process_image()
        print("process_image:", self.nn_app.get_snapshot_dir(), self.process_image_action)
        directory = os.path.join(self.nn_app.get_snapshot_dir(), self.process_image_action)
        if self.do_process_image() and not self.is_on():
            frame_location = None
        else:
            frame_location = os.path.join(directory, str(uuid1()) + '.jpg')
        time.sleep(.5) # ARD: debugging hack
        self.set_process_image(False)
        if self.process_image_action == "DONE":
            return "DONE"
        return frame_location

    # gather_data mode
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
        self.ds_util =  DatasetUtils(self.nn_app.app_name, self.nn_app.app_type)
        if self.nn_app.app_type == "FUNC":
            nn_name = self.nn_app.app_name
        else:
            nn_name = None
        self.current_ds_idx = self.ds_util.dataset_indices(mode=self.nn_app.app_type, nn_name=nn_name, position="NEW")
        print("current_ds_idx:", self.current_ds_idx)

