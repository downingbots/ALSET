import os
import cv2
from uuid import uuid1
import time
from dataset_utils import *
from config import *

class GatherData():
    
    def check_func_movement_restrictions(self):
        if self.action_name is None:
          return True
        for [func, allowed_actions] in self.cfg.func_movement_restrictions:
          if self.nn_name == func:
            if self.action_name not in allowed_actions:
              if self.action_name not in  ["REWARD1", "PENALTY1", "PENALTY2", "REWARD2"]:
                print("#######  WARNING  ########")
                print("func action not in allowed_actions", self.nn_name, self.action_name, allowed_actions)
                print("##########################")
                return False
        return True

    def save_snapshot(self, process_image=False):
      frm_loc = None
      if (process_image):
        self.set_process_image(True)
      if self.nn_app.get_snapshot_dir() is not None and self.action_name is not None:
        directory = os.path.join(self.nn_app.get_snapshot_dir(), self.action_name)
        self.frame_location = os.path.join(directory, str(uuid1()) + '.jpg')
        frm_loc = self.frame_location  # frame_location can get set to None by another thread
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
      # if gather data, store frame jpg in index
## move to webcam.py to avoid race condition
#      if i <= 200 and self.is_on():
#        if frm_loc is not None:
#          ds_line = self.ds_util.dataset_line(frm_loc) + '\n'
#          if self.curr_func_ds_idx is not None:
#            with open(self.curr_func_ds_idx, 'a') as f:
#              f.write(ds_line)
#            print("save to idx:", self.curr_func_ds_idx, ds_line)
#          elif self.curr_dqn_ds_idx is not None:
#            with open(self.curr_dqn_ds_idx, 'a') as f:
#              f.write(ds_line)
#            print("save to idx:", self.curr_dqn_ds_idx, ds_line)
#      else:
#          print("not saved to idx:", i , self.is_on(), frm_loc, self.curr_func_ds_idx, self.curr_dqn_ds_idx, self.action_name)

      print("save snapshot wait time: %f" % (i*.01), frm_loc, process_image)
      return self.process_image_action

    def set_process_image(self, value):
        self.process_image_value = value

    def do_process_image(self):
        return self.process_image_value

    def process_image(self):
        if self.is_on() and self.nn_app.nn_automatic_mode():
          self.process_image_action = self.nn_app.nn_automatic_action(self.action_name)
        else:
          self.process_image_action = self.nn_app.nn_process_image()
        print("process_image:", self.nn_app.get_snapshot_dir(), self.process_image_action)
        directory = os.path.join(self.nn_app.get_snapshot_dir(), self.process_image_action)
        if self.do_process_image() and not self.is_on():
            frame_location = None
        else:
            frame_location = os.path.join(directory, str(uuid1()) + '.jpg')
        # time.sleep(.5) # ARD: debugging hack
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

    def set_action(self, action):
        if self.nn_app.set_action(action):
          self.action_name = action
          if not self.check_func_movement_restrictions():
            print("User Error: setting action to None")
            self.action_name = None

    def set_function_name(self, func_name):
        if self.nn_app.app_type != "APP":
            print("gather_data set_function_name set for non-FunctionalApp:", func_name)
            return
        old_nn_name = self.nn_name
        self.nn_name = func_name
        if old_nn_name != self.nn_name:
          if self.curr_app_ds_idx is not None and self.curr_func_ds_idx is not None:
            with open(self.curr_app_ds_idx, 'a+') as f:
              func_filenm = self.curr_func_ds_idx + '\n'
              f.write(func_filenm)
            print("new FUNC index saved in APP index:", self.curr_func_ds_idx, self.curr_app_ds_idx)
          self.curr_func_ds_idx = self.ds_util.dataset_indices(mode="FUNC", nn_name=self.nn_name, position="NEW")
          self.set_ds_idx(self.curr_func_ds_idx)

    # Video capture is done in __main__()
    def get_ds_idx(self):
          return self.curr_func_ds_idx

    def set_ds_idx(self, f_ds_idx):
          self.curr_func_ds_idx = f_ds_idx 

    def capture_frame_location(self):
          return self.frame_location

    def capture_frame_completed(self):
        if self.frame_location != "DONE":
          self.frame_location = None
        if self.is_on() and self.nn_app.nn_automatic_mode():
          self.nn_app.nn_automatic_post_action(self.action_name)

    def __init__(self, app):
        self.mode = False
        self.nn_app = app
        self.action_name = None
        self.frame_location = None
        self.speed = .5
        self.process_image_value = False
        self.process_image_action = None
        # Video capture is done in __main__()
        self.ds_util =  DatasetUtils(self.nn_app.app_name, self.nn_app.app_type)
        self.cfg = Config()
        if self.nn_app.app_type == "FUNC":
          self.nn_name = self.nn_app.app_name
          self.curr_func_ds_idx = self.ds_util.dataset_indices(mode="FUNC", nn_name=self.nn_name, position="NEW")
          self.curr_app_ds_idx = None
        elif self.nn_app.app_type == "APP":
          self.nn_name = None
          self.curr_func_ds_idx = None
          self.curr_app_ds_idx = self.ds_util.dataset_indices(mode="APP", nn_name=self.nn_app.app_name, position="NEW")
        elif self.nn_app.app_type == "DQN":
          self.nn_name = None
          self.curr_func_ds_idx = None
          self.curr_app_ds_idx = None
        else:
           print("bad app_type: ", self.nn_app.app_type)
           exit
        print("current_ds_idx:", self.curr_func_ds_idx)

