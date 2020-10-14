import os
import time
import cv2
import numpy as np
from .robot import *
from .tabletop_functional_nn import *
from .nn import *
import time

class nn_apps():
    # initialize app registry
    def __init__(self, sir_robot, sir_app_num=None, sir_app_name=None):
      self.app_registry = [[0, "SIRNN", 1], [1, "TT_func", 8]]
      if sir_app_num != None:
          self.app_num = sir_app_num
          if self.app_num >= len(self.app_registry) or self.app_num < 0:
              print("Error: unknown app number %d" % self.app_num)
              exit()
      else:
          found = False
          for [self.app_num, app_name, num_nn] in self.app_registry:
               if sir_app_name == app_name:
                   found = True
                   break
          if not found:
              print("Error: unknown app name %s" % sir_app_name)
              exit()
      self.action_set = ["UPPER_ARM_UP",
                    "UPPER_ARM_DOWN",
                    "LOWER_ARM_UP",
                    "LOWER_ARM_DOWN",
                    "GRIPPER_OPEN",
                    "GRIPPER_CLOSE",
                    "FORWARD",
                    "REVERSE",
                    "LEFT",
                    "RIGHT",
                    "REWARD",
                    "PENALTY"]

      self.app_instance = []
      # 0
      self.app_instance.append(SIRNN(sir_robot, len(self.action_set)))
      # 1
      self.app_instance.append(tabletop_functional_nn(sir_robot))
      robot_dirs = []
      robot_dirs.append("apps")
      for app_reg in self.app_registry:
          robot_dirs.append("apps/" + app_reg[1])
          robot_dirs.append("apps/" + app_reg[1] + "/dataset")
      self.mkdirs(robot_dirs)
      self.mode = False
      self.ready_for_frame = False
      self.frame = None
      self.robot = sir_robot
      self.curr_nn_num = 1 # start with 1
      self.nn_dir = None

    ####################################################
    # COMMON APP CONTROL FUNCTIONS
    ####################################################
    def preprocess(self, camera_value):
         global device, normalize
         mean = 255.0 * np.array([0.485, 0.456, 0.406])
         stdev = 255.0 * np.array([0.229, 0.224, 0.225])
         normalize = torchvision.transforms.Normalize(mean, stdev)

         x = camera_value
         x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
         x = x.transpose((2, 0, 1))
         x = torch.from_numpy(x).float()
         x = normalize(x)
         x = x.to(self.device)
         x = x[None, ...]
         return x

    def is_on(self):
        return self.mode

    def turn_on(self, nn_mode):
        i = 0
        while self.ready_for_frame and nn_mode == False:
            i = i+1
            if i == 1:
                print("Turning NN off when safe")
            time.sleep(.01)
            # 1 second should have produced 20 images; break to prevent hang 
            if i > 100:  
              break
        self.mode = nn_mode

    def wait_for_capture(self):
        self.ready_for_frame = True
        i = 0
        while self.ready_for_frame:
            i = i+1
            time.sleep(.01)
            # 1 second should have produced 20 images; break to prevent hang 
            if i > 100:  
              break
        print("snapshot wait time: %f" % (i*.01))

    def ready_for_capture(self):
        if self.ready_for_frame == True:
            return True
        else:
            return False

    def capture_frame(self, img):
        self.frame = img
        self.ready_for_frame = False

    def set_function(self, func):
        if func == None or func in self.action_set:
            self.function_name = func
        else:
            print("bad function name: %s" % func)
            return False
        return True

    def mkdirs(self, robot_dirs):
        for dir_name in robot_dirs:
          try:
              os.makedirs(dir_name)
              print("mkdir %s" % dir_name)
          except FileExistsError:
            # print('Directory already exists')
            pass

    def get_snapshot_dir(self):
      return self.nn_dir 

    ####################################################
    # PASS-THROUGH FUNCTION CALLS
    ####################################################

    def nn_init(self):
      gather_mode = self.robot.get_gather_data_mode()
      # automatic_mode must be set to false until after the directories
      # are created.
      self.app_instance[self.app_num].nn_set_automatic_mode(False)
      auto_mode, robot_action_dirs = self.app_instance[self.app_num].nn_init(self.curr_nn_num, gather_mode)
      # probably should use os.path.join()
      dataset_dir = "apps/" + self.app_registry[self.app_num][1] + "/dataset"
      robot_dirs = []
      self.nn_dir = dataset_dir + "/NN" + str(self.curr_nn_num)
      robot_dirs.append(self.nn_dir)
      for dir_name in robot_action_dirs:
          robot_dirs.append(self.nn_dir + "/" + dir_name)
      self.mkdirs(robot_dirs)
      print("nn_init: " , self.curr_nn_num, robot_dirs)
      if auto_mode:
          self.app_instance[self.app_num].nn_set_automatic_mode(True)
      return robot_dirs

    def nn_process_image(self):
      action = self.app_instance[self.app_num].nn_process_image(self.curr_nn_num, self.frame)
      if action == None:
          return None
      action_dir = os.path.join(self.nn_dir, action)
# called directly from mcp to better handle automatic mode 
#      if action == "SUCCESS":
#          # do apps-specific transition to next function-specific NN
#          self.curr_nn_num = self.app_instance[self.app_num].nn_upon_reward(self.curr_nn_num)
#      elif action == "FAILURE":
#          # do apps-specific transition to next function-specific NN
#          self.curr_nn_num = self.app_instance[self.app_num].nn_upon_penalty(self.curr_nn_num)
      return action_dir

    def nn_automatic_mode(self):
      return self.app_instance[self.app_num].nn_automatic_mode()

    def nn_automatic_action(self, feedback):
      if feedback == "REWARD":
          self.nn_upon_reward()
          return "REWARD"
      return self.app_instance[self.app_num].nn_automatic_action(self.curr_nn_num, feedback)

    def nn_upon_reward(self):
      self.curr_nn_num = self.app_instance[self.app_num].nn_upon_reward(self.curr_nn_num)
      gather_mode = self.robot.get_gather_data_mode()
      self.nn_init()
      return self.curr_nn_num

    def nn_upon_penalty(self):
      return self.app_instance[self.app_num].nn_upon_penalty(self.curr_nn_num)
