# Flow between
# 
# robot.py
#   only interacts with NN_apps
# 
# NN_apps (nn_apps.py)
#   currently 3 apps are defined:
#      1. a single SIRNN
#      2. one tabletop_nn that will define 8 SIRNN
#      3. DDQN reinforcement learning
#   Each of these Apps has a set of functions that are called
#      by the NN_apps function of the same name in a poor-man's
#      encapsolation.  Eventually, this should be cleaned up
#      so that a superset class defines these functions in
#      an object-oriented manner.
#   sets up directory for dataset
#   stores image in appropriate directory for gather_mode
#   knows curr_nn_num
#      starts at 1, 
#      changes curr_nn_num based upon app's
#         nn_upon_failure, nn_upon_success, automatic_mode
# 
# tabletop (tabletop_functional_nn.py)
#   self.NN[0-7] -> defines a SIRNN for each function performed
#   knows number of NN to define
#   knows actions for each NN
#   knows/calls actual torch NN (SIRNN)
#   defines flow between NN
#    1. Park Arm
#    2. Automatic scan for cube
#    3. Approach cube
#    4. Pick up cube
#    5. Park Arm (with cube in grippers)
#    6. Automatic scan for box (with cube in grippers)
#    7. Approach box (with cube in grippers)
#    8. Drop cube in box and back away
#      -> New park arm (back to 1)
#   defines automatic mode 
# 
# nn.py defines
#   SIRNN -> actual single torch NN
#   also a self-contained single-NN app
import os
import time
import cv2
import numpy as np
from .robot import *
from .sir_ddqn import *
from .functional_app import *
from .automated_funcs import *
from .nn import *
import time

class nn_apps():
    # initialize app registry
    def __init__(self, sir_robot, sir_app_num=None, sir_app_name=None):
      if sir_app_num != None:
          if sir_app_num >= len(self.app_registry) or sir_app_num < 0:
              print("Error: unknown app number %d" % sir_app_num)
              exit()
          [self.app_num, self.app_name, self.num_nn] = self.app_registry[sir_app_num]
      else:
          found = False
          for [self.app_num, self.app_name, self.num_nn] in self.app_registry:
               if sir_app_name == self.app_name:
                   found = True
                   print("app num/name:", self.app_num, self.app_name)
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
      self.app_instance.append(SIRNN(sir_robot, self.action_set))
      # 1
      self.app_instance.append(functional_app(sir_robot))
      # 2
      self.app_instance.append(SIR_DDQN(True, False))
      robot_dirs = []
      robot_dirs.append("apps")
      for app_reg in self.app_registry:
          robot_dirs.append("apps/" + app_reg[1])
          robot_dirs.append("apps/" + app_reg[1] + "/dataset")
      self.mkdirs(robot_dirs)
      self.mode = "TELEOP"
      self.ready_for_frame = False
      self.frame = None
      self.robot = sir_robot
      self.curr_nn_num = 1 # start with 1
      self.nn_dir = None
      self.auto_funcs = AutomatedFuncs()

    ####################################################
    # COMMON APP CONTROL FUNCTIONS
    ####################################################
#    def preprocess(self, camera_value):
#         global device, normalize
#         mean = 255.0 * np.array([0.485, 0.456, 0.406])
#         stdev = 255.0 * np.array([0.229, 0.224, 0.225])
#         normalize = torchvision.transforms.Normalize(mean, stdev)
#
#         x = camera_value
#         x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
#         x = x.transpose((2, 0, 1))
#         x = torch.from_numpy(x).float()
#         x = normalize(x)
#         x = x.to(self.device)
#         x = x[None, ...]
#         return x

    def get_nn_mode(self):
        return self.mode

    def set_nn_mode(self, nn_mode):
        i = 0
        while self.ready_for_frame and nn_mode == "NN":
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
        if func == None or func in self.action_set or func in ["ROBOT_OFF_TABLE_PENALTY", "CUBE_OFF_TABLE_REWARD", "PENALTY", "REWARD"]:
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

    # called to initialize or switch to a new NN for gathering data or exec
    # idempotent
    def nn_init(self):
      gather_mode = self.robot.get_gather_data_mode()
      self.robot.gather_data.set_function(None)
      # automatic_mode must be set to false until after the directories
      # are created.
      self.app_instance[self.app_num].nn_set_automatic_mode(False)

      # print(self.app_instance[self.app_num])
      print(self.app_registry[self.app_num][1])
      print(self.curr_nn_num, gather_mode)
      auto_mode, robot_action_dirs = self.app_instance[self.app_num].nn_init(self.app_registry[self.app_num][1], self.curr_nn_num, gather_mode)
      self.app_instance[self.app_num].nn_set_automatic_mode(auto_mode)
      # probably should use os.path.join()
      robot_dirs = []
      dataset_dir = "apps/" + self.app_registry[self.app_num][1] + "/dataset"
      # curr_nn_num is updated to reflect actual NN_num via nn_upon_reward
      self.nn_dir = dataset_dir + "/NN" + str(self.curr_nn_num)
      robot_dirs.append(self.nn_dir)
      for dir_name in robot_action_dirs:
          robot_dirs.append(self.nn_dir + "/" + dir_name)
      self.mkdirs(robot_dirs)
      print("nn_init: " , self.curr_nn_num, robot_dirs)
      return robot_dirs

    def nn_process_image(self):
      if self.function_name in ["ROBOT_OFF_TABLE_PENALTY", "CUBE_OFF_TABLE_REWARD", "PENALTY", "REWARD"]:
          rew_pen = self.function_name
      else:
          rew_pen = None
      if self.frame is None:
          print("nn_apps process_image None")
      action = self.app_instance[self.app_num].nn_process_image(self.curr_nn_num, self.frame, reward_penalty = rew_pen)
      if action == None:
          return None
# called directly from mcp to better handle automatic mode 
#      if action == "SUCCESS":
#          # do apps-specific transition to next function-specific NN
#          self.curr_nn_num = self.app_instance[self.app_num].nn_upon_reward(self.curr_nn_num)
#      elif action == "FAILURE":
#          # do apps-specific transition to next function-specific NN
#          self.curr_nn_num = self.app_instance[self.app_num].nn_upon_penalty(self.curr_nn_num)
      return action

    def nn_automatic_mode(self):
      return self.app_instance[self.app_num].nn_automatic_mode()

    def nn_automatic_action(self, feedback):
      if feedback == "REWARD":
          self.app_instance[self.app_num].nn_set_automatic_mode(False)
          self.curr_nn_num = self.nn_upon_reward()
          self.nn_init()
          return "REWARD"
      return self.app_instance[self.app_num].nn_automatic_action(self.curr_nn_num, self.frame, feedback)

    def nn_upon_reward(self):
      self.curr_nn_num = self.app_instance[self.app_num].nn_upon_reward(self.curr_nn_num)
      self.nn_init()
      return self.curr_nn_num

    def nn_upon_penalty(self):
      self.curr_nn_num = self.app_instance[self.app_num].nn_upon_penalty(self.curr_nn_num)
      self.nn_init()
      return self.curr_nn_num

    def train(self):
      # TODO: clean this up to keep the abstractions
      self.app_instance[self.app_num].train()

#     return
#      if self.app_name == "TT_NN":
#          # recomputes from scratch every time using dataset loader
#          # combine all datasets into a single NN
#          MODEL_PREFIX = "./apps/TT_NN/"
#          TT_MODEL = "TTNN_model1.pth"
#          DS_PREFIX = "./apps/TT_FUNC/dataset/NN"
#          found = False
#          for [TTFUNC_app_num, TTFUNC_app_name, TTFUNC_num_nn] in self.app_registry:
#               if TTFUNC_app_name == "TT_FUNC":
#                   found = True
#                   break
#          if not found:
#              print("Error: unknown app name %s" % TTFUNC_app_name)
#              exit()
#          model = MODEL_PREFIX + TT_MODEL
#          dataset = []
#          for nn_num in range(1, TTFUNC_num_nn+1):
#              ds = DS_PREFIX + str(nn_num)
#              dataset.append(ds)
#          dataset, model = self.app_instance[self.app_num].get_train_info(dataset, model)
#          self.app_instance[self.app_num].train(dataset, model)
#      elif self.app_name == "TT_FUNC":
#          # recomputes from scratch every time using dataset loader
#          # each datasets has a corresponding NN
#          MODEL_PREFIX = "./apps/TT_FUNC/"
#          TT_MODEL = "TTFUNC_model"
#          DS_PREFIX = "dataset/NN"
#          for nn_num in range(1, self.num_nn+1):
#              model = MODEL_PREFIX + TT_MODEL + str(nn_num) + ".pth"
#              ds = [MODEL_PREFIX + DS_PREFIX + str(nn_num)]
#              self.app_instance[self.app_num].train(nn_num, ds, model)
#      elif self.app_name == "TT_DQN":
#          # recomputes incrementally
#          self.app_instance[self.app_num].train()
