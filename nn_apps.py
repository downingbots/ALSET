# Flow between
# 
# robot.py
#   only interacts with NN_apps
# 
# NN_apps (nn_apps.py)
#   currently 3 apps are defined:
#      1. a single ALSETNN
#      2. one tabletop_nn that will define 8 ALSETNN
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
#  functional_app.py -> generalization of tabletop version
#   self.NN[0-7] -> defines a ALSETNN for each function performed
#   knows number of NN to define
#   knows actions for each NN
#   knows/calls actual torch NN (ALSETNN)
#   defines flow between NN in config file
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
#   ALSETNN -> actual single torch NN
#   also a self-contained single-NN app
import os
import time
import cv2
import numpy as np
from robot import *
from alset_ddqn import *
from functional_app import *
from automated_funcs import *
import nn as alset_nn
from config import *
import time

class nn_apps():
    # initialize app registry
    def __init__(self, alset_robot, alset_app_name=None, alset_app_type=None):
      self.app_name = alset_app_name
      self.app_type = alset_app_type
      self.cfg = Config()
      self.action_set = self.cfg.full_action_set
      if alset_app_type == "FUNC":
        self.nn_name = alset_app_name
        classifier = self.cfg.get_func_value(self.nn_name, "CLASSIFIER")
        if classifier is not None:
          self.classifier_output = classifier[0]
          print("classifier output: ", self.classifier_output)
          self.action_set = self.classifier_output
      # self.app_instance = []
      robot_dirs = []
      robot_dirs.append("apps/")
      robot_dirs.append("apps/FUNC")
      self.dsu = DatasetUtils(alset_app_name, alset_app_type)
      self.dsu.mkdirs(robot_dirs)
      self.app_functions = None
      self.func_flow_model = None
      self.dqn_policy = None
      self.curr_nn_name = None
      if alset_app_type == "FUNC":
        self.app_instance = alset_nn.ALSETNN(alset_robot, self.action_set, self.nn_name, alset_app_type)
        self.curr_nn_name = self.nn_name
      elif alset_app_type == "APP":
        [self.app_functions, self.func_flow_model] = self.cfg.get_value(self.cfg.app_registry, alset_app_name)
        self.app_instance = FunctionalApp(alset_robot, alset_app_name, alset_app_type)
      elif alset_app_type == "DQN":
        [self.app_functions, self.func_flow_model] = self.cfg.get_value(self.cfg.app_registry, alset_app_name)
        [self.dqn_policy] = self.cfg.get_value(self.cfg.DQN_registry, alset_app_name)
        self.app_instance = ALSET_DDQN(alset_robot, True, False, alset_app_name, alset_app_type)
      # for action in self.action_set:
      #     robot_dirs.append("apps/FUNC/" + action)
      self.mode = "TELEOP"
      self.ready_for_frame = False
      self.frame = None
      self.robot = alset_robot
      self.auto_funcs = AutomatedFuncs(self.robot)
      self.auto_done = False
      self.nn_dir = None

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

    # ARD: TODO: rename to set_action() to match current terminology
    def set_action(self, action):
        if action == None or action in self.action_set or action in ["ROBOT_OFF_TABLE_PENALTY", "CUBE_OFF_TABLE_REWARD", "PENALTY1", "REWARD1", "PENALTY2", "REWARD2"]:
            self.action_name = action
        else:
            print("bad action name: %s" % action)
            return False
        return True

    def get_snapshot_dir(self):
      # where to store the images
      if self.app_type in ["APP", "FUNC"]:
        app_dir_type = "FUNC"
        if self.app_type == "FUNC":
          func_nm = self.nn_name
        elif self.app_type == "APP":
          func_nm = self.app_instance.curr_func_name
      else:
        app_dir_type = "DQN"
        func_nm = None
      self.nn_dir = self.dsu.dataset_path(app_dir_type, func_nm)
      return self.nn_dir 

    ####################################################
    # PASS-THROUGH FUNCTION CALLS
    ####################################################

    # called to initialize or switch to a new NN for gathering data or exec
    # idempotent
    def nn_init(self):
      gather_mode = self.robot.get_gather_data_mode()
      self.robot.gather_data.set_action(None)
      # automatic_mode must be set to false until after the directories
      # are created.
      self.app_instance.nn_set_automatic_mode(False)

      # print(self.app_instance[self.app_num])
      print(self.curr_nn_name, gather_mode)
      auto_mode, robot_action_dirs = self.app_instance.nn_init(gather_mode)
      print("auto_mode:", auto_mode)
      self.app_instance.nn_set_automatic_mode(auto_mode)
      # probably should use os.path.join()
      return robot_action_dirs

    def nn_process_image(self):
      if self.action_name in ["ROBOT_OFF_TABLE_PENALTY", "CUBE_OFF_TABLE_REWARD", "PENALTY1", "REWARD1", "PENALTY2", "REWARD2"]:
          rew_pen = self.action_name
      else:
          rew_pen = None
      if self.frame is None:
          print("nn_apps process_image None")
      action = self.app_instance.nn_process_image(self.curr_nn_name, self.frame, reward_penalty = rew_pen)
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
      return self.app_instance.nn_automatic_mode()

    def nn_automatic_action(self, feedback):
      print("auto act", self.curr_nn_name, feedback)
      val,self.auto_done = self.app_instance.nn_automatic_action(self.curr_nn_name, self.frame, feedback)
      return val

    def nn_automatic_post_action(self, feedback):
      # called after storing image & updating indx 
      if feedback == "REWARD1":
          self.app_instance.nn_set_automatic_mode(False)
          self.curr_nn_name = self.nn_upon_reward(feedback)
          self.nn_init()
          return "REWARD1"
      if self.auto_done:
          return self.nn_upon_reward("REWARD1")
      print("nn_automatic_post_action")
      return self.curr_nn_name

    def nn_upon_reward(self, reward):
      self.curr_nn_name = self.app_instance.nn_upon_reward(reward)
      self.nn_init()
      return self.curr_nn_name

    def nn_upon_penalty(self, penalty):
      # self.curr_nn_name = self.app_instance.nn_upon_penalty(self.curr_nn_name, penalty)
      self.curr_nn_name = self.app_instance.nn_upon_penalty(penalty)
      self.nn_init()
      return self.curr_nn_name

    def train(self):
      # TODO: clean this up to keep the abstractions
      self.app_instance.train()

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
