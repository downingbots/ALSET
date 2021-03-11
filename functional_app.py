from .nn import *
from .config import *
from .automated_funcs import *

class FunctionalApp():
  def __init__(self, sir_robot, app_name, app_type):
      # print("TableTop: 8 Functional NNs")
      self.robot = sir_robot
      self.NN = []
      self.app_name = app_name
      self.app_type = app_type
      self.cfg = Config()
      self.ff_nn_num = None
      self.func_name = None
      self.func_automated   = None
      self.func_background  = None
      self.func_outputs     = None
      self.func_attributes  = None
      if app_type not in ["FUNC", "APP", "DQN"]:
        print("App Type must be in [nn, func_app, dqn]. Unknown value:", mode)
        exit()
      self.app_type = app_type

      val = self.cfg.get_value(self.cfg.app_registry, self.app_name)
      if val is not None:
          [self.NN, self.app_flow_model] = val
          self.is_composite_app = True
      elif self.app_name in self.cfg.func_registry:
          self.NN = [self.app_name]
          self.app_flow_model = [
               [[],["START", 0]],
               [[0], ["IF", "REWARD1", "STOP_WITH_REWARD1"] ],
               [[0], ["IF", "REWARD2", "STOP_WITH_REWARD2"] ],
               [[0], ["IF", "PENALTY1", "STOP"] ],
               [[0], ["IF", "PENALTY2", "STOP_WITH_PENALTY2"] ],
               ]
          self.is_composite_app = False
      else:
        print("No such app defined: ", self.app_name)
        exit()
      self.auto_func = AutomatedFuncs()

  def nn_init(self, gather_mode=False):
      # defaults

      gather_mode = False  # ARD: why does gather_mode matter for nn_init?
      outputs = self.cfg.full_action_set
      if not self.is_composite_app:
          print("Stand-alone function execution err. NN num > 1 for NN:", self.app_name)

      else:
        print("App: ", self.app_name, " mode:", self.app_type, " of ", len(self.NN))
        self.robot.sir_robot.stop_all()
        print(" ")
        if self.func_name is None:
          self.nn_upon_reward("REWARD1")
        print(self.func_name)
        print(" ")
        comment = self.cfg.get_func_value(self.func_name, "COMMENT")
        print(comment)
        print("  Press Failure or Success")

        if self.func_name is not None:
            # SIRNN is an actual torch NN. 
            # Torch NN Not needed during gather mode.
            # self.NN.append(SIRNN(self.robot, outputs))
 
            self.func_automated   = self.cfg.get_func_value(self.func_name, "AUTOMATED")
            self.func_subsumption = self.cfg.get_func_value(self.func_name, "SUBSUMPTION")
            self.func_outputs     = self.cfg.get_func_value(self.func_name, "MOVEMENT_RESTRICTIONS")
            self.func_attributes  = self.cfg.get_func_value(self.func_name, "ATTRIBUTES")  # for movement+checks
            self.func_classifier_nn = False  # TODO

            if func_automated:
              self.nn_set_automatic_mode(True)
            else:
              self.nn_set_automatic_mode(False)
            if not automated_func and not classifier_nn:
              # type depends if a classification or not
              self.NN = SIRNN(self.robot, outputs)
              self.NN.nn_init(app_name, NN_name, gather_mode)
            elif classifier_nn:
              self.NN = ClassifierNN(self.robot, outputs)
        return False, outputs

  #############
  # Go through the app func flow model.
  # Starting with ff_nn_num=None and passing in [REWARD1/2, PENALTY1/2] joystick action, 
  # return the next func_name and the output reward type.
  # If func_name is None, then the process flow is completed.
  #############
  def eval_func_flow_model(self, reward_penalty, init=False):
    rew_pen = None
    if init:
       self.ff_nn_num = None
       self.curr_phase = 0
    for [it0, it1] in self.app_flow_model:
      output_rew_pen = None
      if len(it0) == 0 in it0 and self.ff_nn_num is None:
         # starting point
         self.ff_nn_num = it1[1]
         break
      elif ((type(it0) == str and it0 == "ALL") or 
          (type(it0) == list and self.ff_nn_num in it0)):
          if it1[0] == "IF":
            if reward_penalty == it1[1]:
              if type(it2[2])==list:
                self.ff_nn_num = it1[2][1]
                if len(it2[2]) == 2 and it2[2][0] == "GOTO_WITH_REWARD1":
                  output_rew_pen = "REWARD1"
                elif len(it2[2]) == 2 and it2[2][0] == "GOTO_WITH_REWARD2":
                  output_rew_pen = "REWARD2"
                elif len(it2[2]) == 2 and it2[2][0] == "GOTO_WITH_PENALTY1":
                  output_rew_pen = "PENALTY1"
                elif len(it2[2]) == 2 and it2[2][0] == "GOTO_WITH:PENALTY2":
                  output_rew_pen = "PENALTY2"
                break
              elif it1[2] == "NEXT":
                self.ff_nn_num += 1
                break
              elif it1[2] in ["NEXT_WITH_REWARD1"]:
                self.ff_rew1_nn_num.append(ff_nn_num)
                output_rew_pen = "REWARD1"
                self.ff_nn_num += 1
                break
              elif it1[2] == "STOP":
                self.ff_nn_num = None
                break
              elif it1[2] in ["STOP_WITH_REWARD1", "STOP_WITH_REWARD2", "STOP_WITH_PENALTY1", "STOP_WITH_PENALTY2"]:
                self.ff_nn_num = None
                if it1[2] == "STOP_WITH_REWARD1":
                  output_rew_pen = "REWARD1"
                elif it1[2] == "STOP_WITH_REWARD2":
                  output_rew_pen = "REWARD2"
                elif it1[2] == "STOP_WITH_PENALTY1":
                  output_rew_pen = "PENALTY1"
                elif it1[2] == "STOP_WITH_PENALTY2":
                  output_rew_pen = "PENALTY2"
                break
    if self.ff_nn_num is None:
      return [None, output_rew_pen]
    return [nn_list[self.ff_nn_num], output_rew_pen]

  def func_flow_background(self):
    bg = []
    for [it0, it1] in func_flow:
      if it0 == "BACKGROUND_CHECK":
        bg.append(it1)
    return bg

  def nn_upon_penalty(self, penalty):
      [NN_name, penalty] = self.eval_func_flow_model(penalty)
      self.func_name = NN_name
      return NN_name

  def nn_upon_reward(self, reward):
      [NN_name, reward] = self.eval_func_flow_model(reward)
      self.func_name = NN_name
      return NN_name

  def nn_process_image(self, NN_name = None, image=None, reward_penalty=None):
      # run NN
      print("TT process_image %d" % NN_name)
      # allow reward/penalty to be returned by NN by setting to non-None

      return self.NN.nn_process_image(NN_name = NN_name, image=image, reward_penalty="REWARD_PENALTY")

  def nn_set_automatic_mode(self, TF):
      self.automatic_mode = TF

  def nn_automatic_mode(self):
      return self.automatic_mode

  def nn_automatic_action(self, NN_name=None, image=None, function_name=None):
      if self.nn_automatic_mode:
        print("auto_action: ", function_name,  self.NN)
        return self.auto_func.automatic_function(image, self.NN)
      else:
        print("nn_automatic_action called but not in automated mode")
        exit()

  # trains from scratch based on images saved in TT_FUNC dataset
  def train(self):
      for nn_num in range(1, len(self.NN)+1):
        if self.func_name is not None:
            self.nn_init(self.app_name, self.func_name, gather_mode=False)
        # elif self.robot.initialize:
        # elif self.robot.train_new_data:
        ds = self.ds_prefix + str(nn_num)
        dataset_root_list = [ds]
        model = self.model_dir + self.model_prefix + str(nn_num) + ".pth"
        self.NN.train(dataset_root_list, model)
