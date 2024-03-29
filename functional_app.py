from config import *
from automated_funcs import *
from classifier import *
import dataset_utils as ds_utils
import nn as alset_nn

class FunctionalApp():
  def __init__(self, alset_robot, app_name, app_type):
      # print("TableTop: 8 Functional NNs")
      self.robot = alset_robot
      self.NN = []
      self.dsu = ds_utils.DatasetUtils(app_name, app_type)
      self.app_name = app_name
      self.app_type = app_type
      self.app_ds_idx = self.dsu.dataset_indices(mode="APP", nn_name=self.app_name, position="NEW")
      self.cfg = Config()
      self.ff_nn_num = None
      self.curr_func_name = None
      self.nn_init_done = False
      self.func_names       = []
      self.func_app_function= []
      self.func_automated   = []
      self.func_background  = []
      self.func_comment     = []
      self.func_outputs     = []
      self.func_classifier_outputs = []
      self.func_classifier_flow = []
      self.func_attributes  = []
      self.func_subsumption = []
      self.func_attributes  = []
      if app_type not in ["FUNC", "APP", "DQN"]:
        print("App Type must be in [nn, func_app, dqn]. Unknown value:", mode)
        exit()
      self.app_type = app_type
      outputs = self.cfg.full_action_set
      val = self.cfg.get_value(self.cfg.app_registry, self.app_name)
      print("app registry val:", val)
      if val is not None:
        [self.NN, self.app_flow_model] = val
        self.is_composite_app = True
      elif self.app_name in self.cfg.func_registry:
          # not currently used
          self.NN = [self.app_name]
          print("Stand-alone function execution:", self.app_name)
          classifier = self.cfg.get_func_value(self.app_name, "CLASSIFIER")
          if classifier is not None:
            self.classifier_outputs = classifier[0]
            func_flow = classifier[1]
            self.app_flow_model = func_flow
          else:
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
      self.auto_func = AutomatedFuncs(self.robot)

  # creates the metadata for all the NNs/Functions in the APP
  # NN is not instantiated here.
  # calling the nn.nn_init() for each func name actually instantiates the NN. 
  def nn_init(self, gather_mode=False):
      # defaults
      gather_mode = False  # ARD: why does gather_mode matter for nn_init?
      print("App: ", self.app_name, " mode:", self.app_type, " of ", len(self.NN))
      self.robot.alset_robot.stop_all()
      ds_dirs = []
      outputs = self.cfg.full_action_set
      print(" ")
      if not self.nn_init_done:  # previously run
        for nn_num, nn_name in enumerate(self.NN):
            if nn_name in self.func_names:
                continue
            self.func_names.append(nn_name)
            self.func_comment.append(self.cfg.get_func_value(nn_name, "COMMENT"))
            # self.cfg.set_debug(True)
            # print("func_automated append debug on")
            self.func_automated.append(self.cfg.get_func_value(nn_name, "AUTOMATED"))
            # self.cfg.set_debug(False)
            self.func_subsumption.append(self.cfg.get_func_value(nn_name, "SUBSUMPTION"))
            self.func_outputs.append(self.cfg.get_func_value(nn_name, "MOVEMENT_RESTRICTIONS"))
            classifier = self.cfg.get_func_value(nn_name, "CLASSIFIER")
            if classifier is None:
              self.func_classifier_outputs.append(None)
              self.func_classifier_flow.append(None)
              outputs = self.cfg.full_action_set
            else:
              outputs = classifier[0]
              self.func_classifier_outputs.append(outputs)
              func_flow = classifier[1]
              self.func_classifier_flow.append(func_flow)
              self.app_flow_model = func_flow
              print("classifier info: ", outputs, func_flow, len(self.func_classifier_flow))

            # for movement+checks

            ds_path = self.dsu.dataset_path(mode="FUNC", nn_name=nn_name)
            ds_dirs.append(ds_path)
            for act in outputs:
              ds_dirs.append(ds_path + act)

            print("func_automated", nn_name, self.func_automated)
            # if not self.func_automated[nn_num] and self.func_classifier_outputs[nn_num] is None:
            # ARD: if we're in training mode, we still need to create ALSETNN object.
            # ARD: postpone creation until until needed...
            # if not self.func_automated[nn_num] and self.func_classifier_outputs[nn_num] is None:
            if not self.func_automated[nn_num] or self.func_classifier_outputs[nn_num] is not None:
              print("func_classifier_outputs: ", self.func_classifier_outputs[nn_num], nn_num)
              # ALSETNN is an actual torch NN after the call to nn_init. 
              # type depends if a classification or not
              self.func_app_function.append(alset_nn.ALSETNN(self.robot, outputs, self.func_names[nn_num], "FUNC"))
              # Don't instantiate yet by calling nn_init
              # self.func_app_function[-1].nn_init(gather_mode)
            else:
              # TODO: handle a clasifier function!
              self.func_app_function.append(None)
      if not self.nn_init_done:
        self.dsu.mkdirs(ds_dirs)
        if self.curr_func_name is None:
          # reset func flow
          [self.curr_func_name, rew_pen] = self.eval_func_flow_model(reward_penalty="REWARD1", init=True)
          self.robot.gather_data.set_function_name(self.curr_func_name)
          print("curr_func_name:", self.curr_func_name)
          idx = self.func_names.index(self.curr_func_name)
          print("Current Phase: ", self.func_comment[idx])
      else:
          idx = self.func_names.index(self.curr_func_name)
      self.nn_init_done = True
      if self.func_automated[idx]:
        auto_mode=True
      else:
        auto_mode=False
      self.nn_set_automatic_mode(auto_mode)
      return auto_mode, outputs

  #############
  # Go through the app func flow model.
  # Starting with ff_nn_num=None and passing in [REWARD1/2, PENALTY1/2] joystick action, 
  # return the next func_name and the output reward type.
  # If func_name is None, then the process flow is completed.
  #############
  def eval_func_flow_model(self, reward_penalty, init=False):
    rew_pen = None
    nn_output = None
    if init:
       self.ff_nn_num = None
       # self.curr_phase = 0
    if self.ff_nn_num is not None:
      NN_name = self.NN[self.ff_nn_num]
      if (len(self.func_classifier_outputs) > self.ff_nn_num and
         self.func_classifier_outputs[self.ff_nn_num] is None):
        nn_output = self.cfg.full_action_set
      elif len(self.func_classifier_outputs) > self.ff_nn_num:
        nn_output = self.func_classifier_outputs[self.ff_nn_num]
      # print("AFM: nn_output: ", nn_output)
    for [it0, it1] in self.app_flow_model:
      # print("AFM:", reward_penalty, self.ff_nn_num, it0, it1)
      # AFM: ending:  None
      output_rew_pen = None
      if len(it0) == 0 and self.ff_nn_num is None:
         # starting point
         # print("AFM: starting")
         self.ff_nn_num = it1[1]
         break
      elif ((type(it0) == str and it0 == "ALL") or 
          (type(it0) == list and self.ff_nn_num in it0)):
          if it1[0] == "IF":
            # print("AFM: IF")
            if reward_penalty == it1[1]:
              # print("AFM: matching reward")
              if type(it1[2])==list:
                # print("AFM: list compare")
                self.ff_nn_num = it1[2][1]
                if len(it1[2]) == 2 and it1[2][0] == "GOTO_WITH_REWARD1":
                  output_rew_pen = "REWARD1"
                elif len(it1[2]) == 2 and it1[2][0] == "GOTO_WITH_REWARD2":
                  output_rew_pen = "REWARD2"
                elif len(it1[2]) == 2 and it1[2][0] == "GOTO_WITH_PENALTY1":
                  output_rew_pen = "PENALTY1"
                elif len(it1[2]) == 2 and it1[2][0] == "GOTO_WITH:PENALTY2":
                  output_rew_pen = "PENALTY2"
                break
              elif it1[2] == "NEXT":
                # print("AFM: NEXT")
                self.ff_nn_num += 1
                break
              elif it1[2] in ["NEXT_WITH_REWARD1"]:
                # print("AFM: NEXT WITH REW1")
                output_rew_pen = "REWARD1"
                self.ff_nn_num += 1
                break
              elif it1[2] == "STOP":
                # print("AFM: STOP")
                self.ff_nn_num = None
                break
              elif it1[2] in ["STOP_WITH_REWARD1", "STOP_WITH_REWARD2", "STOP_WITH_PENALTY1", "STOP_WITH_PENALTY2"]:
                # print("AFM: STOP w REW/PEN")
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
              elif nn_output is not None and it1[1] in nn_output and it1[2] in self.cfg.full_action_set:
                # print("AFM: output mapping:", it1[1], it1[2])
                output_rew_pen = it1[2]
                break
            else:
                output_rew_pen = reward_penalty
                print("AFT: rew_pen ", output_rew_pen)
    if self.ff_nn_num is None:
      print("AFM: ending: ", output_rew_pen)
      return [None, output_rew_pen]
    print("AFM: eval ", self.ff_nn_num, self.NN[self.ff_nn_num], output_rew_pen)
    return [self.NN[self.ff_nn_num], output_rew_pen]

  def func_flow_background(self):
    bg = []
    for [it0, it1] in func_flow:
      if it0 == "BACKGROUND_CHECK":
        bg.append(it1)
    return bg

  def new_nn(self, nn_name):
      
      self.app_ds_idx 
      new_func_ds_idx = self.ds_util.dataset_indices(mode="FUNC", nn_name=self.nn_name, position="NEW")
      self.robot.gather_data.set_ds_idx(new_func_ds_idx)

  def nn_upon_penalty(self, penalty):
      print("######################################")
      [NN_name, penalty] = self.eval_func_flow_model(penalty)
      self.curr_func_name = NN_name
      self.robot.gather_data.set_function_name(self.curr_func_name)
      idx = self.func_names.index(self.curr_func_name)
      print("Current Phase: ", self.func_comment[idx])
      if self.func_automated[idx]:
        auto_mode=True
      else:
        auto_mode=False
      self.nn_set_automatic_mode(auto_mode)
      return NN_name

  def nn_upon_reward(self, reward):
      print("######################################")
      [NN_name, reward] = self.eval_func_flow_model(reward)
      if NN_name is None:
          # complete APP_IDX entry
          self.robot.gather_data.set_function_name("DUMMY_IDX")
          print("DONE!")
          exit()
      self.curr_func_name = NN_name
      self.robot.gather_data.set_function_name(self.curr_func_name)
      idx = self.func_names.index(self.curr_func_name)
      print("Current Phase: ", self.func_comment[idx])
      if self.func_automated[idx]:
        auto_mode=True
      else:
        auto_mode=False
      self.nn_set_automatic_mode(auto_mode)
      return NN_name

  def nn_process_image(self, NN_name = None, image=None, reward_penalty=None):
      # run NN
      print("Functional App process_image %s" % NN_name)
      if image is None or NN_name is None:
          return None
      if not self.nn_init_done:
          self.nn_init()
      # allow reward/penalty to be returned by NN by setting to non-None
      return self.func_app_function.nn_process_image(NN_name = self.curr_func_name, image=image, reward_penalty=reward_penalty)

  def nn_set_automatic_mode(self, TF):
      self.automatic_mode = TF

  def nn_automatic_mode(self):
      return self.automatic_mode

  def nn_automatic_action(self, NN_name=None, image=None, reward_penalty=None):
      if self.nn_automatic_mode:
        # automatic_function(self, frame, reward_penalty):
        idx = self.func_names.index(self.curr_func_name)
        # [parent_action_func] = self.func_automated[idx]
        # print("auto_action: ", NN_name, reward_penalty, self.curr_func_name, parent_action_func)
        print("auto_action: ", NN_name, reward_penalty, self.curr_func_name)
        # self.auto_func.set_automatic_function(parent_action_func)
        self.auto_func.set_automatic_function(self.curr_func_name)
        val,done = self.auto_func.automatic_function(image, reward_penalty)
        print("auto_action val: ", val)
        return val,done
      else:
        print("nn_automatic_action called but not in automated mode")
        exit()

  # trains each FUNC used by the APP based on images saved in FUNC dataset
  def train(self):
        if self.func_names is None:
            self.nn_init(gather_mode=False)

        for nn_num, nn_name in enumerate(self.func_names):
          print("Train model: ", nn_name)
          nn = self.func_app_function[nn_num]
          if nn is None:
              # JIT creation of ALSETNN
              outputs = self.func_classifier_outputs[nn_num]
              if outputs is None:
                 outputs = self.cfg.full_action_set
              self.func_app_function[nn_num] = alset_nn.ALSETNN(self.robot, outputs, self.func_names[nn_num], "FUNC")
              nn = self.func_app_function[nn_num]
          # nn.nn_init(gather_mode=False)
          nn.nn_init(False)
          if not self.func_automated[nn_num] and not self.func_classifier_outputs[nn_num]:
            ds = self.dsu.dataset_path(mode="FUNC", nn_name=nn_name)
            dataset_root_list = [ds]
            model = self.dsu.best_model(mode="FUNC", nn_name=nn_name)
            action_set = self.cfg.full_action_set
            self.func_app_function[nn_num].train(dataset_root_list, best_model_path=model, full_action_set = action_set)
            self.func_app_function[nn_num].cleanup()
