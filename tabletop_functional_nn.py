## OBSOLETE: replaced by generalized functional_app.py
from .nn import *

class tabletop_functional_nn():
  def __init__(self, sir_robot):
      # print("TableTop: 8 Functional NNs")
      self.robot = sir_robot
      self.NN = []
      self.app_name = "TT_FUNC"
      self.last_arm_action = "LOWER_ARM_DOWN"
      self.curr_automatic_action = "LEFT"
      self.automatic_mode = False
      self.robot_actions = ["UPPER_ARM_UP", "UPPER_ARM_DOWN", "LOWER_ARM_UP", "LOWER_ARM_DOWN",
                "GRIPPER_OPEN", "GRIPPER_CLOSE", "FORWARD", "REVERSE", "LEFT", "RIGHT"]
      self.joystick_actions = ["REWARD","PENALTY"]
      self.automatic_actions = [ "NOOP", "REWARD", "PENALTY"]
      self.automatic_mode_noop_mapping = ["LEFT", "LOWER_ARM_UP", "LOWER_ARM_DOWN"]
      self.full_action_set = self.robot_actions + self.joystick_actions
      self.full_action_set.sort()
      self.model_dir = "./apps/TT_FUNC/"
      self.model_prefix = "TTFUNC_model"
      self.ds_prefix = "./apps/TT_FUNC/dataset/NN"
      self.TTFUNC_NUM_NN = 8

  def nn_init(self, app_name, NN_num, gather_mode=False):
      print("NN_INIT: %d" % NN_num)
      # defaults

      gather_mode = False  # ARD: why does gather_mode matter for nn_init?
      outputs = self.full_action_set
      if NN_num == 1:
        print("TableTop: 8 Functional NNs")
        self.robot.sir_robot.stop_all()
        print("Park Arm: Upper/Lower Arm Up/Down")
        print("  Press Success: parked")
        # outputs = [ "GRIPPER_OPEN", "UPPER_ARM_UP", "UPPER_ARM_DOWN", 
        #            "LOWER_ARM_UP", "LOWER_ARM_DOWN", "REWARD",
        #            "FORWARD", "REVERSE", "LEFT", "RIGHT"] 
        # if gather_mode:
        if len(self.NN) < NN_num:
            # SIRNN is an actual torch NN. 
            # Torch NN Not needed during gather mode.
            # self.NN.append(SIRNN(self.robot, outputs))
            self.NN.append(SIRNN(self.robot, outputs))
            self.NN[NN_num-1].nn_init(app_name, NN_num, gather_mode)
        return False, outputs
      elif NN_num == 2:
        self.robot.sir_robot.stop_all()
        print("Automatic: scan for block")
        print("  Press Failure: scan direction completed")
        print("  Press Success: found block")
        # start with raising the lower arm
        # self.last_arm_action = "LOWER_ARM_UP"
        self.last_arm_action = "LOWER_ARM_DOWN"
        self.curr_automatic_action = "LEFT"
        self.robot.gather_data.set_function(None)
        self.left_count = 45
        # outputs = ["LOWER_ARM_UP", "LOWER_ARM_DOWN","LEFT","REWARD","PENALTY"]
        # if gather_mode:
        if len(self.NN) < NN_num:
            self.NN.append(SIRNN(self.robot, outputs))
            self.NN[NN_num-1].nn_init(app_name, NN_num, gather_mode)
        return True, outputs
      elif NN_num == 3:
        self.robot.sir_robot.stop_all()
        self.robot.gather_data.set_function(None)
        print("Keep block in center while moving forward")
        print("  Press Lower Arm UP/Down; Left; Right; Forward")
        print("  Press Forward: block in center of image")
        print("  Press Success: block in center and close enough to pick up")
        print("  Press Failure: block out of image go back to 1")
        # outputs = ["LOWER_ARM_UP", "LOWER_ARM_DOWN",
        #         "FORWARD", "REVERSE", "LEFT", "RIGHT", "REWARD", "PENALTY"]
        if len(self.NN) < NN_num:
            self.NN.append(SIRNN(self.robot, outputs))
            self.NN[NN_num-1].nn_init(app_name, NN_num, gather_mode)
        return False, outputs
      elif NN_num == 4:
        self.robot.sir_robot.stop_all()
        self.robot.gather_data.set_function(None)
        print("Reach out and grab block")
        print("  Press Upper/Lower Arm UP/Down")
        print("  Press Success: lift up and confirm block in gripper")
        print("  Press Failure: go back to 1")
        # outputs = ["UPPER_ARM_UP", "UPPER_ARM_DOWN", 
        #         "LOWER_ARM_UP", "LOWER_ARM_DOWN",
        #         "GRIPPER_CLOSE", "REWARD", "PENALTY"]
        # if gather_mode:
        if len(self.NN) < NN_num:
            self.NN.append(SIRNN(self.robot, outputs))
            self.NN[NN_num-1].nn_init(app_name, NN_num, gather_mode)
        return False, outputs
      elif NN_num == 5:
        self.robot.sir_robot.stop_all()
        self.robot.gather_data.set_function(None)
        print("Park Arm with block: Upper/Lower Arm Up/Down")
        print("  Press Success: parked")
        # outputs = [ "GRIPPER_OPEN", "UPPER_ARM_UP", "UPPER_ARM_DOWN", "LOWER_ARM_UP", "LOWER_ARM_DOWN", "REWARD", "PENALTY"]
        if len(self.NN) < NN_num:
            self.NN.append(SIRNN(self.robot, outputs))
            self.NN[NN_num-1].nn_init(app_name, NN_num, gather_mode)
        return False, outputs
      elif NN_num == 6:
        self.robot.sir_robot.stop_all()
        self.last_arm_action = "LOWER_ARM_DOWN"
        self.curr_automatic_action = "LEFT"
        self.robot.gather_data.set_function(None)
        self.left_count = 45
        print("Automatic: scan for box")
        print("  Press Failure: scan direction completed")
        print("  Press Success: found block")
        # outputs = ["LOWER_ARM_UP", "LOWER_ARM_DOWN","LEFT","REWARD","PENALTY"]
        if len(self.NN) < NN_num:
            self.NN.append(SIRNN(self.robot, outputs))
            self.NN[NN_num-1].nn_init(app_name, NN_num, gather_mode)
        return True, outputs
      elif NN_num == 7:
        self.robot.sir_robot.stop_all()
        self.robot.gather_data.set_function(None)
        print("Keep box in center while moving forward")
        print("  Press Lower Arm UP/Down; Left; Right; Forward")
        print("  Press Forward: box in center of image")
        print("  Press Success: box in center and close enough to pick up")
        print("  Press Failure: box out of image go back to 5")
        # outputs = ["LOWER_ARM_UP", "LOWER_ARM_DOWN", "FORWARD", "REVERSE", 
        #         "LEFT", "RIGHT", "REWARD", "PENALTY"]
        if len(self.NN) < NN_num:
            self.NN.append(SIRNN(self.robot, outputs))
            self.NN[NN_num-1].nn_init(app_name, NN_num, gather_mode)
        return False, outputs
      elif NN_num == 8:
        self.robot.sir_robot.stop_all()
        self.robot.gather_data.set_function(None)
        print("Reach out arm and drop cube in center and back up")
        # outputs = ["UPPER_ARM_UP", "UPPER_ARM_DOWN", 
        #         "LOWER_ARM_UP", "LOWER_ARM_DOWN",
        #         "LEFT", "RIGHT", "GRIPPER_OPEN",
        #         "REVERSE", "REWARD", "PENALTY"]
        if len(self.NN) < NN_num:
            self.NN.append(SIRNN(self.robot, outputs))
            self.NN[NN_num-1].nn_init(app_name, NN_num, gather_mode)
        return False, outputs

  def nn_upon_penalty(self, NN_num):
      if NN_num == 2 or NN_num == 6:
        print("Penalty in automatic mode?")
      elif NN_num <= 4:
        return 1
      else:
        return 5

  def nn_upon_reward(self, NN_num):
      # reverse twice (no pwm)
      if NN_num == 8:
        return 1
      return NN_num+1

  def nn_process_image(self, NN_num = 0, image=None, reward_penalty=None):
      # run NN
      print("TT process_image %d" % NN_num)
      # allow reward/penalty to be returned by NN by setting to non-None
      return self.NN[NN_num-1].nn_process_image(NN_num = NN_num, image=image, reward_penalty="REWARD_PENALTY")

  def nn_set_automatic_mode(self, TF):
      self.automatic_mode = TF

  def nn_automatic_mode(self):
      return self.automatic_mode

  def nn_automatic_action(self, NN_num, feedback):
      # if feedback == "REWARD":
      #     self.nn_upon_reward(NN_num)
      # elif feedback == "PENALTY":
      #     self.nn_upon_penalty(NN_num)
      if NN_num == 2 or NN_num == 6:
          if (self.curr_automatic_action == "LEFT" and
              self.last_arm_action == "LOWER_ARM_DOWN" and
              self.left_count >= 45):
              # 15 pulses is actually 3 rotation pulses
              self.left_count = 0
              self.curr_automatic_action = "LOWER_ARM_UP"
              self.last_arm_action = "LOWER_ARM_UP"
          elif (self.curr_automatic_action == "LEFT" and
                self.last_arm_action == "LOWER_ARM_UP" and
                self.left_count >= 45):
              # 15 pulses is actually 3 rotation pulses
              self.left_count = 0
              self.curr_automatic_action = "LOWER_ARM_DOWN"
              self.last_arm_action = "LOWER_ARM_DOWN"
          elif (self.curr_automatic_action in ["LOWER_ARM_UP", "LOWER_ARM_DOWN"] and feedback == "PENALTY"):
              self.curr_automatic_action = "LEFT"
              self.left_count += 1
          elif (self.curr_automatic_action == "LEFT"):
              self.curr_automatic_action = "LEFT"
              self.left_count += 1
          else: 
              self.curr_automatic_action = self.last_arm_action
          if self.curr_automatic_action != None:
              print("nn_automatic_action: %s" % self.curr_automatic_action)
          self.robot.gather_data.set_function(self.curr_automatic_action)
          return self.curr_automatic_action
      else:
          return feedback
      return None

  # trains from scratch based on images saved in TT_FUNC dataset
  def train(self):
      for nn_num in range(1, self.TTFUNC_NUM_NN+1):
        if len(self.NN) < nn_num:
            self.nn_init(self.app_name, nn_num, gather_mode=False)
        # elif self.robot.initialize:
        # elif self.robot.train_new_data:
        ds = self.ds_prefix + str(nn_num)
        dataset_root_list = [ds]
        model = self.model_dir + self.model_prefix + str(nn_num) + ".pth"
        self.NN[nn_num-1].train(dataset_root_list, model)
