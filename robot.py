import time
import traitlets
from traitlets.config.configurable import SingletonConfigurable
from mcp23017 import *
from direct_control import *
from motor import *
from alset_joystick import *
from gather_data import *
from nn import *
from docopt import docopt
from nn_apps import *
from alset_ddqn import *
from config import *

#
# webcam->robot->gather_data->nn_apps -> [automatic mode, NN, DQN].process_image
#                   ^
#                   | gather_data shared variables.
#                   v
# joystick->robot->gather_data->[mcp, direct_control] -> robot manipulation
#
class Robot(SingletonConfigurable):
    def __init__(self, args):
        __doc__ = """
        Scripts to drive a Smarter Image Robot and train a model for it.
        Also used by other scripts to generate datasets.

        Usage:
            alset_{teleop,train}.py [--func nn_name] [--app app_name] [--dqn app_name] [--init]

        Options:
            -h --help        Show this screen.
            --func nn_name   name of a single simple NN/automatic action
            --app app_name   App defined as a series of NN/automatic actions in config.py
            --dqn app_name   RL trained on a series of NN/automatic actions in config.py
            --init           Initialize
        """
        print(__doc__)
        print("args:", args)
        # print("kwargs:", kwargs)
        # args = docopt(__doc__)
        self.cfg = Config()
        if args[1] == '--func':
             try:
               self.app_name  = args[2]
               print("app_name:", self.app_name)
               if self.app_name not in self.cfg.func_registry:
                 self.app_name  = args[2]
                 val = self.cfg.get_value(self.cfg.func_registry, self.app_name)
                 if val is None:
                   print("Function name not found: ", self.app_name)
                   print("Known function names: ", self.cfg.func_registry)
                   exit()
               self.app_type  = "FUNC"
             except:
               self.app_type  = None
               self.app_name  = None
        elif args[1] == '--app':
             try:
               self.app_name  = args[2]
               val = self.cfg.get_value(self.cfg.app_registry, self.app_name)
               if val is None:
                 if self.app_name not in self.cfg.func_registry:
                   print("Function name not found: ", self.app_name)
                   print("Known function names: ", self.cfg.app_registry)
                   exit()
               self.app_type  = "APP"
             except:
                 pass
        elif args[1] == '--dqn':
             try:
               self.app_name  = args[2]
               val = self.cfg.get_value(self.cfg.DQN_registry, self.app_name)
               if val is None:
                 print("Function name not found: ", self.app_name)
                 print("Known function names: ", self.cfg.DQN_registry)
                 exit()
               self.app_type  = "DQN"
             except:
               pass
        if self.app_type is None or self.app_name is None:
               print("bad app type/name:", self.app_type, self.app_name)
               exit()
        self.NN_apps  = nn_apps(alset_robot=self, alset_app_name=self.app_name, alset_app_type=self.app_type)
        # only for DQN
        self.initialize = False
        self.train_new_data = False
        if len(args) > 3 and args[3] == '--init':
             try:
               print("init: ", args[3])
               self.initialize = args[3]
               self.train_new_data = True
             except:
               self.initialize = False
               self.train_new_data = False
        # on DQN trains new-data-only, but does so automatically.
        # self.train_new_data = args['--train_new_data']
        # super(Robot, self).__init__(*args, **kwargs)
        self.gather_data = GatherData(self.NN_apps)
        # self.NN = ALSETNN(self)
        if self.cfg.ALSET_MODEL == "S":
          self.alset_robot = MCP_control(self)
        elif self.cfg.ALSET_MODEL == "X":
          self.alset_robot = direct_control(self)

        self.left_motor = Motor(self.alset_robot, "LEFT")
        self.right_motor = Motor(self.alset_robot, "RIGHT")
        self.NN_apps.nn_init()
        self.DQN = None
        alset_joystick_daemon(self)
        
    def set_motors(self, left_speed, right_speed):
        print("set_motors:", left_speed, right_speed)
        if self.gather_data.is_on():
            if left_speed == None:
                left_speed = self.left_motor.get_speed()
                if left_speed == None:
                  left_speed = 0
            if right_speed == None:
                right_speed = self.right_motor.get_speed()
                if right_speed == None:
                  right_speed = 0

            if abs(left_speed) <= 0.1 or abs(right_speed) <= 0.1:
                print("stop")
                self.stop()
            elif left_speed <= 0 and right_speed >= 0:
                print("forward")
                self.forward()
            elif left_speed >= 0 and right_speed <= 0:
                print("backward")
                self.backward()
            elif left_speed >= 0 and right_speed >= 0:
                print("left")
                self.left()
            elif left_speed <= 0 and right_speed <= 0:
                print("right")
                self.right()
            else:
                print("stop")
                self.stop()
        else:
            if self.cfg.ALSET_MODEL == "X":
              # print("l,r:", left_speed, right_speed)
              if left_speed is not None and right_speed is not None:
                  if left_speed < 0 and right_speed > 0:
                     right_speed = -right_speed
                  elif left_speed > 0 and right_speed < 0:
                     right_speed = -right_speed
                  elif left_speed > 0 and right_speed > 0:
                     left_speed = -left_speed
                  elif left_speed < 0 and right_speed < 0:
                     left_speed = -left_speed
                  # print("l2,r2:", left_speed, right_speed)
              elif right_speed is not None:
                right_speed = -right_speed
            self.alset_robot.set_motors(left_speed, right_speed)
        
    def forward(self, speed=1.0, duration=None):
        # if self.NN_apps.app_type == "DQN" and self.gather_data.is_on():
        #     pass
        # elif self.NN_apps.app_type in ["APP", "FUNC"] and self.get_NN_mode() == "NN":
        if self.NN_apps.app_type in ["APP", "FUNC"] and self.get_NN_mode() == "NN":
            pass
        elif self.gather_data.is_on():
            self.gather_data.set_action("FORWARD")
            self.alset_robot.drive_forward(speed)
        else:
            self.alset_robot.drive_forward(speed)

    def backward(self, speed=1.0):
        # if self.NN_apps.app_type == "DQN" and self.gather_data.is_on():
        #     pass
        # elif self.NN_apps.app_type in ["APP", "FUNC"] and self.get_NN_mode() == "NN":
        if self.NN_apps.app_type in ["APP", "FUNC"] and self.get_NN_mode() == "NN":
            pass
        elif self.gather_data.is_on():
            self.gather_data.set_action("REVERSE")
            self.alset_robot.drive_reverse(speed)
        else:
            self.alset_robot.drive_reverse(speed)

    def left(self, speed=1.0):
        # if self.NN_apps.app_type == "DQN" and self.gather_data.is_on():
        #     pass
        # elif self.NN_apps.app_type in ["APP", "FUNC"] and self.get_NN_mode() == "NN":
        if self.NN_apps.app_type in ["APP", "FUNC"] and self.get_NN_mode() == "NN":
            pass
        elif self.gather_data.is_on():
            self.gather_data.set_action("LEFT")
            self.alset_robot.drive_rotate_left(speed)
        else:
            self.alset_robot.drive_rotate_left(speed)

    def right(self, speed=1.0):
        # if self.NN_apps.app_type == "DQN" and self.gather_data.is_on():
        #     pass
        # elif self.NN_apps.app_type in ["APP", "FUNC"] and self.get_NN_mode() == "NN":
        if self.NN_apps.app_type in ["APP", "FUNC"] and self.get_NN_mode() == "NN":
            pass
        elif self.gather_data.is_on():
            self.gather_data.set_action("RIGHT")
            self.alset_robot.drive_rotate_right(speed)
        else:
            self.alset_robot.drive_rotate_right(speed)

    def stop(self):
        self.gather_data.set_action(None)
        self.alset_robot.drive_stop()

    def upper_arm(self,direction):
        # if self.NN_apps.app_type == "DQN" and self.gather_data.is_on():
        #     pass
        # elif self.NN_apps.app_type in ["APP", "FUNC"] and self.get_NN_mode() == "NN":
        if self.NN_apps.app_type in ["APP", "FUNC"] and self.get_NN_mode() == "NN":
            pass
        elif direction == "STOP":
          self.gather_data.set_action(None)
          self.alset_robot.upper_arm(direction)
        else:
          self.alset_robot.upper_arm(direction)
          self.gather_data.set_action("UPPER_ARM_" + direction)

    def lower_arm(self,direction):
        # if self.NN_apps.app_type == "DQN" and self.gather_data.is_on():
        #     pass
        # elif self.NN_apps.app_type in ["APP", "FUNC"] and self.get_NN_mode() == "NN":
        if self.NN_apps.app_type in ["APP", "FUNC"] and self.get_NN_mode() == "NN":
            pass
        elif direction == "STOP":
          self.gather_data.set_action(None)
          self.alset_robot.lower_arm(direction)
        else:
          self.gather_data.set_action("LOWER_ARM_" + direction)
          self.alset_robot.lower_arm(direction)

    def wrist(self,direction):
        self.alset_robot.wrist(direction)

    def chassis(self,direction):
        self.alset_robot.chassis(direction)

    def gripper(self,direction):
        # if self.NN_apps.app_type == "DQN" and self.gather_data.is_on():
        #     pass
        # elif self.NN_apps.app_type in ["APP", "FUNC"] and self.get_NN_mode() == "NN":
        if self.NN_apps.app_type in ["APP", "FUNC"] and self.get_NN_mode() == "NN":
            pass
        elif direction == "STOP":
          self.gather_data.set_action(None)
          self.alset_robot.gripper(direction)
        else:
          self.gather_data.set_action("GRIPPER_" + direction)
          self.alset_robot.gripper(direction)

    def shovel(self,direction):
        # if self.NN_apps.app_type == "DQN" and self.gather_data.is_on():
        #     pass
        # elif self.NN_apps.app_type in ["APP", "FUNC"] and self.get_NN_mode() == "NN":
        if self.NN_apps.app_type in ["APP", "FUNC"] and self.get_NN_mode() == "NN":
            pass
        elif direction == "STOP":
          self.gather_data.set_action(None)
          self.alset_robot.shovel(direction)
        else:
          self.gather_data.set_action("SHOVEL_" + direction)
          self.alset_robot.shovel(direction)

    def sound(self,on_off):
          self.alset_robot.sound(on_off)

    def demo(self,on_off):
          self.alset_robot.demo(on_off)

    def program_mode(self,on_off):
          self.alset_robot.program_mode(on_off)

    def reward(self):
        self.gather_data.set_action("REWARD1")
        self.alset_robot.stop_all()

    def penalty(self):
        self.gather_data.set_action("PENALTY1")
        self.alset_robot.stop_all()

    # gather data / teleop mode
    def set_gather_data_mode(self, mode=True): 
        self.gather_data.turn_on(mode)

    def get_gather_data_mode(self): 
        return self.gather_data.is_on()

    # run pytorch NN to determine next move
    def set_NN_mode(self, mode): 
        self.NN_apps.set_nn_mode(mode)
        if mode == "DQN" and self.DQN == None:
          self.DQN = ALSET_DDQN(self, self.initialize, self.train_new_data)

    # ARD: need to clean up now that generalized beyond TableTop
    def robot_off_table_penalty(self):
        # self.gather_data.set_action("ROBOT_OFF_TABLE_PENALTY")
        self.gather_data.set_action("PENALTY2")
        self.alset_robot.stop_all()

    def cube_off_table_reward(self):
        # self.gather_data.set_action("CUBE_OFF_TABLE_REWARD")
        self.gather_data.set_action("PENALTY2")
        self.alset_robot.stop_all()

    def get_NN_mode(self):
        return self.NN_apps.get_nn_mode()

    # webcam integration
    def get_ds_idx(self):
        return self.gather_data.get_ds_idx()

    def ready_for_capture(self):
        return self.NN_apps.ready_for_capture()

    def capture_frame(self, img):
        return self.NN_apps.capture_frame(img)

    def capture_frame_location(self):
        return self.gather_data.capture_frame_location()

    def capture_frame_completed(self):
        self.gather_data.capture_frame_completed()

    def do_process_image(self):
        return self.gather_data.do_process_image()

    def process_image(self):
        return self.gather_data.process_image()

    def train(self):
        self.NN_apps.train()
