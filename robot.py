import time
import traitlets
from traitlets.config.configurable import SingletonConfigurable
from .mcp23017 import *
from .motor import *
from .sir_joystick import *
from .gather_data import *
from .nn import *
from docopt import docopt
from .nn_apps import *
from .sir_ddqn import *
from .config import *

class Robot(SingletonConfigurable):
    def __init__(self, *args, **kwargs):
        __doc__ = """
        Scripts to drive a Smarter Image Robot and train a model for it.

        Usage:
            sir_robot_{teleop,train}.py [--nn=nn_name] [--app=app_name] [--dqn=app_name]

        Options:
            -h --help        Show this screen.
            --func=nn_name   name of a single simple NN/automatic action
            --app=app_name   App defined as a series of NN/automatic actions in config.py
            --dqn=app_name   RL trained on a series of NN/automatic actions in config.py
        """
        self.cfg = Config()
        # print(__doc__)
        args = docopt(__doc__)
        if args['--func']:
          self.app_name  = args['--func'] 
          if self.app_name not in self.cfg.NN_registry:
            self.app_name  = args['--func'] 
            if self.app_name not in self.cfg.app_registry:
              print("NN:
            print("Function name not found: ", self.app_name)
            print("Known function names: ", self.cfg.NN_registry)
          self.app_type  = "FUNC"
        elif args['--app']:
          self.app_name  = args['--func'] 
          if self.app_name not in self.cfg.app_registry:
            print("Function name not found: ", self.app_registry)
            print("Known function names: ", self.cfg.composit_app)
          self.app_type  = "FUNC"
          name = args['--app'] 
          name = args['--dqn'] 
 

        found = False
        name = args['--app'] 
        name = args['--dqn'] 
        self.NN_apps  = nn_apps(self, find_app_num, find_app_name)
        for [self.app_num,self.app_name,self.num_NN] in self.NN_apps.app_registry:
            print(args['--app_name'] ,self.app_name)
            if args['--app_name'] == self.app_name:
                found = True
                break
            elif args['--app_num'] == self.app_name:
                found = True
                break
        if not found:
            [self.app_num,self.app_name,self.num_NN] = self.NN_apps.app_registry[0]
        # only for DQN
        print("init: ", args['--init'])
        self.initialize = args['--init']
        # on DQN trains new-data-only, but does so automatically.
        # self.train_new_data = args['--train_new_data']
        if self.initialize:
          self.train_new_data = True
        print(self.app_num, self.app_name, self.num_NN)
        # super(Robot, self).__init__(*args, **kwargs)
        self.gather_data = GatherData(self.NN_apps)
        # self.NN = SIRNN(self)
        self.sir_robot = SIR_control(self)
        self.left_motor = Motor(self.sir_robot, "LEFT")
        self.right_motor = Motor(self.sir_robot, "RIGHT")
        self.NN_apps.nn_init()
        self.DQN = None
        sir_joystick_daemon(self)
        
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
            self.sir_robot.set_motors(left_speed, right_speed)
        
    def forward(self, speed=1.0, duration=None):
        if self.NN_apps.app_name == "TT_DQN" and self.gather_data.is_on():
            pass
        elif self.NN_apps.app_name in ["TT_NN", "TT_FUNC"] and self.get_NN_mode() == "NN":
            pass
        elif self.gather_data.is_on():
            self.gather_data.set_function("FORWARD")
            self.sir_robot.drive_forward(speed)
        else:
            self.sir_robot.drive_forward(speed)

    def backward(self, speed=1.0):
        if self.NN_apps.app_name == "TT_DQN" and self.gather_data.is_on():
            pass
        elif self.NN_apps.app_name in ["TT_NN", "TT_FUNC"] and self.get_NN_mode() == "NN":
            pass
        elif self.gather_data.is_on():
            self.gather_data.set_function("REVERSE")
            self.sir_robot.drive_reverse(speed)
        else:
            self.sir_robot.drive_reverse(speed)

    def left(self, speed=1.0):
        if self.NN_apps.app_name == "TT_DQN" and self.gather_data.is_on():
            pass
        elif self.NN_apps.app_name in ["TT_NN", "TT_FUNC"] and self.get_NN_mode() == "NN":
            pass
        elif self.gather_data.is_on():
            self.gather_data.set_function("LEFT")
            self.sir_robot.drive_rotate_left(speed)
        else:
            self.sir_robot.drive_rotate_left(speed)

    def right(self, speed=1.0):
        if self.NN_apps.app_name == "TT_DQN" and self.gather_data.is_on():
            pass
        elif self.NN_apps.app_name in ["TT_NN", "TT_FUNC"] and self.get_NN_mode() == "NN":
            pass
        elif self.gather_data.is_on():
            self.gather_data.set_function("RIGHT")
            self.sir_robot.drive_rotate_right(speed)
        else:
            self.sir_robot.drive_rotate_right(speed)

    def stop(self):
        self.gather_data.set_function(None)
        self.sir_robot.drive_stop()

    def upper_arm(self,direction):
        if self.NN_apps.app_name == "TT_DQN" and self.gather_data.is_on():
            pass
        elif self.NN_apps.app_name in ["TT_NN", "TT_FUNC"] and self.get_NN_mode() == "NN":
            pass
        elif direction == "STOP":
          self.gather_data.set_function(None)
          self.sir_robot.upper_arm(direction)
        else:
          self.sir_robot.upper_arm(direction)
          self.gather_data.set_function("UPPER_ARM_" + direction)

    def lower_arm(self,direction):
        if self.NN_apps.app_name == "TT_DQN" and self.gather_data.is_on():
            pass
        elif self.NN_apps.app_name in ["TT_NN", "TT_FUNC"] and self.get_NN_mode() == "NN":
            pass
        elif direction == "STOP":
          self.gather_data.set_function(None)
          self.sir_robot.lower_arm(direction)
        else:
          self.gather_data.set_function("LOWER_ARM_" + direction)
          self.sir_robot.lower_arm(direction)

    def wrist(self,direction):
        self.sir_robot.wrist(direction)

    def gripper(self,direction):
        if self.NN_apps.app_name == "TT_DQN" and self.gather_data.is_on():
            pass
        elif self.NN_apps.app_name in ["TT_NN", "TT_FUNC"] and self.get_NN_mode() == "NN":
            pass
        elif direction == "STOP":
          self.gather_data.set_function(None)
          self.sir_robot.gripper(direction)
        else:
          self.gather_data.set_function("GRIPPER_" + direction)
          self.sir_robot.gripper(direction)

    def reward(self):
        self.gather_data.set_function("REWARD")
        self.sir_robot.stop_all()

    def penalty(self):
        self.gather_data.set_function("PENALTY")
        self.sir_robot.stop_all()

    # gather data / teleop mode
    def set_gather_data_mode(self, mode=True): 
        self.gather_data.turn_on(mode)

    def get_gather_data_mode(self): 
        return self.gather_data.is_on()

    # run pytorch NN to determine next move
    def set_NN_mode(self, mode): 
        self.NN_apps.set_nn_mode(mode)
        if mode == "DQN" and self.DQN == None:
          self.DQN = SIR_DDQN(self.initialize, self.train_new_data)

    def robot_off_table_penalty(self):
        self.gather_data.set_function("ROBOT_OFF_TABLE_PENALTY")
        self.sir_robot.stop_all()

    def cube_off_table_reward(self):
        self.gather_data.set_function("CUBE_OFF_TABLE_REWARD")
        self.sir_robot.stop_all()

    def get_NN_mode(self):
        return self.NN_apps.get_nn_mode()

    # webcam integration
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
