import torch
import os
import time
import torchvision
import cv2
import numpy as np
from .robot import *
import torch.nn.functional as F
import time

class SIRNN():
    def __init__(self,sir_robot):
        self.model = torchvision.models.alexnet(pretrained=False)
        self.model.classifier[6] = torch.nn.Linear(self.model.classifier[6].in_features, 10)
        self.model.load_state_dict(torch.load('best_model.pth'))

        self.device = torch.device('cuda')
        self.model = self.model.to(self.device)
        self.mode = False
        self.ready_for_frame = False
        self.frame = None
        self.robot = sir_robot

        # mean = 255.0 * np.array([0.485, 0.456, 0.406])
        # stdev = 255.0 * np.array([0.229, 0.224, 0.225])
        # normalize = torchvision.transforms.Normalize(mean, stdev)

    def preprocess(self, camera_value):
#        return camera_value

         global device, normalize
         x = camera_value
         x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
         x = x.transpose((2, 0, 1))
         x = torch.from_numpy(x).float()
#        # x = normalize(x)
         x = x.to(self.device)
         x = x[None, ...]
         return x

    def is_on(self):
        return self.mode

    def turn_on(self, mode):
        self.mode = mode

    def wait_for_capture(self):
        self.ready_for_frame = True
        i = 0
        while self.ready_for_frame:
            i = i+1
            time.sleep(.01)
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
        try:
          if func != None:
            func_dir = self.robot_dir[func]
          self.function_name = func
        except:
            print("bad function name: %s" % func)

        self.frame = camera_frame

    def process_image(self):
        x = self.frame
        x = self.preprocess(x)
        y = self.model(x)
    
        # we apply the `softmax` function to normalize the output vector 
        # so it sums to 1 (which makes it a probability distribution)
        y = F.softmax(y, dim=1)
    
        print(y.flatten) 
        max_prob = 0
        best_action = -1
        for i in range(10):
            prob = float(y.flatten()[i])
            print("PROB", i, prob)
            if max_prob < prob:
                max_prob = prob
                best_action = i
        if best_action == 7:
            print("NN FORWARD") 
            self.robot.forward()
        elif best_action == 3:
            print("NN LEFT") 
            self.robot.left()
        elif best_action == 0:
            print("NN RIGHT") 
            self.robot.right()
        else:
            print("Action:", best_action, max_prob) 
            # unused by Tabletap:
            # robot.backward()
            # robot.backward()
            # robot.upper_arm_up()
            # robot.upper_arm_down()
            # robot.lower_arm_up()
            # robot.lower_arm_down()
            # robot.gripper(self,direction)

            time.sleep(0.08)
            self.robot.stop()
        
