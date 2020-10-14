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
    def __init__(self,sir_robot,num_outputs):
        self.mode = False
        self.ready_for_frame = False
        self.frame = None
        self.robot = sir_robot

        # mean = 255.0 * np.array([0.485, 0.456, 0.406])
        # stdev = 255.0 * np.array([0.229, 0.224, 0.225])
        # normalize = torchvision.transforms.Normalize(mean, stdev)

    def nn_init(self, NN_num, gather_mode=False):
        if gather_mode:
            self.model = torchvision.models.alexnet(pretrained=True)
            self.model.classifier[6] = torch.nn.Linear(self.model.classifier[6].in_features, num_outputs)
            self.model.load_state_dict(torch.load('best_model.pth'))
            self.device = torch.device('cuda')
            self.model = self.model.to(self.device)
        # Class can be used as a single-NN app that can do any action
        return False, ["UPPER_ARM_UP", "UPPER_ARM_DOWN", "LOWER_ARM_UP",
                "LOWER_ARM_DOWN", "GRIPPER_OPEN", "GRIPPER_CLOSE",
                "FORWARD", "REVERSE", "LEFT", "RIGHT", "SUCCESS", "FAILURE"]

    def nn_process_image(self, NN_num = 0, image=None):
        if image == None:
          return
        x = image

        x = self.frame
        x = self.preprocess(x)
        y = self.model(x)
    
        # we apply the `softmax` function to normalize the output vector 
        # so it sums to 1 (which makes it a probability distribution)
        y = F.softmax(y, dim=1)
    
        print(y.flatten) 
        max_prob = 0
        best_action = -1
        for i in range(0, 11):
            prob = float(y.flatten()[i])
            print("PROB", i, prob)
            if max_prob < prob:
                max_prob = prob
                best_action = i
        if best_action == 0:
            print("NN FORWARD") 
            self.robot.forward()
        elif best_action == 1:
            print("NN GRIPPER_CLOSE") 
            self.robot.gripper("CLOSE")
        elif best_action == 2:
            print("NN GRIPPER_OPEN") 
            self.robot.gripper("OPEN")
        elif best_action == 3:
            print("NN LEFT") 
            self.robot.left()
        elif best_action == 4:
            print("NN LOWER ARM DOWN") 
            self.robot.lower_arm("DOWN")
        elif best_action == 5:
            print("NN LOWER ARM UP") 
            self.robot.lower_arm("UP")
        elif best_action == 6:
            print("NN PENALTY") 
            self.NN.failure()
        elif best_action == 7:
            print("NN REVERSE") 
            self.robot.reverse()
        elif best_action == 8:
            print("NN PENALTY") 
            self.NN.success()
        elif best_action == 9:
            print("NN RIGHT") 
            self.robot.right()
            # self.robot.left() # for simplified tabletop NN 
        elif best_action == 10:
            print("NN LOWER ARM DOWN") 
            self.robot.lower_arm("DOWN")
        elif best_action == 11:
            print("NN LOWER ARM UP") 
            self.robot.lower_arm("UP")
        else:
            print("Action:", best_action, max_prob) 
            time.sleep(0.08)
            self.robot.stop()
        
      
    def nn_set_automatic_mode(self, TF):
        pass

    def nn_automatic_mode(self):
        return False

    def nn_automatic_action(self, NN_num, feedback):
        pass

    def nn_before_action_callback(self, NN_num, feedback):
        return None

    def nn_upon_penalty(self, NN_num):
        exit()

    def nn_upon_reward(self, NN_num):
        exit()


