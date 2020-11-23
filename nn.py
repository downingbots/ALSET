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
    def __init__(self,sir_robot,outputs):
        # self.mode = False
        # self.ready_for_frame = False
        # self.frame = None
        self.robot = sir_robot
        self.device = None
        self.model = None
        self.outputs = outputs
        self.num_outputs = len(outputs)

        # mean = 255.0 * np.array([0.485, 0.456, 0.406])
        # stdev = 255.0 * np.array([0.229, 0.224, 0.225])
        # normalize = torchvision.transforms.Normalize(mean, stdev)

    def nn_init(self, app_name, NN_num, gather_mode=False):
        # if gather_mode:
        if True:
            self.model = torchvision.models.alexnet(pretrained=True)
            self.model.classifier[6] = torch.nn.Linear(self.model.classifier[6].in_features, self.num_outputs)
            print(NN_num, str(NN_num))
            model_path = "apps/" + app_name + "/best_model" + str(NN_num) + ".pth"
            try:
              self.model.load_state_dict(torch.load(model_path))
            except:
              torch.save(self.model.state_dict(), model_path)
            self.device = torch.device('cuda')
            self.model = self.model.to(self.device)
            self.full_action_set = ["UPPER_ARM_UP", "UPPER_ARM_DOWN", 
                "LOWER_ARM_UP", "LOWER_ARM_DOWN", 
                "GRIPPER_OPEN", "GRIPPER_CLOSE",
                "FORWARD", "REVERSE", "LEFT", "RIGHT", "SUCCESS", "FAILURE"]
        # Class can be used as a single-NN app that can do any action
        return False, self.full_action_set

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

    def nn_process_image(self, NN_num = 0, image=None):
        # if image == None:
        if image is None:
          return
        x = image
        x = self.preprocess(x)
        y = self.model(x)
    
        # we apply the `softmax` function to normalize the output vector 
        # so it sums to 1 (which makes it a probability distribution)
        y = F.softmax(y, dim=1)
    
        print(y.flatten) 
        max_prob = 0
        best_action = -1
        for i in range(self.num_outputs):
            prob = float(y.flatten()[i])
            print("PROB", i, prob)
            if max_prob < prob:
                max_prob = prob
                for j, name in enumerate(self.full_action_set):
                    if name == self.outputs[i]:
                        best_action = j
                        break
                if best_action == -1:
                    print("invalid action " + self.outputs[i] + "not in " + self.full_action_set)
                    exit()
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


