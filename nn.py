import torch
import os
import time
import torchvision
import cv2
import numpy as np
from .robot import *
from .image_folder2 import *
import torch.nn.functional as F
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms


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
        self.robot_actions = ["UPPER_ARM_UP", "UPPER_ARM_DOWN", "LOWER_ARM_UP", "LOWER_ARM_DOWN", 
                "GRIPPER_OPEN", "GRIPPER_CLOSE", "FORWARD", "REVERSE", "LEFT", "RIGHT"]
        self.joystick_actions = ["REWARD","PENALTY"]
        self.full_action_set = self.robot_actions + self.joystick_actions

        # mean = 255.0 * np.array([0.485, 0.456, 0.406])
        # stdev = 255.0 * np.array([0.229, 0.224, 0.225])
        # normalize = torchvision.transforms.Normalize(mean, stdev)

    def nn_init(self, app_name, NN_num, gather_mode=False):
        # if gather_mode:
        self.model = None
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

    def nn_process_image(self, NN_num = 0, image=None, reward_penalty=None):
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

        # self.robot_actions = { "FORWARD", "REVERSE", "LEFT", "RIGHT", "LOWER_ARM_DOWN", "LOWER_ARM_UP", "UPPER_ARM_DOWN", "UPPER_ARM_UP", "GRIPPER_OPEN", "GRIPPER_CLOSE"}
        action_name = self.full_action_set[best_action]
        if action_name == "FORWARD":
            print("NN FORWARD") 
            self.robot.forward()
        elif action_name == "REVERSE":
            print("NN REVERSE") 
            self.robot.reverse()
        elif action_name == "LEFT":
            print("NN LEFT") 
            self.robot.left()
        elif action_name == "RIGHT":
            print("NN RIGHT") 
            self.robot.right()
            # self.robot.left() # for simplified tabletop NN 
        elif action_name == "LOWER_ARM_DOWN":
            print("NN LOWER ARM DOWN") 
            self.robot.lower_arm("DOWN")
        elif action_name == "LOWER_ARM_UP":
            print("NN LOWER ARM UP") 
            self.robot.lower_arm("UP")
        elif action_name == "UPPER_ARM_DOWN":
            print("NN UPPER ARM DOWN") 
            self.robot.upper_arm("DOWN")
        elif action_name == "UPPER_ARM_UP":
            print("NN UPPER ARM UP") 
            self.robot.upper_arm("UP")
        elif action_name == "GRIPPER_CLOSE":
            print("NN GRIPPER_CLOSE") 
            self.robot.gripper("CLOSE")
        elif action_name == "GRIPPER_OPEN":
            print("NN GRIPPER_OPEN") 
            self.robot.gripper("OPEN")
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

    # train the NN to do imitation learning starting from pre-trained imagenet
    #
    # Train a single NN based upon all 8 of the functional NN from TT_FUNC.
    # Handle new data gathered for TT_FUNC.
    # Use a split of the data so that we know the test accuracy.
    # This test accuracy might help with determining whether there is enough
    # data for TT_DQN.
    def train(self, dataset_root_list=None, best_model_path=None, full_action_set=None, noop_remap=None):
        # nn.py is a primitive NN that can be used by higher layers like tabletop_func_app.
        # Set default parameters if unset by caller
        if dataset_root_list is None:
          TTFUNC_NUM_NN = 8
          dataset_root_list = []
          for nn_num in range(1, TTFUNC_NUM_NN+1):
            ds = "./apps/TT_FUNC/dataset/NN" + str(nn_num)
            dataset_root_list.append(ds)
        if best_model_path is None:
           best_model_path = "./apps/TT_NN/TTNN_model1.pth"
        if full_action_set is None:
           full_action_set = self.full_action_set
        if noop_remap is None:
           pass

        model = models.alexnet(pretrained=True)
        # Dataset transforms
        #    
        # Problems to deal with:
        # - The robot actions must be consistent for all apps.
        # - automatic mode: Determine which images are NOOP operations and
        #   train accordingly.  Automatic mode has only REWARD/PENALTY/NOOP operations.
        #   These need to be mapped from the underlying operations: LEFT, UPPER_ARM_{UP/DOWN}
        # - DQN has 2 additional joystick modes: CUBE_OFF_TABLE_REWARD, ROBOT_OFF_TABLE_PENALTY
        #
        # nn_init needs to store info about additional mappings:
        #  
        # self.robot_actions = ( "FORWARD", "REVERSE", "LEFT", "RIGHT", "LOWER_ARM_DOWN", "LOWER_ARM_UP", "UPPER_ARM_DOWN", "UPPER_ARM_UP", "GRIPPER_OPEN", "GRIPPER_CLOSE")
        # self.joystick_actions = ("REWARD","PENALTY","CUBE_OFF_TABLE_REWARD","ROBOT_OFF_TABLE_PENALTY")
        # self.automatic_mode() returns True/False. If True:
        #   self.automatic_actions = ( "NOOP", "REWARD", "PENALTY")
        #   self.automatic_mode_noop_mapping = ("LEFT", "LOWER_ARM_UP", "LOWER_ARM_DOWN")
        #
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, len(full_action_set))
        device = torch.device('cuda')
        model = model.to(device)
        NUM_EPOCHS = 30
        best_accuracy = 0.0
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        for root in dataset_root_list:
            print("nn train: ", root, best_model_path) 
            # dataset = datasets.ImageFolder2(
            dataset = ImageFolder2(
                root,
                transforms.Compose([
                    transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                full_action_set = full_action_set,
                remap_to_noop = noop_remap
            )
            # Attributes:
            # classes (list): List of the class names sorted alphabetically.
            # class_to_idx (dict): Dict with items (class_name, class_index).
            # imgs (list): List of (image path, class_index) tuples

            # apply transforms
            # full_action_set => revises classes, class_to_idx
            # noop_remap      => revises imgs' class_index
            # 
            # classes = [d.name for d in os.scandir(dir) if d.is_dir()]
            # classes.sort()
            # class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
            # imgs= [image_path, class_index:


            train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - 50, 50])
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=8,
                shuffle=True,
                num_workers=0
            )
            
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=8,
                shuffle=True,
                num_workers=0
            )
            
            for epoch in range(NUM_EPOCHS):
                
                for images, labels in iter(train_loader):
                    images = images.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = F.cross_entropy(outputs, labels)
                    loss.backward()
                    optimizer.step()
                
                # we may not want to do test counts. Take all the data and
                # apply it here.
                test_error_count = 0.0
                for images, labels in iter(test_loader):
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    test_error_count += float(torch.sum(torch.abs(labels - outputs.argmax(1))))
                
                test_accuracy = 1.0 - float(test_error_count) / float(len(test_dataset))
                print('%d: %f' % (epoch, test_accuracy))
                if test_accuracy > best_accuracy:
                    torch.save(model.state_dict(), best_model_path)
                    best_accuracy = test_accuracy
