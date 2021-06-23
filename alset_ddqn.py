# https://github.com/higgsfield/RL-Adventure/blob/master/2.double%20dqn.ipynb

import math, random
# from builtins import bytes
import codecs
from PIL import Image
import cv2

# import gym
import numpy as np
import collections, itertools
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
# import torch.ops
# import torch.ops.image
# from torch import read_file, decode_image

import pickle
from dataset_utils import *
from functional_app import *

# import alset_image 

# from IPython.display import clear_output
# import matplotlib.pyplot as plt
# %matplotlib inline


from collections import deque

# workaround due to ReplayBuffer pickling/unpickling and class evolution
# if sample_start is being used: first call reset() to get buf_id before using
#     buf_id = self.reset_sample(name,start)
#     static_vars.sample_start[buf_id]
def static_vars():
    static_vars.name = []
    static_vars.sample_start = []
static_vars.name = []
static_vars.sample_start = []

# class ReplayBuffer(object):
class ReplayBuffer():
    def __init__(self, capacity, name="replay"):
        self.buffer = deque(maxlen=capacity)
        static_vars.name.append(name)
        static_vars.sample_start.append(0)
    
    def push(self, state, action, reward, next_state, done, q_val):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
            
        self.buffer.append((state, action, reward, next_state, done, q_val))
    
    def random_sample(self, batch_size):
        state, action, reward, next_state, done, q_val = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done, q_val

    def find_done(self, start=0):
        len_buf = len(self.buffer)
        # random_start = random.randint(0,(len_buf-1))
        for i in range(start, len_buf):
           state, action, reward, next_state, done, q_val = zip(*itertools.islice(self.buffer,i,i+1))
           if done[0]:
             # print("find_done:", i, done[0])
             return(i+1)
        return None

    def reset_sample(self, name="replay", start=0):
        for i, nm in enumerate(static_vars.name):
            if name==nm:
               if start is not None:
                  static_vars.sample_start[i] = start
               return i
        print("reset_sample ", name, " returns None buf_id")
        return None

    def get_next_sample(self, batch_size, name="replay", start=None):
        buf_id = self.reset_sample(name,start)
        len_buf = len(self.buffer)
        done_end = None
        done_end = self.find_done(static_vars.sample_start[buf_id])
        if done_end == None:
          static_vars.sample_start[buf_id] = 0
          return None, None, None, None, None, None
        elif done_end - static_vars.sample_start[buf_id] > batch_size:
          done_end = static_vars.sample_start[buf_id] + batch_size 
        # print("gns:", static_vars.sample_start, done_end, len_buf)
        state, action, reward, next_state, done, q_val = zip(*itertools.islice(self.buffer,static_vars.sample_start[buf_id], done_end))
        static_vars.sample_start[buf_id] = done_end
        return np.concatenate(state), action, reward, np.concatenate(next_state), done, q_val

    def clear(self):
        self.buffer.clear()

    def concat(self, replay_buffer):
        # get the last entry and confirm it contains a full fun (done == True)
        done_end = replay_buffer.find_done()
        len_buf = len(replay_buffer)
        # state, action, reward, next_state, done, q_val = zip(replay_buffer.buffer[-1])
        assert done_end == len_buf, "last entry must be completion of run"
        self.buffer += replay_buffer.buffer
        replay_buffer.clear()

    def compute_real_q_values(self, gamma=.99, name="replay", sample_start=0, done_end=None):
        buf_id = self.reset_sample(name,sample_start)
        if done_end is None:
          done_end = len(self.buffer)
        try:
          # NOTE: the state/next_state images are already in memory and not <next_>state_path
          state, action, reward, next_state, done, dummy_q_val = zip(*itertools.islice(self.buffer,sample_start,done_end))
          add_q_val = False
        except:
          state, action, reward, next_state, done = zip(*itertools.islice(self.buffer,sample_start,done_end))
          print("No qval in replay buffer. Adding qval.")
          add_q_val = True
        if not done[done_end-sample_start-1]:
            # print("[state, action, reward, next_state, done]:")
            # print([state, action, reward, next_state, done])
            print("compute_real_q_values: last DONE must be True")
        assert done[done_end-sample_start-1], "Only compute real q values upon completion of run"
        next_q_val = 0
        q_val = 0
        len_reward = len(reward)
        converted = False

        #################
        # ARD: TODO
        #  Following code was prototyped to look into a bug, but it doesn't accurately represent
        #  the Bellman Equation.
        #
        #  Step1: Find current rewards at each step -> sum the rewards from beginning to now.
#        current_reward = 0
#        if sample_start != 0:
#            print("ERROR: sample start not zero:", sample_start)
#        for i,reward_val in enumerate(reward):
#            current_reward += reward_val
#            lst = list(self.buffer[sample_start+offset])
#            lst[5] = current_reward
#            self.buffer[sample_start+offset] = tuple(lst)
#
#
#
#        #  Step2: Find future rewards. Discount for each step backwards
#        discounted_future_reward = 0
#        for i,reward_val in enumerate(reversed(reward)):
#            d = len_reward - i - 1
#            if done[d]:
#              next_q_val = 0
#            else:
#              next_q_val = q_val # "next" because iterating in reverse order
#
#            q_val = reward_val + gamma * next_q_val
#            offset = done_end - sample_start - i - 1
#            lst = list(self.buffer[sample_start+offset])
#            lst[5] += q_val
#            self.buffer[sample_start+offset] = tuple(lst)


        #################

        #  This potential reward is a weighted sum of the expected values of 
        #  the rewards of all future steps starting from the current state.
        q_val_lst = []
        for i,reward_val in enumerate(reversed(reward)):
            d = len_reward - i - 1
            next_q_val = q_val # "next" because iterating in reverse order
            # if done[d]:
            #   next_q_val = 0
            # else:
            #   next_q_val = q_val # "next" because iterating in reverse order

            q_val = reward_val + gamma * next_q_val
            q_val_lst.append(q_val)
            offset = done_end - sample_start - i - 1
            if add_q_val:
              assert len(self.buffer[sample_start+offset])==5,"Wrong number of entries in replay buffer"
              lst = list(self.buffer[sample_start+offset])
              lst.append(q_val)
              self.buffer[sample_start+offset] = tuple(lst)
              assert len((self.buffer[sample_start+offset]))==6,"Wrong # of entries in replay buffer"
              converted = True
            else:
              lst = list(self.buffer[sample_start+offset])
              lst[5] = q_val
              self.buffer[sample_start+offset] = tuple(lst)
        if converted:
            print("Added q_val to replay_buffer")
        print("qval:", q_val_lst)
        print("action:", action)
        print("reward:", reward)

    def __len__(self):
        return len(self.buffer)

    def entry_len(self):
        if len(self.buffer) > 0:
          return len(self.buffer[0])
        return None

##############
# Initially defined for atari env
##############
import torchvision 
import torchvision.io
# import torchvision.io.image
# from torchvision.io.image import read_image
# from torchvision.io import read_image
# import image
# from .alset_image import read_image
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms


# class CnnDQN(nn.Module):
class CnnDQN():
    # def __init__(self, input_shape, num_actions):
    def __init__(self, num_actions, learning_rate):
        # super(CnnDQN, self).__init__()

        # Finetuning vs. feature extraction/transfer learning:
        # In finetuning, we start with a pretrained model and update all of the models parameters
        # for our new task, in essence retraining the whole model. In feature extraction, we start
        # with a pretrained model and only update the final layer weights from which we derive
        # predictions. It is called feature extraction because we use the pretrained CNN as a
        # fixed feature-extractor, and only change the output layer.
        #
        # Jetbot tutorials use Finetuning.
        #
        # In our case, the new dataset is small.  Initially use feature extraction/transfer 
        # learning.  Later you can finetune some of the convolutional stage:
        #  - choose the characteristics of the last layer of the convolutional stage 
        #    and use an SVM or linear classifier... or...
        #  - use features of an earlier layer of the convolutional stage,
        #    as this will be set in more general patterns than the later layers, 
        #    and then use a linear classifier.
        #
        # AlexNet(
        #   (features): Sequential(
        #     (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        #     (1): ReLU(inplace)
        #     (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        #     (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #     (4): ReLU(inplace)
        #     (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        #     (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     (7): ReLU(inplace)
        #     (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     (9): ReLU(inplace)
        #     (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     (11): ReLU(inplace)
        #     (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        #   )
        #   (classifier): Sequential(
        #     (0): Dropout(p=0.5)
        #     (1): Linear(in_features=9216, out_features=4096, bias=True)
        #     (2): ReLU(inplace)
        #     (3): Dropout(p=0.5)
        #     (4): Linear(in_features=4096, out_features=4096, bias=True)
        #     (5): ReLU(inplace)
        #     (6): Linear(in_features=4096, out_features=1000, bias=True)
        #   )
        # )
        # 
        # Replace original Alexnet classifier:
        # (classifier): Sequential(
        #               ...
        #          (6): Linear(in_features=4096, out_features=1000, bias=True)
        #               )
        #############
        # SJB model
        # Start with pretrained Alexnet. If too big for jetbot, use squeezenet.
        #############
        # https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
        feature_extract = True
        # feature_extract = False
        self.alexnet_model = models.alexnet(pretrained=True)
        self.set_parameter_requires_grad(self.alexnet_model, feature_extract)
        # num_features should be 4096
        num_features = self.alexnet_model.classifier[6].in_features
        self.alexnet_model.classifier[6] = torch.nn.Linear(num_features, num_actions)
        # device = torch.device('cuda')
        # self.alexnet_model = self.alexnet_model.to(device)

        # self.input_shape = input_shape
        input_size = 224
        self.num_actions = num_actions

        # From jetbot:
        # dataset = datasets.ImageFolder(
        #    'resources/dataset',
        #        #    transforms.Compose([
        #        transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        #        transforms.Resize((224, 224)),
        #        transforms.ToTensor(),
        #        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #    ])
        # )
        # train_loader = torch.utils.data.DataLoader(
        #     train_dataset,
        #     batch_size=16,
        #     # shuffle=True,
        #     shuffle=False,
        #     num_workers=4
        # )

        # Data augmentation and normalization for training
        # Just normalization for validation
        data_transforms = {
                'train': transforms.Compose([
                    transforms.RandomResizedCrop(input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                # ARD: we won't be doing validation as we're using RL, not pure imitation learning 
                'val': transforms.Compose([
                    transforms.Resize(input_size),
                    transforms.CenterCrop(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
        }
            
        #######################
        # For Imitation Learning (not needed for RL):
        # print("Initializing Datasets and Dataloaders...")
        # Create training and validation datasets
        # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
        # Create training and validation dataloaders
        # dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
        #######################


        #######################
        # Create Optimizer
        #######################
        # Detect if we have a GPU available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Send the model to GPU
        self.alexnet_model = self.alexnet_model.to(device)

        # Gather the parameters to be optimized/updated in this run. If we are
        #  finetuning we will be updating all parameters. However, if we are
        #  doing feature extract method, we will only update the parameters
        #  that we have just initialized, i.e. the parameters with requires_grad
        #  is True.
        params_to_update = self.alexnet_model.parameters()
        print("Params to learn:")
        if feature_extract:
            params_to_update = []
            for name,param in self.alexnet_model.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t",name)
        else:
            for name,param in self.alexnet_model.named_parameters():
                if param.requires_grad == True:
                    print("\t",name)

        # Observe that all parameters are being optimized
        # self.alexnet_optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
        self.alexnet_optimizer = optim.SGD(params_to_update, lr=learning_rate, momentum=0.9)
        # Setup the loss fxn
        criterion = nn.CrossEntropyLoss()

        #######################
        # Train and evaluate (imitation mode only)
        # self.alexnet_model, hist = train_model(self.alexnet_model, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))
        #######################

#######################
# For Atari RL:
#        self.features = nn.Sequential(
#            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
#            nn.ReLU(),
#            nn.Conv2d(32, 64, kernel_size=4, stride=2),
#            nn.ReLU(),
#            nn.Conv2d(64, 64, kernel_size=3, stride=1),
#            nn.ReLU()
#        )
#        self.fc = nn.Sequential(
#            nn.Linear(self.feature_size(), 512),
#            nn.ReLU(),
#            nn.Linear(512, self.num_actions)
#        )


    # if we are feature extracting and only want to compute gradients for the newly 
    # initialized layer then we want all of the other parameters to not require gradients
    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def train(self):
        self.alexnet_model.train()
    def eval(self):
        self.alexnet_model.eval()

    # this trains the Functional NN -> output is the action to perform
    def train_Functional_NN(self, dataloaders, criterion, optimizer):
        since = time.time()
    
        val_acc_history = []
    
        best_model_wts = copy.deepcopy(self.alexnet_model.state_dict())
        best_acc = 0.0
    
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
    
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.alexnet_model.train()  # Set model to training mode
                else:
                    self.alexnet_model.eval()   # Set model to evaluate mode
    
                running_loss = 0.0
                running_corrects = 0
    
                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
    
                    # zero the parameter gradients
                    optimizer.zero_grad()
    
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
    
                        _, preds = torch.max(outputs, 1)
    
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
    
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
    
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
    
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
    
                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.alexnet_model.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc)
    
            print()
    
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
    
        # load best model weights
        self.alexnet_model.load_state_dict(best_model_wts)
        return self.alexnet_model, val_acc_history

    def state_dict(self):
        return self.alexnet_model.state_dict()

    def load_state_dict(self, dict):
        self.alexnet_model.load_state_dict(dict)

    # function for visualizing the feature maps
    def visualize_activation_maps(input, model):
        I = utils.make_grid(input, nrow=1, normalize=True, scale_each=True)
        img = I.permute((1, 2, 0)).cpu().numpy()
    
        conv_results = []
        x = input
        for idx, operation in enumerate(self.current_model.features):
            x = operation(x)
            if idx in {1, 4, 7, 9, 11}:
                conv_results.append(x)
        
        for i in range(5):
            conv_result = conv_results[i]
            N, C, H, W = conv_result.size()
    
            mean_acti_map = torch.mean(conv_result, 1, True)
            mean_acti_map = F.interpolate(mean_acti_map, size=[224,224], mode='bilinear', align_corners=False)
    
            map_grid = utils.make_grid(mean_acti_map, nrow=1, normalize=True, scale_each=True)
            map_grid = map_grid.permute((1, 2, 0)).mul(255).byte().cpu().numpy()
            map_grid = cv2.applyColorMap(map_grid, cv2.COLORMAP_JET)
            map_grid = cv2.cvtColor(map_grid, cv2.COLOR_BGR2RGB)
            map_grid = np.float32(map_grid) / 255
    
            visual_acti_map = 0.6 * img + 0.4 * map_grid
            tensor_visual_acti_map = torch.from_numpy(visual_acti_map).permute(2, 0, 1)
    
            file_name_visual_acti_map = 'conv{}_activation_map.jpg'.format(i+1)
            utils.save_image(tensor_visual_acti_map, file_name_visual_acti_map)
    
        return 0

    def save_state(self, path):
            torch.save(self.alexnet_model.state_dict(), path)

#    # Returns q_value
#    def forward(self, x):
#        x = self.features(x)
#        x = x.view(x.size(0), -1)
#        x = self.fc(x)
#        return x
#    
#    def feature_size(self):
#        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def act(self, state):
        # make an image of size 1
        # batch_img = torch.unsqueeze(transformed_img, 0)
        # print("image batch's shape: " + str(batch_img.shape))

        # put the model to eval mode for testing
        self.alexnet_model.eval()
        # obtain the output of the model
        # RuntimeError: Expected object of device type cuda but got device type cpu for argument #1 'self' in call to _thnn_conv2d_forward

        q_value = self.alexnet_model(state)
        # q_value = self.current_model(state)
        # print("output vector's shape: " + str(q_value.shape))

        # obtain the activation maps
        # visualize_activation_maps(batch_img, self.alexnet_model)
        # sorted, indices = torch.sort(q_value, descending=True)
        # percentage = F.softmax(output, dim=1)[0] * 100.0


        ###############################
        # 
        #               outputs = model(inputs)
        #               loss = criterion(outputs, labels)
        #           _, preds = torch.max(outputs, 1)
        sortact  = q_value.sort(dim=1, descending=True)[1].data[0]
        action  = q_value.max(1)[1].data[0]
        return sortact, q_value


##################
# ALSET_Jetbot (SJB)
##################

# Original dataset for TT_FUNC:
# PH1 - nothing in gripper, no cube/box in vision
#  NN1. Park Arm
#  NN2. Automatic scan for cube
#       add in side avoidance?
# PH2 - nothing in gripper, cube in vision
#  NN3. Approach cube
#  NN4. Pick up cube
#  NN6. Automatic scan for box (with cube in grippers)
#       add in side avoidance? Need to do with cube in gripper?
# PH4 - cube in gripper, and box in vision
#  NN7. Approach box (with cube in grippers)
# PH5 - cube in box, and cube and box in vision
#  NN8. Drop cube in box and back away
# 
import os
import os.path as osp
import importlib.machinery
from config import *
from dataset_utils import *

# class ALSET_DDQN(nn.Module):
class ALSET_DDQN():

    def __init__(self, alset_robot=None, initialize_model=False, do_train_model=False, app_name=None, app_type=None):
        self.app_name = app_name
        self.app_type = app_type
        self.robot = alset_robot
        self.cfg = Config()
        self.dsu = DatasetUtils(self.app_name, self.app_type)
        self.parse_app_ds_details = []
        if app_type == "DQN" or app_type == "APP":
          self.BEST_MODEL_PATH = self.dsu.best_model(mode="DQN")
          self.DQN_PATH_PREFIX = self.dsu.dataset_path()
          self.REPLAY_BUFFER_PATH = self.dsu.dqn_replay_buffer()
          if app_type == "DQN":
            self.DQN_DS_PATH = self.dsu.dataset_path(mode="DQN")
          else:
            self.DQN_DS_PATH = self.dsu.dataset_path(mode="APP")

          # Reward computation constants from config file attributes
          dqn_registry           = self.cfg.get_value(self.cfg.DQN_registry, self.app_name)
          # print("dqn_registry:", dqn_registry)
          # print("DQN_registry:", self.cfg.DQN_registry)
          # print("app_name:", self.app_name)
          DQN_Policy = dqn_registry[0]
        else:
          self.BEST_MODEL_PATH = self.dsu.best_model(mode="FUNC", nn_name=app_name )
          self.DQN_PATH_PREFIX = self.dsu.dataset_path()
          print("alset_ddqn: ", self.BEST_MODEL_PATH, self.DQN_PATH_PREFIX)
          self.REPLAY_BUFFER_PATH = self.dsu.dqn_replay_buffer()
          self.DQN_DS_PATH = self.dsu.dataset_path(mode="FUNC", nn_name=app_name)
          DQN_Policy = self.cfg.FUNC_policy

        self.REPLAY_INITIAL    = self.cfg.get_value(DQN_Policy, "REPLAY_BUFFER_CAPACITY")
        self.REPLAY_PADDING    = self.cfg.get_value(DQN_Policy, "REPLAY_BUFFER_PADDING")
        self.BATCH_SIZE        = self.cfg.get_value(DQN_Policy, "BATCH_SIZE")
        self.GAMMA             = self.cfg.get_value(DQN_Policy, "GAMMA")
        self.LEARNING_RATE     = self.cfg.get_value(DQN_Policy, "LEARNING_RATE")
        print("lr:",self.LEARNING_RATE)
        self.ERROR_CLIP        = self.cfg.get_value(DQN_Policy, "ERROR_CLIP")
        self.DQN_REWARD_PHASES = self.cfg.get_value(DQN_Policy, "DQN_REWARD_PHASES")
        self.REWARD2_REWARD    = self.cfg.get_value(DQN_Policy, "REWARD2")
        self.PENALTY2_PENALTY  = self.cfg.get_value(DQN_Policy, "PENALTY2")
        self.DQN_MOVE_BONUS    = self.cfg.get_value(DQN_Policy, "DQN_MOVE_BONUS")
        self.PER_MOVE_PENALTY  = self.cfg.get_value(DQN_Policy, "PER_MOVE_PENALTY")
        self.MAX_MOVES         = self.cfg.get_value(DQN_Policy, "MAX_MOVES")
        self.MAX_MOVES_EXCEEDED_PENALTY = self.cfg.get_value(DQN_Policy, "MAX_MOVES_EXCEEDED_PENALTY")
        self.ESTIMATED_VARIANCE         = self.cfg.get_value(DQN_Policy, "ESTIMATED_VARIANCE")
        if self.LEARNING_RATE is None:
          self.LEARNING_RATE = 0.001

        # reward variables
        self.standard_mean      = 0.0
        self.standard_variance  = 1.0
        self.estimated_mean     = 0.0
        self.curr_phase   = 0
        self.max_phase   = len(self.DQN_REWARD_PHASES)
        self.total_reward = 0
        self.frame_num    = 0
        self.action       = None
        self.prev_action  = None
        self.state        = None
        self.prev_state   = None
        self.all_rewards  = []
        self.clip_max_reward  = -1
        self.clip_min_reward  = 1

        ############
        # DDQN variables
        ############
        capac = self.REPLAY_INITIAL + self.REPLAY_PADDING
        self.replay_buffer  = ReplayBuffer(capacity=capac, name="replay")
        # allow for 20 rewards in active buffer
        capac = self.MAX_MOVES + self.REPLAY_PADDING
        self.active_buffer  = ReplayBuffer(capacity=capac, name="active")  
        
        # self.epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)
        
        # plt.plot([self.epsilon_by_frame(i) for i in range(1000000)])
        self.num_frames = self.REPLAY_INITIAL

        self.USE_CUDA = torch.cuda.is_available()
        self.Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if self.USE_CUDA else autograd.Variable(*args, **kwargs)
        
        # self.losses = []

        ############
        # FINE TUNE PRETRAINED MODEL USING IMITATION LEARNING
        # add to history
        
        self.mean_cnt = 0
        self.mean_i = 0
        self.num_actions = len(self.cfg.full_action_set)
        self.current_model = None
        self.target_model = None
        self.init_model = initialize_model
        self.train_model = do_train_model
        self.all_loss = []
        self.all_act_rank = []
        self.all_allowed_act_rank = []
        print("DQN initialization: ",self.init_model, self.train_model)
        
    def nn_init(self, gather_mode=False):
        # TODO: allow classification models
        print("lr:",self.LEARNING_RATE)
        self.current_model = CnnDQN(self.num_actions, self.LEARNING_RATE)
        self.target_model = CnnDQN(self.num_actions, self.LEARNING_RATE)

        if (self.init_model):  # probably done on laptop
            # starts from the first dataset
            self.parse_unprocessed_app_datasets(init=True)
            self.save_replay_buffer()
            if (self.train_model):
              self.train_DQN_qvalue()
        else:
            self.load_replay_buffer()
            # get new datasets
            if self.app_type != "FUNC":
              self.parse_unprocessed_app_datasets(init=False)
            self.save_replay_buffer()
            if (self.train_model):
              self.train_DQN_qvalue()
            else:
              self.current_model.load_state_dict(torch.load(self.BEST_MODEL_PATH))

        # self.frame_num = len(self.active_buffer)
        self.frame_num = 0
        # target_model = copy.deepcopy(current_model)
        self.update_target(self.current_model, self.target_model)
        # robot DQN can return self.robot_actions
        # joystick + robot can return self.actions
        return False, self.cfg.full_action_set


    def nn_set_automatic_mode(self, TF):
        pass

    def update_target(self, current_mdl, target_mdl):
        target_mdl.load_state_dict(current_mdl.state_dict())
        # sd = copy.deepcopy(target_mdl.state_dict())
        # model.load_state_dict(sd)
        ## t_state = target_mdl.state_dict()
        ##  t_state.update(current_mdl.state_dict())
    
    def transform_image(self, img, mode='val'):
        # data transforms, for pre-processing the input image before feeding into the net
        # Data augmentation and normalization for training

        # Just normalization for validation
#        input_size = 224
#        data_transforms = {
#            'train': transforms.Compose([
#                transforms.RandomResizedCrop(input_size),
#                transforms.RandomHorizontalFlip(),
#                transforms.ToTensor(),
#                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#            ]),
#            'val': transforms.Compose([
#                transforms.Resize(input_size),
#                transforms.CenterCrop(input_size),
#                transforms.ToTensor(),
#                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#            ]),
#        }
#        transformed_img = {data_transforms[mode](img)}
        data_transforms = transforms.Compose([
            transforms.Resize((224,224)),  # resize to 224x224
            transforms.ToTensor(),         # tensor format
            transforms.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])  ])
        # pre-process the input
        transformed_img = data_transforms(img)
        # transformed_img = transforms(transformed_img.to("cuda")) # done in Variable decl
        return transformed_img

    def transform_image_from_path(self, image_filepath):
        for batch_item_num, path in enumerate(image_filepath):
            img = Image.open(path)
            img = self.transform_image(img)
            # TypeError: float() argument must be a string or a number, not 'set'
            img = self.Variable(torch.FloatTensor(np.float32(img)))
            img = torch.unsqueeze(img, 0)
            if batch_item_num == 0:
                state_image = img
            else:
                state_image = torch.cat((state_image,img),0)
        return state_image

    # import json
    # json.dump(self.replay_buffer, filehandle)
    # json.load(filehandle)
    def save_replay_buffer(self):
        with open(self.REPLAY_BUFFER_PATH, 'wb+') as filehandle:
          # store the data as binary data stream
          pickle.dump(self.replay_buffer, filehandle)
        filehandle.close()

    def load_replay_buffer(self):
        try:
          with open(self.REPLAY_BUFFER_PATH, 'rb') as filehandle:
            self.replay_buffer = pickle.load(filehandle)
          filehandle.close()
        except:
            print("load_replay_buffer: open failed")
        print("loaded replay_buffer. Len = ", self.replay_buffer.entry_len())
        if self.replay_buffer.entry_len() == 5:
            # compatibility with earlier version of buffer format
            # state, action, reward, next_state, done
            print("adding q_values to replay_buffer")
            self.replay_buffer.compute_real_q_values(gamma=self.GAMMA)
            self.save_replay_buffer()

    def train_DQN_qvalue(self):
        loss = 0
        while loss is not None:
          loss = self.compute_td_loss(batch_size = self.BATCH_SIZE, mode = "IMITATION_TRAINING")
        self.current_model.save_state(self.BEST_MODEL_PATH)
        print("Initial model trained by imitation learning")

    def shape(self, tensor):
        # s = tensor.get_shape()
        s = tensor.size()
        print("s: ", s)
        return tuple([s[i].value for i in range(0, len(s))])

    def compute_td_loss(self, batch_size=32, mode="REAL_Q_VALUES"):
        self.current_model.train()  # Set model to training mode
        if mode == "IMITATION_TRAINING":
          # Train based on composite app runs
          state_path, action, rewards, next_state_path, done_val, q_val = self.replay_buffer.get_next_sample(batch_size)
        elif mode == "REAL_Q_VALUES":
          # Train based on runs of DQN datasets
          state_path, action, rewards, next_state_path, done_val, q_val = self.active_buffer.get_next_sample(batch_size=batch_size, name="active")
        elif mode == "RANDOM_FUNCTIONAL_TRAINING":
          # Train based on random runs of func/NN datasets
          self.parse_rand_func_dataset(self, init=False)   # creates a replay buffer
          state_path, action, rewards, next_state_path, done_val, q_val = self.active_buffer.get_next_sample(batch_size=batch_size, name="active")
        elif mode == "EXPERIENCE_REPLAY":
          # Part of DQN algorithm.
          # 
          # The learning phase is then logically separate from gaining experience, and based on 
          # taking random samples from the buffer. 
          #
          # Advantages: More efficient use of previous experience, by learning with it multiple times.
          # This is key when gaining real-world experience is costly, you can get full use of it.
          # The Q-learning updates are incremental and do not converge quickly, so multiple passes 
          # with the same data is beneficial, especially when there is low variance in immediate 
          # outcomes (reward, next state) given the same state, action pair.
          #
          # Disadvantage: It is harder to use multi-step learning algorithms
          state_path, action, rewards, next_state_path, done_val, q_val = self.replay_buffer.random_sample(batch_size)
        else:
          print("Unknown mode for computed td_loss:", mode)
          exit()
        try:
          if state_path is None:
            print("Completed replay buffer training.")
            return None
        except:
          pass


        # 4-D tensors are for batches of images, 3-D tensors for individual images.
        # image_batch is a tensor of the shape (32, 180, 180, 3).  
        # transform state paths to images.  already a Tensor.
        #
        # compute_real_q_values has actual state, not state_path.
        if type(state_path) == np.ndarray and type(state_path[0]) is not np.str_: 
          print("state already converted from path.", type(state_path), type(state_path[0]))
          state = state_path
        else:
          state = self.transform_image_from_path(state_path)
        # next_state   = self.transform_image_from_path(next_state_path)
        # transform action string to number
        action_idx = []
        for a in action:
          # action_idx.append(self.robot_actions.index(a))  # Human readable to integer index
          action_idx.append(self.cfg.full_action_set.index(a))  # Human readable to integer index
        action_index = tuple(action_idx)
        # print("action_idx:",action_idx)
        # print("max action_idx:",len(self.cfg.full_action_set))
        # print("q_val:", q_val)

        action_idx = self.Variable(torch.LongTensor(action_index))
        # reward     = self.Variable(torch.FloatTensor(rewards))
        # done       = self.Variable(torch.FloatTensor(done_val))

        # the real q value is precomputed in q_val
        # real q value computed from done end-pt
        q_val      = self.Variable(torch.FloatTensor(q_val))  

        # Computed q-values from Alexnet
        current_q_values = self.current_model.alexnet_model(state)
        q_value = current_q_values.gather(1, action_idx.unsqueeze(1)).squeeze(1)
        # print("curr_q_val:", current_q_values)
        # print("q_value   :", q_value)

        # determine rankings
        curr_q_vals = current_q_values.detach().cpu().numpy()
        act_rank = []
        allowed_act_rank = []
        if self.cfg.nn_disallowed_actions is None:
          disallowed = []
          feasible_act = list(self.cfg.full_action_set)
        else:
          disallowed = []
          for a in self.cfg.nn_disallowed_actions:
            try:
              disallowed.append(list(self.cfg.full_action_set).index(a))
            except:
              pass
          feasible_act = list(Counter(list(self.cfg.full_action_set)) - Counter(self.cfg.nn_disallowed_actions))

        for i, curr_q_v in enumerate(curr_q_vals):
          act_srt = np.argsort(curr_q_v)
          # print("act_srt1",act_srt, disallowed)
          act_srt = list(Counter(act_srt.tolist()) - Counter(disallowed))
          # print("act_srt2",act_srt)
          # act_srt = list(Counter(act_srt.tolist()) - (Counter(list(self.cfg.full_action_set) - Counter(allowed_actions))))
          # a_r = act_srt.index(action_index[frame_num])
          try:
            a_r = act_srt.index(action_index[i])
            act_rank.append(a_r)
            self.all_act_rank.append(a_r)
          except:
            print("unranked top action:",self.cfg.full_action_set[action_index[i]])
            pass
          # print("ar",a_r)
          for ff_nn, fn, allowed_actions in self.parse_app_ds_details:
            if fn > self.frame_num:
              # print("PARSE_APP_DS: ", ff_nn, fn, self.frame_num, allowed_actions)
              break
          aa_r = []
          for aa in allowed_actions:
            # aa_i = self.cfg.full_action_set.index(aa)  # Human readable to integer index
            try:
              aa_i_s = act_srt.index(aa)
              aa_r.append(aa_i_s)
              allowed_act_rank.append(aa_i_s)
              self.all_allowed_act_rank.append(aa_i_s)
            except:
              # should be a WRIST_ROTATE_LEFT/RIGHT, which is really not an allowed action
              # print("unranked allowed action:",self.cfg.full_action_set[aa])
              pass
          var_aa_r = np.var(aa_r)
          mean_aa_r = np.mean(aa_r)
          # print("action ranking", a_r, mean_aa_r, var_aa_r, len(feasible_act))
          self.frame_num += 1

        mean_act_rank = np.mean(act_rank)
        var_act_rank = np.var(act_rank)
        mean_all_act_rank = np.mean(self.all_act_rank)
        var_all_act_rank = np.var(self.all_act_rank)
        mean_allowed_act_rank = np.mean(allowed_act_rank)
        var_allowed_act_rank = np.var(allowed_act_rank)
        mean_all_allowed_act_rank = np.mean(self.all_allowed_act_rank)
        var_all_allowed_act_rank = np.var(self.all_allowed_act_rank)
        # This should evaluate how good of the last run of the NN was  
        print("FINAL ACTION RANKING:" 
                "mean", mean_all_act_rank, mean_all_allowed_act_rank,
                #       mean_act_rank, mean_allowed_act_rank, 
                "var", var_all_act_rank, var_all_allowed_act_rank,
                #       var_act_rank, var_allowed_act_rank, 
                "numact", len(feasible_act), len(allowed_actions))

        # For DDQN, using target_model to predict q_val (from rladvddqn.py).
        # //github.com/higgsfield/RL-Adventure/blob/master/2.double%20dqn.ipynb
        #
        # current_next_q_values = self.current_model.alexnet_model(next_state)
        # target_next_q_values  = self.target_model.alexnet_model(next_state)
        # target_next_q_state_values = target_model(next_state)
        ## orig:
        ## target_next_q_value = target_next_q_state_values.gather(1, torch.max(target_next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        # target_next_q_value = next_q_state_values.gather(1, next_q_values.max(1)[1].unsqueeze(1)).squeeze(1)

        # target_next_q_value = target_next_q_state_values.gather(1, torch.max(target_next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        # expected_q_value = reward + gamma * target_next_q_value * (1 - done)
        # loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

        # TODO: use the target_model for ddqn per above
        if mode == "IMITATION_TRAINING":
          loss = (q_value - q_val).pow(2).mean()
          print("IMITATION_TRAINING:")
        elif mode == "REAL_Q_VALUES":
          # nondeterministic err: device-side assert triggered: nonzero_finite_vals
          # print("len q_value, q_val:", len(q_value), len(q_val))
          difference = torch.abs(q_value - q_val)
          if self.ERROR_CLIP >= 0:
            quadratic_part = torch.clamp(difference, 0.0, self.ERROR_CLIP)
            linear_part = difference - quadratic_part
            loss = (0.5 * quadratic_part.pow(2) + (self.ERROR_CLIP * linear_part)).mean()
          else:
            loss = (0.5 * difference.pow(2)).mean()
            # loss = (q_value - q_val).pow(2).mean()
          # print("REAL_Q_VALUES:")
          # print("qval diff: ",  difference)
          # print("qval loss: ",  loss)
        elif mode == "RANDOM_FUNCTIONAL_TRAINING":
          loss = (q_value - q_val).pow(2).mean()
          print("RANDOM_FUNC_TRAINING:")
        elif mode == "EXPERIENCE_REPLAY":
          loss = (q_value - q_val).pow(2).mean()
          print("EXPERIENCE_REPLAY FROM TARGET:")

        if loss is not None:
          # Note that following the first .backward call, a second call is only possible after you have performed another forward pass.
          # print("loss training")
          self.current_model.train()  # Set model to training mode
          self.current_model.alexnet_optimizer.zero_grad()
          loss.backward()
          self.current_model.alexnet_optimizer.step()
          # self.current_model.eval()  # Set model to eval mode
          curr_q_vals = current_q_values.detach().cpu().numpy()
          self.all_loss.append(loss.item())
        else:
          print("None loss")
        return loss
    
#    def plot(self, frame_num, rewards, losses):
#        clear_output(True)
#        plt.figure(figsize=(20,5))
#        plt.subplot(131)
#        plt.title('frame %s. reward: %s' % (frame_num, np.mean(rewards[-10:])))
#        plt.plot(rewards)
#        plt.subplot(132)
#        plt.title('loss')
#        plt.plot(losses)
#        plt.show()

    def compute_reward(self, frame_num, action):
                         # phase 0: to first DQN reward; 50 award & 300 allocated moves
                         # phase 1: to second DQN reward; 100 award & 400 allocated moves
                         # more phases allowed
        # ["DQN_REWARD_PHASES", [[50,    300],   [100,   400]]],
        if self.curr_phase < len(self.DQN_REWARD_PHASES):
          PHASE_ALLOCATED_MOVES = self.DQN_REWARD_PHASES[self.curr_phase][1]
          PHASE_REWARD = self.DQN_REWARD_PHASES[self.curr_phase][0]
        else:
          PHASE_ALLOCATED_MOVES = self.DQN_REWARD_PHASES[-1][1]
          PHASE_REWARD = self.DQN_REWARD_PHASES[-1][0]
          print("WARN: curr phase exceeds DQN_REWARD_PHASES.", self.curr_phase, frame_num, action, len(self.DQN_REWARD_PHASES), self.DQN_REWARD_PHASES)
        # print("COMPUTE_REWARD: action, phase, self.DQN_REWARD_PHASES:",  action, frame_num, self.curr_phase, self.DQN_REWARD_PHASES, len(self.DQN_REWARD_PHASES))
        reward = 0
        if frame_num > self.MAX_MOVES:
          return (self.MAX_MOVES_EXCEEDED_PENALTY / self.ESTIMATED_VARIANCE), True
        elif action == "REWARD1":
          done = False
          reward = PHASE_REWARD + max((PHASE_ALLOCATED_MOVES - frame_num),0)*self.DQN_MOVE_BONUS
          self.curr_phase += 1
          print("reward, phase, self.DQN_REWARD_PHASES:",  reward, self.curr_phase, self.DQN_REWARD_PHASES, len(self.DQN_REWARD_PHASES), frame_num)
          # reward, phase, self.DQN_REWARD_PHASES: 52.0 1 [[50, 300], [100, 400]] 2 292
          if self.curr_phase >= len(self.DQN_REWARD_PHASES):
            done = True
          return (reward / self.ESTIMATED_VARIANCE), done
        elif action == "REWARD2":
          return (self.REWARD2_REWARD / self.ESTIMATED_VARIANCE), True
        elif action in ["PENALTY1","PENALTY2"]:
          return (self.PENALTY2_PENALTY / self.ESTIMATED_VARIANCE), True
        elif self.curr_phase < len(self.DQN_REWARD_PHASES):
          return (self.PER_MOVE_PENALTY / self.ESTIMATED_VARIANCE), False
        elif action not in self.cfg.full_action_set:
          print("unknown action: ",  action)
          exit()
    
    #
    #  Can improve scoring over time as collect more output (mean, stddev).
    #    total # moves, # moves to pick up cube, # moves to drop in box,
    #    End game event: too many moves, success, robot off table, cube off table
    #                    auto            success, penalty,         pause (lowerarm  up)
    #                                       left  right            go    (lowerarm down)
    #
    #  Human Labeled events:
    #    pick up cube, drop cube in box, drop cube over edge, off table
    #  Computer labeled events:
    #    each move

    def set_dqn_action(self, action):
        # set by joystick: REWARD1, PENALTY1, REWARD2, PENALTY2
        self.dqn_action = action

    def get_dqn_action(self):
        # set by joystick: REWARD1, PENALTY1, REWARD2, PENALTY2
        return self.dqn_action
    
    # Func: need to factor out common functionality with parse_app_dataset
    #       Currently, just a minor modified clone
    def parse_func_dataset(self, NN_name, init=False, app_mode="FUNC"):
        print(">>>>> parse_func_dataset")
        app_dsu = DatasetUtils(self.app_name, "FUNC")
        if init:
          # start at the beginning
          # e.g., clear TTT_APP_IDX_PROCESSED_BY_DQN.txt
          app_dsu.save_dataset_idx_processed(mode = app_mode, clear = True )

        frame_num = 0
        final_reward_computed = False
        reward = []
        ###################################################
        # iterate through NNs and fill in the active buffer
        while True:
          func_index = app_dsu.dataset_indices(mode=app_mode,nn_name=NN_name,position="NEXT")
          if func_index is None:
            print("parse_func_dataset: done")
            break
          print("Parsing FUNC idx", func_index)
          run_complete = False
          line = None
          next_action = None
          next_line = None
          self.active_buffer.clear()
          self.curr_phase = 0
          nn_filehandle = open(func_index, 'r')
          line = None
          final_reward_computed = False
          while True: # iterate through Img frames in nn
            # read a single line
            next_line = nn_filehandle.readline()
            if not next_line:
                run_complete = True
                print("Function Index completed:", frame_num, NN_name, next_action)
                # Function Index completed: 293 PARK_ARM_RETRACTED_WITH_CUBE REWARD1 PARK_ARM_RETRACTED_WITH_CUBE REWARD1

                if not final_reward_computed:
                  reward, done = self.compute_reward(frame_num, next_action)
                  done = True  # end of func is done in this mode
                  print("completed REWARD phase1", frame_num, next_action, reward, done)
                  self.active_buffer.push(state, action, reward, next_state, done, q_val)
                  final_reward_computed = True
                break
            # get action & next_action
            [tm, app, mode, next_nn_name, next_action, img_name, next_state] = self.dsu.get_dataset_info(next_line, mode="FUNC")
            if line is not None:
              [tm, app, mode, nn_name, action, img_name, state] = self.dsu.get_dataset_info(line, mode="FUNC")
              if action == "NOOP":
                line = next_line
                continue
              elif action == "REWARD1":
                print("Goto next NN; NOOP Reward, curr_NN", action, nn_name)
                line = next_line
                continue
              if next_action == "REWARD1":
                reward, done = self.compute_reward(frame_num, next_action)
                done = True  # end of func is done in this mode
                print("completed REWARD phase2", frame_num, next_action, reward, done)
                final_reward_computed = True
              else:
                # print("compute_reward:", frame_num, action)
                reward, done = self.compute_reward(frame_num, action)
              frame_num += 1
              # add dummy 0 q_val for now. Compute q_val at end of run.
              q_val = 0
              self.active_buffer.push(state, action, reward, next_state, done, q_val)
            if next_action != "REWARD1":
              line = next_line
          # close the pointer to that file
          nn_filehandle.close()
          #################################################
          ## NOT DONE FOR IMMITATION LEARNING
          # if len(self.replay_buffer) > self.REPLAY_INITIAL:
          #   loss = self.compute_td_loss(batch_size, app_path_prefix)
          # if frame_num % 1000 == 0 or done:
          #   self.update_target(self.current_model, self.target_model)
          #################################################
          if run_complete:
              print("SAVING STATE; DO NOT STOP!!!")
              self.active_buffer.compute_real_q_values(gamma=self.GAMMA,  name="active")
              self.active_buffer.reset_sample(name="active", start=0)
              self.frame_num = 0
              for i in range(self.cfg.NUM_EPOCHS):
                loss = 0
                while loss is not None:
                  loss = self.compute_td_loss(batch_size=self.BATCH_SIZE, mode="REAL_Q_VALUES")
                  print("real q values loss: ", i, loss)
              # print("loss: ",loss)
              # print("ACTIVE BUFFER:", self.active_buffer)
              try:
                self.replay_buffer.concat(self.active_buffer)
              except:
                if self.replay_buffer is None:
                    print("self.replay_buffer is None")
                if self.active_buffer is None:
                    print("self.active_buffer is None")
              self.save_replay_buffer()
              print(self.BEST_MODEL_PATH)
              self.current_model.save_state(self.BEST_MODEL_PATH)
              # torch.save(model.state_dict(), self.BEST_MODEL_PATH)
              self.update_target(self.current_model, self.target_model)
              app_dsu.save_dataset_idx_processed(mode = app_mode, nn_name=nn_name)
              print("STATE SAVED")
        return "PROCESSED_FUNC_RUN"

    # for training DQN by processing app dataset (series of functions/NNs)
    def parse_app_dataset(self, init=False, app_mode="APP"):
        print(">>>>> parse_app_dataset")
        app_dsu = DatasetUtils(self.app_name, "APP")
        if init:
          # start at the beginning
          # e.g., clear TTT_APP_IDX_PROCESSED_BY_DQN.txt
          app_dsu.save_dataset_idx_processed(mode = app_mode, clear = True )

        frame_num = 0
        reward = []
        val = self.cfg.get_value(self.cfg.app_registry, self.app_name)
        func_nn_list = val[1]
        func_app = FunctionalApp(alset_robot=self.robot, app_name=self.app_name, app_type=self.app_type)
        func_app.nn_init()
        self.parse_app_ds_details = []

        ###################################################
        # iterate through NNs and fill in the active buffer
        app_index = app_dsu.dataset_indices(mode=app_mode,nn_name=None,position="NEXT")
        if app_index is None:
          print("parse_app_dataset: unknown NEXT index or DONE")
          return None
        print("Parsing APP idx", app_index)
        app_filehandle = open(app_index, 'r')
        run_complete = False
        line = None
        next_action = None
        next_line = None
        self.active_buffer.clear()
        self.curr_phase = 0
        func_flow_reward = None
        first_time_through = True
        self.clip_max_reward  = -1
        self.clip_min_reward  = 1
        while True:  
          # find out what the func flow model expects
          if first_time_through:
            if self.cfg.nn_disallowed_actions is None:
              disallowed = []
              feasible_act = list(self.cfg.full_action_set)
            else:
              disallowed = []
              for a in self.cfg.nn_disallowed_actions:
                try:
                  disallowed.append(list(self.cfg.full_action_set).index(a))  
                except:
                  pass
              feasible_act = list(Counter(list(self.cfg.full_action_set)) - Counter(disallowed))
          else:
            # allowed_actions is different for each nn
            allowed_actions = None
            for [func, func_allowed_actions] in self.cfg.func_movement_restrictions:
              if func_flow_nn_name == func:
                 allowed_actions = []
                 for a in func_allowed_actions:
                   allowed_actions.append(list(self.cfg.full_action_set).index(a))
                 break
            if allowed_actions is None:
              allowed_actions = feasible_act
            self.parse_app_ds_details.append([func_flow_nn_name, frame_num, allowed_actions])

          [func_flow_nn_name, func_flow_reward] = func_app.eval_func_flow_model(reward_penalty="REWARD1", init=first_time_through)
          first_time_through = False
          # get the next function index 
          nn_idx = app_filehandle.readline()
          if not nn_idx:
            if func_flow_nn_name is None:
              if func_flow_reward == "REWARD1":
                if next_action != "REWARD1":
                  print("Soft Error: last action of run is expected to be a reward:", func_flow_action, next_action)
                if func_flow_nn_name is None and func_flow_reward == "REWARD1":
                    # End of Func Flow. Append reward.
                    reward, done = self.compute_reward(frame_num, "REWARD1")
                    if reward > self.clip_max_reward:
                       self.clip_max_reward = reward
                    if reward < self.clip_min_reward:
                       self.clip_min_reward = reward
                    frame_num += 1
                    # add dummy 0 q_val for now. Compute q_val at end of run.
                    q_val = 0
                    done = True
                    self.active_buffer.push(state,"REWARD1", reward, next_state, done, q_val)
                    print("Appending default REWARD1 at end of buffer", func_flow_nn_name)
              run_complete = True
              print("Function flow complete", state, action, reward, next_state, done, q_val)

              break
            else:
              print("Function Flow expected:", func_flow_nn_name, "but app index ended: ", app_index)
              # don't train DQN with incomplete TT results; continue with next idx
              # ARD: TODO: differentiate APP_FOR_DQN and native DQN datasets
              # TT_APP_IDX_PROCESSED.txt vs. TT_APP_IDX_PROCESSED_BY_DQN.txt 
              app_dsu.save_dataset_idx_processed(mode = app_mode)
              return "INCOMPLETE_APP_RUN"
            break
          nn_idx = nn_idx[0:-1]
          # print("remove trailing newline1", nn_idx)
          # file_name = nn_idx[0:-1]   # remove carriage return
          # NN_name = app_dsu.dataset_idx_to_func(file_name)
          NN_name = app_dsu.get_func_name_from_idx(nn_idx)
          if NN_name != func_flow_nn_name:
            print("Func flow / index inconsistency:", NN_name, func_flow_nn_name)
          # file_name = app_dsu.dataset_index_path(mode="FUNC", nn_name=NN_name) + file_name
          # print("Parsing NN idx", nn_idx, file_name)
          # nn_filehandle = open(file_name, 'r')
          print("Parsing NN idx", nn_idx)
          # Parsing NN idx ./apps/FUNC/PARK_ARM_RETRACTED_WITH_CUBE/dataset_indexes/FUNC_PARK_ARM_RETRACTED_WITH_CUBE_21_05_15a.txt
          nn_filehandle = open(nn_idx, 'r')
          line = None
          while True: # iterate through Img frames in nn
            # read a single line
            next_line = nn_filehandle.readline()
            if not next_line:
                print("Function Index completed:", frame_num, NN_name, next_action, func_flow_nn_name, func_flow_reward)
                # Function Index completed: 293 PARK_ARM_RETRACTED_WITH_CUBE REWARD1 PARK_ARM_RETRACTED_WITH_CUBE REWARD1
                break
            # get action & next_action
            [tm, app, mode, next_nn_name, next_action, img_name, next_state] = self.dsu.get_dataset_info(next_line, mode="FUNC")
            if line is not None:
              [tm, app, mode, nn_name, action, img_name, state] = self.dsu.get_dataset_info(line, mode="FUNC")
              # if func_flow_reward is not None:
                # ARD: is this correct? real_reward is never used. Why only called once?
                # ARD: Why break?
                # print("Real reward:", frame_num, func_flow_reward, func_flow_nn_name, next_action)
                # real_reward=True
                # break
              if action == "NOOP":
                # Too common in initial test dataset. Print warning later?
                # print("NOOP action, NN:", action, nn_name)
                line = next_line
                continue
              elif action == "REWARD1":
                print("Goto next NN; NOOP Reward, curr_NN", action, nn_name)
                line = next_line
                continue
              if next_action == "REWARD1" and func_flow_reward == "REWARD1":
                # func_flow determines if a function completes a "reward phase"
                reward, done = self.compute_reward(frame_num, next_action)
                print("completed REWARD phase", frame_num, next_action, reward, done)
                # completed REWARD phase 292 REWARD1 0.17333333333333334 False

              else:
                # print("compute_reward:", frame_num, action)
                reward, done = self.compute_reward(frame_num, action)
              frame_num += 1
              # add dummy 0 q_val for now. Compute q_val at end of run.
              q_val = 0
              self.active_buffer.push(state, action, reward, next_state, done, q_val)
            if next_action != "REWARD1":
              line = next_line
          # close the pointer to that file
          nn_filehandle.close()
        app_filehandle.close()
        #################################################
        ## NOT DONE FOR IMMITATION LEARNING
        # if len(self.replay_buffer) > self.REPLAY_INITIAL:
        #   loss = self.compute_td_loss(batch_size, app_path_prefix)
        # if frame_num % 1000 == 0 or done:
        #   self.update_target(self.current_model, self.target_model)
        #################################################
        if run_complete:
            print("SAVING STATE; DO NOT STOP!!!")
            self.active_buffer.compute_real_q_values(gamma=self.GAMMA,  name="active")
            self.active_buffer.reset_sample(name="active", start=0)
            print("parse_app_ds:", self.parse_app_ds_details)
            for i in range(self.cfg.NUM_EPOCHS):
              loss = 0
              while loss is not None:
                loss = self.compute_td_loss(batch_size=self.BATCH_SIZE, mode="REAL_Q_VALUES")
                mean_loss = np.mean(self.all_loss)
                print("real q values loss: ", mean_loss, i, self.all_loss[-1])
            # print("loss: ",loss)
            # print("ACTIVE BUFFER:", self.active_buffer)
            self.replay_buffer.concat(self.active_buffer)
            self.save_replay_buffer()
            print(self.BEST_MODEL_PATH)
            self.current_model.save_state(self.BEST_MODEL_PATH)
            # torch.save(model.state_dict(), self.BEST_MODEL_PATH)
            self.update_target(self.current_model, self.target_model)
            app_dsu.save_dataset_idx_processed(mode = app_mode)
            print("STATE SAVED")
        return "PROCESSED_APP_RUN"

    def parse_unprocessed_app_datasets(self, init=False):
        # nn_apps sets self.init_model to True in following line:
        #         self.app_instance = ALSET_DDQN(True, False, alset_app_name, alset_app_type)
        # Why? set to False? 
        # Alternatie taken: don't use self.init_model here...
        # init_ds = init
        init_ds = False
        while self.parse_app_dataset(init_ds) in ["PROCESSED_APP_RUN", "INCOMPLETE_APP_RUN"]:
            init_ds = False
            print("process another app dataset")

    def parse_unprocessed_rand_datasets(self, init=False):
        # nn_apps sets self.init_model to True in following line:
        #         self.app_instance = ALSET_DDQN(True, False, alset_app_name, alset_app_type)
        # Why? set to False?
        # Alternatie taken: don't use self.init_model here...
        # init_ds = init
        print(">>>>> parse_rand_dataset")
        init_ds = False
        while self.parse_app_dataset(init=init_ds, app_mode="RAND") in ["PROCESSED_APP_RUN", "INCOMPLETE_APP_RUN"]:
            init_ds = False
            print("process another rand dataset")


    def parse_unprocessed_dqn_datasets(self, init=False):
        init_ds = init
        while self.parse_dqn_dataset(init_ds) in ["PROCESSED_DQN_RUN", "INCOMPLETE_DQN_RUN"]:
            init_ds = False
            print("process another dqn dataset")

    # for training DQN by processing native DQN dataset
    def parse_dqn_dataset(self, init=False):
        print(">>>>> parse_dqn_dataset")
        if init:
          # start at the beginning
          self.dsu.save_dataset_idx_processed(mode = "APP", nn_name = None, dataset_idx = None)
        frame_num = 0
        reward = []
        #######################
        # iterate through NNs
        dqn_index = self.dsu.dataset_indices(mode="APP",nn_name=None,position="NEXT")
        if dqn_index is None:
          # print("parse_func_dataset: unknown NEXT index")
          print("parse_func_dataset: no more indexes")
          return
        print("Parsing DQN idx", dqn_index)
        dqn_filehandle = open(dqn_index, 'r')
        line = None
        while True:  
            # read a single line
            next_line = dqn_filehandle.readline()
            if not next_line:
                done = False
                break
            # get action & next_action
            [tm, app, mode, next_nn_name, next_action, img_name, next_state] = self.dsu.get_dataset_info(next_line)
            if line is not None:
              [tm, app, mode, nn_name, action, img_name, state] = self.dsu.get_dataset_info(line)
              self.replay_buffer.push(state, action, reward, next_state, done)
              reward, done = self.compute_reward(frame_num, action)
            frame_num += 1
  
            if next_action != "REWARD1":
              line = next_line
            if len(self.replay_buffer) > self.REPLAY_INITIAL:
              loss = self.compute_td_loss(batch_size, app_path_prefix)
            if frame_num % 1000 == 0 or done:
              self.update_target(self.current_model, self.target_model)
            if done:
              state = None
              self.all_rewards.append(self.total_reward)
              torch.save(self.current_model.state_dict(), self.BEST_MODEL_PATH)
        # close the pointer to that file
        filehandle.close()

    # for training DQN by processing random NN datasets
    def parse_rand_func_dataset(self, init=False):
        print(">>>>> parse_rand_func_dataset")
        frame_num = 0
        done = False
        reward = []
        val = self.cfg.get_value(self.cfg.app_registry, self.app_name)
        func_nn_list = val[1]
        func_app = FunctionalApp(alset_robot=None, app_name=self.app_name, app_type=self.app_type)
        ff_init = True
        while True:
          # Assume only the primary "successful" func_flow
          [func_flow_nn_name, func_flow_reward] = func_app.eval_func_flow_model(reward_penalty="REWARD1", init=ff_init)
          ff_init = False
          if func_flow_nn_name is None:
            reward, done = self.compute_reward(frame_num, "REWARD1")
            done = True
            break   # done func flow

          # randomly select NN dataset
          nn_idx = self.dsu.dataset_indices(mode="FUNC",nn_name=func_flow_nn_name,position="RANDOM")
          if nn_idx is None:
            break
          print("Parsing NN idx", nn_idx)
          nn_filehandle = open(nn_idx, 'r')
          line = None
          while True: # iterate through Img frames in nn
            # read a single line
            next_line = nn_filehandle.readline()
            if not next_line:
                break
            # get action & next_action
            [tm, app, mode, next_nn_name, next_action, img_name, next_state] = self.dsu.get_dataset_info(next_line)
            if line is not None:
              [tm, app, mode, nn_name, action, img_name, state] = self.dsu.get_dataset_info(line)
              qval = 0  # dummy qval for now until final reward assigned
              self.replay_buffer.push(state, action, reward, next_state, done, qval)
              if action == "REWARD1":
                break
              reward, done = self.compute_reward(frame_num, action)
            frame_num += 1
  
            if next_action != "REWARD1":
              line = next_line
            if len(self.replay_buffer) > self.REPLAY_INITIAL:
              loss = self.compute_td_loss(batch_size, app_path_prefix)
            if frame_num % 1000 == 0 or done:
              self.update_target(self.current_model, self.target_model)
          # close the pointer to that file
          nn_filehandle.close()
          # continue to While loop
        if done:
          state = None
          self.all_rewards.append(self.total_reward)
          torch.save(self.current_model.state_dict(), self.BEST_MODEL_PATH)
  

    def nn_process_image(self, NN_num, next_state, reward_penalty = None):
        # NN_num is unused by ddqn, but keeping nn_apps API

        print("DQN process_image")
        if len(self.replay_buffer) > self.REPLAY_INITIAL:
            # assume immitation learning initialization reduces need for pure random actions
            epsilon_start = .3 * self.REPLAY_INITIAL / len(self.replay_buffer)
        else:
            epsilon_start = 1.0
        epsilon_final = 0.01
        epsilon_decay = 30000   # make a config parameter?
        epsilon_by_frame = lambda frame_idx : epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)
        fr_num = self.frame_num + len(self.replay_buffer)
        epsilon = epsilon_by_frame(fr_num)
        rand_num = random.random()
        next_action = None

        if reward_penalty in ["REWARD1", "PENALTY1", "PENALTY2", "REWARD2"]:
            next_action = reward_penalty
            print("reward/penalty: ", next_action)
        # ARD: for debugging supervised learning, minimize random actions
        elif False and rand_num < epsilon:
            # ARD: We start with supervised learning. The original algorithm starts
            # with pure random actions.  We want to be more focused and add randomness
            # based upon a results of the DQN NN weights.
            while True:
              next_action_num = random.randrange(self.num_actions)
              # next_action = list(self.robot_actions)[next_action_num]
              next_action = list(self.cfg.full_action_set)[next_action_num]
              print("random action: ", next_action, epsilon, rand_num, fr_num, self.frame_num)
              if next_action not in self.cfg.nn_disallowed_actions:
                  break
        else:
            # print("next_state:",next_state)
            # next_state = self.transform_image_from_path(next_state)
            # next_state = self.transform_image(next_state)
            # next_state = torch.unsqueeze(next_state, 0)

            # (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)), weights=((64, 3, 11, 11), (64,)), parameters=23296
            #  weight 64 3 11 11, but got 3-dimensional input of size [224, 224, 3] instead
            # RuntimeError: Given groups=1, weight of size 64 3 11 11, expected input[1, 224, 224, 3] to have 3 channels, but got 224 channels instead

            print("B4 DQN NN exec")
            next_state = next_state.transpose((2, 0, 1))
            next_state_tensor = self.Variable(torch.FloatTensor(np.float32(next_state)))
            next_state_tensor = torch.unsqueeze(next_state_tensor, 0)
            # batch_item_num = 0 
            ## Doesn't use CUDA. Use self.Variable()
            # next_state_tensor = torchvision.transforms.ToTensor()(next_state).unsqueeze(batch_item_num) 
            # next_state_tensor = torch.unsqueeze(next_state_tensor, 0)
            # torch.transpose(next_state_tensor, 0, 1)
            sorted_actions, q_value = self.current_model.act(next_state_tensor)
            # top-level DQN movement_restrictions
            func_restrict = self.cfg.get_func_value(self.app_name, "MOVEMENT_RESTRICTIONS")
            sorted_action_order = sorted_actions.tolist()
            # print("sao:",sorted_action_order)
            sal = []
            tot = 0
            for i, s_a in enumerate(sorted_action_order):
                a = self.cfg.full_action_set[s_a]
                if a in self.cfg.nn_disallowed_actions:
                   continue
                if func_restrict is None or a in func_restrict:
                  q_val = q_value[0].tolist()
                  sal.append([i, a, q_val[s_a]])
                  if q_val[s_a] > 0:
                    tot += q_val[s_a]
            # print("all actions:",self.cfg.full_action_set)
            # print("disallowed:", self.cfg.nn_disallowed_actions)
            # print("restrictions:",self.app_name, func_restrict) 

            # next_action = list(self.robot_actions)[next_action_num]
            # SUM_1_16 = 136
            # ARD: hack
            # epsilon = .1
            if rand_num < epsilon:
              # ARD: Random selection weighted by positive q_val
              # q_val = q_value.data[0]
              # find max random number (sum of positive allowed q_vals)
              # tot = 0
              # for act in sorted_actions:
                  # next_action = list(self.cfg.full_action_set)[act]
                  # if q_val[act] <= 0:
                  #    break
                  # if next_action not in self.cfg.nn_disallowed_actions:
                     # tot += q_val[act]
              # rand_num = random.random() * tot
              # while True:
              if True:
                # rand_num = random.random() * SUM_1_16
                rand_num = random.random() * tot
                # find random selection
                tot2 = 0
                # for i, act in enumerate(sorted_actions):
                for i, act in enumerate(sal):
                  # next_action = list(self.cfg.full_action_set)[act]
                  # if q_val[act] <= 0:
                  #    print("ERR: rand select: ", act, next_action, rand_num, tot2, tot)
                  #    break
                  # tot2 += 16 - i
                  if act[1] in self.cfg.nn_disallowed_actions:
                     continue
                  next_action = act[1]
                  if act[2] > 0:   # problem if all negative 
                    tot2 += act[2]
                  else:
                    next_action = sal[0][1]
                    print("chose highest qval: ", sal[0] )
                    break
                  if rand_num <= tot2:
                        self.mean_cnt += 1
                        self.mean_i += act[0]
                        print("rand select: ", act, rand_num, tot2, tot, (self.mean_i/self.mean_cnt))
                        # rand select:  tensor(0, device='cuda:0') FORWARD 47.237562023512126 55 0
                        break
                # if next_action in func_restrict:
                #     print("allowed1")
                #     break
                # elif next_action not in self.cfg.nn_disallowed_actions:
                #    print("allowed2")
                #    break
                if func_restrict is None or next_action in func_restrict:
                    print("allowed:", next_action)
            else:
              for next_action_num in sorted_actions:
                  next_action = list(self.cfg.full_action_set)[next_action_num]
                  if func_restrict is None or next_action in func_restrict:
                    print("allowed1")
                    break
                  # elif next_action not in self.cfg.nn_disallowed_actions:
                  #     print("next_action:", next_action)
                  #     break
                  print("action disallowed:", next_action)
        print("sel_act:",  next_action, sal)
        if self.frame_num == 0 and self.state == None:
          self.frame_num += 1
          self.state  = next_state
          self.action = next_action
          print("NN3: ", next_action)
          return next_action

        # having figured out our next_action based upon the state next_state,
        # evaluate reward for the preceeding move (self.action) based upon self.state / next_state.
        # Note that events like REWARD/PENALTY aren't a real robot move, so these events 
        # are ignored and the previous real move is used (self.prev_action, self.prev_state)
        phase = self.curr_phase     # changes upon reward
        action_reward, action_done = self.compute_reward(self.frame_num, self.action)
        print("reward: ", self.frame_num, self.action, action_reward, action_done)
        self.total_reward += action_reward

        if self.action != "REWARD1":
          # self.replay_buffer or active_buffer????  
          # add dummy 0 q_val for now. Compute q_val at end of run.
          self.active_buffer.push(self.state, self.action, action_reward, next_state, action_done, 0)
        else:
          # self.replay_buffer or active_buffer????  
          # add dummy 0 q_val for now. Compute q_val at end of run.
          self.active_buffer.push(self.prev_state, self.prev_action, action_reward, self.state, action_done, 0)

        if len(self.replay_buffer) > self.REPLAY_INITIAL:
          loss = self.compute_td_loss(batch_size=self.BATCH_SIZE, mode="EXPERIENCE_REPLAY")
          print("experience replay loss: ",loss)

        # if frame_num % 10000 == 0:
        #    plot(frame_num, self.all_rewards, self.losses)
    
        if action_done:
            self.active_buffer.compute_real_q_values(gamma=self.GAMMA,  name="active")
            self.active_buffer.reset_sample(name="active", start=0)
            loss = 0
            while loss is not None:
              loss = self.compute_td_loss(batch_size=self.BATCH_SIZE, mode="REAL_Q_VALUES")
              print("real q values loss: ",loss)
            print("loss: ",loss)
            self.replay_buffer.concat(self.active_buffer)
            self.save_replay_buffer()
            self.current_model.save_state(self.BEST_MODEL_PATH)
            # torch.save(model.state_dict(), self.BEST_MODEL_PATH)
            self.update_target(self.current_model, self.target_model)
            self.prev_state = None
            self.state = None
            self.curr_phase = 0
            # self.all_rewards.append(self.total_reward)
            self.frame_num = 0
            print("Episode is done; reset robot and cube")
            return "DONE"
        else:
            self.frame_num += 1
            self.prev_state = self.state 
            self.state = next_state 
            self.prev_action = self.action 
            self.action = next_action 
            
        print("NN4: ", next_action)
        return next_action

    # APP training is done automatically at start of DQN.  Just running DQN will pick up
    # any new datasets since it left off.  
    #
    # FUNC training based on a single end-to-end run of a random selection of FUNC/NN runs.
    # FUNC is done by calling:
    #   ./copy_to_python ; python3 ./alset_jetbot_train.py --dqn TTT
    def train(self, number_of_random_func_datasets = 30):
        # Check if any app training left undone:
        print("Checking if any Functional APP training to do...")
        self.parse_unprocessed_app_datasets(init=self.init_model)
        # TODO: RL training of DQN
        # print("Checking if any DQN training to do...")
        # self.parse_unprocessed_dqn_datasets(init=self.init_model)
        print("Train DQN based upon random selection of Functional Runs...")
        self.parse_unprocessed_rand_datasets(init=self.init_model)
        # for i in range(number_of_random_func_datasets):
        #   self.parse_func_datasets(init=self.init_model)
