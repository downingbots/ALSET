# https://github.com/higgsfield/RL-Adventure/blob/master/2.double%20dqn.ipynb

import math, random
# from builtins import bytes
import codecs
from PIL import Image
import cv2

# import gym
import numpy as np
import collections, itertools

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

# import sir_image 

# from IPython.display import clear_output
# import matplotlib.pyplot as plt
# %matplotlib inline


from collections import deque

# workaround due to ReplayBuffer pickling/unpickling and class evolution
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
        state, action, reward, next_state, done, dummy_q_val = zip(*replay_buffer.buffer[-1])
        assert done, "last entry must be completion of run"
        self.buffer += replay_buffer.buffer
        replay.buffer.clear()

    def compute_real_q_values(self, gamma=.99, name="replay", sample_start=0, done_end=None):
        buf_id = self.reset_sample(name,sample_start)
        if done_end is None:
          done_end = len(self.buffer)
        try:
          state, action, reward, next_state, done, dummy_q_val = zip(*itertools.islice(self.buffer,sample_start,done_end))
          add_q_val = False
        except:
          state, action, reward, next_state, done = zip(*itertools.islice(self.buffer,sample_start,done_end))
          print("No qval in replay buffer. Adding qval.")
          add_q_val = True
        print("len done:", len(done), done_end-sample_start, done_end, sample_start)
        assert done[done_end-sample_start-1], "Only compute real q values upon completion of run"
        next_q_val = 0
        q_val = 0
        len_reward = len(reward)
        converted = False
        for i,reward_val in enumerate(reversed(reward)):
            d = len_reward - i - 1
            if done[d]:
              next_q_val = 0
            else:
              next_q_val = q_val # "next" because iterating in reverse order
            q_val = reward_val + gamma * next_q_val
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
# from .sir_image import read_image
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms


# class CnnDQN(nn.Module):
class CnnDQN():
    # def __init__(self, input_shape, num_actions):
    def __init__(self, num_actions):
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
        self.alexnet_optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
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
        for idx, operation in enumerate(model.features):
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

        action  = q_value.max(1)[1].data[0]
        return action


##################
# SIR_Jetbot (SJB)
##################

# Original dataset for TT_FUNC:
# PH1 - nothing in gripper, no cube/box in vision
#  NN1. Park Arm
#  NN2. Automatic scan for cube
#       add in side avoidance?
# PH2 - nothing in gripper, cube in vision
#  NN3. Approach cube
#  NN4. Pick up cube
# PH3 - cube in gripper, no box in vision
#  NN5. Park Arm (with cube in grippers)
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

# class SIR_DDQN(nn.Module):
class SIR_DDQN():

    def __init__(self, initialize_model=False, do_train_model=False):
        self.BEST_MODEL_PATH = './apps/TT_DQN/dataset/TTDQN_model1.pth'
        self.TTDQN_PATH_PREFIX = "./apps/TT_DQN/dataset/"
        self.REPLAY_BUFFER_PATH = './apps/TT_DQN/dataset/replay_buffer.data'

        # these are the directories. Joystick can specify reward/penalty + robot actions.
        # NOOP should not have a directory. TT_FUNC has an index that should translate images
        # into a NOOP for automatic mode.
        # Automatic mode should not use a dataloader for the NN.
        self.actions = ( "FORWARD", "REVERSE", "LEFT", "RIGHT", "LOWER_ARM_DOWN", "LOWER_ARM_UP", "UPPER_ARM_DOWN", "UPPER_ARM_UP", "GRIPPER_OPEN", "GRIPPER_CLOSE", "REWARD", "PENALTY", "ROBOT_OFF_TABLE_PENALTY", "CUBE_OFF_TABLE_REWARD", "NOOP")
        self.robot_actions = ( "FORWARD", "REVERSE", "LEFT", "RIGHT", "LOWER_ARM_DOWN", "LOWER_ARM_UP", "UPPER_ARM_DOWN", "UPPER_ARM_UP", "GRIPPER_OPEN", "GRIPPER_CLOSE")

        # Reward computation constants
        self.ALLOCATED_MOVES_CUBE_PICKUP =  300.0
        self.ALLOCATED_MOVES_CUBE_IN_BOX =  400.0
        self.MAX_MOVES_EXCEEDED_PENALTY  = -100.0
        self.MOVE_BONUS                  =     .25 
        self.PER_MOVE_PENALTY            =    -.25 
        self.CUBE_OFF_TABLE_REWARD       =   25.0
        self.ROBOT_OFF_TABLE_PENALTY     = -300.0

        # Reward States
        self.PRE_CUBE           = 0
        self.POST_CUBE          = 1
        self.CUBE_OFF_TABLE     = 2
        self.ROBOT_OFF_TABLE    = 3
        self.MAX_MOVES_EXCEEDED = 4

        self.MAX_MOVES =  1500

        # reward variables
        self.standard_mean      = 0.0
        self.standard_variance  = 1.0
        self.estimated_mean     = 0.0
        self.estimated_variance = 300.0
        self.curr_phase   = self.PRE_CUBE
        self.total_reward = 0
        self.frame_num    = 0
        self.action       = None
        self.prev_action  = None
        self.state        = None
        self.prev_state   = None

        ############
        # DDQN variables
        ############
        # self.replay_initial = 10000
        self.replay_initial = 5000   # assumes that this is immitation learning data
        self.replay_buffer  = ReplayBuffer(capacity=100000, name="replay")
        # allow for 20 rewards in active buffer
        self.active_buffer  = ReplayBuffer(capacity=self.MAX_MOVES+20, name="active")  
        
        # self.epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)
        
        # plt.plot([self.epsilon_by_frame(i) for i in range(1000000)])
        self.num_frames = 1000000
        self.batch_size = 32
        self.gamma      = 0.99

        self.USE_CUDA = torch.cuda.is_available()
        self.Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if self.USE_CUDA else autograd.Variable(*args, **kwargs)
        
        # self.losses = []
        # self.all_rewards = []

        ############
        # FINE TUNE PRETRAINED MODEL USING IMITATION LEARNING
        # add to history
        
        self.num_actions = len(self.robot_actions)
        self.current_model = None
        self.target_model = None
        self.init_model = initialize_model
        self.train_model = do_train_model
        print("DQN initialization: ",self.init_model, self.train_model)
        
    def nn_init(self, app_name, NN_num, gather_mode=False):
        self.current_model = CnnDQN(self.num_actions)
        self.target_model = CnnDQN(self.num_actions)

        self.robot_actions = ["UPPER_ARM_UP", "UPPER_ARM_DOWN", "LOWER_ARM_UP", "LOWER_ARM_DOWN",
                "GRIPPER_OPEN", "GRIPPER_CLOSE", "FORWARD", "REVERSE", "LEFT", "RIGHT"]
        self.joystick_actions = ["REWARD","PENALTY", "CUBE_OFF_TABLE_REWARD", "ROBOT_OFF_TABLE_PENALTY"]
        self.full_action_set = self.robot_actions + self.joystick_actions
        self.model_dir = "./apps/TT_DQN/"
        self.model_prefix = "TTDQN_model"
        self.ds_prefix = "./apps/TT_FUNC/dataset/NN"

        if (self.init_model):  # probably done on laptop
            # starts from the first dataset
            self.parse_dataset(init=True)
            self.save_replay_buffer()
            if (self.train_model):
              self.train_DQN_qvalue()
        else:
            self.load_replay_buffer()
            # get new datasets
            self.parse_dataset(init=False)
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
        return False, self.actions


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
        # state_image = []
        for batch_item_num, path in enumerate(image_filepath):
            try:
                #filepath = self.TTFUNC_PATH_PREFIX + bytestring.decode(path)
                #filepath = self.TTFUNC_PATH_PREFIX + path.decode("utf-8").strip('\n')
                # filepath = self.TTFUNC_PATH_PREFIX + path.astype("U13").str.strip('\n')
                try:
                  filepath = self.TTFUNC_PATH_PREFIX + path.decode("utf-8").strip('\n')
                except:
                  print("path:", path)
                  filepath = str(self.TTFUNC_PATH_PREFIX) + str(path.astype("U13")).strip('\n')
                # print("1 ", filepath)
                img = Image.open(filepath)
                img = self.transform_image(img)
                # TypeError: float() argument must be a string or a number, not 'set'
                img = self.Variable(torch.FloatTensor(np.float32(img)))
                state_image = torch.unsqueeze(img, 0)
                # unsqueeze to add artificial first dimension
                # img = ToTensor()(img).unsqueeze(batch_item_num) 
                # img = torch.unsqueeze(img, 0)
                # if batch_item_num == 0:
                #    state_image = img
                #else:
                #    state_image = torch.cat((state_image,img),0)

	        # read input image
	        # input_img = cv2.imread(img_path)
	        # do transformations
	        # img = self.transform_image(input_img)
	        # img = transform_image(input_img)["image"]
	        # batch_data = torch.unsqueeze(input_data, 0)

                ## JIT loader doesn't seem to work on Jetson for image file...
                ## lib_dir = osp.abspath(osp.join(osp.dirname(__file__), ".."))
                ## loader_details = (
                ##     importlib.machinery.ExtensionFileLoader,
                ##     importlib.machinery.EXTENSION_SUFFIXES
                ## )
                ## extfinder = importlib.machinery.FileFinder(lib_dir, loader_details)  
                ## ext_specs = extfinder.find_spec("image")
                ## if ext_specs is not None:
                ##   torch.ops.load_library(ext_specs.origin)
                ## img = torch.ops.image.read_file(path)
                ## img = torch.ops.image.decode_image(img)
                # img = read_file(path)
                # img = decode_image(img)
                # img = torch.ops.read_file(path)
                # img = torch.ops.decode_image(img)
                # img = read_image(filepath)
                # img = decode_image(img)
                # img = torchvision.io.image.read_image(filepath)
                # img = torchvision.io.image.decode_image(img)
                # img = Image.open(filepath)
                # filehandle = open(filepath, 'r')
                # img  = cv2.imread(filepath)
                # load the raw data from the file as a string
                # img = tf.io.read_file(file_path)
                # img = tf.image.decode_jpeg(img, channels=3)

            except:
                filepath = self.TTDQN_PATH_PREFIX + path.decode("utf-8").strip('\n')
                # print("2 ", filepath)
                img = Image.open(filepath)
                img = self.transform_image(img)
                img = torch.unsqueeze(img, 0)
                if batch_item_num == 0:
                    state_image = img
                else:
                    state_image = torch.cat((state_image,img),0)
            # close the pointer to that file
            # img.close()
        # state_image = tuple(state_image)
        return state_image

    # import json
    # json.dump(self.replay_buffer, filehandle)
    # json.load(filehandle)
    def save_replay_buffer(self):
        with open(self.REPLAY_BUFFER_PATH, 'wb') as filehandle:
          # store the data as binary data stream
          pickle.dump(self.replay_buffer, filehandle)
        filehandle.close()

    def load_replay_buffer(self):
        with open(self.REPLAY_BUFFER_PATH, 'rb') as filehandle:
          self.replay_buffer = pickle.load(filehandle)
        filehandle.close()
        print("loaded replay_buffer. Len = ", len(self.replay_buffer))
        if self.replay_buffer.entry_len() == 5:
          # state, action, reward, next_state, done
          print("adding q_values to replay_buffer")
          self.replay_buffer.compute_real_q_values(gamma=self.gamma)
          self.save_replay_buffer()

    def train_DQN_qvalue(self):
        loss = 0
        while loss is not None:
          loss = self.compute_td_loss(batch_size = self.batch_size, mode = "IMITATION_TRAINING")
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
          state_path, action, rewards, next_state_path, done_val, q_val = self.replay_buffer.get_next_sample(batch_size)
        elif mode == "REAL_Q_VALUES":
          state_path, action, rewards, next_state_path, done_val, q_val = self.active_buffer.get_next_sample(batch_size=batch_size, name="active")
        elif mode == "EXPERIENCE_REPLAY":
          # The learning phase is then logically separate from gaining experience, and based on 
          # taking random samples from the buffer. 
          #
          # Advantages: More efficient use of previous experience, by learning with it multiple times.
          # This is key when gaining real-world experience is costly, you can get full use of it.
          # The Q-learning updates are incremental and do not converge quickly, so multiple passes 
          # with the same data is beneficial, especially when there is low variance in immediate 
          # outcomes (reward, next state) given the same state, action pair.
          #
          # Disadvangage: It is harder to use multi-step learning algorithms
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

        # transform state paths to images
        # print(state_path, next_state_path)
        # print(rewards, action, done_val)
        state        = self.transform_image_from_path(state_path)
        next_state   = self.transform_image_from_path(next_state_path)
        # transform action string to number
        # print("self.actions:",self.robot_actions)
        action_idx = []
        for a in action:
          # action_idx.append(self.robot_actions.index(a))  # Human readable to integer index
          action_idx.append(self.full_action_set.index(a))  # Human readable to integer index
        action_idx = tuple(action_idx)
        # print("action:",action)
        # print("action_idx:",action_idx)

        # 4-D tensors are for batches of images, 3-D tensors for individual images.
        # image_batch is a tensor of the shape (32, 180, 180, 3)
        # already a Tensor
        # state      = self.Variable(torch.FloatTensor(np.float32(state)))
        # next_state = self.Variable(torch.FloatTensor(np.float32(next_state)))

        action_idx = self.Variable(torch.LongTensor(action_idx))
        reward     = self.Variable(torch.FloatTensor(rewards))
        done       = self.Variable(torch.FloatTensor(done_val))
        q_val      = self.Variable(torch.FloatTensor(q_val))  # real q value computed from done end-pt

        # Computed q-values from Alexnet
        current_q_values      = self.current_model.alexnet_model(state)
        current_next_q_values = self.current_model.alexnet_model(next_state)
        target_next_q_values  = self.target_model.alexnet_model(next_state)
        # print(current_q_values)
        # print(current_next_q_values)
        # print(target_next_q_values)

        q_value      =     current_q_values.gather(1, action_idx.unsqueeze(0)).squeeze(1)
        # print("q_value", q_value)
        next_q_value = target_next_q_values.gather(1, torch.max(current_next_q_values, 1)[1].unsqueeze(1)).squeeze(1)

        # the real q value is precomputed in q_val
        # the real q value is likely dependent on a reward beyond the batch size.
        expected_q_value = reward + self.gamma * next_q_value * (1-done)
        # print("expected_q_value: ", expected_q_value)
        # print("q_value         : ", q_value)
        # print("next_q_value    : ", next_q_value)
        print("reward: ", reward)
        print("q_val : ", q_val)
        if mode == "IMITATION_TRAINING":
          loss = (q_value - q_val).pow(2).mean()
          print("q_value:", q_value)
          print("IMITATION_TRAINING:", loss)
        elif mode == "REAL_Q_VALUES":
          loss = (q_value - q_val).pow(2).mean()
          print("q_value:", q_value)
          print("REAL_Q_VALUES:", loss)
        elif mode == "EXPERIENCE_REPLAY":
          loss = (q_value - expected_q_value).pow(2).mean()
          print("exp_q_value:", expected_q_value)
          print("EXPERIENCE_REPLAY FROM TARGET:", loss)
          # loss = (q_value - expected_q_value.data).pow(2).mean()
          # loss = (q_value - q_val).pow(2).mean()
          # print("EXPERIENCE_REPLAY WITH REAL Q_VAL:", loss)

        self.current_model.train()  # Set model to training mode
        self.current_model.alexnet_optimizer.zero_grad()
        loss.backward()
        self.current_model.alexnet_optimizer.step()
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
        reward = 0
        if frame_num > self.MAX_MOVES:
          return (self.MAX_MOVES_EXCEEDED_PENALTY / self.estimated_variance), True
        elif action == "REWARD":
          done = False
          if self.curr_phase == self.PRE_CUBE:
            # Cube picked up
            reward = 50 + max((self.ALLOCATED_MOVES_CUBE_PICKUP - frame_num),0)*self.MOVE_BONUS
            self.curr_phase = self.POST_CUBE
          elif self.curr_phase == self.POST_CUBE:
            reward = 100 + max((self.ALLOCATED_MOVES_CUBE_IN_BOX - frame_num),0)*self.MOVE_BONUS
            done = True
            self.curr_phase = self.PRE_CUBE
          return (reward / self.estimated_variance), done
        elif action == "CUBE_OFF_TABLE_REWARD":
          return (self.CUBE_OFF_TABLE_REWARD / self.estimated_variance), True
        elif action == "ROBOT_OFF_TABLE_PENALTY" or action == "PENALTY":
          return (self.ROBOT_OFF_TABLE_PENALTY / self.estimated_variance), True
        elif self.curr_phase == self.POST_CUBE or self.curr_phase == self.PRE_CUBE:
          return (self.PER_MOVE_PENALTY / self.estimated_variance), False
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
        # set by joystick: REWARD, PENALTY, ROBOT_OFF_TABLE, CUBE_OFF_TABLE
        self.dqn_action = action

    def get_dqn_action(self):
        # set by joystick: REWARD, PENALTY, ROBOT_OFF_TABLE, CUBE_OFF_TABLE
        return self.dqn_action
    
    # for parsing either TT_FUNC or TT_DQN app dataset
    def parse_dataset(self, init=False):
        ds_util = DatasetUtils("TT_DQN")
        if init:
          # start at the beginning
          ds_util.save_dataset_idx_processed(app_nm = "TT_DQN", dataset_idx=None)
        while True:
          full_path, ds_name = ds_util.next_dataset_idx(app_nm = "TT_DQN", init=init)
          if ds_name is None:
            break
          filehandle = open(full_path, 'r')
          frame_num = 0
          reward = []
          line = filehandle.readline()
          while True:
            # read a single line
            next_line = filehandle.readline()
            if not next_line:
                break
            [tm, state, action, nn_num] = ds_util.get_dataset_info(line)
            [tm, next_state, next_action, nn_num] = ds_util.get_dataset_info(next_line)
            if action == "REWARD":
                if NN_num not in NN_REAL_REWARD:
                  print("action to NOOP, NN#", action, NN_num)
                  line_action = "NOOP"
            reward, done = compute_reward(frame_num, action)
            frame_num += 1
            self.replay_buffer.push(state, action, reward, next_state, done)
  
            if next_action != "REWARD":
              line = next_line
            if len(self.replay_buffer) > self.replay_initial:
              loss = compute_td_loss(batch_size, app_path_prefix)
            if frame_num % 1000 == 0 or done:
              update_target(self.current_model, self.target_model)
            if done:
              state = None
              self.all_rewards.append(self.total_reward)
              torch.save(model.state_dict(), self.BEST_MODEL_PATH)
  
          # close the pointer to that file
          filehandle.close()

    def nn_process_image(self, NN_num, next_state, reward_penalty = None):
        # NN_num is unused by ddqn, but keeping nn_apps API

        if len(self.replay_buffer) > self.replay_initial:
            # assume immitation learning initialization reduces need for pure random ations
            epsilon_start = .3 * self.replay_initial / len(self.replay_buffer)
        else:
            epsilon_start = 1.0
        epsilon_final = 0.01
        epsilon_decay = 30000
        epsilon_by_frame = lambda frame_idx : epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)
        fr_num = self.frame_num + len(self.replay_buffer)
        epsilon = epsilon_by_frame(fr_num)
        rand_num = random.random()

        if reward_penalty in ["REWARD", "PENALTY", "ROBOT_OFF_TABLE_PENALTY", "CUBE_OFF_TABLE_REWARD"]:
            next_action = reward_penalty
            print("reward/penalty: ", next_action)
        elif rand_num < epsilon:
            next_action_num = random.randrange(self.num_actions)
            # next_action = list(self.robot_actions)[next_action_num]
            next_action = list(self.full_action_set)[next_action_num]
            print("random action: ", next_action, epsilon, rand_num, fr_num, self.frame_num)
        else:
            # print("next_state:",next_state)
            # next_state = self.transform_image_from_path(next_state)
            # next_state = self.transform_image(next_state)
            # next_state = torch.unsqueeze(next_state, 0)

            # (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)), weights=((64, 3, 11, 11), (64,)), parameters=23296
            #  weight 64 3 11 11, but got 3-dimensional input of size [224, 224, 3] instead
            # RuntimeError: Given groups=1, weight of size 64 3 11 11, expected input[1, 224, 224, 3] to have 3 channels, but got 224 channels instead


            next_state = next_state.transpose((2, 0, 1))
            next_state_tensor = self.Variable(torch.FloatTensor(np.float32(next_state)))
            next_state_tensor = torch.unsqueeze(next_state_tensor, 0)
            # batch_item_num = 0 
            ## Doesn't use CUDA. Use self.Variable()
            # next_state_tensor = torchvision.transforms.ToTensor()(next_state).unsqueeze(batch_item_num) 
            # next_state_tensor = torch.unsqueeze(next_state_tensor, 0)
            # torch.transpose(next_state_tensor, 0, 1)
            next_action_num = self.current_model.act(next_state_tensor)
            # next_action = list(self.robot_actions)[next_action_num]
            next_action = list(self.full_action_set)[next_action_num]
            print("NN: ", next_action)
        if self.frame_num == 0 and self.state == None:
          self.frame_num += 1
          self.state  = next_state
          self.action = next_action
          return next_action

        # having figured out our next_action based upon the state next_state,
        # evaluate reward for the preceeding move (self.action) based upon self.state / next_state.
        # Note that events like REWARD/PENALTY aren't a real robot move, so these events 
        # are ignored and the previous real move is used (self.prev_action, self.prev_state)
        phase = self.curr_phase     # changes upon reward
        action_reward, action_done = self.compute_reward(self.frame_num, self.action)
        print("reward: ", self.frame_num, self.action, action_reward, action_done)
        self.total_reward += action_reward

        if self.action != "REWARD":
          # add dummy 0 q_val for now. Compute q_val at end of run.
          self.active_buffer.push(self.state, self.action, action_reward, next_state, action_done, 0)
        else:
          # add dummy 0 q_val for now. Compute q_val at end of run.
          self.active_buffer.push(self.prev_state, self.prev_action, action_reward, self.state, action_done, 0)

        if len(self.replay_buffer) > self.replay_initial:
          loss = self.compute_td_loss(batch_size=self.batch_size, mode="EXPERIENCE_REPLAY")
          print("experience replay loss: ",loss)

        # if frame_num % 10000 == 0:
        #    plot(frame_num, self.all_rewards, self.losses)
    
        if action_done:
            self.active_buffer.compute_real_q_values(gamma=self.gamma)
            self.active_buffer.reset_sample(name="active", start=0)
            loss = 0
            while loss is not None:
              loss = self.compute_td_loss(batch_size=self.batch_size, mode="REAL_Q_VALUES")
              print("real q values loss: ",loss)
            print("loss: ",loss)
            self.replay_buffer.concat(self.active_buffer)
            self.save_replay_buffer()
            self.current_model.save_state(self.BEST_MODEL_PATH)
            # torch.save(model.state_dict(), self.BEST_MODEL_PATH)
            self.update_target(self.current_model, self.target_model)
            self.prev_state = None
            self.state = None
            self.curr_phase = self.PRE_CUBE
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
            
        return next_action

    def train(self):
        self.train_model = True
        self.nn_init("TT_DQN", 2, False)

# TODO: Allow states self.CUBE_OFF_TABLE, self.ROBOT_OFF_TABLE, REWARD, PENALTY 
# add syntax to call DDQN, PRETRAIN
