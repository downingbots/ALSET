# https://github.com/higgsfield/RL-Adventure/blob/master/2.double%20dqn.ipynb

import math, random

# import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

# from IPython.display import clear_output
# import matplotlib.pyplot as plt
# %matplotlib inline


from collections import deque

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
            
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
    
    def __len__(self):
        return len(self.buffer)


##############
# Initially defined for atari env
##############
import torchvision
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


    def train_model(self, dataloaders, criterion, optimizer):
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
    
    def transform_image(self, image_filepath):
        # data transforms, for pre-processing the input image before feeding into the net
        data_transforms = transforms.Compose([
                                              transforms.Resize((224,224)),  # resize to 224x224
                                              transforms.ToTensor(),         # tensor format
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])  ])
        # open the testing image
        img = Image.open(image_filepath)
        # pre-process the input
        transformed_img = data_transforms(img)
        return transformed_img

    def act(self, state, epsilon):
        if random.random() < self.epsilon:
            action = random.randrange(self.num_actions)
            return action
        self.act(state)

    def act(self, state):
        # state is an image file, preappend path prefix
        image_filepath = self.PATH_PREFIX + state
        transformed_img = transform_image(image_filepath)
        # print("transformed image's shape: " + str(transformed_img.shape))
        # form a batch with only one image
        batch_img = torch.unsqueeze(transformed_img, 0)
        # print("image batch's shape: " + str(batch_img.shape))

        # put the model to eval mode for testing
        self.alexnet_model.eval()
        # obtain the output of the model
        q_value = self.alexnet_model(batch_img)
        # print("output vector's shape: " + str(q_value.shape))

        # obtain the activation maps
        # visualize_activation_maps(batch_img, self.alexnet_model)
        # sorted, indices = torch.sort(q_value, descending=True)
        # percentage = F.softmax(output, dim=1)[0] * 100.0
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

# class SIR_DDQN(nn.Module):
class SIR_DDQN():

    def __init__(self, initialize_model=False):
        self.TTFUNC_DATASET_INDEX_FILE = "./apps/tt_func/dataset/run_details_sorted.txt"
        self.TTFUNC_PATH_PREFIX = './apps/tt_func/dataset/'
        self.BEST_MODEL_PATH = './apps/tt_ddq/dataset/best_model.pth'
        self.TTDQN_DATASET_INDEX_FILE = "./apps/tt_dqn/dataset/run_details_sorted.txt"
        self.TTDQN_PATH_PREFIX = './apps/tt_dqn/dataset/'

        # these are the directories. Joystick can specify reward/penalty + robot actions.
        self.actions = { "FORWARD", "REVERSE", "LEFT", "RIGHT", "LOWER_ARM_DOWN", "LOWER_ARM_UP", "UPPER_ARM_DOWN", "UPPER_ARM_UP", "GRIPPER_OPEN", "GRIPPER_CLOSE", "REWARD", "PENALTY", "ROBOT_OFF_TABLE_PENALTY", "CUBE_OFF_TABLE_REWARD", "NOOP"}
        self.robot_actions = { "FORWARD", "REVERSE", "LEFT", "RIGHT", "LOWER_ARM_DOWN", "LOWER_ARM_UP", "UPPER_ARM_DOWN", "UPPER_ARM_UP", "GRIPPER_OPEN", "GRIPPER_CLOSE"}

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
        self.replay_initial = 10000
        self.replay_buffer  = ReplayBuffer(100000)
        
        # self.epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)
        
        # plt.plot([self.epsilon_by_frame(i) for i in range(1000000)])
        self.num_frames = 1000000
        self.batch_size = 32
        self.gamma      = 0.99

        self.Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
        
        # self.losses = []
        # self.all_rewards = []

        ############
        # FINE TUNE PRETRAINED MODEL USING IMITATION LEARNING
        # add to history
        
        self.num_actions = len(self.robot_actions)
        self.current_model = CnnDQN(self.num_actions)
        self.target_model = CnnDQN(self.num_actions)
        if (initialize_model):
            self.parse_dataset(self.TTFUNC_DATASET_INDEX_FILE, self.TTDQN_PATH_PREFIX)
        else:
            self.current_model.load_state_dict(torch.load(self.BEST_MODEL_PATH))

        # target_model = CnnDQN(env.observation_space.shape, env.action_space.n)
        # if USE_CUDA:
        #     current_model = current_model.cuda()
        #     target_model  = target_model.cuda()
            
        # target_model = copy.deepcopy(current_model)
        self.update_target(self.current_model, self.target_model)
        

    def update_target(self, current_mdl, target_mdl):
        target_mdl.load_state_dict(current_mdl.state_dict())
        # sd = copy.deepcopy(target_mdl.state_dict())
        # model.load_state_dict(sd)
        ## t_state = target_mdl.state_dict()
        ##  t_state.update(current_mdl.state_dict())
    
    def compute_td_loss(self, batch_size, path_prefix):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        # transform state paths to images
        img_filepath = path_prefix + state
        state = self.transform_image(img_filepath)
        img_filepath = path_prefix + next_state
        next_state = self.transform_image(img_filepath)
        # transform action string to number
        action = self.actions.index(action)

        state      = self.Variable(torch.FloatTensor(np.float32(state)))
        next_state = self.Variable(torch.FloatTensor(np.float32(next_state)))
        action     = self.Variable(torch.LongTensor(action))
        reward     = self.Variable(torch.FloatTensor(reward))
        done       = self.Variable(torch.FloatTensor(done))
    
        q_values      = self.current_model.act(state)
        next_q_values = self.current_model.act(next_state)
        next_q_state_values = self.target_model.act(next_state) 
    
        q_value       = q_values.gather(1, action.unsqueeze(1)).squeeze(1) 
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = reward + gamma * next_q_value * (1 - done)
        
        loss = (q_value - self.Variable(expected_q_value.data)).pow(2).mean()
            
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
        elif action == "ROBOT_OFF_TABLE_PENALTY":
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
    
    
    def get_action(self, line):
        NN_NUM_OFFSET = 13
        ACTION_OFFSET = 15
        NN_REAL_REWARD = [4, 8]  # pick_up_cube, drop_cube_in_box
        found = False
        for action in self.actions:
          line_action = line[ACTION_OFFSET:ACTION_OFFSET+len(action)]
          if line_action == action:
            found = True
            if action == "REWARD":
              NN_num = int(line[NN_NUM_OFFSET:NN_NUM_OFFSET+1])
              if NN_num not in NN_REAL_REWARD:
                # print("action to NOOP, NN#", action, NN_num)
                action = "NOOP"
            break
        if not found:
          print("action not found: ", line)
        return line_action 
    
    # for parsing either TT_FUNC or TT_DQN app dataset
    def parse_dataset(self, filename, app_path_prefix):
        # open the file for reading
        filehandle = open(filename, 'r')
        #
        # format of DATASET_INDEX_FILE:
        #  "18:31:28 ./NN1/LOWER_ARM_DOWN/a1099b28-4334-11eb-8cce-3413e860d1ff.jpg"
        #   0123456789012345
        # # fixed size offsets
        # file   = line[9] 
        # NN#    = line[13]
        # action = line[15]
        # 
        FILE_OFFSET   = 9
      
        frame_num = 1
        reward = 0
        done = True
        while True:
          if done:
            # initialize; no reward on first move!
            frame_num = 1
            self.curr_phase = self.PRE_CUBE
            prev_action = None
            prev_state = None
            line = filehandle.readline()
            action = self.get_action(line)
            state = line[FILE_OFFSET:]
            done = False
          # read a single line
          next_line = filehandle.readline()
          if not next_line:
              print("no next line")
              break
          next_action = self.get_action(next_line)
          if next_action == "NOOP":
            # NOOP is only for multi-function NN apps in automatic mode and no replay_buffer.
            print("NOOP")
            continue
          next_state = next_line[FILE_OFFSET:]
          # nn_num = int(line[NN_NUM_OFFSET, NN_NUM_OFFSET]) 
          # print("action ", action, " NN# ", nn_num)
          print("action ", action, frame_num, self.curr_phase)
          reward, done = self.compute_reward(frame_num, action)
          if reward != 0:
            print("reward:", reward*self.estimated_variance, done)
          if action == "REWARD":
            self.replay_buffer.push(prev_state, prev_action, reward, state, done)
          else:
            self.replay_buffer.push(state, action, reward, next_state, done)
          frame_num += 1
          prev_state  = state
          prev_action = action
          state  = next_state
          action = next_action
          if len(self.replay_buffer) > self.replay_initial:
            loss = compute_td_loss(batch_size, app_path_prefix)
          if frame_num % 1000 == 0 or done:
            self.update_target(self.current_model, self.target_model)
          if done:
            # self.all_rewards.append(self.total_reward)
            self.current_model.save_state(self.BEST_MODEL_PATH)
            print("completed run")

        # close the pointer to that file
        filehandle.close()
    
    def process_frame(self, next_state):
        epsilon_start = 1.0
        epsilon_final = 0.01
        epsilon_decay = 30000
        epsilon_by_frame = lambda frame_idx : epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)
        epsilon = epsilon_by_frame(self.frame_num)

        next_action = current_model.act(next_state, epsilon)
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
        reward, done = compute_reward(self.frame_num, self.action)
        self.total_reward += reward

        if self.action != "REWARD":
          self.replay_buffer.push(self.state, self.action, reward, next_state, done)
        else:
          self.replay_buffer.push(self.prev_state, self.prev_action, reward, self.state, done)

        if len(self.replay_buffer) > self.replay_initial:
            loss = compute_td_loss(batch_size, TTDQN_PATH_PREFIX)
            
        # if frame_num % 10000 == 0:
        #    plot(frame_num, self.all_rewards, self.losses)
    
        if frame_num % 1000 == 0 or phase != self.curr_phase or done:
           self.update_target(self.current_model, self.target_model)

        if done:
            self.prev_state = None
            self.state = None
            self.curr_phase = self.PRE_CUBE
            # self.all_rewards.append(self.total_reward)
            self.frame_num = 0
            self.current_model.save_state(self.BEST_MODEL_PATH)
            print("Episode is done; reset robot and cube")
        else:
            self.frame_num += 1
            self.prev_state = self.state 
            self.state = next_state 
            self.prev_action = self.action 
            self.action = next_action 
            
        return next_action

# TODO: Allow states self.CUBE_OFF_TABLE, self.ROBOT_OFF_TABLE, REWARD, PENALTY 
# add syntax to call DDQN, PRETRAIN
