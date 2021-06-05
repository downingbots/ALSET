import time
import torchvision
import cv2
import numpy as np
from robot import *
from image_folder2 import *
from config import *
from sir_ddqn import *
import torch.nn.functional as F
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms



class SIRNN():
    def __init__(self,sir_robot,outputs, nn_name, app_type):
        self.robot = sir_robot
        self.app_name = None
        self.nn_name = nn_name
        self.app_name = nn_name
        self.app_type = app_type
        self.sir_dqn = None
        self.device = None
        self.model = None
        self.optimizer = None
        self.outputs = outputs   # outputs may be much smaller than full_action_set for a function
        self.num_outputs = len(outputs)
        # self.joystick_actions = ["REWARD","PENALTY"]
        self.dsu = DatasetUtils(self.app_name, self.app_type, self.nn_name)
        self.cfg = Config()
        self.automatic_mode = False
        self.auto_func = None

    def nn_init(self, gather_mode=False):
        self.model = torchvision.models.alexnet(pretrained=True)
        self.model.classifier[6] = torch.nn.Linear(self.model.classifier[6].in_features, self.num_outputs)
        model_path = self.dsu.best_model(mode=self.app_type, nn_name=self.nn_name)
        try:
          self.model.load_state_dict(torch.load(model_path))
          print("Loaded model state.")
        except:
          print("Starting from new Alexnet model.")
          torch.save(self.model.state_dict(), model_path)
        self.device = torch.device('cuda')
        self.model = self.model.to(self.device)
        robot_dirs = []
        self.nn_dir = self.dsu.dataset_path(mode=self.app_type, nn_name=self.nn_name)
        robot_dirs.append(self.nn_dir)
        for dir_name in self.cfg.full_action_set:
          robot_dirs.append(self.nn_dir + dir_name)
        self.dsu.mkdirs(robot_dirs)
        print("nn_init: " , robot_dirs)
        # Class can be used as a single-NN app that can do any action
        if self.is_automated_function():
          self.auto_func = AutomatedFuncs(self.robot)
          self.auto_func.set_automatic_function(self.nn_name)
        return self.is_automated_function(), self.cfg.full_action_set

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

    def nn_process_image(self, NN_name=None, image=None, reward_penalty=None):
        # self.nn_process_image_cnn(NN_name=None, image=None, reward_penalty=None)
        if image is None:
          print("NN process image None", NN_name, reward_penalty)
          return "NOOP"
        return self.nn_process_image_dqn(NN_name, image, reward_penalty)

    def nn_process_image_dqn(self, NN_name=None, image=None, reward_penalty=None):
        print("NN process image")
        if self.sir_dqn is None:
          self.sir_dqn = SIR_DDQN(initialize_model=False, do_train_model=False, app_name=NN_name, app_type="FUNC")
          self.sir_dqn.nn_init(gather_mode=False)
        return self.sir_dqn.nn_process_image(NN_name, image, reward_penalty)
        # NN_name=None, image=None, reward_penalty=None)


    def nn_process_image_cnn(self, NN_name=None, image=None, reward_penalty=None):
        # if image == None:
        if image is None:
          print("NN process image None")
          return
        print("NN process image")
        x = image
        x = self.preprocess(x)
        y = self.model(x)
    
        # we apply the `softmax` function to normalize the output vector 
        # so it sums to 1 (which makes it a probability distribution)
        y = F.softmax(y, dim=1)
    
        print(y.flatten) 
        max_prob = 0
        best_action = -1
        # for i in range(self.num_outputs):
        for i in range(len(self.cfg.full_action_set)):
            prob = float(y.flatten()[i])
            print("PROB", i, self.cfg.full_action_set[i], prob)
            if max_prob < prob:
                for j, name in enumerate(self.cfg.full_action_set):
                    # if name == self.outputs[i]:
                    if name == self.cfg.full_action_set[i]:
                        if (reward_penalty is not None or 
                            name not in ["REWARD1", "PENALTY1", "REWARD2", "PENALTY2"]):
                          max_prob = prob
                          best_action = j
                          break
                if best_action == -1:
                    print("invalid action " + self.cfg.full_action_set[i] + "not in " + self.cfg.full_action_set)
                    exit()
        action_name = self.cfg.full_action_set[best_action]
        return action_name
#        if action_name == "FORWARD":
#            print("NN FORWARD") 
#            self.robot.forward()
#        elif action_name == "REVERSE":
#            print("NN REVERSE") 
#            self.robot.reverse()
#        elif action_name == "LEFT":
#            print("NN LEFT") 
#            self.robot.left()
#        elif action_name == "RIGHT":
#            print("NN RIGHT") 
#            self.robot.right()
#            # self.robot.left() # for simplified tabletop NN 
#        elif action_name == "LOWER_ARM_DOWN":
#            print("NN LOWER ARM DOWN") 
#            self.robot.lower_arm("DOWN")
#        elif action_name == "LOWER_ARM_UP":
#            print("NN LOWER ARM UP") 
#            self.robot.lower_arm("UP")
#        elif action_name == "UPPER_ARM_DOWN":
#            print("NN UPPER ARM DOWN") 
#            self.robot.upper_arm("DOWN")
#        elif action_name == "UPPER_ARM_UP":
#            print("NN UPPER ARM UP") 
#            self.robot.upper_arm("UP")
#        elif action_name == "GRIPPER_CLOSE":
#            print("NN GRIPPER_CLOSE") 
#            self.robot.gripper("CLOSE")
#        elif action_name == "GRIPPER_OPEN":
#            print("NN GRIPPER_OPEN") 
#            self.robot.gripper("OPEN")
#        else:
#            print("Action:", best_action, max_prob, action_name) 
#            time.sleep(0.08)
#            self.robot.stop()
      
    def is_automated_function(self):
        try:
          idx1 = self.cfg.func_registry.index(self.nn_name)
          for [func,parent_func] in self.cfg.func_automated:
            if func == self.nn_name:
              print("is_automated_func True")
              return True
          print("is_automated_func False")
          return False
        except:
          print("is_automated_func False")
          return False

    def nn_set_automatic_mode(self, TF):
        if not self.is_automated_function:
          print("not an automated function: ", self.nn_name)
          exit()
        self.automatic_mode = TF

    def nn_automatic_mode(self):
        return self.automatic_mode

    def nn_automatic_action(self, NN_name, frame, feedback):
        if not self.is_automated_function():
          print("not an automated function: ", NN_name)
          exit()
        return self.auto_func.automatic_function(frame, feedback)

    def nn_upon_penalty(self, penalty):
        # goto teleop mode
        self.nn_set_automatic_mode(False)

    def nn_upon_reward(self, reward):
        exit()

    # train the NN to do imitation learning starting from pre-trained imagenet
    #
    # Train a single NN based upon all 8 of the functional NN from TT_FUNC.
    # Handle new data gathered for TT_FUNC.
    # Use a split of the data so that we know the test accuracy.
    # This test accuracy might help with determining whether there is enough
    # data for TT_DQN.
    def train(self, dataset_root_list=None, best_model_path=None, full_action_set=None, noop_remap=None, only_new_images=True):
        # self.train_cnn(dataset_root_list, best_model_path, full_action_set, noop_remap, only_new_images)
        self.train_dqn(dataset_root_list, best_model_path, full_action_set, noop_remap, only_new_images)

    def train_dqn(self, dataset_root_list=None, best_model_path=None, full_action_set=None, noop_remap=None, only_new_images=True):
        if dataset_root_list is None:
          # num_NN = self.cfg.func_registry.index(self.app_name)
          dataset_root_list = []
          dsp = self.dsu.dataset_path("FUNC", self.nn_name)
          dataset_root_list.append(dsp)
        for ds_root in dataset_root_list:
           func_name = self.dsu.get_func_name_from_full_path(ds_root)
           sir_dqn = SIR_DDQN(initialize_model=False, do_train_model=False, app_name=func_name, app_type="FUNC")
           sir_dqn.nn_init(gather_mode=False)
           sir_dqn.parse_func_dataset(func_name, False, "FUNC")
           # sir_dqn.parse_func_dataset(init=False, app_name=ds_root, app_mode="FUNC")
           #   def parse_func_dataset(self, init=False, app_name=None, app_mode="FUNC"):
           # TypeError: parse_func_dataset() got an unexpected keyword argument 'app_name'



    def train_cnn(self, dataset_root_list=None, best_model_path=None, full_action_set=None, noop_remap=None, only_new_images=True):
        # nn.py is a primitive NN that can be used by higher layers like tabletop_func_app.
        # Set default parameters if unset by caller
        if dataset_root_list is None:
          # num_NN = self.cfg.func_registry.index(self.app_name)
          dataset_root_list = []
          dsp = self.dsu.dataset_path("FUNC", self.nn_name)
          dataset_root_list.append(dsp)
        best_model = self.dsu.best_model("FUNC", self.nn_name)
        if full_action_set is None:
           full_action_set = self.cfg.full_action_set
        if noop_remap is None:
           pass

        #######################
        # idx = self.dsu.dataset_indices(mode="FUNC", nn_name=self.nn_name, position="NEXT")
        # idx_processed = self.dsu.dataset_idx_processed(mode="FUNC", nn_name=self.nn_name)
        #######################

        # if self.robot.initialize:
        # elif self.robot.train_new_data:

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
        # model = models.alexnet(pretrained=True)
        # model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, len(full_action_set))
        # self.model is already loaded from nn_init() = which is a prerequisite
        if self.model is None:
            print("nn_init() not called before training.")
            exit
        NUM_EPOCHS = 30
        # NUM_EPOCHS = 3   # for debugging
        best_accuracy = 0.0
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        for root in dataset_root_list:
            while True:
              print("nn train: ", root, best_model_path) 
              # dataset = datasets.ImageFolder2(
              dataset = ImageFolder2(
                  root,
                  self.nn_name,
                  self.app_type,
                  transforms.Compose([
                      transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                      transforms.Resize((224, 224)),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                  ]),
                  full_action_set = full_action_set,
                  remap_to_noop = noop_remap,
                  only_new_images = only_new_images
              )
              if len(dataset.imgs) == 0:
                if dataset.all_images_processed("FUNC", self.nn_name):
                  print("no more images")
                  break
                print("next index")
                dataset.save_images_processed("FUNC", self.nn_name)
                continue
              print("num dataset imgs:", len(dataset.imgs))
              print("len dataset     :", len(dataset))
              if len(dataset) > len(dataset.imgs):
                print("dataset[0]:")
                print(dataset[0])
                print("datasetimg:")
                print(dataset.imgs[0])
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
  
              # problem: we're currently training incrementally. So, dataset may not be big
              # enough to split into train/test.
              # However, we're starting on a pretrained alexnet model and trying to
              # specialize the model from there.  The original code that this was based
              # upon was training from scratch.
              if len(dataset) > 500:
                train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - 50, 50])
              elif (len(dataset) > 50):
                  # test on a subset of the dataset
                  train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - 50, 50])
                  train_dataset = dataset    # train on full incremental dataset
              else:
                  train_dataset = dataset    # train on full incremental dataset
                  test_dataset = dataset     # test on full incremental dataset

              print("len train_dataset:", len(train_dataset))
              print("len test_dataset :", len(test_dataset))

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
                      images = images.to(self.device)
                      labels = labels.to(self.device)
                      optimizer.zero_grad()
                      outputs = self.model(images)
                      loss = F.cross_entropy(outputs, labels)
                      loss.backward()
                      optimizer.step()
                  
                  # we may not want to do test counts. Take all the data and
                  # apply it here.
                  test_error_count = 0.0
                  for images, labels in iter(test_loader):
                      images = images.to(self.device)
                      labels = labels.to(self.device)
                      outputs = self.model(images)
                      test_error_count += float(torch.sum(torch.abs(labels - outputs.argmax(1))))
                  
                  test_accuracy = 1.0 - float(test_error_count) / float(len(test_dataset))
                  print('%d: %f' % (epoch, test_accuracy))
                  if test_accuracy > best_accuracy:
                      print("bm, bmp", best_model, best_model_path)
                      best_accuracy = test_accuracy
              torch.save(self.model.state_dict(), best_model)
              dataset.save_images_processed("FUNC", self.nn_name)

    # wipe_memory(self)
    def cleanup(self):
        del self.model
        self._optimizer_to(torch.device('cpu'))
        del self.optimizer
        gc.collect()
        torch.cuda.empty_cache()
        # you can check that memory was released using nvidia-smi

    def _optimizer_to(self, device):
      for param in self.optimizer.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(self.device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(self.device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(self.device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
