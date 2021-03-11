# Examples using TableTop (TT) app. The code is not specific to the TT app
# and uses definitions in Config.py
#
##############
##################################################
# NN mode:
#
# ./apps/NN/NN_SEARCH_FOR_CUBE_MODEL.pth
##  a single NN trained across all SEARCH_FOR_CUBE runs.
#
# ./apps/NN/dataset/SEARCH_FOR_CUBE/LOWER_ARM_DOWN/a1099b28-4334-11eb-8cce-3413e860d1ff.jpg
##  the dataset repository for an atomic NN / autonomous action.
#
# ./apps/NN/dataset_indexes/SEARCH_FOR_CUBE_NN_IDX_YY_MM_DDa.txt
##  contains lines like (when run in TT_FUNC or NN modes only):
##  18:31:28 ./apps/NN/dataset/SEARCH_FOR_CUBE/LOWER_ARM_DOWN/a1099b28-4334-11eb-8cce-3413e860d1ff.jpg
#  
# ./apps/NN/dataset_indexes/NN_SEARCH_FOR_CUBE_IDX_PROCESSED.txt
##  ./apps/NN/dataset_indexes/NN_SEARCH_FOR_CUBE_IDX_YY_MM_DDa.txt
#
##################################################
# FUNC mode:
#
# ./apps/TT_FUNC/dataset_indexes/TT_FUNC_IDX.txt
##  This index is mainly used by DQN to train on end-to-end NN runs.
##  contains lines for each TT NN like (when run in TT_FUNC mode only):
##  1 1 ./apps/NN/dataset_indexes/NN_SEARCH_FOR_CUBE_IDX_YY_MM_DDa.txt
##  1 2 ./apps/NN/dataset_indexes/NN_DRIVE_TO_CUBE_IDX_YY_MM_DDa.txt
##  ...
##  1 8 ./apps/NN/dataset_indexes/NN_DROP_CUBE_IN_BOX_IDX_YY_MM_DDa.txt
##  2 1 ./apps/NN/dataset_indexes/NN_SEARCH_FOR_CUBE_IDX_YY_MM_DDb.txt
##  ...
##  2 8 ./apps/NN/dataset_indexes/NN_DROP_CUBE_IN_BOX_IDX_YY_MM_DDb.txt
#
# ./apps/TT_FUNC/dataset_indexes/TT_FUNC_IDX_PROCESSED_BY_DQN.txt
##  contains single line of the final most recent TT_FUNC run processed by DQN like:
##  2 8 ./apps/NN/dataset_indexes/NN_DROP_CUBE_IN_BOX_IDX_YY_MM_DDb.txt
#
# ./apps/TT_FUNC/TT_FUNC_MODEL.pth
##  a single NN trained across all 8 NNs.
#
# ./apps/TT_FUNC/dataset_indexes/TT_FUNC_IDX_PROCESSED.txt
##  contains 1 line for each NN used to train the TT_FUNC_MODEL.pth.
##  Contents not specific to a TT_FUNC run.  It can contain stand-alone NN runs:
##  ./apps/NN/dataset_indexes/NN_SEARCH_FOR_CUBE_IDX_YY_MM_DDa.txt
##  ./apps/NN/dataset_indexes/NN_DRIVE_TO_CUBE_IDX_YY_MM_DDc.txt
##  ...
##  ./apps/NN/dataset_indexes/NN_DROP_CUBE_IN_BOX_IDX_YY_MM_DDb.txt
#  
##################################################
# DQN
# 
# ./apps/TT_DQN/dataset/LOWER_ARM_DOWN/a1099b28-4334-11eb-8cce-3413e860d1ff.jpg
# DQN dataset as a result of a DQN run
# 
# ./apps/TT_DQN/dataset_indexes/TT_DQN_IDX_YY_MM_DDa.txt
##  contains lines like:
##  18:31:28 ./apps/TT_DQN/dataset/LOWER_ARM_DOWN/a1099b28-4334-11eb-8cce-3413e860d1ff.jpg
#
# ./apps/TT_DQN/dataset_indexes/TT_DQN_NN_TRAINING_COMBOS.txt
##  random sets of NN runs are played back in TT_FUNC order. Contains lines like:
##  ./apps/NN/dataset_indexes/NN_SEARCH_FOR_CUBE_IDX_YY_MM_DDa.txt
##  ./apps/NN/dataset_indexes/NN_DRIVE_TO_CUBE_IDX_YY_MM_DDa.txt
##  ...
#
# ./apps/TT_DQN/dataset_indexes/dataset_idx_processed.txt
##  contains single line containing the most recently processed DQN run like:
##  ./apps/TT_DQN/dataset_indexes/TT_DQN_IDX_YY_MM_DDa.txt
#
# ./apps/TT_DQN/TT_DQN_MODEL.pth
##  dump of the DQN model.
#
# ./apps/TT_DQN/replay_buffer.data
##  dump of the replay_buffer
#
##################################################
##############
#  All datasets are stored in the format of the least-common denominator (TT_NN). 
#  Each NN is trained from its dataset in TT_NN/datasets/<NN_name> and
#  stored in <NN_name>_NN_MODEL.pth
#
#  TT_FUNC combines the actions from the 8 NNs together to form a single NN (TT_FUNC)
#  and stored in <NN_name>_NN_model.pth
#
#  TT_DQN processes individual actions in the original order created across all 8 NN.
#  Afterwards, TT_DQN uses the replay buffer and the apps/TT_DQN/dataset/NN1
#  and stored in TT_NN_model1.pth
#
#  In the future, TT_DQN/dataset/NN1 could also be used to train TTDQN_model1.pth and TT_NN_m
#  TT_FUNC cannot be trained from TT_DQN files.
#
#  TT_FUNC Datasets are indexed by files: dataset_idx_%y_%m_%d[a-zA-Z].txt 
#  and contain data of the format:
#  18:31:31 ./NN1/LOWER_ARM_DOWN/a3258980-4334-11eb-8cce-3413e860d1ff.jpg
#
#  The datasets are processed in order of creation. Upon processing, the file
#  dataset_idx_processed is updated so the entries are only processed once.
#

# Import the datetime module
from datetime import datetime
from dataset_utils import *
from config import *
import os

class DatasetUtils():

    def __init__(self, app_name, app_type):
        self.app_name = app_name
        self.app_type = app_type
        self.cfg = Config()
        # Note: for non-TT apps, STOP is a legitimate (non)action.  For example, you might
        # be waiting for an action by another robot to happen.  For object-tracking, you 
        # might need to stop if the object stops.
        self.processed_datasets = []
        self.processed_dataset_idx = None

    ################################
    # Dataset Index
    # contains a list of files for a full run at NN, FUNC, and/or DQN level
    # Name contains date of the run and a letter for unique ordering  
    ################################
    # ./apps/NN/dataset_index/
    # ./apps/TT_FUNC/dataset_index/
    # ./apps/TT_DQN/dataset_index/
    def dataset_index_path(self, mode="DQN", nn_name=None):
        if mode in ["DQN", "APP"]:
          ds_idx_path = self.cfg.APP_DIR + self.app_name + "_" + mode + self.cfg.DATASET_IDX_DIR 
        elif mode == "FUNC":
          ds_idx_path = self.cfg.APP_DIR + mode + self.cfg.DATASET_IDX_DIR 
        return ds_idx_path

    # ./apps/NN/dataset_indexes/NN_SEARCH_FOR_CUBE_IDX_YY_MM_DDa.txt
    ## contains multiple images, in order obtained, across actions while gathering data for function
    # ./apps/TT_FUNC/dataset_indexes/TT_FUNC_IDX_YY_MM_DDa.txt
    ## contains multiple NN_SEARCH_FOR_CUBE_IDX_YY_MM_DDa.tt in order obtained
    ##  1 1 ./apps/NN/dataset_indexes/NN_SEARCH_FOR_CUBE_IDX_YY_MM_DDa.txt
    # ./apps/TT_DQN/dataset_indexes/TT_DQN_IDX_YY_MM_DDa.txt
    ## contains multiple images, in order obtained, across actions
    # position in ["NEW","NEXT","OLDEST", "NEWEST", "RANDOM"]
    def dataset_indices(self, mode="DQN", nn_name=None, position="RANDOM"):
        ds_idx_pth = self.dataset_index_path() 
        if mode == "FUNC":
          ds_idx_nm = "FUNC_" + nn_name + "_IDX_" 
        elif mode in ["APP","DQN"]:
          ds_idx_nm = self.app_name + "_" + mode + "_IDX_" 
        if position == "NEXT":
          full_ds_idx = self.next_processed_dataset_idx(ds_idx_pth, ds_idx_nm, mode, nn_name)
        elif position == "NEW":
          full_ds_idx = self.new_dataset_idx_name(ds_idx_pth, ds_idx_nm, mode, nn_name)
        elif position in ["NEWEST","OLDEST","RANDOM"]:
          full_ds_idx = self.oldest_newest_random_processed_dataset_idx(ds_idx_pth, ds_idx_nm, mode, nn_name, position)
        else:
          print("Incorrect position option:", position, ds_idx_pth, ds_idx_nm)
        return full_ds_idx

    def next_dataset_idx(self, ds_idx_pth, ds_idx_nm, mode="DQN", nn_name=None):
        # Sort the list in ascending order of dates
        idx_list = os.listdir(ds_idx_pth)
        idx_list.sort()
        lastdataset = self.last_dataset_idx_processed(mode, nn_name)
        if lastdataset is not None:
          i = idx_list.index(lastdataset) # else value index error
          if i+1 >= len(idx_list):
            print("all datasets processed", lastdataset)
            return None
          else:
            next_dataset = idx_list[i+1,:]
        else:
          # should be there unless explicitly skipped by command line option
          if not init:
            print("last data set processed not found")
            return None
          found = False
          for idx in idx_list:
            if idx[0:len(ds_idx)] == ds_idx and len(idx) == len(ds_idx) + len("YY_MM_DDa.txt"):
              next_dataset = idx
              found = True
              break
          if not found:
            print("No datasets")
            return None
        full_path = ds_idx_pth + next_dataset
        return next_dataset

    def new_dataset_idx_name(self, ds_idx_pth, ds_idx_nm, mode, nn_name):
        try:
          os.makedirs(ds_idx_pth)
          print("mkdir %s" % ds_idx_pth)
        except FileExistsError:
          # print('Directory already exists')
          pass

        idx_list = os.listdir(ds_idx_pth)
        idx_list.sort()
        if len(idx_list) > 0:
          last_ds_idx = idx_list[-1]
          len_idx = len(last_ds_idx)
        else:
          last_ds_idx = None
        print("last ds idx", last_ds_idx)
        now = datetime.now()
        today = now.strftime("%d_%m_%y")
        letter = "a"
        name = ds_idx_nm + today + letter + ".txt"
        while name in idx_list:
          letter = chr(ord(letter.upper())+1)
          name = ds_idx_nm + today + letter + ".txt"
          print("duplicate name: ", name)
        print("New name: ", name)
        full_path = ds_idx_pth + name
        return full_path

    def oldest_newest_random_dataset_idx_name(self, ds_idx_pth, ds_idx_nm, mode, nn_name, position):
        idx_list = os.listdir(ds_idx_pth)
        idx_list.sort()
        oldest = None
        newest = None
        for i, idx in enumerate(idx_list):
          if idx.startswith(ds_idx_nm) and idx_list.endswith(".txt"):
            if start is None:
              oldest = i
            newest = i
        if oldest is None:
          return None
        if position == "RANDOM":
          idx_num = random.randint(oldest,newest)
        elif position == "OLDEST":
          idx_num = oldest
        elif position == "NEWEST":
          idx_num = newest
        else:
          print("incorrect position specified: ", position)
          exit()
        full_path = ds_idx_pth + idx_list[idx_num]
        print("random selection of nn idx:", nn_name, full_path)
        return full_path

    ################################
    # Dataset Index Processed
    # Fixed name. 1 file for each of NN nn_name, FUNC, or DQN.
    # contains a single files with the name of the last Dataset Index that was processed
    ################################
    # ./apps/TT_DQN/dataset_indexes/dataset_idx_processed.txt
    ##  contains single line containing the most recently processed DQN run like:
    ##  ./apps/TT_DQN/dataset_indexes/TT_DQN_IDX_YY_MM_DDa.txt
    def dataset_idx_processed(self, mode="DQN", nn_name=None):
        if mode == "FUNC":
          ds_idx_p = self.dataset_index_path() + mode + "_" + nn_name + "_IDX_PROCESSED.txt"
        elif mode == "APP":
          ds_idx_p = self.dataset_index_path() + self.app_name + "_" + mode + "_IDX_PROCESSED_BY_DQN.txt"
        elif mode == "DQN":
          ds_idx_p = self.dataset_index_path() + self.app_name + "_" + mode + "_IDX_PROCESSED.txt"
        return ds_idx_p

    def last_dataset_idx_processed(self, mode="DQN", nn_name=None):
        filename = self.dataset_idx_processed(mode,nn_name)
        filehandle = open(full_path, 'r')
        frame_num = 0
        line = filehandle.readline()
        # verify line
        [time, app, mode, nn_name, action, img_name, img_path] = self.ds_util.get_dataset_info(line)
        return line

    # save last dataset_index that has been processed for an App 
    def save_dataset_idx_processed(self, mode = "DQN", nn_name = None, dataset_idx = None):
        filename = self.dataset_indices(mode=mode, nn_name=nn_name, position="LAST")
        with open(filename, 'w') as file:
          file.write( dataset_idx )


    ################################
    # NN or DQN store image names.  
    ################################
    def dataset_line(self, img_nm, tm = None):
        if tm is None:
          tm = datetime.now().strftime("%H:%M:%S")
        line = tm + " " + img_nm
        return line

    def get_dataset_info(self, ds_line, mode="DQN"):
        #  "18:31:28 ./apps/TT_DQN/dataset/LOWER_ARM_DOWN/a1099b28-4334-11eb-8cce-3413e860d1ff.jpg"
        dataset_info = ds_line.split()
        ds_line_time = dataset_info[0]
        full_img_path = dataset_info[1] 
        dataset_info = full_img_path.split("/")
        app_mode = dataset_info[1]
        if app_mode.endswith("APP"):
          # ./apps/NN/dataset/PICK_UP_CUBE/LOWER_ARM_DOWN/a1099b28-4334-11eb-8cce-3413e860d1ff.jpg"
          ds_line_mode = "APP"
          ds_line_nn = dataset_info[3]
          ds_line_action = dataset_info[4]
          ds_line_img = dataset_info[5]
        elif app_mode.endswith("DQN"):
          # ./apps/TT_DQN/dataset/LOWER_ARM_DOWN/a1099b28-4334-11eb-8cce-3413e860d1ff.jpg"
          ds_line_app = app_mode[:-4]
          ds_line_mode = "DQN"
          ds_line_nn = None
          ds_line_action = dataset_info[4]
          ds_line_img = dataset_info[5]
        elif app_mode.endswith("FUNC"):
          ds_line_app = app_mode[:-5]
          ds_line_mode = "FUNC"
          ds_line_nn = None
          ds_line_action = dataset_info[4]
          ds_line_img = dataset_info[5]
        return [ds_line_time, ds_line_app, ds_line_mode, ds_line_nn, ds_line_action, ds_line_img, full_img_path]

    # return a tuple of new images since last dataset training
    def get_dataset_images(self, mode="DQN", nn_name=None, position="NEXT"):
        # open the file for reading
        idx_list = []
        full_path = self.ds_util.dataset_indices(mode,nn_name,position)
        idx_list.append(full_path)
        filehandle = open(full_path, 'r')
        new_image = []
        if mode in ["DQN", "APP"]:
          while True:
            nn_line = filehandle.readline()
            if not nn_line:
                break
            [time, app, mode, nn_name, action, img_name, img_path] = self.ds_util.get_dataset_info(nn_line)
            new_image.append(image_path)
        elif mode == "FUNC":
          while True:
            nn_idx = filehandle.readline()
            if not nn_idx:
              break
            nn_filehandle = open(nn_idx, 'r')
            idx_lst.append(nn_idx)
            while True:
              nn_line = nn_filehandle.readline()
              if not nn_line:
                  break
              [time, app, mode, nn_name, action, img_name, img_path] = self.ds_util.get_dataset_info(nn_line)
              new_image.append(image_path)

        # close the pointer to that file
        filehandle.close()
        new_image = tuple(new_image)
        return new_image, indx_lst

    def remove_dataset_images(self, mode="DQN", nn_name=None, position="NEWEST"):
        if position not in ["OLDEST", "NEWEST"]:
          print("incorrect position specified: ", position)
          exit()
        # open the file for reading
        new_imgs,idx_lst = self.ds_util.get_dataset_indices(mode,nn_name,position=position)

        for img in len(new_imgs):
            # os.remove(img) 
            print("virtual remove image: ",img) 
        for idx in idx_lst:
          # os.remove(idx)
          print("virtual remove index: ",idx)

    ##################################################
    ##  a single NN trained across all SEARCH_FOR_CUBE runs.
    ##  dump of the DQN model.
    def best_model_path(self, mode="DQN", nn_name=None):
        if mode not in ["DQN", "APP", "FUNC"]:
          print("Mode must be one of DQN, NN, FUNC. Received: ", mode)
          exit()
        if mode == "FUNC":
          bmp = self.cfg.APP_DIR + mode + "/" 
        else:
          bmp = self.cfg.APP_DIR + self.app_name + "_" + mode + "/" 
        return bmp

    def best_model(self, mode="DQN", nn_name=None):
        if mode in ["DQN", "APP"]:
          bm = self.best_model_path(mode, nn_name) + self.app_name + "_" + mode + "_" + self.cfg.MODEL_POST_FIX 
        else:
          # nn_name assumed to already been validated
          bm = self.best_model_path(mode, nn_name) + "FUNC_" + nn_name + self.cfg.MODEL_POST_FIX 
        return bm

    # ./apps/NN/dataset/FUNC_DRIVE_TO_CUBE
    # ./apps/TT_DQN/dataset
    def dataset_path(self, mode="DQN", nn_name=None):
        if mode == "FUNC":
          ds_idx_path = self.cfg.APP_DIR + mode + self.cfg.DATASET_PATH + nn_name + "/"
        elif mode in ["APP", "DQN"]:
          ds_idx_path = self.cfg.APP_DIR + self.app_name + "_" + mode + self.cfg.DATASET_PATH 
        return ds_idx_path

    # ./apps/TT_DQN/dataset_indexes/TT_DQN_replay_buffer.data
    def dqn_replay_buffer(self):
        dqn_replay_buff = self.cfg.APP_DIR + self.app_name + "_DQN" + self.cfg.REPLAY_BUFFER
        return dqn_replay_buff

    # ./apps/TT_DQN/dataset_indexes/TT_DQN_NN_TRAINING_COMBOS.txt
    def dqn_nn_training_combos(self):
        dqn_nn_train_combo = self.dqn_dataset_path() + self.app_name + self.cfg.DQN_NN_COMBOS 
        return dqn_nn_train_combo

