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
from config import *
import math, random
import os

class DatasetUtils():

    def __init__(self, app_name, app_type, nn_name=None):
        self.app_name = app_name
        self.app_type = app_type
        self.cfg = Config()
        # Note: for non-TT apps, STOP is a legitimate (non)action.  For example, you might
        # be waiting for an action by another robot to happen.  For object-tracking, you 
        # might need to stop if the object stops.
        self.processed_datasets = []
        self.processed_dataset_idx = None
        if nn_name is not None:
          self.last_ds_idx = self.last_dataset_idx_processed(app_type, nn_name)
        else:
          self.last_ds_idx = self.last_dataset_idx_processed(app_type, app_name)

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
          if mode is None or self.app_name is None:
              print("dataset_index_path mode,app_name:", mode, self.app_name)
          ds_idx_path = self.cfg.APP_DIR + self.app_name + "_" + mode + self.cfg.DATASET_IDX_DIR 
        elif mode == "FUNC":
          if mode is None or self.app_name is None:
              print("dataset_index_path mode,nn_name:", mode, nn_name)
          ds_idx_path = self.cfg.APP_DIR + mode + "/" + nn_name + self.cfg.DATASET_IDX_DIR 
        else:
          print("dataset_index_path: unknown mode", mode)
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
        ds_idx_pth = self.dataset_index_path(mode, nn_name) 
        if mode == "FUNC":
          # ds_idx_nm = "FUNC_" + nn_name + "_IDX_" 
          ds_idx_nm = "FUNC_" + nn_name  + "_"
        elif mode in ["APP","DQN"]:
          # ds_idx_nm = self.app_name + "_" + mode + "_IDX_" 
          ds_idx_nm = self.app_name + "_" + mode + "_IDX_"
        if position == "NEXT":
          full_ds_idx = self.next_dataset_idx(ds_idx_pth, ds_idx_nm, mode, nn_name)
        elif position == "NEW":
          full_ds_idx = self.new_dataset_idx_name(ds_idx_pth, ds_idx_nm, mode, nn_name)
        elif position == "LAST_PROCESSED":
          return self.last_ds_idx
        elif position in ["NEWEST","OLDEST","RANDOM"]:
          full_ds_idx = self.oldest_newest_random_dataset_idx_name(ds_idx_pth, ds_idx_nm, mode, nn_name, position)
        else:
          print("Incorrect position option:", position, ds_idx_pth, ds_idx_nm)
        self.last_ds_idx = full_ds_idx
        return full_ds_idx

    def next_dataset_idx(self, ds_idx_pth, ds_idx_nm, mode="DQN", nn_name=None):
        # Sort the list in ascending order of dates
        idx_list = os.listdir(ds_idx_pth)
        idx_list.sort()
        lastdataset = self.last_dataset_idx_processed(mode, nn_name)
        today = datetime.now().strftime("%d_%m_%y")
        # initialize 
        next_dataset = ds_idx_nm + today + "a.txt"
        print("lastdataset:", lastdataset)
        if lastdataset == "DUMMY_IDX":
          i = 0
          while True and i < len(idx_list):
            next_dataset = idx_list[i]
            # confirm proper format for dataset index name
            if next_dataset[0:len(ds_idx_nm)] == ds_idx_nm and len(next_dataset) == len(ds_idx_nm) + len("YY_MM_DDa.txt"):
              print("next_dataset:", next_dataset)
              break
            i = i+1
        elif lastdataset is not None and len(lastdataset) > 0:
          print("idx_list:", idx_list)
          lastdatasetname = self.get_filename_from_full_path(lastdataset)
          print("lastdatasetname:", lastdatasetname)
          i = idx_list.index(lastdatasetname) # else value index error
          # print("i = ", i)
          if i+1 >= len(idx_list):
            print("all datasets processed", lastdatasetname)
            return None
          else:
            while True:
              i = i+1
              if i < len(idx_list):
                next_dataset = idx_list[i]
                # confirm proper format for dataset index name
                if next_dataset[0:len(ds_idx_nm)] == ds_idx_nm and len(next_dataset) == len(ds_idx_nm) + len("YY_MM_DDa.txt"):
                  print("next_dataset:", next_dataset)
                  break
              else:
                return None
        else:
          # should be there unless explicitly skipped by command line option
          # if not init:
          #   print("last data set processed not found")
          #   return None
          found = False
          print("idx_list:", idx_list, ds_idx_nm)
          for idx in idx_list:
            if idx[0:len(ds_idx_nm)] == ds_idx_nm and len(idx) == len(ds_idx_nm) + len("YY_MM_DDa.txt"):
              next_dataset = idx
              print(len(idx), len(ds_idx_nm), len("YY_MM_DDa.txt"))
              print("ds_idx_nm:", ds_idx_nm)
              print("idx:", idx)
              print("next_dataset2:", next_dataset)
              found = True
              break
          if not found:
            print("No datasets")
            return None
        full_path = ds_idx_pth + next_dataset
        print("full_path:", full_path)
        return full_path

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
        today = now.strftime("%y_%m_%d")
        letter = "a"
        name = ds_idx_nm + today + letter + ".txt"
        lower_done = False
        while name in idx_list:
          print("duplicate name: ", name)
          if letter == "z":
            letter = "A"
          elif letter == "Z":
            letter = "1"
          else:
            # letter = chr(ord(letter.upper())+1)  # lower to upper
            letter = chr(ord(letter)+1)
          name = ds_idx_nm + today + letter + ".txt"
        print("New name: ", name)
        full_path = ds_idx_pth + name
        return full_path

    def oldest_newest_random_dataset_idx_name(self, ds_idx_pth, ds_idx_nm, mode, nn_name, position):
        idx_list = os.listdir(ds_idx_pth)
        idx_list.sort()
        oldest = None
        newest = None
        for i, idx in enumerate(idx_list):
          if idx.startswith(ds_idx_nm) and idx.endswith(".txt") and len(idx) == len(ds_idx_nm) + len("YY_MM_DDa.txt"):
            if oldest is None:
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
        print("selection of nn idx:", position, nn_name, full_path)
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
          ds_idx_p = self.dataset_index_path(mode=mode, nn_name=nn_name) + mode + "_" + nn_name + "_IDX_PROCESSED.txt"
        elif mode == "APP":
          ds_idx_p = self.dataset_index_path(mode=mode, nn_name=nn_name) + self.app_name + "_" + mode + "_IDX_PROCESSED_BY_DQN.txt"
        elif mode == "DQN":
          ds_idx_p = self.dataset_index_path(mode=mode, nn_name=nn_name) + self.app_name + "_" + mode + "_IDX_PROCESSED.txt"
        return ds_idx_p

    def last_dataset_idx_processed(self, mode="DQN", nn_name=None):
        filename = self.dataset_idx_processed(mode,nn_name)
        try:
          with open(filename, 'r') as file:
            line = file.readline()
        except:
          # first_processed = self.dataset_indices(mode, nn_name, position="OLDEST")
          first_processed = "DUMMY_IDX"
          return first_processed

        # verify line
        if line:
          [time, app, mode, nn_name, action, img_name, img_path] = self.get_dataset_info(line, mode=mode)
        return line

    # save last dataset_index that has been processed for an App 
    def save_dataset_idx_processed(self, mode="DQN", nn_name=None, clear=False):
        filename = self.dataset_idx_processed(mode,nn_name)
        last_processed = self.dataset_indices(mode=mode, nn_name=nn_name, position="LAST_PROCESSED")
        if last_processed is None or last_processed == "DUMMY_IDX":
          print("save_dataset_idx_processed: nothing to save")
          return
        with open(filename, 'w+') as file:
          if clear:
            file.truncate()
          else:
            file.write( last_processed )


    ################################
    # NN or DQN store image names.  
    ################################
    def dataset_line(self, img_nm, tm = None):
        if tm is None:
          tm = datetime.now().strftime("%H:%M:%S")
        line = tm + " " + img_nm
        return line

    def get_filename_from_full_path(self, full_path):
        dataset_info = full_path.split("/")
        return dataset_info[-1]

    def get_dataset_info(self, ds_line, mode="DQN"):
        #  "18:31:28 ./apps/TT_DQN/dataset/LOWER_ARM_DOWN/a1099b28-4334-11eb-8cce-3413e860d1ff.jpg"
        dataset_info = ds_line.split()
        if len(dataset_info) != 2:
          print("get_dataset_info:",len(dataset_info), dataset_info)
          return [None, None, None, None, None, None, None]
        ds_line_time = dataset_info[0]
        full_img_path = dataset_info[1] 
        dataset_info = full_img_path.split("/")
        if not ((len(dataset_info) == 7 and mode == "APP") or
                (len(dataset_info) == 6 and mode == "DQN") or
                (len(dataset_info) == 8 and mode == "FUNC")):
          print("get_dataset_info error:",mode, len(dataset_info), dataset_info)
          return undefined_var
          return [None, None, None, None, None, None, None]
        app_mode = dataset_info[2]
        if app_mode.endswith("APP"):
          # ./apps/TT_APP/dataset/PICK_UP_CUBE/LOWER_ARM_DOWN/a1099b28-4334-11eb-8cce-3413e860d1ff.jpg"
          ds_line_app = app_mode[:-4]
          # print("ds_line_app", ds_line_app)
          ds_line_mode = "APP"
          ds_line_nn = dataset_info[4]
          ds_line_action = dataset_info[5]
          ds_line_img = dataset_info[6]
        elif app_mode.endswith("DQN"):
          # ./apps/TT_DQN/dataset/LOWER_ARM_DOWN/a1099b28-4334-11eb-8cce-3413e860d1ff.jpg"
          ds_line_app = app_mode[:-4]
          # print("ds_line_app", ds_line_app)
          ds_line_mode = "DQN"
          ds_line_nn = None
          ds_line_action = dataset_info[4]
          ds_line_img = dataset_info[5]
        elif app_mode.endswith("FUNC"):
          # ./apps/FUNC/PICK_UP_CUBE/dataset/PICK_UP_CUBE/LOWER_ARM_DOWN/a1099b28-4334-11eb-8cce-3413e860d1ff.jpg"
          # print("dataset_info:", dataset_info)
          ds_line_app = None
          # print("ds_line_app", ds_line_app)
          ds_line_mode = "FUNC"
          ds_line_nn = dataset_info[5]
          ds_line_action = dataset_info[6]
          ds_line_img = dataset_info[7]
        else:
          print("app_mode", app_mode)
        return [ds_line_time, ds_line_app, ds_line_mode, ds_line_nn, ds_line_action, ds_line_img, full_img_path]

    # return a tuple of new images since last dataset training
    def get_dataset_images(self, mode="DQN", nn_name=None, position="NEXT"):
        # open the file for reading
        full_path = self.dataset_indices(mode,nn_name,position)
        if full_path is None:
          return [], None
        # full_path = self.dataset_idx_processed(mode,nn_name)
        dataset_idx_file = full_path
        print("idx_proc: ", full_path)
        filehandle = open(full_path, 'r')
        # new_image = ["DUMMY.jpg"]
        new_image = []
        if mode in ["DQN", "APP"]:
          while True:
            nn_line = filehandle.readline()
            if not nn_line:
                break
            [time, app, mode, nn_name, action, img_name, img_path] = self.get_dataset_info(nn_line,mode=mode)
            new_image.append(image_path)
        elif mode == "FUNC":
          while True:
            nn_idx = filehandle.readline()
            if not nn_idx:
              print("nn_idx done")
              break
            # remove time plus carriage return
            # nn_full_path = nn_idx[len("18:32:23 "):-1]
            # nn_filehandle = open(nn_full_path, 'r')
            # idx_list.append(nn_full_path)
            # print("nn_full_path:",nn_full_path)
            [time, app, mode, nn_name, action, img_name, img_path] = self.get_dataset_info(nn_idx, mode="FUNC")
            # print("imginfo:",time, app, mode, nn_name, action, img_name, img_path)
            new_image.append(img_path)

#            while True:
#              # jpg
#              nn_line = nn_filehandle.readline()
#              if not nn_line:
#                  print("nn_line done")
#                  break
#              [time, app, mode, nn_name, action, img_name, img_path] = self.get_dataset_info(nn_line)
#              print("img:",img_name)
#              new_image.append(image_path)
          print("dataset_idx_file:", dataset_idx_file)
          print("len new_image:", len(new_image))

        # close the pointer to that file
        filehandle.close()
        new_image = tuple(new_image)
        return new_image, dataset_idx_file

    def all_indices_processed(self, mode="DQN", nn_name=None):
        filename = self.dataset_idx_processed(mode,nn_name)
        last_processed = self.dataset_indices(mode=mode, nn_name=nn_name, position="LAST_PROCESSED")
        if last_processed is None:
          print("save_dataset_idx_processed: all processed")
          return True
        try:
          with open(filename, 'r') as file:
            prev_processed = file.readline()
            print("idx:", filename)
        except:
          prev_processed = None
        if last_processed == "DUMMY_IDX":
          last_processed = self.dataset_indices(mode=mode, nn_name=nn_name, position="NEXT")
          print("all_indx_processed1:", last_processed)
          if last_processed is not None:
            return False
          else:
            return True
        elif last_processed == prev_processed:
          next_processed = self.dataset_indices(mode=mode, nn_name=nn_name, position="NEXT")
          print("all_indx_processed2:", last_processed, next_processed)
          if next_processed is not None:
            return False
          else:
            return True
        else:
          print("all_indx_processed3:", last_processed, prev_processed)
          return False

    def dataset_images_processed(self, mode="DQN", nn_name=None):
        self.save_dataset_idx_processed(mode, nn_name)

    def remove_dataset_images(self, mode="DQN", nn_name=None, position="NEWEST", do_remove=False):
        if position not in ["OLDEST", "NEWEST"]:
          print("incorrect position specified: ", position)
          exit()
        # open the file for reading
        new_imgs,idx_lst = self.get_dataset_images(mode,nn_name,position=position)

        for img in new_imgs:
          if do_remove:
            os.remove(img) 
            print("removed image: ",img) 
          else:
            print("virtually removed image: ",img) 
        if do_remove:
            os.remove(idx_lst)
            print("removed index: ",idx_lst)
        else:
            print("virtually removed image: ",img) 

    ##################################################
    ##  a single NN trained across all SEARCH_FOR_CUBE runs.
    ##  dump of the DQN model.
    def best_model_path(self, mode="DQN", nn_name=None):
        if mode not in ["DQN", "APP", "FUNC"]:
          print("Mode must be one of DQN, NN, FUNC. Received: ", mode)
          exit()
        if mode == "FUNC":
          bmp = self.cfg.APP_DIR + mode + "/" + nn_name + "/"
        else:
          bmp = self.cfg.APP_DIR + self.app_name + "_" + mode + "/" 
        return bmp

    def best_model(self, mode="DQN", nn_name=None):
        if mode in ["DQN", "APP"]:
          bm = self.best_model_path(mode, nn_name) + self.app_name + "_" + mode + self.cfg.MODEL_POST_FIX 
        else:
          # nn_name assumed to already been validated
          bm = self.best_model_path(mode, nn_name) + "FUNC_" + nn_name + self.cfg.MODEL_POST_FIX 
        return bm

    # FUNC_PARK_ARM_HIGH_20_12_20a.txt -> PARK_ARM_HIGH
    def dataset_idx_to_func(self, ds_idx):
        nn_name = ds_idx[len("FUNC_"): -len("_YY_MM_DDa.txt")]
        print("dataset_idx_to_func:", ds_idx, nn_name)
        return nn_name

    # ./apps/NN/dataset/FUNC_DRIVE_TO_CUBE
    # ./apps/TT_DQN/dataset
    def dataset_path(self, mode="DQN", nn_name=None):
        if mode == "FUNC":
          dir_pth = self.best_model_path(mode=mode, nn_name=nn_name)
          ds_idx_path = dir_pth[:-1] + self.cfg.DATASET_PATH + nn_name + "/"
        elif mode in ["APP", "DQN"]:
          dir_pth = self.best_model_path(mode=mode)
          # extra / after dir_pth ?  ... still works
          ds_idx_path = dir_pth[:-1] + self.cfg.DATASET_PATH 
        return ds_idx_path

    # ./apps/TT_DQN/dataset_indexes/TT_DQN_replay_buffer.data
    def dqn_replay_buffer(self):
        dir_pth = self.best_model_path(mode="DQN")
        dqn_replay_buff = dir_pth + self.app_name + "_DQN" + self.cfg.REPLAY_BUFFER
        return dqn_replay_buff

    # ./apps/TT_DQN/dataset_indexes/TT_DQN_NN_TRAINING_COMBOS.txt
    def dqn_nn_training_combos(self):
        dqn_nn_train_combo = self.dataset_path("DQN") + self.app_name + self.cfg.DQN_NN_COMBOS 
        return dqn_nn_train_combo

    def mkdirs(self, robot_dirs):
        for dir_name in robot_dirs:
          try:
              os.makedirs(dir_name)
              print("mkdir %s" % dir_name)
          except FileExistsError:
            # print('Directory already exists')
            pass

