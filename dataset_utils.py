# Examples using TableTop (TT) app. The code is not specific to the TT app
# and uses definitions in Config.py
#
##############
##################################################
# FUNC mode:
#
# The lowest level of autonomous functionality is FUNC model
# You might train to SEARCH_FOR_CUBE, GOTO_CUBE, PARK_ARM_RETRACTED, etc.
#
# Dataset indices are stored in files like:
##  ./apps/FUNC/dataset_indexes/NN_SEARCH_FOR_CUBE_IDX_YY_MM_DDa.txt
#
# The dataset for each of the indices are stored in the following directory:
##  ./apps/FUNC/dataset
#
# The NN for the functionality is:
##  ./apps/FUNC/FUNC_MODEL.pth
#
##################################################
# APP
##################################################
# DQN
# 
#
##################################################
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
        self.nn_name = nn_name
        self.cfg = Config()
        # Note: for non-TT apps, STOP is a legitimate (non)action.  For example, you might
        # be waiting for an action by another robot to happen.  For object-tracking, you 
        # might need to stop if the object stops.
        self.processed_datasets = []
        self.processed_dataset_idx = None
        if nn_name is not None:
          self.last_ds_idx = self.last_dataset_idx_processed(app_type, nn_name)
        else:
          if app_type == "DQN":
            self.last_ds_idx = self.last_dataset_idx_processed("DQN", app_name)
            if self.last_ds_idx is None:
              self.last_ds_idx = self.last_dataset_idx_processed("APP", app_name)
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
        if mode in ["RAND"]:
          if self.app_name is None:
              print("dataset_index_path mode,app_name:", mode, self.app_name)
          ds_idx_path = self.cfg.APP_DIR + self.app_name + "_" + "APP" + self.cfg.DATASET_IDX_DIR
        elif mode in ["DQN", "APP"]:
          if self.app_name is None:
              print("dataset_index_path mode,app_name:", mode, self.app_name)
          ds_idx_path = self.cfg.APP_DIR + self.app_name + "_" + mode + self.cfg.DATASET_IDX_DIR 
        elif mode == "FUNC" or mode == "FPEN":
          if self.app_name is None:
              print("dataset_index_path mode,nn_name:", mode, nn_name)
          # print("dip: ", self.cfg.APP_DIR, mode, nn_name , self.cfg.DATASET_IDX_DIR)
          ds_idx_path = self.cfg.APP_DIR + "FUNC" + "/" + nn_name + self.cfg.DATASET_IDX_DIR 
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
        if mode in ["FUNC","FPEN"]:
          # ds_idx_nm = "FUNC_" + nn_name + "_IDX_" 
          ds_idx_nm = mode + "_" + nn_name  + "_"
        elif mode in ["APP","DQN","RAND"]:
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
        # sort indx with same dates with all lowercase, then all uppercase
        idx_list.sort(key=str.swapcase)
        # idx_list.sort()
        if mode == "DQN":
          # could be APP or DQN (or RAND?) Depends on ds_idx_pth.
          real_mode = self.get_mode_from_path(ds_idx_pth)
          # print("real_mode:", real_mode)
          # print("idx_list:", idx_list)
          lastdataset = self.last_dataset_idx_processed(real_mode, nn_name)
        else:
          lastdataset = self.last_dataset_idx_processed(mode, nn_name)
        # today = datetime.now().strftime("%d_%m_%y")
        today = datetime.now().strftime("%y_%m_%d")
        # initialize 
        next_dataset = ds_idx_nm + today + "a.txt"
        print("lastdataset:", lastdataset)
        if lastdataset == "DUMMY_IDX":
          i = 0
          while True and i < len(idx_list):
            next_dataset = idx_list[i]
            if (next_dataset.endswith("_DQN_IDX_PROCESSED.txt")
                or next_dataset.endswith("_IDX_PROCESSED_BY_DQN.txt")):
              i = i+1
              continue
            # confirm proper format for dataset index name
            if next_dataset[0:len(ds_idx_nm)] == ds_idx_nm and len(next_dataset) == len(ds_idx_nm) + len("YY_MM_DDa.txt"):
              print("next_dataset:", next_dataset)
              break
            i = i+1
        elif lastdataset is not None and len(lastdataset) > 0:
          # print("idx_list:", idx_list)
          lastdatasetname = self.get_filename_from_full_path(lastdataset)
          print("lastdatasetname:", lastdatasetname)
          try:
            i = idx_list.index(lastdatasetname) # else value index error
          except Exception as e:
            print("not in idx_list",e)
            return None
          # print("i = ", i)
          if i+1 >= len(idx_list):
            print("all datasets processed", lastdatasetname)
            return None
          else:
            while True:
              i = i+1
              if i < len(idx_list):
                next_dataset = idx_list[i]
                if (next_dataset.endswith("_DQN_IDX_PROCESSED.txt")
                    or next_dataset.endswith("_IDX_PROCESSED_BY_DQN.txt")):
                  continue
                # confirm proper format for dataset index name
                if next_dataset[0:len(ds_idx_nm)] == ds_idx_nm and len(next_dataset) == len(ds_idx_nm) + len("YY_MM_DDa.txt"):
                  print("next_dataset:", next_dataset)
                  break
              else:
                # maybe dataset_idx_processed(self, mode="DQN", nn_name=None)
                # if so, all done.
                print("all done?")
                return None
        else:
          # should be there unless explicitly skipped by command line option
          # if not init:
          #   print("last data set processed not found")
          #   return None
          found = False
          # print("idx_list:", idx_list, ds_idx_nm)
          for idx in idx_list:
            if (idx[0:len(ds_idx_nm)] == ds_idx_nm 
                and len(idx) == len(ds_idx_nm) + len("YY_MM_DDa.txt")
                and not idx.endswith("_DQN_IDX_PROCESSED.txt")):
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
        # find valid ds_idx_nm
        now = datetime.now()
        today = now.strftime("%y_%m_%d")
        letter = "a"
        name = ds_idx_nm + today + letter + ".txt"
#        i = 1
#        while True:
#          if len(idx_list) > 0:
#            last_ds_idx = idx_list[-i]
#            len_idx = len(last_ds_idx)
#          else:
#            last_ds_idx = None
#          if len(last_ds_idx) == len(name) and last_ds_idx.startswith(ds_idx_name):
#            break
#          i = i + 1
#        print("last ds idx", last_ds_idx)
        lower_done = False
        while name in idx_list:
          # print("duplicate name: ", name)
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
        ds_idx_p = None
        if mode == "FUNC":
          ds_idx_p = self.dataset_index_path(mode=mode, nn_name=nn_name) + mode + "_" + nn_name + "_IDX_PROCESSED.txt"
        elif mode == "FPEN":
          # ARD: FPEN isn't processed by FUNC mode; it's part of RAND dataset idx in APP mode.
          ds_idx_p = self.dataset_index_path(mode=mode, nn_name=nn_name) + mode + "_" + nn_name + "_IDX_PROCESSED.txt"
        elif mode == "RAND":
          ds_idx_p = self.dataset_index_path(mode=mode, nn_name=nn_name) + self.app_name + "_" + mode + "_IDX_PROCESSED_BY_DQN.txt"
        elif mode == "APP":
          ds_idx_p = self.dataset_index_path(mode=mode, nn_name=nn_name) + self.app_name + "_" + mode + "_IDX_PROCESSED_BY_DQN.txt"
        elif mode == "DQN":
          ds_idx_p = self.dataset_index_path(mode=mode, nn_name=nn_name) + self.app_name + "_" + mode + "_IDX_PROCESSED.txt"
        print("dataset_idx_processed: ", ds_idx_p)
        return ds_idx_p

    def last_dataset_idx_processed(self, mode="DQN", nn_name=None):
        filename = self.dataset_idx_processed(mode,nn_name)
        try:
          with open(filename, 'r') as file:
            last_ds_idx_processed = file.readline()
          last_ds_idx_processed = last_ds_idx_processed.strip()
          print("last_dataset_idx_processed: ", filename, last_ds_idx_processed, mode)
        except:
          # first_processed = self.dataset_indices(mode, nn_name, position="OLDEST")
          first_processed = "DUMMY_IDX"
          return first_processed

        # verify line
        # if line:
        #   [time, app, mode, nn_name, action, img_name, img_path] = self.get_dataset_info(line, mode=mode)
        # return line
        return last_ds_idx_processed

    # save last dataset_index that has been processed for an App 
    def save_dataset_idx_processed(self, mode="DQN", nn_name=None, clear=False, ds_idx=None):
        filename = self.dataset_idx_processed(mode,nn_name)
        if ds_idx is None:
          last_processed = self.dataset_indices(mode=mode, nn_name=nn_name, position="LAST_PROCESSED")
        else:
          last_processed = ds_idx
        if last_processed is None or last_processed == "DUMMY_IDX":
          print("save_dataset_idx_processed: nothing to save")
          return
        with open(filename, 'w+') as file:
          if clear:
            file.truncate()
            print("save_dataset_idx_processed: truncated")
          else:
            file.write( last_processed )
            print("save_dataset_idx_processed: ", filename, last_processed, mode)


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

    def get_func_name_from_full_path(self, full_path):
        dataset_info = full_path.split("/")
        return dataset_info[3]

    def get_func_name_from_idx(self, idx):
        ds_idx_info = idx.split("/")
        print("ds_idx_info", ds_idx_info)
        return ds_idx_info[3]

    def get_mode_from_path(self, full_path):
        dataset_info = full_path.split("/")
        mode = None
        if dataset_info[2].endswith("DQN"):
          mode = "DQN"
        elif dataset_info[2].endswith("APP"):
          mode = "APP"
        # print("ds_mode:", mode, dataset_info)
        return mode

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
                (len(dataset_info) == 7 and mode == "DQN") or
                (len(dataset_info) == 8 and mode == "FUNC") or
                (len(dataset_info) == 8 and mode == "RAND")):
          print("get_dataset_info error:",mode, len(dataset_info), dataset_info)
          return undefined_var
          return [None, None, None, None, None, None, None]
        app_mode = dataset_info[2]
        if app_mode.endswith("APP"):
          ds_line_app = app_mode[:-4]
          # print("ds_line_app", ds_line_app)
          ds_line_mode = "APP"
          ds_line_nn = dataset_info[4]
          ds_line_action = dataset_info[5]
          ds_line_img = dataset_info[6]
        elif app_mode.endswith("DQN"):
          ds_line_app = app_mode[:-4]
          # print("ds_line_app", ds_line_app)
          ds_line_mode = "DQN"
          ds_line_nn = None
          ds_line_action = dataset_info[5]
          ds_line_img = dataset_info[6]
        elif app_mode.endswith("RAND"):
          ds_line_app = app_mode[:-4]
          print("ds_line_app", ds_line_app)
          ds_line_mode = "RAND"
          ds_line_nn = dataset_info[4]
          ds_line_action = dataset_info[5]
          ds_line_img = dataset_info[6]
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
    def get_dataset_images(self, mode="DQN", nn_name=None, position="NEXT", exclude_FPEN=False):
        # open the file for reading
        while True:
          full_path = self.dataset_indices(mode,nn_name,position)
          fnm = self.get_filename_from_full_path(full_path)
          if exclude_FPEN and fnm.startswith("FPEN_"):
              print("excluding ", fnm)
              continue
          else:
              break
        if full_path is None:
          return [], None
        # full_path = self.dataset_idx_processed(mode,nn_name)
        dataset_idx_file = full_path
        print("idx_proc: ", full_path)
        filehandle = open(full_path, 'r')
        # new_image = ["DUMMY.jpg"]
        new_image = []
        if mode in ["DQN", "APP", "RAND"]:
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
        # no separate model for RAND. RAND is only used by DQN.
        if mode not in ["DQN", "APP", "FUNC"]:
          print("Mode must be one of DQN, NN, FUNC. Received: ", mode)
          exit()
        if mode == "FUNC":
          bmp = self.cfg.APP_DIR + mode + "/" + nn_name + "/"
        else:
          bmp = self.cfg.APP_DIR + self.app_name + "_" + mode + "/" 
        return bmp

    def best_model(self, mode="DQN", nn_name=None, classifier=False):
        if mode in ["DQN", "APP"]:
          bm = self.best_model_path(mode, nn_name) + self.app_name + "_" + mode + self.cfg.MODEL_POST_FIX 
        else:
          if classifier:
            # nn_name assumed to already been validated
            bm = self.best_model_path(mode, nn_name) + "FCLASS_" + nn_name + self.cfg.MODEL_POST_FIX 
          else:
            bm = self.best_model_path(mode, nn_name) + "FUNC_" + nn_name + self.cfg.MODEL_POST_FIX 
        return bm

    # FUNC_PARK_ARM_HIGH_20_12_20a.txt -> PARK_ARM_HIGH
    def dataset_idx_to_func(self, ds_idx):
        nn_name = ds_idx[len("FUNC_"): -len("_YY_MM_DDa.txt")]
        print("dataset_idx_to_func:", ds_idx, nn_name)
        return nn_name

    # ./apps/FUNC/dataset/FUNC_DRIVE_TO_CUBE
    # ./apps/TT_DQN/dataset
    def dataset_path(self, mode="DQN", nn_name=None, dqn_idx_name=None):
        if mode == "FUNC":
          dir_pth = self.best_model_path(mode=mode, nn_name=nn_name)
          ds_idx_path = dir_pth[:-1] + self.cfg.DATASET_PATH + nn_name + "/"
        elif mode in ["APP", "DQN"]:
          dir_pth = self.best_model_path(mode=mode)
          # extra / after dir_pth ?  ... still works
          ds_idx_path = dir_pth[:-1] + self.cfg.DATASET_PATH 
          if dqn_idx_name is not None:
            ds_idx_path = ds_idx_path + dqn_idx_name + "/"
        return ds_idx_path

    # ./apps/TT_DQN/dataset_indexes/TT_DQN_replay_buffer.data
    def dqn_replay_buffer(self):
        if self.app_type == "FUNC": 
          dir_pth = self.best_model_path(mode="FUNC", nn_name=self.app_name)
          dqn_replay_buff = dir_pth + self.app_name + "_DQN" + self.cfg.REPLAY_BUFFER
        else:
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

