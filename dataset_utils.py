#  All datasets are stored in the format of the least-common denominator (TT_FUNC). 
#  TT_FUNC stores from NN1 through NN8, with subdirectories for the actions.
#  Eight NNs are trained from the dataset in TT_FUNC/datasets/NN[1-8] and 
#  stored in TTFUNC_model[1-8].pth
#
#  TT_NN combines the actions from the 8 NNs together to form a single NN (TT_NN)
#  and stored in TT_NN_model1.pth
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

# Import the datetime module
from datetime import datetime
import os

class DatasetUtils():

    def __init__(self, app_nm):
        # 0123456789012345678901234567890123456789
        self.TTFUNC_DATASET_IDX_DIR = "./apps/TT_FUNC/dataset/dataset_indexes/"
        self.TTFUNC_DATASET_IDX_PROCESSED = "dataset_idx_processed.txt"
        self.TTFUNC_DATASET_IDX_PREFIX = "dataset_idx_%y_%m_%d"
        self.TTFUNC_PATH_PREFIX = './apps/TT_FUNC/dataset/'
        self.TTDQN_PATH_PREFIX = './apps/TT_DQN/dataset/'
        self.REPLAY_BUFFER_PATH = './apps/TT_DQN/dataset/replay_buffer.data'
        TTFUNC_MODEL_NUM_OFFSET = 33
        self.actions = ( "FORWARD", "REVERSE", "LEFT", "RIGHT", "LOWER_ARM_DOWN", "LOWER_ARM_UP", "UPPER_ARM_DOWN", "UPPER_ARM_UP", "GRIPPER_OPEN", "GRIPPER_CLOSE", "REWARD", "PENALTY", "ROBOT_OFF_TABLE_PENALTY", "CUBE_OFF_TABLE_REWARD", "NOOP")
        self.processed_datasets = []
        self.processed_dataset_idx = None

    def last_processed_datasets_idx(self, app_nm = "TT_DQN"):
        filename = self.TTFUNC_DATASET_IDX_DIR + self.TTFUNC_DATASET_IDX_PROCESSED
        filehandle = open(filename, 'r')
        while True:
          line = filehandle.readline()
          if not line:
              break
          dataset_info = line.split()
          if dataset_info[0] not in ["TT_FUNC", "TT_DQN", "TT_NN"]:
              print("incorrect app name:", processed_datasets[0], filename)
              return None
          self.processed_dataset_idx.append(dataset_info)
        filehandle.close()
        return self.processed_dataset_idx

    def next_dataset_idx(self, app_nm = "TT_DQN", init=False):
        # Sort the list in ascending order of dates
        path = self.TTFUNC_DATASET_IDX_DIR
        idx_list = os.listdir(path)
        idx_list.sort()
        lastdataset = self.last_processed_datasets_idx(app_nm)
        if lastdataset is not None:
          i = idx_list.index(lastdataset) # else value index error
          if i+1 >= len(idx_list):
            print("all datasets processed", lastdataset)
            return None, None
          else:
            next_dataset = idx_list[i+1,:]
        else:
          # should be there unless explicitly skipped by command line option
          if not init:
            print("last data set processed not found")
            return None, None
          found = False
          for ds_idx in idx_list:
            if ds_idx[0,len("dataset_idx_")] == "dataset_idx_" and len(ds_idx) == len("dataset_idx_YY_MM_DDa.txt"):
              next_dataset = ds_idx
              found = True
              break
          if not found:
            print("No datasets")
            return None, None
        full_path = self.TTFUNC_DATASET_IDX_DIR + next_dataset
        return full_path, next_dataset

    def new_dataset_idx_name(self, app_nm = "TT_DQN"):
        last_ds_idx = self.last_processed_datasets_idx(app_nm=app_nm)
        if last_ds_idx is None:
          last_ds_idx = "dataset_idx_"
        now = datetime.now()
        today = now.strftime("%d_%m_%y")
        prefix = last_ds_idx[0:len("dataset_idx_")] + today
        if last_ds_idx[0:len(prefix)] == prefix:
          letter = last_ds_idx[len(prefix)]
          next_letter = chr(ord(letter.upper())+1)
        else:
          next_letter = "a"
        ds_idx = prefix + next_letter + ".txt"
        return ds_idx

    def save_dataset_idx_processed(self, app_nm = "TT_DQN", dataset_idx = None):
        last_ds_idx = self.last_processed_datasets_idx(app_nm)
        if last_ds_idx is None:
          last_ds_idx = [[app_nm, dataset_idx]] 
        else:
          found = False
          for i in range(3):
            if last_ds_idx[i][0] == app_nm:
              last_ds_idx[i][1] = app_nm
              found = True
              break
          if not found:
            i = len(last_ds_idx)
            if i > 2:
              print("bad app_nm", app_nm)
              return False
            last_ds_idx.append([app_nm, dataset_idx]) 
        
          with open('stats.txt', 'w') as file:
            for i in len(last_ds_idx):
              line = last_ds_idx[i][0] + " " + last_ds_idx[i][1]
              file.writeline( line )

    def dataset_line(self, img_nm, tm = None):
        if tm is None:
          tm = datetime.now().strftime("%H:%M:%S")
        line = tm + " " + img_nm
        return line

    def get_dataset_info(self, ds_line):
        # format of DATASET_INDEX_FILE:
        #  "18:31:28 ./NN1/LOWER_ARM_DOWN/a1099b28-4334-11eb-8cce-3413e860d1ff.jpg"
        #   012345678901234567890
        # # fixed size offsets
        FILE_OFFSET   = 9
        NN_NUM_OFFSET = 13
        ACTION_OFFSET = 15
        NN_REAL_REWARD = [4, 8]  # pick_up_cube, drop_cube_in_box

        time = ds_line[0,8]
        state = ds_line[FILE_OFFSET,:]
        nn_num = int(ds_line[NN_NUM_OFFSET:NN_NUM_OFFSET+1])
        found = False
        for action in self.actions:
          line_action = ds_line[ACTION_OFFSET:ACTION_OFFSET+len(action)]
          if line_action == action:
            found = True
            break
        if not found:
          print("action not found: ", ds_line)
        return [time, state, action, nn_num]

    # for parsing TT_FUNC dataset
    def DQN_parse_dataset(self, filename, app_path_prefix):
        # open the file for reading
        full_path, ds_name = self.ds_util.next_dataset_idx(app_nm = "TT_DQN")
        filehandle = open(full_path, 'r')
        frame_num = 0
        reward = []
        line = filehandle.readline()
        while True:
          # read a single line
          next_line = filehandle.readline()
          if not next_line:
              break
          [tm, state, action, nn_num] = self.ds_util.get_dataset_info(line)
          [tm, next_state, next_action, nn_num] = self.ds_util.get_dataset_info(next_line)
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
          ##########################

        # close the pointer to that file
        filehandle.close()


