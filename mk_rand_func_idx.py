#!/usr/bin/python
import sys
from robot import *
from functional_app import *
from dataset_utils import *
from config import *
import shutil

print("Number of arguments:", len(sys.argv), "arguments.")
print("Argument List:", str(sys.argv))
if str(sys.argv[1]) == "--app":
  app_name = str(sys.argv[2])
else:
  print("expected --app app_name")
  exit()
cfg = Config()
ds_util = DatasetUtils(app_name,"APP")

add_penalty_func_name = None
if len(sys.argv) > 3 and str(sys.argv[3]) == "--add_penalty":
  if len(sys.argv) > 4:
    add_penalty_func_name = str(sys.argv[4])
  else:
    add_penalty_func_name = app_name

elif len(sys.argv) > 3:
  print("expected [--app=app_name] [--dqn=app_name] --add_penalty [func_name]")
  exit()
  # python 3 alset_jetbot_{teleop,train}.py [--func=nn_name] [--app=app_name] [--dqn=app_name] [--init]

args = ["alset_train.py", sys.argv[1], sys.argv[2]]
alset_robot = Robot(args)
func_app = FunctionalApp(alset_robot, app_name, "APP")
rand_func_ds_idx = ds_util.dataset_indices(mode="RAND", nn_name=app_name, position="NEW")
print(rand_func_ds_idx,":")

with open(rand_func_ds_idx, 'w+') as idx_file:
# if True:
  init = True # reset func flow
  while True:
    [func_name, rew_pen] = func_app.eval_func_flow_model(reward_penalty="REWARD1", init=init)
    if func_name is None:
      print("DONE")
      exit()
    init = False
    func_dir = ds_util.dataset_index_path(mode="FUNC", nn_name=func_name)
  
    # Sort the func index list in ascending order of dates
    idx_list = os.listdir(func_dir)
    idx_list.sort()
    while True:
      r = random.randint(0,(len(idx_list)-1))
      if (idx_list[r].startswith("FUNC") and not idx_list[r].endswith("IDX_PROCESSED.txt")
          and idx_list[r].endswith(".txt")
          and len(idx_list[r]) == len("FUNC__21_06_22b.txt") + len(add_penalty_func_name)):
        print("from_file:", idx_list[r])
        break
    print("idx:", idx_list, r, func_name, add_penalty_func_name)
    if func_name == add_penalty_func_name:
      # copy an index file
      from_file = func_dir + idx_list[r]
      to_file = ds_util.dataset_indices(mode="FPEN", nn_name=func_name, position="NEW")
      print("from, to:", from_file, to_file)
      # get illegal actions
      allowed_actions = []
      for [func, func_allowed_actions] in cfg.func_movement_restrictions:
         if func_name == func:
            for a in func_allowed_actions:
              allowed_actions.append(a)
            break
      if allowed_actions is None:
        print("no illegal actions for function ", func_name)
        exit()
      illegal_actions = []
      for action in cfg.full_action_set:
          if action not in allowed_actions:
            if action not in cfg.nn_disallowed_actions:  # PENALTY1, REWARD1,...
              illegal_actions.append(action)
      lines = []
      for line in open(from_file): 
        lines.append(line)
      replace_line_num = 0
      if len(lines) > 0:
        replace_line_num = random.randint(0,(len(lines)-1))
      count = 0
      with open(to_file, 'w+') as tofile:
       while True:
        if count == replace_line_num:
          replace_action = random.randint(0,(len(illegal_actions)-1))
          # ARD: This is an index, not a line in the index. returns Nones:
          print("old imgpath:", lines[count])
          [ds_time, ds_app, ds_mode, ds_nn_name, ds_action, ds_img_name, ds_img_path] = ds_util.get_dataset_info(lines[count], mode="FUNC")
          # copy img_name to illegal_action and PENALTY1 directories
          new_img_name = "FPEN_" + ds_img_name
          nn_dir = ds_util.dataset_path("FUNC", func_name)
          illegal_action_path = os.path.join(nn_dir, illegal_actions[replace_action])
          ds_util.mkdirs([illegal_action_path])
          illegal_action_path = os.path.join(illegal_action_path, new_img_name)
          try:
            shutil.copyfile(ds_img_path, illegal_action_path)
          except Exception as e:
              print("copyfile failed", ds_img_path, illegal_action_path, e)
          penalty_path = os.path.join(nn_dir, "PENALTY1")
          ds_util.mkdirs([penalty_path])
          penalty_path = os.path.join(penalty_path, new_img_name)
          try:
            shutil.copyfile(ds_img_path, penalty_path)
          except Exception as e:
              print("copyfile failed", ds_img_path, penalty_path, e)
          print("penalty:", penalty_path)
          print("illegal act:", illegal_action_path)
          print("to_file:", to_file)
          # update the FPEN index 
          illegal_action_line = ds_util.dataset_line(illegal_action_path, tm = ds_time)
          tofile.write(illegal_action_line + "\n")
          penalty_line = ds_util.dataset_line(penalty_path, tm = ds_time)
          tofile.write(penalty_line + "\n")
          break
        else:
          tofile.write(lines[count])
          count += 1
      # update the App index
      idx_file.write(to_file + "\n")
      exit()
    else:
      # randomly selected an index for the function, and write it to the App index
      idx_file.write(func_dir + idx_list[r] + "\n")
      print(func_dir + idx_list[r])
  
