#!/usr/bin/python
import sys
from robot import *
from functional_app import *
from dataset_utils import *

print("Number of arguments:", len(sys.argv), "arguments.")
print("Argument List:", str(sys.argv))
ds_util = DatasetUtils("TT","APP")
if str(sys.argv[1]) == "--app":
  app_name = str(sys.argv[2])
else:
  print("expected --app app_name")
  exit()

alset_robot = Robot()
func_app = FunctionalApp(alset_robot, app_name, "APP")
rand_func_ds_idx = ds_util.dataset_indices(mode="RAND", nn_name=app_name, position="NEW")
print(rand_func_ds_idx,":")

with open(rand_func_ds_idx, 'w+') as file:
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
    r = random.randint(0,(len(idx_list)-1))
    file.write(func_dir + idx_list[r] + "\n")
    print(func_dir + idx_list[r])
  
