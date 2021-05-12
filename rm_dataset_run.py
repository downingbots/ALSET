#!/usr/bin/python

import sys

from dataset_utils import *

# print 'Number of arguments:', len(sys.argv), 'arguments.'
# print 'Argument List:', str(sys.argv)
ds_util = DatasetUtils("TT","DQN")
if str(sys.argv) == "--mode":
  position = "OLD"
position = "NEW"
mode = "DQN"
if str(sys.argv[1]).startswith("--oldest_"):
  position = "OLDEST"
elif str(sys.argv[1]).startswith("--newest_"):
  position = "NEWEST"
elif str(sys.argv[1]).startswith("--next_"):
  position = "NEXT"
else:
  print("bad option:", str(sys.argv[1]))
nn_name = None
if str(sys.argv[1]).endswith("app"):
  mode = "APP"
  if len(sys.argv) <= 3:
    print("missing nn_name: ", sys.argv)
  nn_name = str(sys.argv[2])
elif str(sys.argv[1]).endswith("func"):
  mode = "FUNC"
  if len(sys.argv) < 3:
    print("missing nn_name: ", sys.argv)
  nn_name = str(sys.argv[2])
elif str(sys.argv[1]).endswith("dqn"):
  mode = "DQN"
else:
  print("bad option:", str(sys.argv))
if str(sys.argv[len(sys.argv)-1]) == "--rm":
  do_remove=True
else:
  do_remove=False

ds_util.remove_dataset_images(mode, nn_name, position, do_remove)
