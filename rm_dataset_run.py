#!/usr/bin/python

import sys

from .dataset_utils import *

# print 'Number of arguments:', len(sys.argv), 'arguments.'
# print 'Argument List:', str(sys.argv)
ds_util = DatasetUtils("TT","DQN")
if str(sys.argv) == "--mode":
  position = "OLD"
position = "NEW"
mode = "DQN"
if str(sys.argv[0]).startswith("--oldest_"):
  position = "OLD"
elif str(sys.argv[0]) == "--newest_":
  position = "NEW"
elif str(sys.argv[0]) == "--next_":
  position = "NEXT"
else:
  print("bad option:", str(sys.argv[0]))
nn_name = None
if str(sys.argv[0]).endswith("app"):
  mode = "app"
  if len(sys.argv) != 2:
    print("missing nn_name: ", sys.argv)
  nn_name = str(sys.argv[1])
elif str(sys.argv[0]).endswith("func"):
  mode = "func"
elif str(sys.argv[0]).endswith("dqn"):
  mode = "dqn"
else:
  print("bad option:", str(sys.argv))
ds_util.remove_dataset_images(mode, nn_name, position)
