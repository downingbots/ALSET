from jetbot import *
import sys

args = ["alset_train.py", sys.argv[1], sys.argv[2]]
r = Robot(args)
r.train()

