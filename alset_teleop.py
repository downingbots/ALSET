import sys
from jetbot import *
# from jetbot import webcam

args = ["alset_train.py", sys.argv[1], sys.argv[2]]
r = Robot(args)
webcam_run(r)
