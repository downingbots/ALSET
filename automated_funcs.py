import numpy as np
import cv2
import math
from .config import *

class AutomatedFuncs():

  def __init__(self):
      self.init()
      self.cfg = Config()

  def init(self):
      self.automatic_function_name = None
      self.curr_automatic_action = None
      self.last_arm_action = None
      self.left_count = 0
      self.forward_count = 0
      self.random_goal = 0
      self.prev_frame = None
      self.curr_frame = None
      self.upper_arm_count = 0
      self.lower_arm_count = 0
      self.gripper_count = 0
      self.nonmovement_count = 0
      self.search_and_relocate = "QUICK_SEARCH"
      self.rew_pen = None

  def set_automatic_function(self, function_name):
      self.init()
      self.automatic_function_name = function_name

  def get_automatic_function(self):
      return self.automatic_function_name

  def automatic_function(self, frame, reward_penalty):
      self.prev_frame = self.curr_frame
      self.curr_frame = frame
      self.rew_pen = reward_penalty
      if self.automatic_function_name == "HIGH_SLOW_SEARCH":
        return self.high_slow_search()
      elif self.automatic_function_name == "QUICK_SEARCH":
        return self.quick_search()
      elif self.automatic_function_name == "RELOCATE":
        return self.relocate()
      elif self.automatic_function_name == "SEARCH_AND_RELOCATE":
        if self.automatic_function_name == "QUICK_SEARCH":
          return self.quick_search()
        elif self.automatic_function_name == "RELOCATE":
          return self.relocate()
      elif self.automatic_function_name == "PARK_ARM_COMPACT":
        return self.park_arm_compact()
      elif self.automatic_function_name == "CLOSE_GRIPPER":
        return self.close_gripper()

  def high_slow_search(self):
      if (self.curr_automatic_action == "LEFT" and
          self.last_arm_action == "LOWER_ARM_DOWN" and
          self.left_count >= 45):
          # 15 pulses is actually 3 rotation pulses
          self.left_count = 0
          self.curr_automatic_action = "LOWER_ARM_UP"
          self.last_arm_action = "LOWER_ARM_UP"
      elif (self.curr_automatic_action == "LEFT" and
            self.last_arm_action == "LOWER_ARM_UP" and
            self.left_count >= 45):
          # 15 pulses is actually 3 rotation pulses
          self.last_arm_action = "LOWER_ARM_DOWN"
      elif (self.curr_automatic_action in ["LOWER_ARM_UP", "LOWER_ARM_DOWN"] and self.rew_pen == "PENALTY1"):
          self.curr_automatic_action = "LEFT"
          self.left_count += 1
      elif (self.curr_automatic_action == "LEFT"):
          self.curr_automatic_action = "LEFT"
          self.left_count += 1
      else: 
          self.curr_automatic_action = self.last_arm_action
      if self.curr_automatic_action != None:
          print("nn_automatic_action: %s" % self.curr_automatic_action)
      self.robot.gather_data.set_function(self.curr_automatic_action)
      return self.curr_automatic_action

  def quick_search(self):
      self.search_and_relocate = "QUICK_SEARCH"
      if (self.rew_pen == "PENALTY1"):
         self.random_relocate()
      self.curr_automatic_action = "LEFT"
      self.left_count += 1
      self.robot.gather_data.set_function(self.curr_automatic_action)
      return self.curr_automatic_action

  def relocate(self):
      self.search_and_relocate = "RELOCATE"
      if (self.rew_pen == "PENALTY1"):
         self.quick_search()
      self.curr_automatic_action = "FORWARD"
      self.forward_count += 1
      return self.curr_automatic_action

  def optflow(self, old_frame, new_frame):
      if old_frame is None:
        return True
      # cap = cv.VideoCapture('slow.flv')
      # params for ShiTomasi corner detection
      feature_params = dict( maxCorners = 100,
                             qualityLevel = 0.3,
                             minDistance = 7,
                             blockSize = 7 )
      # Parameters for lucas kanade optical flow
      lk_params = dict( winSize  = (15,15),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
      # Create some random colors
      color = np.random.randint(0,255,(100,3))
      # Take first frame and find corners in it
      # ret, old_frame = cap.read()
      old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
      p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
      # Create a mask image for drawing purposes
      mask = np.zeros_like(old_frame)
  
      frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
      # calculate optical flow
      try:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
      except:
        print("OPT FLOW FAILS")
        return False
      # Select good points
      good_new = p1[st==1]
      good_old = p0[st==1]
      # draw the tracks
      dist = 0
      numpts = 0
      frame1 = new_frame
      for i,(new,old) in enumerate(zip(good_new,good_old)):
          a,b = new.ravel()
          c,d = old.ravel()
          dist += math.hypot(a-c,b-d)
          numpts += 1
          # mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
          # frame1 = cv2.circle(frame1,(a,b),5,color[i].tolist(),-1)
      img = cv2.add(new_frame,mask)
      # cv2.imshow('frame',img)
      k = cv2.waitKey(30) & 0xff
      # Now update the previous frame and previous points
      # old_gray = frame_gray.copy()
      p0 = good_new.reshape(-1,1,2)
      # cv2.destroyAllWindows()
      if numpts != 0:
        dist /= numpts
      else:
        dist = 0
      print("optflow dist %f minthrot %d maxthrot %d" % (dist,self.minthrottle,self.maxthrottle))
      # note: PPF also used to ensure that moving
      # tried 0.75, 0.9, 1
      # OPTFLOWTHRESH = 0.8
      if dist > cfg.OPTFLOWTHRESH:
        return True
      else:
        return False

  def check_movement(self):
      moved = self.optflow(self.prev_frame, self.curr_frame)
      if not moved:
        self.nonmovement_count += 1
      if self.nonmovement_count > self.cfg.MAX_NON_MOVEMENT:
        self.curr_automatic_action = None

  # prototype of "DO ACTION UNTIL REACHED LIMIT"
  def close_gripper(self):
      self.last_arm_action = self.curr_automatic_action
      moved = self.optflow(self.prev_frame, self.curr_frame)
      if self.rew_pen == "PENALTY1":
          self.gripper_count = self.cfg.MAX_NON_MOVEMENT + 1
          self.curr_automatic_action = None
      elif not moved and self.last_arm_action == "GRIPPER_CLOSE":
          if self.gripper_count > self.cfg.MAX_NON_MOVEMENT:
            self.curr_automatic_action = None
          else:
            self.gripper_count += 1
            self.curr_automatic_action = "GRIPPER_CLOSE"
      else:
          self.curr_automatic_action = "GRIPPER_CLOSE"
      return self.curr_automatic_action

  # This is designed to be 100% automatic. Should probably be first move in any sequence.
  def park_arm_compact(self):
      # determine state
      self.last_arm_action = self.curr_automatic_action
      moved = self.optflow(self.prev_frame, self.curr_frame)
      # this is designed to be 100% automatic, but background movement can cause problems
      # PENALTY1 is an error condition from the user saying "don't do that again"
      if self.rew_pen == "PENALTY1":
        if self.last_arm_action == "UPPER_ARM_UP":
          self.upper_arm_count = self.cfg.MAX_NON_MOVEMENT + 1
        elif self.last_arm_action == "LOWER_ARM_UP":
          self.lower_arm_count = self.cfg.MAX_NON_MOVEMENT + 1
        elif self.last_arm_action == "GRIPPER_CLOSE":
          self.gripper_count = self.cfg.MAX_NON_MOVEMENT + 1
        elif self.last_arm_action == "GRIPPER_OPEN":
          self.gripper_count = self.cfg.MAX_NON_MOVEMENT + 1
      elif not moved and self.last_arm_action == "UPPER_ARM_UP":
        self.upper_arm_count += 1
      elif not moved and self.last_arm_action == "LOWER_ARM_DOWN":
        self.lower_arm_count += 1
      elif not moved and self.last_arm_action == "GRIPPER_CLOSE":
        self.gripper_count += 1
      elif not moved and self.last_arm_action == "GRIPPER_OPEN":
        self.gripper_count += 1
      if self.lower_arm_count > self.cfg.MAX_NON_MOVEMENT:
        lower_arm_done = True
      else:
        lower_arm_done = False
      if self.lower_arm_count > self.cfg.MAX_NON_MOVEMENT:
        lower_arm_done = True
      else:
        lower_arm_done = False
      if self.gripper_count > self.cfg.MAX_NON_MOVEMENT:
        gripper_done = True
      else:
        gripper_done = False

      # determine next move
      if gripper_done and lower_arm_done and upper_arm_done:
        # All done
        self.curr_automatic_action = None
      elif not gripper_done and not lower_arm_done and not upper_arm_done:
        # First: close gripper
        self.curr_automatic_action = "GRIPPER_CLOSE"
      elif not lower_arm_done and not upper_arm_done:
        # Alternate between raising upper arm and lowering lower arm
        if self.last_arm_action == "LOWER_ARM_DOWN":
           self.curr_automatic_action = "UPPER_ARM_DOWN"
        elif self.last_arm_action == "UPPER_ARM_UP":
           self.curr_automatic_action = "LOWER_ARM_DOWN"
      elif not lower_arm_done:
        self.gripper_count = 0  # gripper eligible to be closed next frame
        self.curr_automatic_action = "LOWER_ARM_DOWN"
      elif not upper_arm_done:
        self.gripper_count = 0  # gripper eligible to be closed next frame
        self.curr_automatic_action = "UPPER_ARM_UP"
      elif lower_arm_done and upper_arm_done and not gripper_done:
        # Last: open gripper
        self.curr_automatic_action = "GRIPPER_OPEN"
      return self.curr_automatic_action

