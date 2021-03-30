import numpy as np
import cv2
import math
from config import *

class AutomatedFuncs():

  def __init__(self, sir_robot):
      self.cfg = Config()
      self.robot = sir_robot
      self.init()
  
  def init(self):
      self.prev_frame = None
      self.curr_frame = None
      self.automatic_function_name = None
      self.curr_automatic_action = None
      self.last_arm_action = None
      self.left_count = 0
      self.forward_count = 0
      self.random_goal = 0
      self.upper_arm_count = 0
      self.lower_arm_count = 0
      self.gripper_count = 0
      self.nonmovement_count = 0
      self.phase = 0
      self.phase_count = 0
      self.max_nonmovement_count = self.cfg.MAX_NON_MOVEMENT
      self.search_and_relocate = "QUICK_SEARCH"
      self.rew_pen = None

  def set_automatic_function(self, function_name):
      if (function_name != self.automatic_function_name):
        self.init()
      self.automatic_function_name = function_name
      # attr_lst = self.cfg.get_func_value(function_name, "ATTRIBUTES")
      # self.max_nonmovement_count = self.cfg.get_value(attr_lst, "MAX_NON_MOVEMENT")
      # print("Auto Func:", function_name, self.max_nonmovement_count)

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
      elif self.automatic_function_name == "QUICK_SEARCH_AND_RELOCATE":
        return self.quick_search()
      elif self.automatic_function_name == "PARK_ARM_RETRACTED":
        return self.park_arm_retracted()
      elif self.automatic_function_name == "PARK_ARM_RETRACTED_WITH_CUBE":
        return self.park_arm_retracted(False)
      elif self.automatic_function_name == "CLOSE_GRIPPER":
        return self.close_gripper()
      else:
        print("Error: unknown automatic function", self.automatic_function_name)
        exit

  def high_slow_search(self):
      done = False
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
      return self.curr_automatic_action, done

  def quick_search(self):
      self.search_and_relocate = "QUICK_SEARCH"
      done = False
      if (self.rew_pen == "PENALTY1"):
         return self.relocate()
      self.curr_automatic_action = "LEFT"
      self.left_count += 1
      self.robot.gather_data.set_function(self.curr_automatic_action)
      print("quick_search action:", self.curr_automatic_action)
      return self.curr_automatic_action, done

  def relocate(self):
      done = False
      self.search_and_relocate = "RELOCATE"
      if (self.rew_pen == "PENALTY1"):
         return self.quick_search()
      self.curr_automatic_action = "FORWARD"
      self.forward_count += 1
      self.robot.gather_data.set_function(self.curr_automatic_action)
      print("quick_search action:", self.curr_automatic_action)
      return self.curr_automatic_action, done

  def optflow(self, old_frame, new_frame):
      if old_frame is None:
        print("optflow: old_frame None")
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
      # k = cv2.waitKey(30) & 0xff
      # Now update the previous frame and previous points
      # old_gray = frame_gray.copy()
      p0 = good_new.reshape(-1,1,2)
      # cv2.destroyAllWindows()
      if numpts != 0:
        dist /= numpts
      else:
        dist = 0
      print("optflow dist %f " % (dist))
      # note: PPF also used to ensure that moving
      # tried 0.75, 0.9, 1
      # OPTFLOWTHRESH = 0.8
      if dist > self.cfg.OPTFLOWTHRESH:
        return True
      else:
        return False

  def check_movement(self):
      moved = self.optflow(self.prev_frame, self.curr_frame)
      if not moved:
        self.nonmovement_count += 1
      if self.nonmovement_count > self.max_nonmovement_count:
        self.curr_automatic_action = None

  # prototype of "DO ACTION UNTIL REACHED LIMIT"
  def close_gripper(self):
      done = False
      self.last_arm_action = self.curr_automatic_action
      moved = self.optflow(self.prev_frame, self.curr_frame)
      if self.rew_pen == "PENALTY1":
          self.gripper_count = self.max_nonmovement_count + 1
          self.curr_automatic_action = None
      elif not moved and self.last_arm_action == "GRIPPER_CLOSE":
          if self.gripper_count > self.max_nonmovement_count:
            self.curr_automatic_action = None
            done = True
          else:
            self.gripper_count += 1
            self.curr_automatic_action = "GRIPPER_CLOSE"
      else:
          self.curr_automatic_action = "GRIPPER_CLOSE"
      return self.curr_automatic_action, done

  # This is designed to be 100% automatic. Should probably be first move in any sequence.
  def park_arm_retracted(self, final_gripper_open=True):
      # determine state
      self.last_arm_action = self.curr_automatic_action
      moved = self.optflow(self.prev_frame, self.curr_frame)
      done = False
      print("moved:", moved)
      # this is designed to be 100% automatic, but background movement can cause problems
      # PENALTY1 is an error condition from the user saying "don't do that again"
      if self.rew_pen == "PENALTY1":
        if self.last_arm_action in ["UPPER_ARM_UP", "UPPER_ARM_DOWN"]:
          self.upper_arm_count = self.max_nonmovement_count + 1
        elif self.last_arm_action in ["LOWER_ARM_UP", "LOWER_ARM_DOWN"]:
          self.lower_arm_count = self.max_nonmovement_count + 1
        elif self.last_arm_action in ["GRIPPER_CLOSE", "GRIPPER_OPEN"]:
          self.gripper_count = self.max_nonmovement_count + 1
      elif not moved and self.last_arm_action in ["UPPER_ARM_UP", "UPPER_ARM_DOWN"]:
        self.upper_arm_count += 1
      elif not moved and self.last_arm_action in ["LOWER_ARM_UP", "LOWER_ARM_DOWN"]:
        self.lower_arm_count += 1
      elif not moved and self.last_arm_action in ["GRIPPER_CLOSE", "GRIPPER_OPEN"]:
        self.gripper_count += 1
      print("counts:", self.gripper_count, self.lower_arm_count, self.upper_arm_count)
      if self.lower_arm_count > self.max_nonmovement_count:
        lower_arm_done = True
      else:
        lower_arm_done = False
      if self.upper_arm_count > self.max_nonmovement_count:
        upper_arm_done = True
      else:
        upper_arm_done = False
      if self.gripper_count > self.max_nonmovement_count:
        gripper_done = True
      else:
        gripper_done = False
      print("done  :", gripper_done, lower_arm_done, upper_arm_done)

      # determine next move
      if gripper_done and lower_arm_done and upper_arm_done:
        # All done
        self.curr_automatic_action = None

      elif self.phase == 0:
        if gripper_done:
          self.gripper_count = 0
          self.phase = 1
        else:
          self.curr_automatic_action = "GRIPPER_CLOSE"
      elif self.phase == 1:
        if upper_arm_done:
          self.upper_arm_count = 0
          self.phase = 2
        else:
          self.curr_automatic_action = "UPPER_ARM_UP"
      elif self.phase == 2:
        if upper_arm_done or self.phase_count >= 3:
          self.upper_arm_count = 0
          self.phase = 3
          self.phase_count = 0
        else:
          # if stopped due to being stuck, unstick
          self.phase_count += 1
          self.curr_automatic_action = "UPPER_ARM_DOWN"
      elif self.phase == 3:
        if lower_arm_done:
          self.lower_arm_count = 0
          self.phase = 4
        else:
          self.curr_automatic_action = "LOWER_ARM_DOWN"
      elif self.phase == 4:
        if lower_arm_done or self.phase_count >= 3:
          self.lower_arm_count = 0
          self.phase = 5
          self.phase_count = 0
        else:
          # if stopped due to being stuck, unstick
          self.phase_count += 1
          self.curr_automatic_action = "LOWER_ARM_UP"
      elif self.phase == 5:
        if upper_arm_done:
          self.upper_arm_count = 0
          self.phase = 6
        else:
          self.curr_automatic_action = "UPPER_ARM_UP"
      elif self.phase == 6:
        if gripper_done or not final_gripper_open:
          self.upper_arm_count = 0
          self.phase = 7
        else:
          self.curr_automatic_action = "GRIPPER"
      elif self.phase == 7:
        done = True
        print("DONE!")

      self.robot.gather_data.set_function(self.curr_automatic_action)
      return self.curr_automatic_action, done

#      elif not gripper_done and not lower_arm_done and not upper_arm_done:
#        # First: close gripper
#        self.curr_automatic_action = "GRIPPER_CLOSE"
#      elif not lower_arm_done:
#
#      elif not lower_arm_done and not upper_arm_done:
#        # Alternate between raising upper arm and lowering lower arm
#        if self.last_arm_action == "LOWER_ARM_DOWN":
#           self.curr_automatic_action = "UPPER_ARM_UP"
#        elif self.last_arm_action == "UPPER_ARM_UP":
#           self.curr_automatic_action = "LOWER_ARM_DOWN"
#        else: 
#           self.curr_automatic_action = "UPPER_ARM_UP"
#      elif not lower_arm_done:
#        self.gripper_count = 0  # gripper eligible to be closed next frame
#        self.curr_automatic_action = "LOWER_ARM_DOWN"
#      elif not upper_arm_done:
#        self.gripper_count = 0  # gripper eligible to be closed next frame
#        self.curr_automatic_action = "UPPER_ARM_UP"
#      elif lower_arm_done and upper_arm_done and not gripper_done:
#        # Last: open gripper
#        self.curr_automatic_action = "GRIPPER_OPEN"

