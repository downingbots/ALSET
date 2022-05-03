#!/usr/bin/env python
from alset_state import *
from arm_nav import *
from matplotlib import pyplot as plt
import cv2
import STEGO.src
from alset_stego import *
import numpy as np
import copy
from math import sin, cos, pi, sqrt
from utilborders import *
from cv_analysis_tools import *
from dataset_utils import *
from PIL import Image
import imutils
from skimage.feature import peak_local_max
from scipy import ndimage
from scipy.cluster.hierarchy import ward
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN
from sklearn import metrics, linear_model
import matplotlib.image as mpimg
from sortedcontainers import SortedList, SortedSet, SortedDict

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

class AnalyzeLines(object):
  def line_intersection(self, line1, line2):
      xdiff = (line1[0][0] - line1[0][2], line2[0][0] - line2[0][2])
      ydiff = (line1[0][1] - line1[0][3], line2[0][1] - line2[0][3])
  
      def det(a, b):
          return a[0] * b[1] - a[1] * b[0]
  
      div = det(xdiff, ydiff)
      if div == 0:
         return None
         # raise Exception('lines do not intersect')
  
      # d = (det(line1[0][0:1], line1[0][2:3]), det(line2[0][0:1],line2[0][2:3]))
      d = (det(line1[0][0:2], line1[0][2:4]), det(line2[0][0:2],line2[0][2:4]))
      x = det(d, xdiff) / div
      y = det(d, ydiff) / div
      # print("intersection: ", line1, line2, [x,y])
      if ( x >= max(line1[0][0], line1[0][2]) + 2
        or x <= min(line1[0][0], line1[0][2]) - 2
        or x >= max(line2[0][0], line2[0][2]) + 2
        or x <= min(line2[0][0], line2[0][2]) - 2
        or y <= min(line1[0][1], line1[0][3]) - 2
        or y >= max(line1[0][1], line1[0][3]) + 2
        or y <= min(line2[0][1], line2[0][3]) - 2
        or y >= max(line2[0][1], line2[0][3])) + 2:
        # intersection point outside of line segments' range
        return None
         
      xdiff = (line1[0][0] - line1[0][2], line2[0][0] - line2[0][2])
      ydiff = (line1[0][1] - line1[0][3], line2[0][1] - line2[0][3])
      return [x, y]
  
  def is_same_line(self, line1, line2):
      if self.is_parallel(line1, line2) and self.line_intersection(line1, line2) is not None:
        # print("same line: line1, line2", line1, line2)
        return True
  
  def is_broken_line(self, line1, line2):
      def det(a, b):
          return a[0] * b[1] - a[1] * b[0]
      def get_dist(x1,y1, x2, y2):
          return sqrt((x2-x1)**2 + (y2-y1)**2)
  
      if not self.is_parallel(line1, line2):
        return None
      if self.line_intersection(line1, line2) is not None:
        return None
  
      dist0 = get_dist(line1[0][0], line1[0][1], line2[0][0], line1[0][1])
      dist1 = get_dist(line1[0][0], line1[0][1], line2[0][2], line1[0][3])
      dist2 = get_dist(line1[0][2], line1[0][3], line2[0][0], line1[0][1])
      dist3 = get_dist(line1[0][2], line1[0][3], line2[0][2], line1[0][3])
  
      extended_line = None
      if dist0 == max(dist0,dist1,dist2,dist3):
        extended_line = [[line1[0][0], line1[0][1], line2[0][0], line1[0][1]]]
      elif dist1 == max(dist0,dist1,dist2,dist3):
        extended_line = [[line1[0][0], line1[0][1], line2[0][2], line1[0][3]]]
      elif dist2 == max(dist0,dist1,dist2,dist3):
        extended_line = [[line1[0][2], line1[0][3], line2[0][0], line1[0][1]]]
      elif dist3 == max(dist0,dist1,dist2,dist3):
        extended_line = [[line1[0][2], line1[0][3], line2[0][2], line1[0][3]]]
     
      if not (is_parallel(line1, extended_line) and is_parallel(line2, extended_line)):
        return None
      # print("broken line: ", line1, line2, extended_line)
      return extended_line
  
  def is_parallel(self, line1, line2):
      angle1 = np.arctan2((line1[0][0]-line1[0][2]), (line1[0][1]-line1[0][3]))
      angle2 = np.arctan2((line2[0][0]-line2[0][2]), (line2[0][1]-line2[0][3]))
      allowed_delta = .1
      if abs(angle1-angle2) <= allowed_delta:
        # print("is_parallel line1, line2", line1, line2, angle1, angle2)
        return True
      if abs(np.pi-abs(angle1-angle2)) <= allowed_delta:
        # note: .01 and 3.14 should be considered parallel
        # print("is_parallel line1, line2", line1, line2, angle1, angle2)
        return True
      return False
  
  def parallel_dist(self, line1, line2, dbg=False):
      if not is_parallel(line1, line2):
        return None
      # line1, line2 [[151 138 223 149]] [[ 38  76 139  96]]
  
      # y = mx + c
      # pts = [(line1[0][0], line1[0][2]), (line1[0][1], line1[0][3])]
      pts = [(line1[0][0], line1[0][1]), (line1[0][2], line1[0][3])]
      x_coords, y_coords = zip(*pts)
      A = np.vstack([x_coords,np.ones(len(x_coords))]).T
      l1_m, l1_c = np.linalg.lstsq(A, y_coords)[0]
      if dbg:
        print("x,y,m,c", x_coords, y_coords, l1_m, l1_c)
  
      pts = [(line2[0][0], line2[0][1]), (line2[0][2], line2[0][3])]
      x_coords, y_coords = zip(*pts)
      A = np.vstack([x_coords,np.ones(len(x_coords))]).T
      l2_m, l2_c = np.linalg.lstsq(A, y_coords)[0]
      if dbg:
        print("x,y,m,c", x_coords, y_coords, l2_m, l2_c)
  
      # coefficients = np.polyfit(x_val, y_val, 1)
      # Goal: set vert(y) the same on both lines, compute horiz(x).
      # with a vertical line, displacement will be very hard to compute
      # unless same end-points are displaced.
      if ((line1[0][0] >= line2[0][0] >= line1[0][2]) or
          (line1[0][0] <= line2[0][0] <= line1[0][2])):
        x1 = line2[0][0]
        y1 = line2[0][1]
        y2 = y1
        x2 = (y2 - l1_c) / l1_m
        # y2 = l1_m * x1 + l1_c
        # x2 = (y1 - l2_c) / l2_m
      elif ((line1[0][0] >= line2[0][2] >= line1[0][2]) or
            (line1[0][0] <= line2[0][2] <= line1[0][2])):
        x1 = line2[0][2]
        y1 = line2[0][3]
        y2 = y1
        x2 = (y2 - l1_c) / l1_m
        # y2 = l1_m * x1 + l1_c
        # x2 = (y1 - l2_c) / l2_m
      elif ((line2[0][0] >= line1[0][0] >= line2[0][2]) or
            (line2[0][0] <= line1[0][0] <= line2[0][2])):
        x1 = line1[0][0]
        y1 = line1[0][1]
        y2 = y1
        x2 = (y2 - l2_c) / l2_m
      elif ((line2[0][0] >= line1[0][2] >= line2[0][2]) or
            (line2[0][0] <= line1[0][2] <= line2[0][2])):
        x1 = line1[0][2]
        y1 = line1[0][3]
        y2 = y1
        x2 = (y2 - l2_c) / l2_m
      elif ((line1[0][1] >= line2[0][1] >= line1[0][3]) or
            (line1[0][1] <= line2[0][1] <= line1[0][3])):
        x1 = line2[0][0]
        y1 = line2[0][1]
        y2 = y1
        x2 = (y2 - l1_c) / l1_m
      elif ((line1[0][1] >= line2[0][3] >= line1[0][3]) or
            (line1[0][1] <= line2[0][3] <= line1[0][3])):
        y1 = line2[0][3]
        x1 = line2[0][2]
        y2 = y1
        x2 = (y2 - l1_c) / l1_m
      elif ((line2[0][1] >= line1[0][1] >= line2[0][3]) or
            (line2[0][1] <= line1[0][1] <= line2[0][3])):
        y1 = line1[0][1]
        x1 = line1[0][0]
        y2 = y1
        x2 = (y2 - l2_c) / l2_m
      elif ((line2[0][1] >= line1[0][3] >= line2[0][3]) or
            (line2[0][1] <= line1[0][3] <= line2[0][3])):
        y1 = line1[0][3]
        x1 = line1[0][2]
        y2 = y1
        x2 = (y2 - l2_c) / l2_m
      else:
        return None
      # print("parallel_dist", (x1-x2),(y1-y2))
      return x1-x2, y1 - y2
  
  ##########################################################
  # Using line analysis and Stego, track various obstacles
  ##########################################################
  #       ___   w2 = 3
  #      /   \
  #     /     \   h = 3D
  #    /       \
  #    123456789   w1=9
  #
  #    w1/d1 = w2/d2  => but d1 is near zero????. depends on angle of camera.
  #                      get pi camera angle
  #                      pi FOV: 62.2deg x 48.8deg
  #    w1/w2 = d1/d2 = (d2-d1)
  #                   => use to get distance traveled
  #                   => # pixels moved?
  #
  #    If you know the end width, you can compute the distance to the end
  #
  # vanishing point 
  # [1  0  0  0   [ D       [Dx
  #  0  1  0  0  *  0 ]  =   Dy
  #  0  0  1  0]             Dz]
  #
  def track_table(self, img):
      # perspective, width/length scale
      pass
  
  def track_wall(self, img):
      pass
  
  def track_ground_barrier(self, img):
      pass
  
  def track_lines(self, img):
      pass
  
  def track_road(self, img):
      pass
  
  
  def get_hough_lines(self, img, max_line_gap = 10):
        # rho_resolution = 1
        # theta_resolution = np.pi/180
        # threshold = 155
        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        #  threshold = 15  # minimum number of votes (intersections in Hough grid cell)
        threshold = 10  # minimum number of votes (intersections in Hough grid cell)
        # min_line_length = 50  # minimum number of pixels making up a line
        min_line_length = 40  # minimum number of pixels making up a line
        # max_line_gap = 10  # maximum gap in pixels between connectable line segments
        # max_line_gap = 5  # maximum gap in pixels between connectable line segments
  
        # Output "lines" is an array containing endpoints of detected line segments
        hough_lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                      min_line_length, max_line_gap)
        return hough_lines
  
  def get_hough_lines_img(self, hough_lines, gray):
      hough_lines_image = np.zeros_like(gray)
      if hough_lines is not None:
        for line in hough_lines:
          for x1,y1,x2,y2 in line:
            cv2.line(hough_lines_image,(x1,y1),(x2,y2),(255,0,0),3)
      else:
        print("No houghlines")
      return hough_lines_image
  
  def get_box_lines(self, curr_img, gripper_img = None):
        gray = cv2.cvtColor(curr_img.copy(), cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 200, None, 3)
        edges = cv2.dilate(edges,None,iterations = 1)
        cv2.imshow("edges1", edges)
        if gripper_img is not None: 
          edges = cv2.bitwise_and(edges, cv2.bitwise_not(gripper_img))
          cv2.imshow("edges2", edges)
        hough_lines = self.get_hough_lines(edges)
        hough_lines_image = np.zeros_like(curr_img)
        # print("hough_lines", hough_lines)
        if hough_lines is None:
          return None
        for line in hough_lines:
          for x1,y1,x2,y2 in line:
            cv2.line(hough_lines_image,(x1,y1),(x2,y2),(255,0,0),5)
        cv2.imshow("box lines houghlines", hough_lines_image)
        # cv2.waitKey(0)
        return hough_lines
  
  def exact_line(self, l1, l2):
      if l1[0][0]==l2[0][0] and l1[0][1]==l2[0][1] and l1[0][2]==l2[0][2] and l1[0][3]==l1[0][3]:
        return True
      return False
  
  def display_lines(self, img_label, line_lst, curr_img):
      lines_image = np.zeros_like(curr_img)
      for line in line_lst:
        print("line:", line)
        # for x1,y1,x2,y2 in line[0]:
        x1,y1,x2,y2 = line[0]
        cv2.line(lines_image,(x1,y1),(x2,y2),(255,0,0),5)
      cv2.imshow(img_label, lines_image)
  
  def display_line_pairs(self, img_label, line_pair_lst, curr_img, mode=2):
      lines_image = np.zeros_like(curr_img)
      for line0, line1, rslt in line_pair_lst:
        if mode == 0:
          print("line0:", img_label, line0)
        elif mode == 1:
          print("line1:", img_label, line1)
        elif mode == 2:
          print("line0,line1:", img_label, line0, line1)
        if mode == 0 or mode == 2:
          x1,y1,x2,y2 = line0[0]
          cv2.line(lines_image,(x1,y1),(x2,y2),(255,0,0),3)
        if mode == 1 or mode == 2:
          x1,y1,x2,y2 = line1[0]
          cv2.line(lines_image,(x1,y1),(x2,y2),(130,0,0),5)
      cv2.imshow(img_label, lines_image)
  
  
  def analyze_moved_lines(self, best_moved_lines, num_ds, num_imgs, gripper_img, actions):
  
      # gripper_lines are umoved between frames.
      # These lines show up consistently unmoved between frames.
      # Other less-consistent lines associated with the gripper are possible.
      gripper_lines = self.get_hough_lines(gripper_img)
      gripper_lines_image = np.zeros_like(gripper_img)
      for line in gripper_lines:
        for x1,y1,x2,y2 in line:
          cv2.line(gripper_lines_image,(x1,y1),(x2,y2),(255,0,0),5)
      cv2.imshow("gripper lines", gripper_lines_image)
  
      # 0 = unknown
      # 1 = gripper / unmoving
      # 2+ = group of lines moving as a group between frames
  
      # Note: a line must of appeared in at least two consec frames
      # a particular line can be ID'ed by (ds_num, img_num, ml_num)
  
      def is_known_line(known_lines, line_id, moved_line):
          # line = 
          # for kl_id, kl in enumerate(known_lines):
            # if exact_line(l1, l2):
          pass
  
      def add_line(line_groups, known_line, curr_key, prev=None,next=None):
          # add a known_line to image index
          try:
            # current entry for known line already exists
            # add link to previous image of known line
            if prev is not None:
              known_line[curr_key][0].append(prev)
            # add link to next image of known line
            if next is not None:
              known_line[curr_key][1].append(curr)
          except:
            # create current entry for known line
            # add link to previous image of known line
            if prev is not None and curr is not None:
              known_line[curr_key] = [[prev],[curr]]
            elif prev is not None:
              known_line[curr_key] = [[prev],[]]
            elif curr is not None:
              known_line[curr_key] = [[],[curr]]
            else:
              raise
          if prev is not None:
            prev_key = (curr_key[0], curr_key[1]-1, prev)
            try:
              # prev entry for known line already exists
              known_line[prev_key][1].append(curr_key[2])
            except:
              # create prev entry for known line
              known_line[prev_key] = [[prev],[curr]]
            try:
              found = line_group["BOX"]["LINES"][prev_key]
              line_group["BOX"]["LINES"].append(curr_key)
            except:
              pass
          if next is not None:
            next_key = (curr_key[0], curr_key[1]+1, next)
            try:
              # prev entry for known line already exists
              known_line[next_key][0].append(curr_key[2])
            except:
              # create prev entry for known line
              known_line[next_key] = [[curr_key[2]],[]]
  
      # problem: multiple known_line mappings?
      #   -> make prev/next a list
      unknown_lines = {} 
      known_lines = {}   # (ds_num, img_num, ml_num) = [[prev],[next]]
      num_other_lg = 0
      line = []
      for ds_num in range(num_ds):
        action_cnt = -1
        # use Arm Navigation to figure out angle to box
        actions = {"UPPER_ARM_UP", "UPPER_ARM_DOWN", "LOWER_ARM_UP", "LOWER_ARM_DOWN", "FORWARD", "REVERSE", "LEFT", "RIGHT"}
        for img_num in range(num_imgs):
          min_disp = 1000000000
          action_cnt += 1
          for ml_num, lst in enumerate(best_moved_lines[ds_num][img_num]):
            if lst is None:
              continue
            disparity = best_moved_lines[ds_num][img_num][ml_num][2]
            x_y_disp.append(disparity)
            x_y_disp_imgnum.append(img_num)
            try:  # No gripper actions
              action = actions[ds_num][img_num]
              x_y_disp_action[action].append(disparity)
              x_y_disp_action_imgnum[action].append(img_num)
            except:
              pass
  
            # print("line disparity", [img_num, ml_num], prevline, disparity)
            # print("line disparity",[img_num, ml_num],moved_lines[ds_num][img_num][ml_num])
        # print("x_y_disp", len(x_y_disp))
  
        #########
        # BEST_LINE DISPARITY gets a set of clusters for approx different line groups over 
        # frames 
        #########
        print("BEST_LINE DISPARITY")
        best_clusters, best_cluster = analyze_x_y_dist(x_y_disp, x_y_disp_imgnum)
  
        #########
        # The actions make a difference in how the lines are viewed frame to frame
        # Seems hard to come up with reliable rules of movement
        # Probably need estimated camera and line positions (e.g., based on intersections)
        #########
  
        #########
        # While clustering helped verify some theories and provide guidelines, when you restrict
        # to "BEST LINES" and per action, then all lines are clumped into the same cluster as the
        # lines move over time.
        #
        # Do line grouping based on "BEST LINES" and x_y disparity and gripper analysis.
        # Do intersections to get points for scale.  
        # Do scale by comparing dist between parallel lines within same image and comparing dist across images.
        # 
        # gripper -> compute gripper bounds. Look for lines with (0,0) +-1 within / across /below bounds.
        # 
   
        ##################################################33
        # Gripper Check
        # are the lines potentially part of the gripper?
        action_cnt = -1
        print("Gripper check")
        for ds_num in range(num_ds):
          if ds_num == 0:
            pass
          for img_num in range(num_imgs):
            img_line_group = []
            action_cnt += 1
            print("dsnum, cnt, len", ds_num, action_cnt, len(actions))
            print("action[ds_num] len", ds_num, action_cnt, len(actions[ds_num]))
  
            for bml_num, lst in enumerate(best_moved_lines[ds_num][img_num]):
              if lst is None:
                continue
              bm_lines= best_moved_lines[ds_num][img_num][bml_num][1] 
              disparity = best_moved_lines[ds_num][img_num][bml_num][2] 
              print("bml:", bml_num, bm_lines, disparity)

              # best_moved_line contains [best_prev_line, curr_line, disparity])
  
            cv2.destroyAllWindows()
            lines_image = np.zeros_like(gripper_img)
  
      return disparity
  
  def analyze_box_lines(self, dataset, box_lines, actions, gripper_img, drop_off_img, img_paths):
      angle_sd = SortedDict()
      box_line_intersections = []
      world_lines = []
      arm_pos = []
      num_ds = len(box_lines)
      prev_lines = None
      hough_lines = None
      num_passes = 4
      # ret, gripper_img = cv2.threshold(gripper_img, 100, 255, cv2.THRESH_TOZERO)
      ret, drop_off_img = cv2.threshold(drop_off_img, 100, 255, cv2.THRESH_TOZERO)
      line_group = [[]]  # includes gripper line_group[0]
      gripper_lines = self.get_hough_lines(gripper_img)
      gripper_lines2 = self.get_hough_lines(gripper_img)
      gripper_lines_image = np.zeros_like(gripper_img)
      min_gripper_y = 100000000
      for line in gripper_lines2:
        for x1,y1,x2,y2 in line:
          cv2.line(gripper_lines_image,(x1,y1),(x2,y2),(255,0,0),5)
          for y in [y1,y2]:
            if y < min_gripper_y:
              MIN_GRIPPER_Y = y
  
      cv2.imshow("gripper lines", gripper_lines_image)
      unknown_lines = []
      same_frame_parallel = []
      diff_frame_parallel = []
      min_diff_frame_parallel = []
      broken_lines = []
      gripper_lines = []
      gripper_lines2 = []
      drop_off_gray = cv2.cvtColor(drop_off_img, cv2.COLOR_BGR2GRAY)
      drop_off_lines = self.get_hough_lines(drop_off_gray)
      drop_off_lines_image = np.zeros_like(drop_off_gray)
      for line in drop_off_lines:
        for x1,y1,x2,y2 in line:
          cv2.line(drop_off_lines_image,(x1,y1),(x2,y2),(255,0,0),5)
      cv2.imshow("drop_off lines", drop_off_lines_image)
      # cv2.waitKey(0)
  
      action_cnt = -1
      persist_line_stats = []
      persist_line = []
      persist_line_most_recent = [] # [[ds_num, img_num]]
      persist_line_min_max_ds_img = []
      persist_line_counter = []
      for ds_num, ds in enumerate(dataset):
        for pass_num in range(num_passes):
          num_imgs = len(box_lines[ds_num])
          if pass_num == 0:
            unknown_lines.append([])
            same_frame_parallel.append([])
            diff_frame_parallel.append([])
            min_diff_frame_parallel.append([])
            broken_lines.append([])
          for img_num in range(num_imgs):
            new_img_num = True
            if pass_num == 0:
              unknown_lines[ds_num].append([])
              same_frame_parallel[ds_num].append([])
              diff_frame_parallel[ds_num].append([])
              min_diff_frame_parallel[ds_num].append([])
              broken_lines[ds_num].append([])
            action_cnt += 1
            prev_lines = hough_lines
            hough_lines = box_lines[ds_num][img_num]
            if hough_lines is None:
              hough_lines = []
            if img_num > 0 and box_lines[ds_num][img_num-1] is not None:
              prev_hough_lines = box_lines[ds_num][img_num-1]
            else:
              prev_hough_lines = []
            for hl_num, h_line in enumerate(hough_lines):
              # print(pass_num, hl_num, "new h_line:", h_line)
              # Gripper lines now computed differently
              unknown_lines[ds_num][img_num].append(copy.deepcopy(h_line))
              
              # Gripper Lines should be the same from frame to frame
              if len(h_line) == 0:
                continue
              if pass_num == 0 and False:
                # Gripper lines now computed differently
                unknown_lines[ds_num].append([])
                # find gripper (line_group[0])
                found = False
                # TODO: increase strength of unmoved lines.
                # remove moved lines, esp over FWD/REV/L/R.
                for g_line in line_group[0]:
                  if len(g_line) == 0:
                    continue
                  if self.is_same_line(h_line, g_line):
                    # gripper_lines.append(g_line)
                    gripper_lines2.append(copy.deepcopy(g_line))
                    # line_group[0].append(g_line)
                    found = True
                    break
                # if found:
                #   continue
                # for g_line in gripper_lines:
                for g_line in gripper_lines2:
                  if len(g_line) == 0:
                    continue
                  if self.is_same_line(h_line, g_line):
                    # gripper_lines.append(g_line)
                    gripper_lines2.append(copy.deepcopy(g_line))
                    # line_group[0].append(g_line)
                    found = True
                    break
                # if found:
                #   continue
  
                for g_line in prev_hough_lines:
                  if len(g_line) == 0:
                    continue
                  if self.is_same_line(h_line, g_line):
                    # gripper_lines.append(g_line)
                    gripper_lines2.append(copy.deepcopy(g_line))
                    # line_group[0].append(g_line)
                    found = True
                # if found:
                #   continue
  
              if pass_num == 1:
                # for u_line in unknown_lines[ds_num][-3:-1]:
                for u_line in unknown_lines[ds_num][img_num]:
                  if len(u_line) == 0:
                    continue
                  # broken lines
                  broken_line = self.is_broken_line(u_line, h_line)
                  if broken_line is not None:
                    found = False
                    for x1_line, x2_line, b_line in broken_lines[ds_num][img_num]:
                      if self.exact_line(x1_line, h_line) and self.exact_line(x2_line, u_line):
                        found = True
                        break
                    if not found:
                      # print("broken_line", ds_num, img_num)
                      broken_lines[ds_num][img_num].append([copy.deepcopy(u_line), copy.deepcopy(h_line), broken_line])
                min_dist = 100000000
                best_p_line = None
                for p_line in prev_hough_lines:
                    if len(p_line) == 0:
                      continue
                    dist = parallel_dist(p_line, h_line)
                    if dist is not None:
                      found = False
                      for x1_line, x2_line, d in diff_frame_parallel[ds_num][img_num]:
                        if self.exact_line(x1_line, h_line) and self.exact_line(x2_line, p_line):
                          found = True
                          break
                      if not found :
                        print("diff_frame_parallel", ds_num, img_num)
                        diff_frame_parallel[ds_num][img_num].append([copy.deepcopy(p_line), copy.deepcopy(h_line), dist])
                        if min_dist > sqrt(abs(dist[0])+abs(dist[1])):
                          min_dist = sqrt(abs(dist[0])+abs(dist[1]))
                          best_p_line = copy.deepcopy(p_line)
                if best_p_line is not None:
                  disp = parallel_dist(best_p_line, h_line)
                  min_diff_frame_parallel[ds_num][img_num].append([copy.deepcopy(best_p_line), copy.deepcopy(h_line), disp])
                # cv2.waitKey(0)
  
              if pass_num == 2:
                # display broken lines
                self.display_line_pairs("broken lines", broken_lines[ds_num][img_num], drop_off_img)
                # note: we could generalize this analysis to be
                # based upon edge contours instead of lines.
                # display diff_frame_parallel lines
                self.display_line_pairs("best moved lines", min_diff_frame_parallel[ds_num][img_num], drop_off_img)
                if img_num == num_imgs-1:
                  disparity = self.analyze_moved_lines(min_diff_frame_parallel, num_ds, num_imgs, gripper_img, actions)
  
            if pass_num == 0:
              if img_num == num_imgs-1:
                print("##################")
                print("Image Sorted Lines")
                print("##################")
                for is_ds_num, ds in enumerate(dataset):
                  for is_img_num in range(num_imgs):
                    hough_lines = box_lines[is_ds_num][is_img_num]
                    if hough_lines is None:
                      print("No hough_lines for", is_ds_num, is_img_num)
                      continue
                    else:
                      # for hl_num, h_line in enumerate(hough_lines):
                        # print((is_ds_num, is_img_num, hl_num), h_line)
                      pass
  
            if pass_num == 2:
              print("pass_num", pass_num)
        
              ###################
              # Angle Sorted Dict
              ###################
              hough_lines = box_lines[ds_num][img_num]
              if hough_lines is None:
                print("No hough_lines for", ds_num, img_num)
                continue
              else:
                # print("##################")
                # print("Angle Sorted Lines")
                # print("##################")
                for hl_num, h_line in enumerate(hough_lines):
                  angle = np.arctan2((h_line[0][0]-h_line[0][2]), (h_line[0][1]-h_line[0][3]))
                  if angle <= 0:
                    angle += np.pi
                  try_again = True
                  while try_again:
                    try:
                      # ensure slightly unique angle so no lines are overwritten
                      val = angle_sd[angle]
                      angle += .000001
                      try_again = True
                    except:
                      try_again = False
                  angle_sd[angle] = (ds_num, img_num, h_line)
  
    
              if img_num == num_imgs-1:
                print("##################")
                print("Angle Sorted Lines")
                print("##################")
                skv = angle_sd.items()
                for item in skv:
                  print(item)
    
              ###########
              # Intersections within same image
              if hough_lines is None:
                print("No hough_lines for", ds_num, img_num)
                # continue
              else:
                for hl_num, h_line in enumerate(hough_lines):
                  for hl_num2, h_line2 in enumerate(hough_lines):
                    if hl_num2 > hl_num:
                      xy_intersect = self.line_intersection(h_line, h_line2)
                      if xy_intersect is not None:
                        box_line_intersections.append([ds_num, img_num, xy_intersect, h_line, h_line2])
              if img_num == num_imgs-1:
                print("##################")
                print("Line Intersections")
                print("##################")
                for bli in box_line_intersections:
                   print(bli)
  
              ############
              # find lines
              ############
              if img_num == num_imgs-1:
                num_lines = 0
                # persist_line = []
                # persist_line_most_recent = [] # [[ds_num, img_num]]
                # persist_line[lineno] = {(ds_num, img_num), [key1, key2]}
                # new persist_line[lineno][(d,i for d in range(num_ds) for i in range(num_imgs))] = []
                persist_line_3d = []  
                # 3d_persist_line[line_num] = [composite_3D_line]
                max_angle_dif = .1   # any bigger, can be purged from cache
                line_cache = []      # [(line_num, angle)] 
                akv = angle_sd.items()
                min_dist_allowed = 3
                a_cnt = 0
                for angle, item in akv:
                  a_cnt += 1
                  if a_cnt % 50 == 0 or len(persist_line) < 7:
                    print(a_cnt, "angle:", angle, len(persist_line))
                  # angle, pl_angle => too big to be considered parallel even though same line.... 
                  ds_num, img_num, a_line = item
                  min_dist = 1000000000000
                  best_pl_num = -1
                  best_pl_line = -1
                  best_pl_ds_num = -1
                  best_pl_img_num = -1
                  max_counter = -1
                  max_counter_num = -1
                  # instead of going through each persist line,
                  # go to persist line with angles above/below curr angle.
                  for pl_num, pl in enumerate(persist_line):
                    mr_pl_ds_num, mr_pl_img_num = persist_line_most_recent[pl_num]
                    pl_angle_lst = persist_line[pl_num][(mr_pl_ds_num, mr_pl_img_num)]
                    if len(persist_line) < 7:
                      print("pl ds,img num:", mr_pl_ds_num, mr_pl_img_num)
                      # print("pl_angle_lst:", pl_angle_lst)
                    for pl_angle in pl_angle_lst:
                      pl_item = angle_sd[pl_angle]
                      pl_ds_num, pl_img_num, pl_line = pl_item
                      dist = parallel_dist(pl_line, a_line)
                      # if len(persist_line) < 7:
                      if True:
                        cont = True
                        for i in range(4):
                          if abs(pl_line[0][i] - a_line[0][i]) > 4:
                           cont = False
                        cont = False
                        if cont:
                          dist = parallel_dist(pl_line, a_line, True)
                          print("pl_angle, item:", pl_angle, pl_item)
                          print("pl_line:", pl_line, a_line)
                          print("pl_dist:", dist)
                          if dist is not None:
                            print(min_dist,sqrt(dist[0]**2+dist[1]**2))
                      if dist is None:
                        # lines may be broken continutions of each other 
                        extended_line = self.is_broken_line(pl_line, a_line)
                        if extended_line is not None:
                          dist = parallel_dist(pl_line, extended_line)
                          if len(persist_line) < 7:
                            print("dist:", dist, pl_line, extended_line)
                      if dist is not None and min_dist > sqrt(dist[0]**2+dist[1]**2):
                        min_dist = sqrt(dist[0]**2+dist[1]**2)
                        best_pl_num = pl_num
                        best_pl_ds_num = mr_pl_ds_num
                        best_pl_img_num = mr_pl_img_num
                  # if ds_num == best_pl_ds_num: 
                  #   print("min_dist", min_dist, abs(img_num - best_pl_img_num), best_pl_img_num)
                  if ds_num == best_pl_ds_num and min_dist < abs(img_num - best_pl_img_num+1) * min_dist_allowed:
                    # same line
                    # print("p_l",persist_line[best_pl_num])
                    # print("p_l2", persist_line[best_pl_num][(best_pl_ds_num, best_pl_img_num)])
                    try:
                      lst = persist_line[best_pl_num][(ds_num, img_num)]
                      lst.append(angle)
                    except:
                      persist_line[best_pl_num][(ds_num, img_num)] = []
                      persist_line[best_pl_num][(ds_num, img_num)].append(angle)
                    persist_line_most_recent[best_pl_num] = [ds_num, img_num]
                  else:
                    persist_line_most_recent.append([ds_num, img_num])
                    if len(persist_line) < 7:
                        print("best pl, ds, img:", best_pl_num, best_pl_ds_num, best_pl_img_num)
                    persist_line.append({})
                    persist_line[-1][(ds_num, img_num)] = []
                    persist_line[-1][(ds_num, img_num)].append(angle)
                # persist_line_stats = []
                mean_gripper_line = []
                mean_gripper_line1 = []
                mean_gripper_line2 = []
                # persist_line_counter = []
                # persist_line_min_max_ds_img = []
                non_gripper = []
                none_disp = []
                big_count = []
                density = []
                for pl_num, pl in enumerate(persist_line):
                  print("PERSISTENT LINE #", pl_num)
                  persist_line_stats.append(None)
                  counter = 0
                  running_sum_x = 0
                  running_sum_y = 0
                  running_sum2_x = 0
                  running_sum2_y = 0
                  got_disp = False
                  dispcnt = 0
                  running_sum_counter = 0
                  running_sum2_counter = 0
                  running_sum_disp_x = 0
                  running_sum_disp_y = 0
                  running_sum2_disp_x = 0
                  running_sum2_disp_y = 0
                  running_line_length = 0
                  running_line = [0,0,0,0]
                  running_angle = 0
                  # for ds = 0, get max/min img#
                  persist_line_min_max_ds_img.append([1000000, -1])
                  a_ds_num = -1
                  a_img_num = -1
                  a_line = []
                  for pl_ds_num in range(num_ds):
                    if pl_ds_num > 0:
                      # TODO: num_imgs depends on ds_num; get ds_num 0 to work first.
                      print("pl_ds_num > 0", pl_ds_num)
                      break
                    for pl_img_num in range(num_imgs):
                      try:
                        angle_list = pl[(pl_ds_num, pl_img_num)]
                      except:
                        continue
                      # first and last line appearance
                      if pl_img_num  < persist_line_min_max_ds_img[pl_num][0]:
                        persist_line_min_max_ds_img[pl_num][0] = pl_img_num
                      if pl_img_num  > persist_line_min_max_ds_img[pl_num][1]:
                        persist_line_min_max_ds_img[pl_num][1] = pl_img_num
  
                      for pl_angle in angle_list:
                        prev_a_line = copy.deepcopy(a_line)
                        prev_ds_num = a_ds_num
                        prev_img_num = a_img_num
                        asd_item = angle_sd[pl_angle]
                        a_ds_num, a_img_num, a_line = asd_item
                        if prev_ds_num != -1:
                          disp = parallel_dist(prev_a_line, a_line)
                          if disp is not None:
                            got_disp = True
                            running_sum_disp_x += disp[0]
                            running_sum_disp_y += disp[1]
                            running_sum2_disp_x += disp[0] * disp[0]
                            running_sum2_disp_y += disp[1] * disp[1]
                            dispcnt += 1
                        running_sum_x += (a_line[0][0] + a_line[0][2])/2
                        running_sum_y += (a_line[0][1] + a_line[0][3])/2
                        for i in range(4):
                          running_line[i] += a_line[0][i]
                        x_dif = abs(a_line[0][0] - a_line[0][2])
                        y_dif = abs(a_line[0][1] - a_line[0][3])
                        running_line_length += sqrt(x_dif*x_dif + y_dif*y_dif)
                        running_angle += pl_angle
                        counter += 1
                  if counter == 0:
                    print("counter:", counter)
                    print("angle_list:", angle_list)
                    print("pl, ds, img:", pl_num, a_ds_num, a_img_num)
                    continue
                  if dispcnt == 0:
                    stddev_disp_x = None
                    stddev_disp_y = None
                    mean_disp_x = None
                    mean_disp_y = None
                  else:
                    stddev_disp_x = sqrt(running_sum2_disp_x / dispcnt - running_sum_disp_x * running_sum_disp_x / dispcnt / dispcnt)
                    stddev_disp_y = sqrt(running_sum2_disp_y / dispcnt - running_sum_disp_y * running_sum_disp_y / dispcnt / dispcnt)
                    mean_disp_x = running_sum_disp_x / dispcnt
                    mean_disp_y = running_sum_disp_y / dispcnt
                  mean_x = running_sum_x / counter
                  mean_y = running_sum_y / counter
                  mean_line_length = running_line_length / counter
                  mean_angle = running_angle / counter
                  mean_line = [[0,0,0,0]]
                  for i in range(4):
                    mean_line[0][i] = int(running_line[i]/counter)
                  print("mean disp, angle, linlen:", got_disp, mean_disp_x, mean_disp_y, mean_angle, mean_line_length, mean_line, counter) 
                  persist_line_stats[pl_num] = [mean_disp_x, mean_disp_y, stddev_disp_x, stddev_disp_y, mean_x, mean_y, copy.deepcopy(mean_line), mean_line_length, mean_angle, counter, copy.deepcopy(persist_line_min_max_ds_img[pl_num])]
                  persist_line_counter.append(counter)
                  running_sum_counter += counter
                  running_sum2_counter += counter * counter
                  if got_disp and (abs(mean_disp_x) + abs(mean_disp_y) < 1):
                    mean_gripper_line2.append(mean_line)
                    if got_disp and (abs(mean_disp_x) + abs(mean_disp_y) < .5):
                      mean_gripper_line1.append(mean_line)
                      if got_disp and (abs(mean_disp_x) + abs(mean_disp_y) < .001):
                        mean_gripper_line.append(mean_line)
                  elif got_disp:
                    non_gripper.append(mean_line)
                  else:
                    none_disp.append(mean_line)
                  
                  if counter > 50:
                    big_count.append(mean_line)
                    density.append((persist_line_min_max_ds_img[pl_num][1] - persist_line_min_max_ds_img[pl_num][0] +1) / counter)
                    if max_counter < counter:
                      max_counter = counter
                      max_counter_num = pl_num
                print("pl density:", density)
                sum_density = 0
                for dense in density:
                  sum_density += dense
                print("mean pl density:", (sum_density/len(big_count)))
                counter_cnt = len(persist_line_counter)
                mean_counter = running_sum_counter / counter_cnt
                stddev_counter = sqrt(running_sum2_counter / counter_cnt - running_sum_counter * running_sum_counter / counter_cnt / counter_cnt)
                print("counter mean, stdev", mean_counter, stddev_counter)
                # MIN_GRIPPER_Y
                for pl_num in [max_counter_num]:
                  print("PERSIST LINE", pl_num)
                  for pl_item in persist_line[pl_num].items():
                    pl_key, angle_list = pl_item
                    for a_num, pl_angle in enumerate(angle_list):
                      asd_item = angle_sd[pl_angle]
                      a_ds_num, a_img_num, a_line = asd_item
                      print(pl_key, a_num, a_line, persist_line_min_max_ds_img[pl_num])
  
                # print("mean_gripper_line1", len(mean_gripper_line1))
                # print("mean_gripper_line2", len(mean_gripper_line2))
                # print("non_gripper_line", len(non_gripper))
                print("none_disp", len(none_disp))
                print("mean_gripper_line", len(mean_gripper_line))
                self.display_lines("Mean_Gripper_Lines", mean_gripper_line, drop_off_img)
                cv2.waitKey(0)
                # display_lines("Mean_Gripper_Lines1", mean_gripper_line1, drop_off_img)
                # display_lines("Mean_Gripper_Lines2", mean_gripper_line2, drop_off_img)
                # display_lines("Mean_NonGripper_Line", non_gripper, drop_off_img)
                self.display_lines("Mean_BigCount", big_count, drop_off_img)
                cv2.waitKey(0)
  
                mean_counter = running_sum_counter / counter_cnt
                stddev_counter = sqrt(running_sum2_counter / counter_cnt - running_sum_counter * running_sum_counter / counter_cnt / counter_cnt)
                print("counter mean, stdev", mean_counter, stddev_counter)
                # MIN_GRIPPER_Y
              pl_last_img_line = {}
              for pl_ds_num in range(num_ds):
                for pl_img_num in range(num_imgs):
                  bb = []
                  bb_maxw, bb_minw, bb_maxh, bb_minh = -1, 10000, -1, 10000
                  for pl_num in range(len(persist_line)):
                    pl_stats = persist_line_stats[pl_num]
                    if pl_stats is not None:
                      [mean_disp_x, mean_disp_y, stddev_disp_x, stddev_disp_y, mean_x, mean_y, mean_line, mean_line_length, mean_angle, counter, [pl_min_img_num, pl_max_img_num]] = pl_stats
                      print("PL", pl_num, counter, mean_line, pl_min_img_num, pl_max_img_num)
                    else:
                      continue
                    if mean_y > MIN_GRIPPER_Y:
                      continue
                    if counter < mean_counter:
                      continue
                    a_line = None
                    try:
                      print("pl angle_list:")
                      angle_list = persist_line[pl_num][(pl_ds_num, pl_img_num)]
                      # print(angle_list)
                      l_maxw, l_minw, l_maxh, l_minh = -1, 10000, -1, 10000
                      for a_num, pl_angle in enumerate(angle_list):
                        asd_item = angle_sd[pl_angle]
                        a_ds_num, a_img_num, a_line = asd_item
                        l_maxw = max(a_line[0][0], a_line[0][2], l_maxw)
                        l_minw = min(a_line[0][0], a_line[0][2], l_minw)
                        l_maxh = max(a_line[0][1], a_line[0][3], l_maxh)
                        l_minh = min(a_line[0][1], a_line[0][3], l_minh)
                        if l_maxh > MIN_GRIPPER_Y:
                          l_maxh = MIN_GRIPPER_Y
                      pl_last_img_line[pl_num] = [l_maxw, l_minw, l_maxh, l_minh]
                    except:
                      print("except pl angle_list:")
                      try:
                        [l_maxw, l_minw, l_maxh, l_minh] = pl_last_img_line[pl_num]
                      except:
                        continue
                    if l_maxw == -1 or l_maxh == -1:
                      print("skipping PL", pl_num)
                      continue
                  bb = make_bb(bb_maxw, bb_minw, bb_maxh, bb_minh)
                  img_path = img_paths[pl_ds_num][pl_img_num]
                  img = cv2.imread(img_path)
                  bb_img = get_bb_img(img, bb)
                  print(pl_img_num, "bb", bb)
                  cv2.imshow("bb", bb_img)
                  # cv2.waitKey(0)
  
          # angle_list = pl[(pl_ds_num, pl_img_num)]
          # pl_last_img_line[pl_num] = [l_maxw, l_minw, l_maxh, l_minh]
          # display_lines("Mean_Gripper_Lines", mean_gripper_line, drop_off_img)
          # [max_counter_num]:
          # for pl_item in persist_line[pl_num].items():
          #   pl_key, angle_list = pl_item
          #   for a_num, pl_angle in enumerate(angle_list):
          #     asd_item = angle_sd[pl_angle]
          #     a_ds_num, a_img_num, a_line = asd_item
          # big_count.append(mean_line)
          # angle_list = persist_line[pl_num][(pl_ds_num, pl_img_num)]
          # persist_line_stats[pl_num] = [mean_disp_x, mean_disp_y, stddev_disp_x, stddev_disp_y, mean_x, mean_y, copy.deepcopy(mean_line), mean_line_length, mean_angle, counter, copy.deepcopy(persist_line_min_max_ds_img[pl_num])]
          # persist_line_metadata[pl_num] = [first_img, last_img, largest_gap, largest_consec, num_img]
  
          if False:
            return
            x = 1/0
            cv2.waitKey(0)
          if False:
                pass
  
                # display gripper lines
                # display_lines("gripper lines", gripper_lines[ds_num][img_num], drop_off_img)
                # display_lines("gripper lines", gripper_lines2, drop_off_img)
  
  
                # display parallel lines
                self.display_line_pairs("parallel lines", same_frame_parallel[ds_num][img_num], drop_off_img)
  
                # display diff_frame_parallel lines
                self.display_line_pairs("moved lines", diff_frame_parallel[ds_num][img_num], drop_off_img)
                found = (len(same_frame_parallel[ds_num][img_num])
                       + len(diff_frame_parallel[ds_num][img_num])
                       + len(broken_lines[ds_num][img_num]))
                if found and new_img_num: 
                  print("ds_num, img_num", ds_num, img_num)
                  cv2.waitKey(0)
                  new_img_num = False
                
              ######################
              # elif pass_num == 2:
  
        
              
  
    
              # TODO: eliminate moving line groups
    
              # find relationship of line groups
              # is one behind/above another?
    
              # estimate drop point into box
  
              # estimate position of camera
  
  def __init__(self):
      self.arm_nav = ArmNavigation()
  
      self.alset_state = AlsetState()
      self.cvu = CVAnalysisTools(self.alset_state)
      func_idx_file = "sample_code/TT_BOX.txt"
      dsu = DatasetUtils(app_name="TT", app_type="FUNC")
      dataset = [[]]
      self.curr_dataset = dataset[0]
      self.fwd_actions = [[]]
      self.arm_pos = []
      unmoved_pix = None
      slow_moved_pix = None
      self.delta_arm_pos = {"UPPER_ARM_UP":0,"UPPER_ARM_DOWN":0,
                            "LOWER_ARM_UP":0,"LOWER_ARM_DOWN":0}
      self.final_delta_arm_pos = []
      self.arm_nav.set_current_position(self.delta_arm_pos, update_plot=True)
      drop_off_img = []
      drop_off_img.append(None)
      num_datasets = 0
      num_images = 0
      func_name = ""
      prev_func_name = ""
      unique_color = {}
      curr_ds_num = 0
      with open(func_idx_file, 'r') as file1:
        while True:
          ds_idx = file1.readline()
          if not ds_idx:
            break
          if ds_idx[-1:] == '\n':
            ds_idx = ds_idx[0:-1]
          # ./apps/FUNC/GOTO_BOX_WITH_CUBE/dataset_indexes/FUNC_GOTO_BOX_WITH_CUBE_21_05_16a.txt
          prev_func_name = func_name
          func_name = dsu.get_func_name_from_idx(ds_idx)
          # if func_name == "GOTO_BOX_WITH_CUBE" and prev_func_name != "GOTO_BOX_WITH_CUBE":
          if prev_func_name == "DROP_CUBE_IN_BOX" and func_name != "DROP_CUBE_IN_BOX":
            # func_name = dsu.dataset_idx_to_func(ds_idx)
            self.final_delta_arm_pos.append(copy.deepcopy(self.delta_arm_pos))
            if curr_ds_num == 1:
              print("BREAK")
              break
            self.delta_arm_pos = {"UPPER_ARM_UP":0,"UPPER_ARM_DOWN":0,
                                  "LOWER_ARM_UP":0,"LOWER_ARM_DOWN":0}
            self.arm_nav.set_current_position(self.delta_arm_pos, update_plot=True)
            drop_off_img.append(None)
            curr_ds_num += 1
            self.fwd_actions.append([])
            dataset.append([])
            self.curr_dataset = dataset[curr_ds_num]
          self.arm_pos = []
          with open(ds_idx, 'r') as file2:
            while True:
              img_line = file2.readline()
              # 21:24:34 ./apps/FUNC/QUICK_SEARCH_FOR_BOX_WITH_CUBE/dataset/QUICK_SEARCH_FOR_BOX_WITH_CUBE/LEFT/9e56a302-b5fe-11eb-83c4-16f63a1aa8c9.jpg
              if not img_line:
                 break
              print("img_line", img_line)
              self.curr_dataset.append(img_line)
              [time, app, mode, func_name, action, img_name, img_path] = dsu.get_dataset_info(img_line,mode="FUNC") 
              self.fwd_actions[curr_ds_num].append(action)
              if action == "GRIPPER_OPEN":
                img = cv2.imread(img_path)
                drop_off_img[curr_ds_num] = img
                print("drop_off_img", curr_ds_num, len(drop_off_img))
              if action in ["UPPER_ARM_UP", "UPPER_ARM_DOWN", "LOWER_ARM_UP", "LOWER_ARM_DOWN"]:
                self.delta_arm_pos[action] += 1
                print(self.delta_arm_pos.items())
                img = cv2.imread(img_path)
                self.arm_nav.set_current_position(self.delta_arm_pos, update_plot=True, img=img)
              self.arm_pos.append(self.delta_arm_pos.copy())
  
      # print("dataset",dataset)
      for doi in drop_off_img:
        if doi is None:
          print("drop_off_img None, cnt", len(drop_off_img))
          print("actions", self.fwd_actions)
          
      img = None
      img_paths = []
      prev_img = None
      num_passes = 4
      box_lines = []
      actions = []
      # img_copy = []
      edge_copy = []
      rl_copy = []
      delta_arm_pos_copy = self.delta_arm_pos.copy()
      compute_gripper = True
      for pass_num in range(num_passes):
        # if pass_num > 0:
        #   self.delta_arm_pos = delta_arm_pos_copy.copy()
        for ds_num, ds in enumerate(dataset):
          # img_copy_num = len(ds) // 24
          ################################
          # REMOVE to handle more ds_nu
          ################################
          self.delta_arm_pos = copy.deepcopy(self.final_delta_arm_pos[ds_num])
          self.arm_nav.set_current_position(self.delta_arm_pos, update_plot=True)
          if pass_num == 0:
            box_lines.append([])
            actions.append([])
            img_paths.append([])
          for img_num, img_line in enumerate(reversed(ds)):
            # note: img_num is really the reversed img num!
            # img_num = len(ds) - rev_img_num
            [time, app, mode, func_name, action, img_name, img_path] = dsu.get_dataset_info(img_line,mode="FUNC") 
            prev_action = action
            if img is not None:
              prev_img = img
            img = cv2.imread(img_path)
            # print("img_path", img_path)
            if action in ["UPPER_ARM_UP", "UPPER_ARM_DOWN", "LOWER_ARM_UP", "LOWER_ARM_DOWN"]:
              self.delta_arm_pos[action] -= 1   # going in reversed(ds) order
              self.arm_nav.set_current_position(self.delta_arm_pos, update_plot=True, img=img)
            if pass_num == 0:
              adj_img,mean_diff,rl = self.cvu.adjust_light(img_path)
              if rl is not None:
                rl_img = img.copy()
                mask = rl["LABEL"]==rl["LIGHT"]
                mask = mask.reshape((rl_img.shape[:2]))
                # print("mask",mask)
                rl_img[mask==rl["LIGHT"]] = [0,0,0]
                # adj_img = rl_img
                center = np.uint8(rl["CENTER"].copy())
                rl_copy = rl["LABEL"].copy()
                res    = center[rl_copy.flatten()]
                rl_img2  = res.reshape((img.shape[:2]))
              if prev_img is not None:
                gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 200, None, 3)
                edges = cv2.dilate(edges,None,iterations = 1)
                d_edges = cv2.dilate(edges,None,iterations = 3)
                edge_copy.append(d_edges.copy())
                actions[ds_num].append(action)
              box_lines[ds_num].append([])
  
            ##################################
            #  Compute Gripper
            ##################################
            elif pass_num == 1:
              if compute_gripper:
                compute_gripper = False
                num_ors = int(np.sqrt(len(edge_copy)))
                # 3, 12, 13
                print("num edge_copy", num_ors, len(edge_copy), len(ds))
                running_ors = None
                running_ands = None
                do_or = 0
                for i2 in range(2, len(edge_copy)-2):
                  if do_or == 0:
                    running_ors = cv2.bitwise_or(edge_copy[i2-1], edge_copy[i2])
                    # running_ors = cv2.bitwise_or(edge_copy[i2-1], robot_light[i2])
                    do_or = 2
                  else:
                    running_ors = cv2.bitwise_or(running_ors, edge_copy[i2])
                    # running_ors = cv2.bitwise_or(running_ors, robot_light[i2])
                    do_or += 1
                  if do_or == num_ors:
                    do_or = 0
                    if running_ands is None:
                      running_ands = running_ors
                    else:
                      running_ands = cv2.bitwise_and(running_ands, running_ors)
                gripper = running_ands
                cv2.imshow("running_gripper", running_ands)
                # cv2.waitKey(0)
              # overwrite previous attempt at getting gripper.
              print("box_lines sz", len(box_lines), len(box_lines[ds_num]), img_num)
              # 1,0,0
              box_lines[ds_num][img_num] = self.get_box_lines(img, gripper)
              box_lines_image = self.get_hough_lines_img(box_lines[ds_num][img_num], img)
              cv2.imshow("box hough lines", box_lines_image)
            elif pass_num == 2:
              self.analyze_box_lines(dataset, box_lines, actions, gripper, drop_off_img[ds_num], img_paths)
  
            elif False and pass_num == 3:
              stego_img, unique_color = stego(img, unique_color)
              cv2.imshow("stego orig input img", img)
              cv2.imshow("stego img", stego_img)
              stego_gray = cv2.cvtColor(stego_img, cv2.COLOR_BGR2GRAY)
              stego_hough_lines = self.get_box_lines(stego_img)
              stego_hough_lines_image = self.get_hough_lines_img(stego_hough_lines, stego_gray)
              # cv2.imshow("stego hough lines", stego_hough_lines_image)
              cv2.imshow("stego gray", stego_gray)
  
              # cv2.imshow("stego adj input img", adj_img)
              # cv2.imshow("stego input img", rl_img)
              # cv2.imshow("stego rl img", rl_img2)
              cv2.waitKey(0)
  
  
  # Map lines to stego_number_list
  # map stego_number_list to role list:
  #   - Top of box
  #     - arm pos on top of box
  #     - camera angle
  #     - bounding box
  #   - box boundary (line around top)
  #   - table
  #     - front/side of full box
  #   - off table
  #   - gripper: 
  #     - gripper with cube 
  #     - gripper without cube 
  #   - front/side of box
  #     - Size when arm is parked
  #     - in front of top
  #     - bounding box
  #
  #   - outside table
  #   - unknown
  #
  # Arm Estimated Position
  #   - estimated camera angle
  #
  # Map stego categories from beginning:
  #   - likely-cube
  #   - confirmed-cube
  #   - rectangular-table
  #   - off table
  #   - likely-box
  #   - confirmed-box
  
  
  ###########################################
  # pickup -> bounding box
  # pickup -> work backwards
  #
  ###########################################
  # estimate distance to pick-up object
  # estimate location in rectangle
  # 
  ###########################################
  # Circle around, find rectangular bounds
  # Find little thing within boundary
  # Find Big think within boundary
  # Put little thing on/in big thing
  # obstacle avoidance
  
if __name__ == '__main__':
  box = AnalyzeLines()
