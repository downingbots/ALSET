import numpy as np
import cv2
import math
import statistics
from imutils import *
from config import *
from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern
from shapely.geometry import *
from pprint import pprint
from utilborders import *

class CVAnalysisTools():
  def __init__(self, alset_state=None):
      # foreground/background for movement detection
      self.alset_state = alset_state
      self.background = None
      self.foreground = None
      self.unmoved_pix = None
      self.INFINITE = 1000000000000000000
      self.BLACK = [0,0,0]
      self.prev_foreground = None
      self.cfg = Config()
      # store the number of points and radius for color histogram
      self.n_points = None
      self.radius = None
      self.TEXTURE_METHOD = 'uniform'
      self.textures = {}
      self.MeanLight = None
      self.MeanLightCnt = None

  def load_state(self):
      blckbrd = self.alset_state.get_blackboard("CV_LIGHT")
      if blckbrd is None:
        self.MeanLightCnt = 0 
        self.MeanLight = 0
      else:
        [self.MeanLightCnt, self.MeanLight] = blckbrd

  ###############################################
  # should remove from automated func
  # True means moving. False means not moving.
  def optflow(self, old_frame_path, new_frame_path, add_edges=False, thresh=None):
      if old_frame_path is None:
        print("optflow: old_frame None")
        return True
      # old_frame = cv2.imread(old_frame_path)
      # new_frame = cv2.imread(new_frame_path)
      old_frame,mean_diff,rl = self.adjust_light(old_frame_path)
      new_frame,mean_diff,rl = self.adjust_light(new_frame_path)
      opt_flow_results = self.optflow_pts(old_frame, new_frame, add_edges, thresh)
      return opt_flow_results["result"]

  def optflow_pts(self, old_frame, new_frame, add_edges=False, thresh=None):
      # cap = cv2.VideoCapture('slow.flv')
      # params for ShiTomasi corner detection
      optflow_results = {}

      feature_params = dict( maxCorners = 100,
                             qualityLevel = 0.3,
                             minDistance = 7,
                             blockSize = 7 )
      # Parameters for lucas kanade optical flow
      lk_params = dict( winSize  = (15,15),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
      # Take first frame and find corners in it
      # ret, old_frame = cap.read()
      old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
      p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
      # Create a mask image for drawing purposes
      mask = np.zeros_like(old_frame)
  
      frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
      if add_edges:
        old_gray = cv2.Canny(old_gray, 50, 200, None, 3)
        frame_gray = cv2.Canny(frame_gray, 50, 200, None, 3)
      # calculate optical flow
      try:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
      except:
        print("OPT FLOW FAILS")
        optflow_results["result"] = False
        optflow_results["prevpts"] = None
        optflow_results["currpts"] = None
        return False
      # Select good points
      good_new = p1[st==1]
      good_old = p0[st==1]
      # draw the tracks
      dist = 0
      numpts = 0
      # color = np.random.randint(0,255,(100,3))
      frame1 = new_frame
      gn_pts = []
      go_pts = []
      for i,(new,old) in enumerate(zip(good_new,good_old)):
          a,b = new.ravel()
          c,d = old.ravel()
          gn_pts.append([int(a), int(b)])
          go_pts.append([int(c), int(d)])
          dist += math.hypot(a-c,b-d)
          numpts += 1
          # mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
          # frame1 = cv2.circle(frame1,(a,b),5,color[i].tolist(),-1)
      gn_pts = np.array(gn_pts)
      go_pts = np.array(go_pts)
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
# BUG: optflow is too variable. Need a way to make it more predictable
# or replace it with something else...  For now, look for a successful
# behavior of PARK_ARM_RETRACTED.
      if thresh is None:
        thresh = self.cfg.OPTFLOWTHRESH
      if dist > thresh:
        optflow_results["result"] = True
        optflow_results["prevpts"] = go_pts
        optflow_results["currpts"] = gn_pts
      else:
        optflow_results["result"] = False
        optflow_results["prevpts"] = go_pts
        optflow_results["currpts"] = gn_pts
      return optflow_results

  # ARD: does not work well
  def moved_pixels_over_time(self, prev_img_path, curr_img_path, init=False):
      # height, width, channels = prev_img.shape
      if self.background is None or init:
        prev_img = cv2.GaussianBlur(prev_img, (5, 5), 0)
        self.background = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        print("init")
      elif init and self.foreground is not None:
        # self.background = cv2.absdiff(self.foreground_inv, self.background)
        self.background = cv2.add(self.foreground, self.background)
        print("bg")
      elif self.prev_foreground is not None:
        fgdiff = cv2.absdiff(self.foreground, self.prev_foreground)
        # self.foreground = cv2.absdiff(self.foreground, fgdiff)
        # cv2.imshow("Gripper FGdiff", fgdiff)
        # self.background = cv2.absdiff(self.background, fgdiff)
        # self.background = cv2.add(fgdiff, self.background)
        # cv2.accumulateWeighted(fgdiff, self.background.astype(float), 0.75)
        print("prevfg")
        pass
      if self.foreground is not None:
         self.prev_foreground = self.foreground.copy()
      curr_img = cv2.GaussianBlur(curr_img, (5, 5), 0)
      gray_curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
      diff = cv2.absdiff(gray_curr_img, self.background)
      # diff = cv2.absdiff(self.background, gray_curr_img)
      # diff = cv2.add(gray_curr_img, self.background)
      self.foreground_inv = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY_INV)[1]
      self.foreground_bin = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)[1]
      self.foreground = self.foreground_inv
      #  self.foreground = self.foreground_inv
      # else:
      #   self.foreground = self.foreground_bin
      # self.foreground = cv2.absdiff(self.background, self.foreground)
      # self.foreground = cv2.add(self.foreground, self.background)
      # cv2.accumulateWeighted(self.foreground, self.background.astype(float), 0.1)
      # cv2.accumulateWeighted(self.foreground, self.background.astype(float), 0.5)
#     #### Adding Contours didn't help!
#      contours, hierarchy = cv2.findContours(self.foreground, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#      contour_list = []
#      for contour in contours:
#        M = cv2.moments(contour)
#        area = cv2.contourArea(contour)
#        # if area > 100 :
#        #     contour_list.append(contour)
#        contour_list.append(contour)
#      image = self.foreground.copy()
#      cv2.drawContours(image, contour_list, -1, (0, 255, 0), 2)
#      cv2.imshow("Gripper Output", image)
      # cv2.imshow("Gripper FG", self.foreground)
      # cv2.imshow("Gripper prev", prev_img)
      # cv2.imshow("Gripper curr", curr_img)
      # cv2.imshow("Gripper Output", self.foreground)
      # cv2.imshow("Gripper Background", self.background)
      # cv2.waitKey(0)
      return self.foreground

  def moved_pixels(self, prev_img_path, curr_img_path, init=False, add_edges=False):
      # prev_img = cv2.imread(prev_img_path)
      # curr_img = cv2.imread(curr_img_path)
      curr_img,mean_diff,rl = self.adjust_light(curr_img_path)
      prev_img,mean_diff,rl = self.adjust_light(prev_img_path)
      if add_edges:
        prev_img = cv2.Canny(prev_img, 50, 200, None, 3)
        curr_img = cv2.Canny(curr_img, 50, 200, None, 3)
      # thresh = 10
      thresh = 20
      prev_img = cv2.GaussianBlur(prev_img, (5, 5), 0)
      self.background = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
      curr_img = cv2.GaussianBlur(curr_img, (5, 5), 0)
      gray_curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
      diff = cv2.absdiff(gray_curr_img, self.background)
      self.foreground = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY_INV)[1]
      # cv2.imshow("Gripper FG", self.foreground)
      # cv2.imshow("Gripper IM", gray_curr_img)
      # cv2.waitKey(0)
      # invert mask and combine with original image - this makes the black outer edge white
      mask_inv = cv2.bitwise_not(self.foreground)
      moved_pix = cv2.bitwise_or(gray_curr_img, mask_inv)
      # cv2.imshow("moved pix", moved_pix)
      left_bb, right_bb, moved_pix = self.get_gripper_bounding_box(moved_pix, curr_img)
      return left_bb, right_bb, moved_pix

  def get_gripper_bounding_box(self, unmoved_pix, image):
      ret, contour_thresh = cv2.threshold(unmoved_pix.copy(), 125, 255, 0)
      contour_thresh = cv2.dilate(contour_thresh,None,iterations = 10)

      contours, hierarchy = cv2.findContours(contour_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
      # print("hierarchy:", hierarchy)

      # cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
      #   cv2.CHAIN_APPROX_SIMPLE)

      # for accuracy in range(10):
      #   accuracy_factor = .1 * (accuracy+1)
      # accuracy_factor = .01
      accuracy_factor = .1
      x,y,ch = image.shape
      max_left_gripper_bounding_box = [[[0, int(y*.4)]], [[0, y-1]],      
                        [[ int(x/2), y-1 ]], [[ int(x/2), int(y*.4)]]]
      max_right_gripper_bounding_box =[[[int(x/2),int(y*.4)]],[[x-1,int(y*.4)]],
                                       [[x-1, y-1]], [[int(x/2), y-1]]]
      left_bb = max_left_gripper_bounding_box
      right_bb = max_right_gripper_bounding_box
      left_found, right_found = False, False
      for c in contours:
        if len(contours) > 1:
          area = cv2.contourArea(c)
          if area < 16:
            continue
          # Calculate accuracy as a percent of the contour perimeter
          accuracy = accuracy_factor * cv2.arcLength(c, True)
          approx = cv2.approxPolyDP(c, accuracy, True)
          if len(approx) < 2:
            continue
          if check_gripper_bounding_box(max_left_gripper_bounding_box, approx):
            print("Found Left Gripper Bounding Box")
            if len(approx) == 2:
              maxx = max(approx[0][0][0], approx[1][0][0])
              minx = min(approx[0][0][0], approx[1][0][0])
              maxy = max(approx[0][0][1], approx[1][0][1])
              miny = min(approx[0][0][1], approx[1][0][1])
              print("0maxy, miny, maxx, minx:", maxy, miny, maxx, minx)
            else:
              maxx, minx, maxy, miny = get_min_max_borders(approx)
              print("1maxx, minx, maxy, miny:", maxx, minx, maxy, miny)
            if left_found:  
              # gripper is represented by multiple polygons
              maxx = max(maxx, left_bb[2][0][0])
              miny = min(miny, left_bb[0][0][1])
            left_bb = [[[0, miny]],[[0, y-1]],
                       [[maxx, y-1]], [[maxx, miny]]]
            left_found = True
            print("approx:", approx)
            print("l_bb:", left_bb)
          elif check_gripper_bounding_box(max_right_gripper_bounding_box,approx):
            print("Found Right Gripper Bounding Box")
            if len(approx) == 2:
              maxx = max(approx[0][0][0], approx[1][0][0])
              minx = min(approx[0][0][0], approx[1][0][0])
              maxy = max(approx[0][0][1], approx[1][0][1])
              miny = min(approx[0][0][1], approx[1][0][1])
              print("2maxy, miny, maxx, minx:", maxy, miny, maxx, minx)
            else:
              maxx, minx, maxy, miny = get_min_max_borders(approx)
            print("3maxx, minx, maxy, miny:", maxx, minx, maxy, miny)
            if right_found:  
              # gripper is represented by multiple polygons
              minx = min(minx, right_bb[0][0][0])
              miny = min(miny, right_bb[0][0][1])
            right_bb = [[[minx, miny]],[[x-1, miny]],
                        [[x-1, y-1]],[[minx, y-1]]]
            print("approx:", approx)
            print("r_bb:", right_bb)
            right_found = True
      # confirm symmetrical gripper
      if left_found and right_found:
        lminy = left_bb[0][0][1]
        lmaxx = left_bb[2][0][0]
        rminx = right_bb[0][0][0]
        rminy = right_bb[0][0][1]

        miny = min(lminy, rminy)
        if lmaxx < x - rminx - 1:
          lmaxx = x - rminx - 1
        else:
          rminx = x - lmaxx - 1

        left_bb = [[[0, miny]],[[0, y-1]], [[lmaxx, y-1]], [[lmaxx, miny]]]
        right_bb = [[[rminx, miny]],[[x-1, miny]],[[x-1, y-1]],[[rminx, y-1]]]
      l_bb = np.int0(left_bb)
      r_bb = np.int0(right_bb)
      print("l_bb, r_bb", left_bb, right_bb)

      image = cv2.drawContours(image.copy(), [l_bb], 0, (0, 255, 0), 2)
      image = cv2.drawContours(image, [r_bb], 0, (0, 255, 0), 2)
      return left_bb, right_bb, image

  def unmoved_pixels(self, prev_img_path, curr_img_path, init=False, init_pix=None):
      try:
        # prev_img = cv2.imread(prev_img_path)
        # curr_img = cv2.imread(curr_img_path)
        curr_img,mean_diff,rl = self.adjust_light(curr_img_path)
        prev_img,mean_diff,rl = self.adjust_light(prev_img_path)
      except:
        prev_img = prev_img_path.copy()
        curr_img = curr_img_path.copy()
      gray = cv2.cvtColor(curr_img.copy(), cv2.COLOR_BGR2GRAY)
      edges = cv2.Canny(gray, 50, 200, None, 3)
      # ARD: dilation also works, and is possibly better in some situations
      # edges = cv2.dilate(edges,None,iterations = 10)
      prev_gray = cv2.cvtColor(prev_img.copy(), cv2.COLOR_BGR2GRAY)
      prev_edges = cv2.Canny(prev_gray, 50, 200, None, 3)
      # ARD: dilation also works, and is possibly better in some situations
      # prev_edges = cv2.dilate(prev_edges,None,iterations = 10)
      if init and init_pix is not None:
        self.unmoved_pix = init_pix
      if init and self.unmoved_pix is None:
        self.unmoved_pix = prev_edges
        # self.unmoved_pix = edges
      if True:
        # print("unmoved pix:", self.unmoved_pix)
        # cv2.imshow("unmoved pix", self.unmoved_pix)
        # cv2.imshow("edges", edges)
        # cv2.waitKey(0)
        diff = cv2.absdiff(self.unmoved_pix, edges)
        prev_diff = cv2.absdiff(prev_edges, edges)
        for h in range(prev_img.shape[0]):
          for w in range(prev_img.shape[1]):
            if int(diff[h][w]) > 30 and self.unmoved_pix[h][w] > 50:
              # edge is 0 or 255
              # diff can result in increase or decrease depending on edge
              self.unmoved_pix[h][w] = int(.5*self.unmoved_pix[h][w] + .5*edges[h][w])
              # print("edge, dif, prevdif:", edges[h][w], diff[h][w], prev_diff[h][w])
            elif int(diff[h][w]) > 30 and prev_diff[h][w] > 30:
              self.unmoved_pix[h][w] = int(.9*self.unmoved_pix[h][w] + .1*edges[h][w])
              # self.unmoved_pix[h][w] = int(.75*self.unmoved_pix[h][w] + .25*edges[h][w])
#            elif (int(diff[h][w]) > 30):
#              for i in range(-1,2):
#                for j in range(-1,2):
#                  x = min(max(0,h+i), prev_img.shape[0]-1)
#                  y = min(max(0,w+i), prev_img.shape[1]-1)
#                  if self.unmoved_pix[x][y] > 50:
#                    self.unmoved_pix[x][y] = int(.9*self.unmoved_pix[x][y] + .1*edges[h][w])

      # contours, image = self.unmoved_pixel_contours(self.unmoved_pix, curr_img.copy())
      left_bb, right_bb, image = self.get_gripper_bounding_box(self.unmoved_pix, curr_img.copy())
      # cv2.imshow("bounding box:", image)
      # cv2.imshow("unmoved_pix2", self.unmoved_pix)
      # cv2.imshow("Gripper CI", curr_img)
      # cv2.waitKey(0)

      left_bb, right_bb, image = self.get_gripper_bounding_box(self.unmoved_pix, curr_img.copy())
      return self.unmoved_pix.copy(), left_bb, right_bb

  ###############################################
  # Color histogram for texture mapping (e.g., roads, floor)
  ###############################################
  # plot the color histograms using opencv
# Moved to analyze_color.py
#  def draw_image_histogram(image, channels, color='k'):
#      hist = cv2.calcHist([image], channels, None, [256], [0, 256])
#      plt.plot(hist, color=color)
#      plt.xlim([0, 256])
#  
#  def show_color_histogram(image):
#      for i, col in enumerate(['b', 'g', 'r']):
#          draw_image_histogram(image, [i], color=col)
#      plt.show()
#
#  def kullback_leibler_divergence(self, p, q):
#      p = np.asarray(p)
#      q = np.asarray(q)
#      filt = np.logical_and(p != 0, q != 0)
#      return np.sum(p[filt] * np.log2(p[filt] / q[filt]))
#  
#  def match_texture(self, img):
#      best_score = 10
#      best_name = None
#      lbp = local_binary_pattern(img, self.n_points, self.radius, self.TEXTURE_METHOD)
#      n_bins = int(lbp.max() + 1)
#      hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
#      for label in self.textures.keys():
#          ref_hist, _ = np.histogram(self.textures[label], density=True, bins=n_bins,
#                                     range=(0, n_bins))
#          score = kullback_leibler_divergence(hist, ref_hist)
#          # lower score is better
#          if score < best_score:
#              best_score = score
#              best_name = label
#      return best_name
# 
#  def register_texture(self, label, image, radius):
#      lbp = local_binary_pattern(image, self.n_points, self.radius, self.TEXTURE_METHOD)
#      self.textures[label] = lbp
#
  ###############################################
  def get_lines(self, img):
      # Canny: Necessary parameters are:
      #   image: Source/Input image of n-dimensional array.
      #   threshold1: It is the High threshold value of intensity gradient.
      #   threshold2: It is the Low threshold value of intensity gradient.
      # Canny: Optional parameters are:
      #   apertureSize: Order of Kernel(matrix) for the Sobel filter. 
      #      Its default value is (3 x 3), and its value should be odd between 3 and 7. 
      #      It is used for finding image gradients. Filter is used for smoothening and 
      #      sharpening of an image.
      #   L2gradient: This specifies the equation for finding gradient magnitude. 
      #      L2gradient is of boolean type, and its default value is False.
      # edges = cv2.Canny(img, 75, 200, None, 3)
      # edges = cv2.Canny(img, 50, 200, None, 3)
      # edges = cv2.Canny(img.copy(), 50, 200, None, 3)
      edges = cv2.Canny(img.copy(), 10, 245, None, 3)
      # edges = cv2.Canny(img,100,200)
      # Copy edges to the images that will display the results in BGR
      imglinesp = np.copy(img)
      # HoughLinesP Parameters:
      #   image: 8-bit, single-channel binary source image. 
      #   lines:Output vector of lines.
      #   rho: Distance resolution of the accumulator in pixels.
      #   theta: Angle resolution of the accumulator in radians.
      #   threshold: Accumulator threshold parameter. Only those lines are returned that get 
      #       enough votes ( >threshold ).
      #   minLineLength: Minimum line length. Line segments shorter than that are rejected.
      #   maxLineGap: Maximum allowed gap between points on the same line to link them.
      #
      #   linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 10, 10)
      # linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 20, 20)
      linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 40, 40)
      HB = HoughBundler()
      mergedLinesP = HB.process_lines(linesP, edges)
      mlLinesP = []                 # put back in HoughLineP format
      if mergedLinesP is not None:
          print("num linesP:", len(linesP), len(mergedLinesP))
          for i in range(0, len(mergedLinesP)):
              l0 = mergedLinesP[i][0]
              l1 = mergedLinesP[i][1]
              cv2.line(imglinesp, (l0[0], l0[1]), (l1[0], l1[1]), (0,255,0), 3, cv2.LINE_AA)
              mlLinesP.append([[l0[0], l0[1], l1[0], l1[1]]])
      # cv2.imshow("lines:", imglinesp)
      # cv2.waitKey(0)
      return mlLinesP, imglinesp

  def find_longest_line(self, linesP, border=None):
      max_dist = 0
      map_line = None
      map_dx = None
      map_dy = None
      map_slope = None
      in_brdr_cnt = 0
      for [[l0,l1,l2,l3]] in linesP:
          if border is None or self.line_in_border(border, (l0,l1), (l2,l3)):
            # print("in border:", l0,l1,l2,l3)
            in_brdr_cnt += 1
            dx = l0 - l2
            dy = l1 - l3
            dist = np.sqrt(dx**1 + dy**2)
            if max_dist < dist:
              map_line = [l0,l1,l2,l3]
              map_dx = dx
              map_dy = dy
              map_slope = np.arctan2(map_dx, map_dy)
              if map_slope < 0:   # keep all slopes positive
                map_slope = 2 * np.pi + map_slope
              max_dist = dist
      return max_dist, map_line, map_dx, map_dy, map_slope, in_brdr_cnt

  def rectangle_within_image(self, img):
      # the following is not what we're looking for mapping, but may serve 
      # as a prototype for now. The following cuts away more than the borders
      # to get a clean image. We just want to know what the external borders are.
      #
      # convert the stitched image to grayscale and threshold it
      # such that all pixels greater than zero are set to 255
      # (foreground) while all others remain 0 (background)

      # input image is expected to already be gray
      # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

      thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)[1]

      # find all external contours in the threshold image then find
      # the *largest* contour which will be the contour/outline of
      # the image
      cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
      	cv2.CHAIN_APPROX_SIMPLE)
      cnts = imutils.grab_contours(cnts)
      c = max(cnts, key=cv2.contourArea)
      # allocate memory for the mask which will contain the
      # rectangular bounding box of the image region
      mask = np.zeros(thresh.shape, dtype="uint8")
      (x, y, w, h) = cv2.boundingRect(c)
      cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
      # create two copies of the mask: one to serve as our actual
      # minimum rectangular region and another to serve as a counter
      # for how many pixels need to be removed to form the minimum
      # rectangular region
      minRect = mask.copy()
      sub = mask.copy()
      # keep looping until there are no non-zero pixels left in the
      # subtracted image
      while cv2.countNonZero(sub) > 0:
      	# erode the minimum rectangular mask and then subtract
      	# the thresholded image from the minimum rectangular mask
      	# so we can count if there are any non-zero pixels left
      	minRect = cv2.erode(minRect, None)
      	sub = cv2.subtract(minRect, thresh)
      # find contours in the minimum rectangular mask and then
      # extract the bounding box (x, y)-coordinates
      cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL,
      	cv2.CHAIN_APPROX_SIMPLE)
      cnts = imutils.grab_contours(cnts)
      c = max(cnts, key=cv2.contourArea)
      (x, y, w, h) = cv2.boundingRect(c)
      return (x, y, w, h) 

  
  def color_quantification(self, img, num_clusters):
      # try color quantification
      Z = img.copy()
      Z = Z.reshape((-1,3))
      # convert to np.float32
      Z = np.float32(Z)
      # define criteria, number of clusters(K) and apply kmeans()
      criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
      K = num_clusters
      # compactness : sum of squared distance from each point to their centers.
      # labels : the label array where each element marked '0', '1'.....
      # centers : This is array of centers of clusters.
      ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
      # Now convert back into uint8, and make original image
      # print("compactness, centers:", ret, center)
      # ret is a single float, label is ?, center is RGB
      center = np.uint8(center)
      res = center[label.flatten()]
      res2 = res.reshape((img.shape))
      return res2

  def order_points(self, pts):
      # initialzie a list of coordinates that will be ordered
      # such that the first entry in the list is the top-left,
      # the second entry is the top-right, the third is the
      # bottom-right, and the fourth is the bottom-left
      rect = np.zeros((4, 2), dtype = "float32")

      # the top-left point will have the smallest sum, whereas
      # the bottom-right point will have the largest sum
      s = pts.sum(axis = 1)
      rect[0] = pts[np.argmin(s)]
      rect[2] = pts[np.argmax(s)]

      # now, compute the difference between the points, the
      # top-right point will have the smallest difference,
      # whereas the bottom-left will have the largest difference
      diff = np.diff(pts, axis = 1)
      rect[1] = pts[np.argmin(diff)]
      rect[3] = pts[np.argmax(diff)]

      # return the ordered coordinates
      return rect

  def four_point_transform(self, image, pts):
      # obtain a consistent order of the points and unpack them
      # individually
      rect = self.order_points(pts)
      (tl, tr, br, bl) = rect
      # compute the width of the new image, which will be the
      # maximum distance between bottom-right and bottom-left
      # x-coordiates or the top-right and top-left x-coordinates
      widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
      widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
      maxWidth = max(int(widthA), int(widthB))
      # compute the height of the new image, which will be the
      # maximum distance between the top-right and bottom-right
      # y-coordinates or the top-left and bottom-left y-coordinates
      heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
      heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
      maxHeight = max(int(heightA), int(heightB))
      # now that we have the dimensions of the new image, construct
      # the set of destination points to obtain a "birds eye view",
      # (i.e. top-down view) of the image, again specifying points
      # in the top-left, top-right, bottom-right, and bottom-left
      # order
      dst = np.array([
            [0, 0], [maxWidth-1, 0], [maxWidth-1, maxHeight-1], [0, maxHeight-1]
            ], dtype = "float32")
     
      # compute the perspective transform matrix and then apply it
      M = cv2.getPerspectiveTransform(dst, rect)
      warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight),  cv2.WARP_INVERSE_MAP)
      # return the warped image
      return warped

  #########################
  # Contours
  #########################
  def get_contours(self,img):
      blurred = cv2.GaussianBlur(img, (5, 5), 0)
      gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
      lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
      thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
      # find contours in the thresholded image
      cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
              cv2.CHAIN_APPROX_SIMPLE)
      cnts = imutils.grab_contours(cnts)
      return cnts
      
  def draw_contours(self,img,cnt,text="",def_clr=(0,255,0)):
      for i,c in enumerate(cnt):
          # compute the center of the contour
          M = cv2.moments(c)
          # cX = int((M["m10"] / M["m00"]) )
          # cY = int((M["m01"] / M["m00"]) )
          c = c.astype("int")
          # print(i,"c",c)
          itext = text + str(i)
          cv2.drawContours(img, [c], -1, def_clr, 2)
          # cv2.putText(img, itext, (cX, cY),
          #       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

  ###################
  # Light Adjustment
  # Robot arm has a light.
  # check for light.
  # find center of light.
  # do bi-modal light adjustment if necessary.
  ###################
  def adjust_light(self, frame_path, add_to_mean=False, gripper_state="FULLY_OPEN", force_bb=False):
      try:
        img = cv2.imread(frame_path)
      except Exception as e:
        try:
          img = frame_path
        except:
          print("Error: unable to read:", frame_path, e)
          return None, None, None
      # font = cv2.FONT_HERSHEY_SIMPLEX
      try:
        hsv  = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)
      except:
        # grayscale: no mean to set
        return img, 0, None
      h, s, v = cv2.split(hsv)
      mean_v = cv2.mean(v)[0]
      if self.MeanLight is None or self.MeanLightCnt is None:
        blkbrd = self.alset_state.get_blackboard("CV_LIGHT")
        if blkbrd is None:
          self.MeanLight = 0
          self.MeanLightCnt = 0
        else:
          [self.MeanLightCnt, self.MeanLight] = blkbrd
      if add_to_mean:
        # compute overall mean over time
        self.MeanLight = (self.MeanLightCnt*self.MeanLight + mean_v)/(self.MeanLightCnt+1)
        self.MeanLightCnt += 1
        self.alset_state.set_blackboard("CV_LIGHT", [self.MeanLightCnt, self.MeanLight])
      adjusted_img  = img
      mean_dif = int(abs(np.round(self.MeanLight - mean_v)))

      # find robot light through kmeans analysis

      Z = v.reshape((-1,1))
      # convert to np.float32
      Z = np.float32(Z)

      # define criteria, number of clusters(K) and apply kmeans()
      criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
      K = 2
      compactness,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
      rl = {}
      # Compactness is the sum of squared distance from each pt to their center
      rl["COMPACTNESS"] = compactness
      rl["LABEL"] = label
      rl["CENTER"] = center
      if center[0] > center[1]:
       rl["LIGHT"] = 0
      else:
       rl["LIGHT"] = 1
      # print("compactness, center, label cnt:", int(compactness), center, len([label==1]))
      return adjusted_img, mean_dif, rl

# Now convert back into uint8, and make original image
# center = np.uint8(center)
# res = center[label.flatten()]
# res2 = res.reshape((v.shape))
# print("res2:", res2)
# A = Z[label==0]
# print("A:",A)
# mask = Z[label==1]
# img = Z[label != 1]
# cv2.imshow('KMeans',res2)
# cv2.waitKey(0)
#        colors = np.zeros((1, cluster_n, 3), np.uint8)
#        colors[0,:] = 255
#        colors[0,:,0] = np.arange(0, 180, 180.0/cluster_n)
#        colors = cv2.cvtColor(colors, cv2.COLOR_HSV2BGR)[0]
#        for (x, y), label in zip(np.int32(Z), labels.ravel()):
#            c = list(map(int, colors[label]))
#            cv2.circle(img, (x, y), 1, c, -1)
#####################3
#      if self.MeanLightCnt > 10 or force_bb:
#        mean_dif = int(abs(np.round(self.MeanLight - mean_v)))
#        print("big change in lighting.  mean_dif:", mean_dif, self.MeanLight)
#        if abs(mean_dif) > .1 * abs(self.MeanLight) or force_bb:
#          # big light difference. Likely due the robot arm LED.
#          orig = img.copy()
#          gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
#          # perform a naive attempt to find the (x, y) coordinates of
#          # the area of the image with the largest intensity value
#          # (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
#          # display the results of the naive attempt
#          # cv2.circle(img, maxLoc, 5, (255, 0, 0), 2)
#          # cv2.imshow("Naive brightest spot", img)
#          # apply a Gaussian blur to the image then find the brightest region
#          # if diameter % 2 != 1:
#          #   diameter += 1
#          # radius = int(diameter/2 + 1)
#
#          # radius = int(min(gray.shape[0], gray.shape[1]) / 2) - 1
#          radius = 5
#          print("gray radius", radius)
#          gray = cv2.GaussianBlur(gray, (radius, radius), 0)
#          (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
#          rl_img = orig.copy()
#          cv2.circle(rl_img, maxLoc, 4, (255, 0, 0), 2)
#          # display the results of our newly improved method
#          cv2.imshow("Robust brightest spot", rl_img)
#          cv2.waitKey(0)
#
#          # starting from the brightest point, find the radius of the robot LED light
#          # maximum radius is distance to the edge of the image.
#          max_radius = int(np.round(min([img.shape[0]-maxLoc[0], img.shape[1]-maxLoc[1],
#                           maxLoc[0], maxLoc[1]])))
#          print("radius:", max_radius, img.shape[0]-maxLoc[0], img.shape[1]-maxLoc[1],
#                           maxLoc[0], maxLoc[1])
#          # max_radius = int(np.round(min([img.shape[xy2hw(0)]-maxLoc[0], img.shape[xy2hw(1)]-maxLoc[1],
#          #     maxLoc[0], maxLoc[1]])))
#          min_radius = 5
#          best_rl_mean_v = 0
#          prev_rl_mean_v = 1/self.cfg.INFINITE
#          rl_mean_dif = None
#          best_robot_light_img = None
#          best_robot_light_bounding_box = None
#          max_v_dif = 0
#          # compute mean light within radius, but don't count the corners of the
#          # square that are beyond the radius
#          for r_i, radius in enumerate(range(min_radius, max_radius)):
#            diameter = radius * 2
#            rl_minx = int(np.round(max(0, maxLoc[0] - radius)))
#            rl_maxx = int(np.round(min(rl_img.shape[0],rl_minx + diameter)))
#            # print("rlimgshaper, rlmaxx:", rl_img.shape[1],rl_minx + diameter)
#            rl_miny = int(np.round(max(0, maxLoc[1] - radius)))
#            rl_maxy = int(np.round(min(rl_img.shape[1],rl_miny + diameter)))
#            robot_light_bounding_box = [[[rl_minx, rl_miny]],[[rl_minx, rl_maxy]], [[rl_maxx, rl_maxy]], [[rl_maxx, rl_miny]]]
#
#            # there are cases where the light results in an elyptical shape (not round).
#            # ex1: the light is partially on the cube, and casts a shadow on the 
#            #      ground before the rest of the light is on the ground.
#            # ex2: A round shape is only when gripper lowered from above.
#            #      What if the object is grabbed from the side or diagnal?
#            #
#            # We want mean light within radius to be counted
#
#            # print(r_i,"robot_light_bounding_box:", robot_light_bounding_box)
#            # print(r_i,"rlmaxy, rlminy, rlmaxx, rlminx:",rl_maxy, rl_miny, rl_maxx, rl_minx)
#            robot_light_img = np.zeros((rl_maxy-rl_miny, rl_maxx-rl_minx, 3),dtype="uint8")
#            # print("img.shape:", img.shape, rl_maxy-rl_miny, rl_maxx-rl_minx)
#            robot_light_img[0:rl_maxy-rl_miny, 0:rl_maxx-rl_minx,:] = img[rl_miny:rl_maxy, rl_minx:rl_maxx,:]
#
#            # find mean dif in brightest spots
#            robot_light_hsv  = cv2.cvtColor(robot_light_img.copy(), cv2.COLOR_BGR2HSV)
#            rl_h, rl_s, rl_v = cv2.split(robot_light_hsv)
#            # rl_mean_v = cv2.mean(rl_v)[0]
#            inside_radius = []
#            outside_radius = []
#            for x in range(rl_maxx-rl_minx):
#              for y in range(rl_maxy-rl_miny):
#                if np.sqrt(x*x + y*y) <= radius-1:
#                  inside_radius.append(1.0*rl_v[x,y])  # make float (not uint8)
#                if np.sqrt(x*x + y*y) > radius-1:
#                  outside_radius.append(1.0*rl_v[x,y])  # make float (not uint8)
#            rl_inside_mean_v = statistics.mean(inside_radius)
#            rl_outside_mean_v = statistics.mean(outside_radius)
#            rl_mean_v = statistics.mean(inside_radius+outside_radius)
#            print(r_i, "rl mean v: inside/outside:", rl_inside_mean_v, rl_outside_mean_v, rl_mean_v)
#            # (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
#            # mean_v_dif = (rl_mean_v - prev_rl_mean_v) / prev_rl_mean_v
#            # print(r_i,"mean_dif", max_v_dif, mean_v_dif, rl_mean_v, prev_rl_mean_v)
#            # mean_dif = int(abs(np.round(self.MeanLight - rl_mean_v)))
#            # if abs(mean_dif) > .4 * abs(self.MeanLight) and mean_v_dif < 0 and r_i > 5:
#            if abs(rl_outside_mean_v/rl_inside_mean_v) < 1 and r_i >= 5:
#              # bigger sudden decrease in robot light
#              # print("RL decrease:", mean_dif/self.MeanLight, mean_dif, self.MeanLight)
#              # print("RL decrease:",mean_v_dif, max_v_dif, mean_v_dif, self.MeanLight)
#              print("RL decrease:",abs(rl_outside_mean_v/rl_inside_mean_v))
#              rl_mean_dif = int(abs(np.round(self.MeanLight - rl_mean_v)))
#              break
#            else:
#              best_rl_mean_v = rl_mean_v
#              best_robot_light_img = robot_light_img.copy()
#              best_robot_light_bounding_box = robot_light_bounding_box.copy()
#            
#              rl_mean_dif = int(abs(np.round(self.MeanLight - rl_mean_v)))
#              adjusted_rl_v = rl_v.copy()
#              adjusted_rl_v[adjusted_rl_v <= rl_mean_dif] = 0
#              adjusted_rl_v[adjusted_rl_v > rl_mean_dif] -= rl_mean_dif
#            # if abs(mean_v_dif) > abs(max_v_dif):
#            #   max_v_dif = mean_v_dif
#            # prev_rl_mean_v = rl_mean_v
#
#            # we've failed to compute the robot light image
#            # print("FAILED TO COMPUTE THE ROBOT LIGHT IMAGE")
#            # rl_mean_v = 0
#            # rl_mean_dif = 0
#            # robot_light_bounding_box = None
#
#          print("best_robot_light_bb:", best_robot_light_bounding_box)
#          # compute mean outside region
#
#          if best_robot_light_bounding_box is not None:
#            rl_maxx, rl_minx, rl_maxy, rl_miny = get_min_max_borders(best_robot_light_bounding_box)
#            unlit_mean_v = mean_v * img.shape[0] * img.shape[1] - best_rl_mean_v * (rl_maxy-rl_miny) * (rl_maxx-rl_minx)
#          # unlit_mean_v /= ((img.shape[1]-(rl_maxx-rl_minx)) * (img.shape[0]-(rl_maxy-rl_miny)))
#            print("areas", img.shape[1]*img.shape[0], (rl_maxx-rl_minx) * (rl_maxy-rl_miny))
#            unlit_mean_v /= ((img.shape[0]*img.shape[1])-(rl_maxy-rl_miny)*(rl_maxx-rl_minx))
#          else:
#            unlit_mean_v = mean_v
#
#          unlit_mean_v = int(abs(np.round(unlit_mean_v)))
#          unlit_mean_dif = int(abs(np.round(self.MeanLight - unlit_mean_v)))
#          print("MEANV, mean_v, rl_mean_v, unlit_mean_v:",self.MeanLight, mean_v, best_rl_mean_v, unlit_mean_v)
#          print("rl_mean_dif, unlit_mean_dif:",rl_mean_dif, unlit_mean_dif )
#
#          # adjust mean lights
#          lim = 255 - unlit_mean_dif
#          unlit_v = v.copy()
#          unlit_v[unlit_v > lim] = 255
#          unlit_v[unlit_v <= lim] += unlit_mean_dif
#          # ValueError: could not broadcast input array from shape (54,54) into shape (75,54)
#          # print("rl_miny,rl_maxy, (rl_maxy-rl_miny):", rl_miny,rl_maxy, (rl_maxy-rl_miny))
#          # Too discontinuous
#          # unlit_v[rl_miny:rl_maxy, rl_minx:rl_maxx] = adjusted_rl_v[0:(rl_maxy-rl_miny), 0:(rl_maxx-rl_minx)]
#
#          final_hsv = cv2.merge((h, s, unlit_v))
#          adjusted_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
#          # cv2.imshow("Best Robot Light bb ", best_robot_light_img)
#          # cv2.imshow("Light Adjusted Image", adjusted_img)
#          # cv2.imshow("Orig Image", img)
#          # cv2.waitKey(0)
#          return adjusted_img, mean_dif, best_robot_light_bounding_box
#
#        elif abs(mean_dif) > .06 * abs(self.MeanLight):
#          lim = 255 - mean_dif
#          v[v > lim] = 255
#          v[v <= lim] += mean_dif
#
#          final_hsv = cv2.merge((h, s, v))
#          adjusted_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
#          # cv2.imshow("Orig Image", img)
#          # cv2.imshow("Light Adjusted Image", adjusted_img)
#          # cv2.waitKey(0)
#      return adjusted_img, mean_dif, None

  def mean_sq_err(self, imageA, imageB):
      # the 'Mean Squared Error' between the two images is the
      # sum of the squared difference between the two images;
      # NOTE: the two images must have the same dimension
      # print("np.sum diff sqr", err)
      # Normalized Least Squared Error
      if True:
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= np.sum((imageA.astype("float")) ** 2)
      elif False and rl is not None:
        # rl is a kmeans label representing the robot light cluster
        err = np.sum((imageA.astype("float")[rl["LABEL"]!=rl["LIGHT"]] - imageB.astype("float")[rl["LABEL"]!=1]) ** 2)
        err /= np.sum((imageA.astype("float")[rl["LABEL"]!=rl["LIGHT"]]) ** 2)

#        print("cpimg:",imageA.shape, exclude_bb)
#        exclude_imgA = get_bb_img(imageA, exclude_bb)
#        exclude_imgB = get_bb_img(imageB, exclude_bb)
#        maxw, minw, maxh, minh = get_min_max_borders(exclude_bb)
#        print("cpimg:",imageA.shape, maxw, minw, maxh, minh)
#        # cv2.imshow('b4 eximga',exclude_imgA)
#        # cv2.imshow('b4 eximgb',exclude_imgB)
#        exclude_imgA = cv2.copyMakeBorder(exclude_imgA, 
#                       top=minh, bottom=imageA.shape[0] - maxh,
#                       left=minw, right=imageA.shape[1] - maxw,
#                       borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])  # black
#        exclude_imgB = cv2.copyMakeBorder(exclude_imgB, 
#                       top=minh, bottom=imageB.shape[0] - maxh,
#                       left=minw, right=imageB.shape[1] - maxw,
#                       borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])  # black
#        # print("imgA:", imageA.shape[0] - maxw, imageA.shape, exclude_imgA.shape, maxh, maxw)
#        # print("imgB:", imageB.shape[0] - maxw, imageB.shape, exclude_imgB.shape, maxh, maxw)
#        cv2.imshow('eximga',exclude_imgA)
#        # cv2.imshow('eximgb',exclude_imgB)
#        cv2.imshow('imga',imageA)
#        # cv2.imshow('imgb',imageB)
#        # k = cv2.waitKey(0)
#        includedErr = np.sum(((imageA.astype("float") - exclude_imgA.astype("float"))
#                    - (exclude_imgB.astype("float") - exclude_imgB.astype("float"))) ** 2)
#        includedA = imageA.astype("float") - exclude_imgA.astype("float") 
#        err = (includedErr) / np.sum(includedA ** 2)
      else:
        # debug: compare slow way results to fast way
        mse_sum = 0.0
        mse_sum3 = 0.0
        mse_sumA = 0.0
        mse_cnt = 0
        zero_cnt = 0
        zero_cntA = 0
        zero_cntB = 0
        for h in range(imageA.shape[0]):
          for w in range(imageA.shape[1]):
              if (self.is_black(imageA[h,w].astype("float")) and 
                  self.is_black(imageB[h,w].astype("float"))):
                zero_cnt +=1
              elif (self.is_black(imageA[h,w].astype("float")) and 
                    self.is_black(imageB[h,w].astype("float"))):
                zero_cntA +=1
              elif (self.is_black(imageA[h,w].astype("float")) and 
                    self.is_black(imageB[h,w].astype("float"))):
                zero_cntB +=1
              else:
                mse_sum += np.sum((imageA[h,w].astype("float") - imageB[h,w].astype("float"))**2)
                mse_cnt += 1
                mse_sumA += np.sum((imageA[h,w].astype("float"))**2)
                mse_err = mse_sum / mse_sumA
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= np.sum((imageA.astype("float")) ** 2)
        print("mse cnts: AB, A, B, mse:", zero_cnt, zero_cntA, zero_cntB, mse_cnt)
        print("mse_err: ", mse_err, err)
        # err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        # err /= np.sum((imageA.astype("float")) ** 2)
      # print("mse shape", float(imageA.shape[0] * imageA.shape[1]), np.sum((imageA.astype("float")) ** 2))
      # return the MSE, the lower the error, the more "similar"
      # the two images are
      return err
  
  def is_black(self, pixel):
      if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
        return True
      return False

# from: https://stackoverflow.com/questions/45531074/how-to-merge-lines-after-houghlinesp
class HoughBundler:
    '''Clasterize and merge each cluster of cv2.HoughLinesP() output
    a = HoughBundler()
    foo = a.process_lines(houghP_lines, binary_image)
    '''

    def get_orientation(self, line):
        '''get orientation of a line, using its length
        https://en.wikipedia.org/wiki/Atan2
        '''
        orientation = math.atan2(abs((line[0] - line[2])), abs((line[1] - line[3])))
        return math.degrees(orientation)

    def checker(self, line_new, groups, min_distance_to_merge, min_angle_to_merge):
        '''Check if line have enough distance and angle to be count as similar
        '''
        for group in groups:
            # walk through existing line groups
            for line_old in group:
                # check distance
                if self.get_distance(line_old, line_new) < min_distance_to_merge:
                    # check the angle between lines
                    orientation_new = self.get_orientation(line_new)
                    orientation_old = self.get_orientation(line_old)
                    # if all is ok -- line is similar to others in group
                    if abs(orientation_new - orientation_old) < min_angle_to_merge:
                        group.append(line_new)
                        return False
        # if it is totally different line
        return True

    def DistancePointLine(self, point, line):
        """Get distance between point and line
        http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
        """
        px, py = point
        x1, y1, x2, y2 = line

        def lineMagnitude(x1, y1, x2, y2):
            'Get line (aka vector) length'
            lineMagnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
            return lineMagnitude

        LineMag = lineMagnitude(x1, y1, x2, y2)
        if LineMag < 0.00000001:
            DistancePointLine = 9999
            return DistancePointLine

        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / (LineMag * LineMag)

        if (u < 0.00001) or (u > 1):
            #// closest point does not fall within the line segment, take the shorter distance
            #// to an endpoint
            ix = lineMagnitude(px, py, x1, y1)
            iy = lineMagnitude(px, py, x2, y2)
            if ix > iy:
                DistancePointLine = iy
            else:
                DistancePointLine = ix
        else:
            # Intersecting point is on the line, use the formula
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            DistancePointLine = lineMagnitude(px, py, ix, iy)

        return DistancePointLine

    def get_distance(self, a_line, b_line):
        """Get all possible distances between each dot of two lines and second line
        return the shortest
        """
        dist1 = self.DistancePointLine(a_line[:2], b_line)
        dist2 = self.DistancePointLine(a_line[2:], b_line)
        dist3 = self.DistancePointLine(b_line[:2], a_line)
        dist4 = self.DistancePointLine(b_line[2:], a_line)

        return min(dist1, dist2, dist3, dist4)

    def merge_lines_pipeline_2(self, lines):
        'Clusterize (group) lines'
        groups = []  # all lines groups are here
        # Parameters to play with
        min_distance_to_merge = 30
        min_angle_to_merge = 30
        # first line will create new group every time
        groups.append([lines[0]])
        # if line is different from existing gropus, create a new group
        for line_new in lines[1:]:
            if self.checker(line_new, groups, min_distance_to_merge, min_angle_to_merge):
                groups.append([line_new])

        return groups

    def merge_lines_segments1(self, lines):
        """Sort lines cluster and return first and last coordinates
        """
        orientation = self.get_orientation(lines[0])

        # special case
        if(len(lines) == 1):
            return [lines[0][:2], lines[0][2:]]

        # [[1,2,3,4],[]] to [[1,2],[3,4],[],[]]
        points = []
        for line in lines:
            points.append(line[:2])
            points.append(line[2:])
        # if vertical
        if 45 < orientation < 135:
            #sort by y
            points = sorted(points, key=lambda point: point[1])
        else:
            #sort by x
            points = sorted(points, key=lambda point: point[0])

        # return first and last point in sorted group
        # [[x,y],[x,y]]
        return [points[0], points[-1]]

    def process_lines(self, lines, img):
        '''Main function for lines from cv.HoughLinesP() output merging
        for OpenCV 3
        lines -- cv.HoughLinesP() output
        img -- binary image
        '''
        lines_x = []
        lines_y = []
        # for every line of cv2.HoughLinesP()
        for line_i in [l[0] for l in lines]:
                orientation = self.get_orientation(line_i)
                # if vertical
                if 45 < orientation < 135:
                    lines_y.append(line_i)
                else:
                    lines_x.append(line_i)

        lines_y = sorted(lines_y, key=lambda line: line[1])
        lines_x = sorted(lines_x, key=lambda line: line[0])
        merged_lines_all = []

        # for each cluster in vertical and horizantal lines leave only one line
        for i in [lines_x, lines_y]:
                if len(i) > 0:
                    groups = self.merge_lines_pipeline_2(i)
                    merged_lines = []
                    for group in groups:
                        merged_lines.append(self.merge_lines_segments1(group))

                    merged_lines_all.extend(merged_lines)

        return merged_lines_all


