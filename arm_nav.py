"""
Based on: https://github.com/AtsushiSakai/PythonRobotics
Note: started with n_joint_arm_to_point_control.py, but 
  simple approach purely based upon jacobian inverses didn't 
  handle joint limits or the base (obstacle).  Added here.

Obstacle navigation using A* on a toroidal grid

Author: Daniel Ingram (daniel-s-ingram)
        Tullio Facchinetti (tullio.facchinetti@unipv.it)
"""
import cv2
from math import pi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import from_levels_and_colors
import sys
from config import *
from operator import itemgetter


class ArmNavigation(object):

    def __init__(self):
        # Simulation parameters
        # circle.obstacles = [[1.75, 0.75, 0.6], [0.55, 1.5, 0.5], [0, -1, 0.7]]
        plt.ion()
        self.circle_obstacles = []
        self.square_obstacles = []
        self.line_segment_obstacles = []
        self.delta_arm_pos = {"UPPER_ARM_UP":0,"UPPER_ARM_DOWN":0,
                                "LOWER_ARM_UP":0,"LOWER_ARM_DOWN":0}
        # changle in angle in radians
        self.arm_delta = {"UPPER_ARM_UP":0.085,"UPPER_ARM_DOWN":-0.085,
                          "LOWER_ARM_UP":-0.200,"LOWER_ARM_DOWN":0.200}
        # Arm geometry in the working space
        self.cfg = None
        self.link_length, self.link_angle_limit, self.init_link_angle, self.base = self.alset_arm()
        # there are three "spaces" here:
        # - actual size (in inches, from config.py)
        # - plot space (Normalized along length of arm to 2)
        # - grid space (2 plot_space == 200 grid_space)

        # convert to horiz axis
        # for i in range(len(self.init_link_angle)):
        #   self.init_link_angle[i] += np.pi
 
        # actual size
        total_link_len = 0
        for i in range(len(self.link_length)):
          total_link_len += self.link_length[i]
        
        self.plot_size = 2   # to the negative and positive
        self.normalize_factor = self.plot_size/total_link_len
        self.M = 100
        self.grid_factor = self.M / (self.plot_size*2)
        self.pivot_point = None
        self.pivot_point_len = 0
        # convert
        for i in range(len(self.link_length)):
          self.link_length[i] = self.link_length[i] * self.normalize_factor
        for i in range(len(self.base)):
          # base is a series of line segments defined by points
          # i=0: Fixed Arm joint
          # i=1: Main robot body (min, max)
          # i=2: Ground
          self.base[i][0][0] = self.base[i][0][0] * self.normalize_factor
          self.base[i][0][1] = self.base[i][0][1] * self.normalize_factor
          self.base[i][1][0] = self.base[i][1][0] * self.normalize_factor
          self.base[i][1][1] = self.base[i][1][1] * self.normalize_factor
        base_node = self.pos_to_node(self.base[1][1])
        ground_node = self.pos_to_node(self.base[2][1])
        # (36, 39) (0, 29)
        #
        print("base_node, ground_node:", base_node, ground_node)
        self.n_links = len(self.link_length)
        self.min_max_goals = [self.M, -1, self.M, -1]
        # self.neighbor_buffer = 5  # max allowed diff
        # self.neighbor_buffer = 2  # max allowed diff
        # self.neighbor_buffer = 10  # max allowed diff
        self.neighbor_buffer = 20   # max allowed diff
        self.arm = NLinkArm(self.link_length.copy(), self.init_link_angle, self.base)
        self.world_arm = NLinkArm(self.link_length.copy(), self.init_link_angle, self.base)
        self.link_angle = self.init_link_angle.copy()
        # (x, y) co-ordinates in the joint space [cell]
        self.base_pos = self.base[0][0] # (0, 0)
        self.base_node = self.pos_to_node(self.base_pos)
        self.start_pos = self.forward_kinematics() 
        self.start_node = self.pos_to_node(self.start_pos)
        print("start_node:", self.start_pos, self.start_node)
        # only compute once, and make copy for each goal point
        self.master_grid, self.grid_angles = self.get_occupancy_grid(self.world_arm)
        # while True:
        if True:
          self.grid = self.master_grid.copy()
          # self.goal_pos, self.goal_node = self.get_random_goal()  
          self.goal_pos = None
          self.goal_node = None
          # print("INIT goal:", self.goal_pos, self.goal_node)
          # self.route = self.astar_torus(self.circle_obstacles)
          # print("route:", self.route)

          # if len(self.route) >= 0:
          #     self.animate(self.arm, self.route, self.circle_obstacles)
          # cv2.waitKey(0) 

        # Animate the grid point that has the most alternative angle combinations 
        # self.animate_pivot_pt(self.arm)

    def animate_random_goal(self):
        self.goal_pos, self.goal_node = self.get_random_goal()  
        print("INIT goal:", self.goal_pos, self.goal_node)
        self.route = self.astar_torus(self.circle_obstacles)
        print("route:", self.route)
        if len(self.route) >= 0:
          self.animate(self.arm, self.route, self.circle_obstacles)
          cv2.waitKey(0) 
        # Animate the grid point that has the most alternative angle combinations 
        self.animate_pivot_pt(self.arm)
    
    ###################################
    # ALSET WORLD: Plot the external world
    ###################################
    def clear_world(self):
        self.delta_arm_pos = {"UPPER_ARM_UP":0,"UPPER_ARM_DOWN":0,
                              "LOWER_ARM_UP":0,"LOWER_ARM_DOWN":0}
        self.line_segment_obstacles = []

    def arm_pos_to_points(self):
        # self.arm = NLinkArm(self.link_length.copy(), self.init_link_angle, self.base)
        uau_delta = self.arm_delta["UPPER_ARM_UP"]
        uad_delta = self.arm_delta["UPPER_ARM_DOWN"]
        lau_delta = self.arm_delta["LOWER_ARM_UP"]
        lad_delta = self.arm_delta["LOWER_ARM_DOWN"]

        ua_angle = self.cfg.ROBOT_ARM_INIT_ANGLES[0]
        la_angle = self.cfg.ROBOT_ARM_INIT_ANGLES[1]
        ga_angle = self.cfg.ROBOT_ARM_INIT_ANGLES[2]
        ua_angle += (self.delta_arm_pos["UPPER_ARM_UP"]   * uau_delta +
                    self.delta_arm_pos["UPPER_ARM_DOWN"] * uad_delta)
        la_angle -= (self.delta_arm_pos["LOWER_ARM_UP"]   * lau_delta +
                    self.delta_arm_pos["LOWER_ARM_DOWN"] * lad_delta)
        joint_list = [ua_angle, la_angle, ga_angle]
        self.world_arm.update_joints(joint_list)

    def add_line_segment_to_world(self, line_segment):
        self.line_segment_obstacles.append(line_segment)

    def set_arm_pos(self, arm_delta):
        self.delta_arm_pos = arm_delta
        self.arm_pos_to_points()

    def get_camera_pos(self):
        camera_angle = self.ang_diff(np.sum(link_angle[:]), np.pi/2)
        camera_pos = self.forward_kinematics(self.link_length, link_angle)
        print("cam,goal angle: ", camera_angle, camera_pos)
        return camera_angle, camera_pos

    def get_camera_pov(self):
        pass

    def plot_world(self):
        self.plot_arm(self.world_arm, goal=None, line_segments=self.line_segment_obstacles)

    ###################################
    # Plot the external world
    ###################################

    def pos_to_node(self,pos):
        node = (int((pos[0]+self.plot_size)*self.grid_factor),
                (self.M - int((pos[1]+self.plot_size)*self.grid_factor)))
        return node
        
    def node_to_pos(self,node):
        pos = ((node[0]) / self.grid_factor - self.plot_size,
               (self.M - node[1]) / self.grid_factor - self.plot_size)
        return pos

    def press(self, event):
        """Exit from the simulation."""
        if event.key == 'q' or event.key == 'Q':
            print('Quitting upon request.')
            sys.exit(0)
    
    # returns goal in grid in float precision
    def get_random_goal(self):
        from random import random

        ground_x1 = self.base[2][1][0]
        ground_y  = self.base[2][1][1]
        base_x1   = self.base[1][1][0]
        print("ground_x1, ground_y, base_x1, MAX_POS:", ground_x1, ground_y, base_x1)
        while True:
          goal_pos = (((ground_x1 - base_x1) * random() + base_x1), ground_y)
          goal_node = self.pos_to_node(goal_pos)
          if (self.grid[goal_node[0]][goal_node[1]] != 7): # 7 is UNREACHABLE
            return goal_pos, goal_node
          print("unreachable goal_node", goal_node, self.grid[goal_node])
    
    def alset_arm(self):
        self.cfg = Config()
        return self.cfg.ROBOT_ARM_LENGTHS, self.cfg.ROBOT_ARM_ANGLE_LIMITS, self.cfg.ROBOT_ARM_INIT_ANGLES, self.cfg.ROBOT_BASE
    
    def camera_angle_delta(self, link_angle):
        camera_angle = self.ang_diff(np.sum(link_angle[:]), np.pi/2)
        # camera_angle = np.sum(link_angle[:])
        curr_pos = self.forward_kinematics(self.link_length, link_angle)
        goal_angle = self.ang_diff(np.arctan2((self.goal_pos[0]-curr_pos[0]), (self.goal_pos[1]-curr_pos[1])), 0)
        delta = abs(self.ang_diff(camera_angle, goal_angle))
        print("cam,goal angle: ", camera_angle, goal_angle, delta)
        return delta
    
    def forward_kinematics(self, link_length = None, link_angle = None):
        if link_length is None:
          link_length = self.link_length
        if link_angle is None:
          link_angle = self.link_angle
        x = y = 0
        # print("arm pt 0", self.pos_to_node((x,y)))
        for i in range(1, len(link_length)+1):
            x += link_length[i-1] * np.cos(np.sum(link_angle[:i]))
            y += link_length[i-1] * np.sin(np.sum(link_angle[:i]))
            # print("arm pt",i, self.pos_to_node((x,y)))
        # for i in range(1, len(link_length) + 1):
        #     x += link_length[i - 1] * np.cos(np.sum(link_angle[:i]))
        #     y += link_length[i - 1] * np.sin(np.sum(link_angle[:i]))
        return np.array([x, y]).T
    
    def get_move_delta_angle(self):
        # should be based on alset stats for current position's angle changes
        # est_delta_angle = [["UPPER_ARM_UP", 1/32],["UPPER_ARM_DOWN", -1/32],
        #                    ["LOWER_ARM_UP", 1/32],["LOWER_ARM_DOWN", -1/32]]
        est_delta_angle = [["UPPER_ARM_UP", 1],["UPPER_ARM_DOWN", -1],
                           ["LOWER_ARM_UP", 1],["LOWER_ARM_DOWN", -1]]
        return est_delta_angle

    def ang_diff(self, theta1, theta2):
        """
        Returns the difference between two angles in the range -pi to +pi
        """
        return (theta1 - theta2 + np.pi) % (2 * np.pi) - np.pi
    
    def animate_pivot_pt(self, arm):
        print("number angles in pivot point: ",self.pivot_point_len)
        pivot_angles = sorted(self.grid_angles[self.pivot_point[0]][self.pivot_point[1]], key=itemgetter(0))
        for angles in pivot_angles:
            arm.update_joints(angles)
            plt.subplot(1, 2, 2)
            arm.plot_arm(plt, self.goal_pos)
            plt.xlim(-2.0, 2.0)
            plt.ylim(-3.0, 3.0)
            plt.show()
            # Uncomment here to save the sequence of frames
            # plt.savefig('frame{:04d}.png'.format(i))
            plt.pause(.5)
   

    def animate(self, arm, route, obstacles):
        fig, axs = plt.subplots(1, 2)
        fig.canvas.mpl_connect('key_press_event', self.press)
        #         unobstr  obstacle explor neigh   start     goal     route     unreach
        colors = ['white', 'black', 'red', 'pink', 'yellow', 'green', 'orange', 'tan']
        levels = [0,       1,       2,     3,      4,        5,        6,       7,    8]
        cmap, norm = from_levels_and_colors(levels, colors)
        for i, node in enumerate(route):
            if node == self.start_node:
              joint_angles = self.init_link_angle.copy()
            else:
              # already sorted to prioritize the best camera angle to front ([0])
              joint_angles = self.grid_angles[node[0]][node[1]][0].copy()
            arm.update_joints(joint_angles)

            plt.subplot(1, 2, 1)
            self.grid[node] = 6  # route
            plt.cla()
            plt.imshow(self.grid, cmap=cmap, norm=norm, interpolation=None)
            print("Animate node:", node)
            # self.get_arm_theta(node)
            plt.subplot(1, 2, 2)
            arm.plot_arm(plt, self.goal_pos, line_segments=self.line_segment_obstacles)
            plt.xlim(-2.0, 2.0)
            plt.ylim(-3.0, 3.0)
            plt.show()
            # Uncomment here to save the sequence of frames
            # plt.savefig('frame{:04d}.png'.format(i))
            plt.pause(.5)
        plt.pause(5)
    
    
    def detect_collision_with_base_or_ground(self, line_segment):
        # eventually support any polygon of line segments to define an obstacle.
        # note: doing collision detection based on pos space, not grid space
        # curr_pos = self.forward_kinematics(self.link_length, arm.joint_angles)
        for i in range(2):
          # note the ranges are between [-2, +2].
          if (line_segment[i][0] >= self.base[1][0][0] and line_segment[i][0] <= self.base[1][1][0] and line_segment[i][1] <= self.base[1][0][1]):
            print("A line_seg, base", line_segment, self.base)
            return True
          # assumes flat ground. Lower number is lower on y axis.
          if (line_segment[i][1] < self.base[2][0][1]):
            print("B line_seg, base", line_segment, self.base[2][0][1])
            return True
        return False
    
    def detect_collision(self, line_seg, circle):
        """
        Determines whether a line segment (arm link) is in contact
        with a circle (obstacle).
        Credit to: http://doswa.com/2009/07/13/circle-segment-intersectioncollision.html
        Args:
            line_seg: List of coordinates of line segment endpoints e.g. [[1, 1], [2, 2]]
            circle: List of circle coordinates and radius e.g. [0, 0, 0.5] is a circle centered
                    at the origin with radius 0.5
    
        Returns:
            True if the line segment is in contact with the circle
            False otherwise
        """
        a_vec = np.array([line_seg[0][0], line_seg[0][1]])
        b_vec = np.array([line_seg[1][0], line_seg[1][1]])
        c_vec = np.array([circle[0], circle[1]])
        radius = circle[2]
        line_vec = b_vec - a_vec
        line_mag = np.linalg.norm(line_vec)
        circle_vec = c_vec - a_vec
        proj = circle_vec.dot(line_vec / line_mag)
        if proj <= 0:
            closest_point = a_vec
        elif proj >= line_mag:
            closest_point = b_vec
        else:
            closest_point = a_vec + line_vec * proj / line_mag
        if np.linalg.norm(closest_point - c_vec) > radius:
            return False
        return True
    
    # Note: currently unused
    def get_hypot(len1, len2, angle):
        # hypot: c^2 = a^2 + b^2 - 2ab*cos(C)
        hypot = len1**2 + len2**2 - 2*len1*len2*np.cos(angle)
        return hypot

    # Note: currently unused
    # we know len1, len2, and hypot (from get_hypot)
    # we know "fixed" angle betw len1, len2
    # we need angle betw len1 and hypot
    #
    # The law of sines says that the ratio of the sine of one angle to
    # the opposite side is the same ratio for all three angles.
    #
    # sin(hypot_angle) / hypot = sin(len2_angle) / len2
    # sin(len2_angle) = sin(hypot_angle) * len2 / hypot 
    # len2_angle = arcsin(sin(hypot_angle) * len2 / hypot)
    def get_link_angle(len2, hypot, hypot_angle):
        if len2 < hypot:
          ang = np.sin(hypot_angle) * len2 / hypot
        else:
          ang = np.sin(hypot_angle) * hypot / len2
        # ang should be between 1,-1
        print("get_link_angle:", len2, hypot, hypot_angle, ang)
        len2_angle = np.arcsin(ang)
        return len2_angle
            
    def get_occupancy_grid(self, arm, obstacles=[]):
        """

        Discretizes joint space into M values from -pi to +pi
        and determines whether a given coordinate in joint space
        would result in a collision between a robot arm and relatively unchanging
        base/obstacles in its environment.

        Only compute once.  New localized obstacles should be factored in
        by self.astar_torus.
    
        Args:
            arm: An instance of NLinkArm
            obstacles: A list of obstacles, with each obstacle defined as a list
                       of xy coordinates and a radius. 
    
        Returns:
            Occupancy grid in joint space
        """
        grid = [[7 for _ in range(self.M)] for _ in range(self.M)]
        grid_angles = [[[] for _ in range(self.M)] for _ in range(self.M)]

        grid[self.start_node[0]][self.start_node[0]] = 0
        arm_reach = sum(self.link_length) # longest possible reach of arm 
        arm_node_reach = int(arm_reach*self.grid_factor)
        theta_list = [2 * i * np.pi / self.M for i in range(-self.M // 2, self.M // 2 + 1)]
        max_counter = [self.M for n in range(self.n_links)]
        min_counter = [0 for n in range(self.n_links)]
        for n in range(self.n_links):
          if self.link_angle_limit[n] is not None and self.link_angle_limit[n][0] is not None:
            max_counter[n] = int((self.link_angle_limit[n][0]+np.pi) * self.M / (2 * np.pi))-1
          if self.link_angle_limit[n] is not None and self.link_angle_limit[n][1] is not None:
            min_counter[n] = int((self.link_angle_limit[n][1]+np.pi) * self.M / (2 * np.pi))+1
        counters = [min_counter[n] for n in range(self.n_links)]
         
        while True:
          joint_list = []
          for n in range(self.n_links):
            joint_list.append(theta_list[counters[n]])
          arm.update_joints(joint_list)
          points = arm.points
          collision_detected = False
          for k in range(len(points) - 1):
                line_seg = [points[k], points[k + 1]]
                for obstacle in obstacles:
                    collision_detected = detect_collision(line_seg, obstacle)
                    if collision_detected:
                        break
                collision_detected2 = self.detect_collision_with_base_or_ground(line_seg)
                if collision_detected or collision_detected2:
                    print("collision detected", counters, collision_detected, collision_detected2)
                    collision_detected = True
                    break
          gripper_pos = points[-1]
          gripper_node = self.pos_to_node(gripper_pos)
          if gripper_node[0] < self.M and gripper_node[1] < self.M:
              if not collision_detected:
                grid_angles[gripper_node[0]][gripper_node[1]].append(joint_list)
                grid[gripper_node[0]][gripper_node[1]] = int(collision_detected)
                if len(grid_angles[gripper_node[0]][gripper_node[1]]) > self.pivot_point_len:
                  self.pivot_point_len = len(grid_angles[gripper_node[0]][gripper_node[1]]) 
                  self.pivot_point = gripper_node
          
          # increment to next counter
          done = False
          for n in range(self.n_links):
            c = self.n_links - n - 1
            counters[c] += 1
            if counters[c] > max_counter[c]:
              if c == 0:
                done = True
                break
              else:
                counters[c] = min_counter[c]
            else:
              break
          if done:
            break
        # print out some stats
        for i in range(self.M):
          for j in range(self.M):
            if len(grid_angles[i][j]) > 0:
              print("Grid Angles", (i,j), grid_angles[i][j])
        cnts = [0 for _ in range(8)]
        for i in range(self.M):
          for j in range(self.M):
            cnts[grid[i][j]] += 1 
        for c, val in enumerate(["UNOBSTRUCTED", "OBSTACLE    ", "EXPLORED    ",
                       "NEIGHBOR    ", "START       ", "GOAL        ",
                       "ROUTE       ", "UNREACHABLE "]):
          print("COUNTS:", val, cnts[c])
        # print("UNREACHABLE COUNTS:", unreachable_cnt)
        return np.array(grid), grid_angles

    def astar_torus(self, circle_obstacles=[]):
        """
        Finds a path between an initial and goal joint configuration using
        the A* Algorithm on a tororiadal grid.
    
        Args:
            grid: An occupancy grid (ndarray)
            start_node: Initial joint configuration (tuple)
            goal_node: Goal joint configuration (tuple)
    
        Returns:
            Obstacle-free route in joint space from start_node to goal_node.
            Favor those angles that point the camera at the goal.
        """
        # START_NODE = yellow ; GOAL_NODE = green ; UNOBSTRUCTED = white ; EXPLORED = red ; 
        # CURRENT NEIGHBOR = pink ; OBSTACLE = black ; UNREACHABLE = orange
        #         unobstr  obstacle explor  neigh   start     goal     route      unreach

        colors = ['white', 'black', 'red', 'pink', 'yellow', 'green', 'orange', 'tan']
        levels = [0,       1,       2,     3,      4,        5,       6,         7,   8]
        cmap, norm = from_levels_and_colors(levels, colors)
    
        parent_map = [[() for _ in range(self.M)] for _ in range(self.M)]
        # heuristic_map = self.calc_heuristic_map()
        heuristic_map = self.calc_heuristic_map2()
    
        explored_heuristic_map = np.full((self.M, self.M), np.inf)
        distance_map = np.full((self.M, self.M), np.inf)
        explored_heuristic_map[self.start_node] = heuristic_map[self.start_node]
        print("explored_heuristic_map[self.start_node]", explored_heuristic_map[self.start_node])
        distance_map[self.start_node] = 0
        print("start/goal grid val", self.grid[self.start_node], self.grid[self.goal_node])
        while True:
            self.grid[self.start_node] = 4
            self.grid[self.goal_node] = 5
    
            current_node = np.unravel_index(
                np.argmin(explored_heuristic_map, axis=None), explored_heuristic_map.shape)
            min_distance = np.min(explored_heuristic_map)
            if (current_node == self.goal_node) or np.isinf(min_distance):
                break
    
            self.grid[current_node] = 2
            explored_heuristic_map[current_node] = np.inf
    
            i, j = current_node[0], current_node[1]
    
            neighbors = self.find_neighbors(i, j)
    
            for neighbor in neighbors:
                if self.grid[neighbor] == 0 or self.grid[neighbor] == 5:
                    distance_map[neighbor] = distance_map[current_node] + 1
                    explored_heuristic_map[neighbor] = heuristic_map[neighbor]
                    parent_map[neighbor[0]][neighbor[1]] = current_node
                    print("parent_map: ", current_node)
                    self.grid[neighbor] = 3
                # else:
                #     print("failed neighbor: ", neighbor, current_node)
    
        if np.isinf(explored_heuristic_map[self.goal_node]):
            route = []
            print("No route found.")
        else:
            route = [self.goal_node]
            while parent_map[route[0][0]][route[0][1]] != ():
                route.insert(0, parent_map[route[0][0]][route[0][1]])
    
            print("The route found covers %d grid cells." % len(route))
            for i in range(1, len(route)):
                self.grid[route[i]] = 6
                plt.cla()
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                        lambda event: [exit(0) if event.key == 'escape' else None])
                plt.imshow(self.grid, cmap=cmap, norm=norm, interpolation=None)
                plt.show()
                plt.pause(1e-2)
    
        return route

    def find_neighbors(self, i, j):
        neighbors = []
        if i - 1 >= 0:
            neighbors.append((i - 1, j))
            i1 = i-1
        else:
            neighbors.append((self.M - 1, j))
            i1 = self.M-1
    
        if i + 1 < self.M:
            neighbors.append((i + 1, j))
            i2 = i+1
        else:
            neighbors.append((0, j))
            i2 = 0
    
        if j - 1 >= 0:
            neighbors.append((i, j - 1))
            j1 = j-1
        else:
            neighbors.append((i, self.M - 1))
            j1 = self.M-1
    
        if j + 1 < self.M:
            neighbors.append((i, j + 1))
            j2 = j+1
        else:
            neighbors.append((i, 0))
            j2 = 0

        # include diagnal neighbors
        neighbors.append((i1, j1))
        neighbors.append((i2, j1))
        neighbors.append((i1, j2))
        neighbors.append((i2, j2))
    
        return neighbors

    # Original. Unused.
    def calc_heuristic_map(self):
        X, Y = np.meshgrid([i for i in range(self.M)], [i for i in range(self.M)])
        heuristic_map = np.abs(X - self.goal_node[1]) + np.abs(Y - self.goal_node[0])
        print("heuristic_map:", heuristic_map)
        print("heuristic_map[goal]:", heuristic_map[self.goal_node], self.goal_node)
        print("heuristic_map shape:", heuristic_map.shape)
        for i in range(heuristic_map.shape[0]):
          for j in range(heuristic_map.shape[1]):
            heuristic_map[i, j] = min(heuristic_map[i, j],
                                      i + 1 + heuristic_map[self.M - 1, j],
                                      self.M - i + heuristic_map[0, j],
                                      j + 1 + heuristic_map[i, self.M - 1],
                                      self.M - j + heuristic_map[i, 0]
                                      )

        return heuristic_map


    
    # Try to have last arm link (with camera) facing the goal node
    def calc_heuristic_map2(self):
        X, Y = np.meshgrid([i for i in range(self.M)], [i for i in range(self.M)])
        print("X, Y:", X, Y)
        # prioritize those closest to goal
        heuristic_map = np.abs(X - self.goal_node[1]) + np.abs(Y - self.goal_node[0])
        print("heuristic_map:", heuristic_map, len(heuristic_map))
        print("heuristic_map shape", heuristic_map.shape)
        print("heuristic_map[start]", heuristic_map[self.start_node], self.start_node)
        print("heuristic_map[goal]", heuristic_map[self.goal_node], self.goal_node)
        arm_reach = np.sum(self.link_length) # longest possible reach of arm
        arm_node_reach = int(arm_reach*self.grid_factor)

        # occupancy grid heuristic map used just a straight line for both links, pointing at i
        # theta_list = [2 * i * pi / M for i in range(-M // 2, M // 2 + 1)]
        for i in range(heuristic_map.shape[0]):
            for j in range(heuristic_map.shape[1]):
                x_diff = i - self.start_node[0]
                y_diff = j - self.start_node[1]
                if self.grid[i][j] == 7:
                  heuristic_map[i, j] = self.cfg.INFINITE
                  continue
                if int(np.hypot(x_diff, y_diff)) > arm_node_reach:
                  print("arm_node_reach:", int(np.hypot(x_diff, y_diff)), arm_node_reach)
                  heuristic_map[i, j] = self.cfg.INFINITE
                  continue
                min_angle_dif = 2*pi
                best_link_angles = None
                best_link_num = None
                camera_angle_dif = 1
                for link_angle_num, link_angles in enumerate(self.grid_angles[i][j]):
                  print("link_angles:", link_angles, int(np.hypot(x_diff, y_diff)), arm_node_reach)
                  if link_angles is None:
                    heuristic_map[i, j] = self.cfg.INFINITE
                    continue
                  camera_angle_dif = self.camera_angle_delta(link_angles)
                  if camera_angle_dif < min_angle_dif:
                    best_link_angles = link_angles.copy()
                    best_link_num = link_angle_num
                    min_angle_dif = camera_angle_dif 

                # place best link angles to front of list
                if best_link_num is not None and best_link_num != 0:
                   print("b4 self.grid_angles:",(i,j), self.grid_angles[i][j])
                   reordered_grid_angles = []
                   for k in range(len(self.grid_angles[i][j])):
                     if k == 0:
                       reordered_grid_angles.append(self.grid_angles[i][j][best_link_num].copy())
                     elif k <= best_link_num:
                       reordered_grid_angles.append(self.grid_angles[i][j][k-1].copy())
                     elif k > best_link_num:
                       reordered_grid_angles.append(self.grid_angles[i][j][k].copy())
                   self.grid_angles[i][j] = reordered_grid_angles
                   print("af self.grid_angles:",(i,j), self.grid_angles[i][j])
          
                # original heuristic map repeats from 4 different (symetric?) angles
                heuristic_map[i, j] = min(heuristic_map[i, j],
                                         i + 1 + heuristic_map[self.M - 1, j],
                                         self.M - i + heuristic_map[0, j],
                                         j + 1 + heuristic_map[i, self.M - 1],
                                         self.M - j + heuristic_map[i, 0]
                                         ) * camera_angle_dif
                # heuristic_map[i, j] *= camera_angle_dif
                print("camera_angle_dif:",(i,j), heuristic_map[i, j], camera_angle_dif)
    
        print("2:heuristic_map[goal]", heuristic_map[self.goal_node], self.goal_node)
        return heuristic_map

    # unused
    def distance_to_goal(self, current_pos, goal_pos):
        x_diff = goal_pos[0] - current_pos[0]
        y_diff = goal_pos[1] - current_pos[1]
        return np.array([x_diff, y_diff]).T, np.hypot(x_diff, y_diff)
    
    # unused
    def jacobian_inverse(self, link_lengths, joint_angles):
        n_links = len(link_lengths)
        J = np.zeros((2, n_links))
        for i in range(n_links):
            J[0, i] = 0
            J[1, i] = 0
            for j in range(i, n_links):
                J[0, i] -= link_lengths[j] * np.sin(np.sum(joint_angles[:j]))
                J[1, i] += link_lengths[j] * np.cos(np.sum(joint_angles[:j]))
        return np.linalg.pinv(J)

  #################################################
  # Public interface for setting real robot state
  #################################################
    def update_arm_plot(self, img=None):
        colors = ['white', 'black', 'red', 'pink', 'yellow', 'green', 'orange', 'tan']
        levels = [0,       1,       2,     3,      4,        5,        6,       7,    8]
        cmap, norm = from_levels_and_colors(levels, colors)
        if img is not None:
          sp_cnt = 4
        else:
          sp_cnt = 3
        plt.subplot(1, sp_cnt, 1)
        # self.grid[node] = 6  # route
        plt.cla()
        plt.imshow(self.grid, cmap=cmap, norm=norm, interpolation=None)
        # print("Animate node:", node)
        # self.get_arm_theta(node)
        plt.subplot(1, sp_cnt, (2,3))
        self.world_arm.plot_arm(plt, self.goal_pos, line_segments=self.line_segment_obstacles)
        # plt.xlim(-2.0, 2.0)
        # plt.ylim(-3.0, 3.0)
        # self.arm.plot_arm(plt, self.goal_pos, circle_obstacles=obstacles)
        if img is not None:
          plt.subplot(1, sp_cnt, 4)
          plt.imshow(img)
  
    def set_obstacles(self, line_obstacles):
        self.line_segment_obstacles = line_obstacles
        update_arm_plot()
  
    # ground-level: y = 0
    def set_goal_position(self, x, y):
        ground_x1 = self.base[2][1][0]
        ground_y  = self.base[2][1][1]
        base_x1   = self.base[1][1][0]
        # max dist = ground_x1 - base_x1
        self.goal_pos = ((base_x1 + x), ground_y + y)
        if True:
              if img is not None:
                sp_cnt = 3
              else:
                sp_cnt = 2
              plt.subplot(1, sp_cnt, 1)
              self.grid[node] = 6  # route
              plt.cla()
              plt.imshow(self.grid, cmap=cmap, norm=norm, interpolation=None)
              print("Plot node:", node)
              # self.get_arm_theta(node)
              plt.subplot(1, sp_cnt, 2)
              arm.plot_arm(plt, self.goal_pos, circle_obstacles=obstacles)
              plt.xlim(-2.0, 2.0)
              plt.ylim(-3.0, 3.0)
  
        self.world_arm.plot_arm(plt, self.goal_pos, line_segments=self.line_segment_obstacles)
        if img is not None:
          plt.subplot(1, sp_cnt, 3)
          plt.imshow(img)
  
    def set_action_delta(self, new_arm_delta, update_plot=True):
        # format:
        # self.arm_delta = {"UPPER_ARM_UP":.10,"UPPER_ARM_DOWN":-.10,
        #                   "LOWER_ARM_UP":.10,"LOWER_ARM_DOWN":-.10}
        self.arm_delta = new_arm_delta
        # Arm geometry in the working space
        self.arm_pos_to_points()
        self.update_arm_plot()
        if update_plot:
          self.update_arm_plot()
  
    def get_current_position(self, new_arm_delta):
        return self.delta_arm_pos
  
    def reset_current_position(self, update_plot=True):
        for action in ["UPPER_ARM_UP","UPPER_ARM_DOWN","LOWER_ARM_UP","LOWER_ARM_DOWN"]:
          self.delta_arm_pos[action] = 0
        self.arm_pos_to_points()
        if update_plot:
          self.update_arm_plot()
  
    def set_current_position(self, arm_pos, update_plot=True, img=None):
        self.delta_arm_pos = arm_pos
        self.arm_pos_to_points()
        if update_plot:
          self.update_arm_plot(img)
  
    def update_move(self, action, update_plot=True, img=None):
        self.delta_arm_pos[action] += 1
        self.arm_pos_to_points()
        if update_plot:
          self.update_arm_plot(img)
  
    def plan_next_move(self):
        print("plan goal:", self.goal_pos, self.goal_node)
        self.route = self.astar_torus(self.circle_obstacles)
        print("route:", self.route)
        for i, desired_node in enumerate(route):
          curr_node = self.pos_to_node(self.end_effector)
          found_action = None
          for action in ["UPPER_ARM_UP","UPPER_ARM_DOWN","LOWER_ARM_UP","LOWER_ARM_DOWN"]:
            backup_arm_pos = copy.deepcopy(self.delta_arm_pos)
            proposed_arm_pos = copy.deepcopy(self.delta_arm_pos)
            proposed_arm_pos[action] += 1
            self.set_current_position(proposed_arm_pos, update_plot=False)
            eff_node = self.pos_to_node(self.end_effector)
            if eef_node == desired_node:
              found_action = action
              break
          if found_action is None:
            print("failed to find action", curr_node, eff_node)
          else:
            print("Found action", found_action, curr_node, eff_node)
          # restore state
          self.set_current_position(backup_arm_pos, update_plot=False)
          return found_action
  


class NLinkArm(object):
    """
    Class for controlling and plotting a planar arm with an arbitrary number of links.
    """

    def __init__(self, link_lengths, joint_angles, base):
        self.n_links = len(link_lengths)
        
        if self.n_links != len(joint_angles):
            raise ValueError()

        self.link_lengths = np.array(link_lengths)
        self.joint_angles = np.array(joint_angles)
        self.base = base
        self.points = [[0, 0] for _ in range(self.n_links + 1)]

        self.lim = sum(link_lengths)
        self.update_points()
        print("INIT points", self.points)
        print("INIT link lengths", self.link_lengths)
        print("INIT joint angles", self.joint_angles)

    def update_joints(self, joint_angles):
        self.joint_angles = joint_angles
        self.update_points()

    def update_points(self):
        for i in range(1, self.n_links + 1):
            self.points[i][0] = self.points[i - 1][0] + \
                self.link_lengths[i - 1] * \
                np.cos(np.sum(self.joint_angles[:i]))
            self.points[i][1] = self.points[i - 1][1] + \
                self.link_lengths[i - 1] * \
                np.sin(np.sum(self.joint_angles[:i]))
        self.end_effector = np.array(self.points[self.n_links]).T

    def plot_arm(self, myplt, goal, circle_obstacles=[], square_obstacles=[], line_segments=[]):  
        # pragma: no cover
        # plot_world(self):
        myplt.cla()

        for obstacle in circle_obstacles:
            circle = myplt.Circle(
                (obstacle[0], obstacle[1]), radius=0.5 * obstacle[2], fc='k')
            myplt.gca().add_patch(circle)

        for obstacle in square_obstacles:
            square = myplt.Rectangle(
                (obstacle[0], obstacle[1]), 0.5*obstacle[2], 0.5*side*obstacle[2], fc='k', ec='k')
            myplt.gca().add_patch(square)

        # myplt.plot(goal[1], goal[0], 'go')
        if goal is not None:
          myplt.plot(goal[0], goal[1], 'go')
          print("goal: ", goal)
        print("points: ", self.points)

        # plot fixed base, ground in black
        for [pt1,pt2] in self.base:
          myplt.plot(pt1[0], pt1[1], 'ko')
          myplt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'k-')
          myplt.plot(pt2[0], pt2[1], 'ko')

        for i in range(self.n_links + 1):
            if i is not self.n_links:
                myplt.plot([self.points[i][0], self.points[i + 1][0]],
                           [self.points[i][1], self.points[i + 1][1]], 'r-')
            myplt.plot(self.points[i][0], self.points[i][1], 'k.')

        myplt.xlim([-self.lim, self.lim])
        myplt.ylim([-self.lim, self.lim])
        myplt.draw()
        myplt.pause(1e-5)
        # plt.pause(.5)

if __name__ == '__main__':
    arm_nav = ArmNavigation()
