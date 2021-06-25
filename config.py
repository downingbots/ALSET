import copy

class Config():
    def __init__(self):
      ###################################
      # Action Sets: 
      ###################################
      # self.ALSET_MODEL = "S"
      self.ALSET_MODEL = "X"
      ###################################
      if self.ALSET_MODEL == "S":
        # self.IP_ADDR = "10.0.0.31"
        self.IP_ADDR = "192.168.50.182"
        self.arm_actions_no_wrist  = ["UPPER_ARM_UP", "UPPER_ARM_DOWN", "GRIPPER_OPEN", "GRIPPER_CLOSE",
                                       "LOWER_ARM_UP", "LOWER_ARM_DOWN"]
        self.arm_actions = self.arm_actions_no_wrist + ["WRIST_ROTATE_LEFT", "WRIST_ROTATE_RIGHT"]
        self.nn_disallowed_actions = ["REWARD1", "REWARD2", "PENALTY1", "PENALTY2", "WRIST_ROTATE_LEFT", "WRIST_ROTATE_RIGHT"]
        # Note: Wrist Rotate is unreliable. Should just support horiz and vert positioning
        # via a NN.
      ###################################
      if self.ALSET_MODEL == "X":
        self.IP_ADDR = "192.168.50.86"
        self.arm_actions  = ["UPPER_ARM_UP", "UPPER_ARM_DOWN", "SHOVEL_UP", "SHOVEL_DOWN",
                             "LOWER_ARM_UP", "LOWER_ARM_DOWN", "CHASSIS_LEFT", "CHASSIS_RIGHT"]
        self.nn_disallowed_actions = ["REWARD1", "REWARD2", "PENALTY1", "PENALTY2"] 

      ###################################
      self.modes = ["TELEOP", "NN"]
      self.gather_data_modes = ["GATHER_DATA_ON", "GATHER_DATA_OFF"]
      self.base_actions  = ["FORWARD", "REVERSE", "LEFT", "RIGHT"]
      self.robot_actions = self.base_actions + self.arm_actions
      self.joystick_actions  = ["REWARD1", "REWARD2", "PENALTY1", "PENALTY2"]
      self.full_action_set = self.robot_actions + self.joystick_actions
      self.full_action_set.sort()
      self.full_action_set = tuple(self.full_action_set)
      self.debug = False

      # NOOP: obsolete?
      self.NOOP = "NOOP"

      #####################################################
      # SIMPLE PRETRAINED ATOMIC NNs OR AUTOMATED FUNCTIONS
      #####################################################
      self.FUNC_policy = [
                         ["DQN_REWARD_PHASES", [[125,    300],   [275,   400]]],
                         ["DQN_MOVE_BONUS", 1],
                         ["PER_MOVE_PENALTY", -1],
                         ["REWARD2", 50],  # "CUBE_OFF_TABLE_REWARD"
                         ["PENALTY2", -250],  # "ROBOT_OFF_TABLE_PENALTY"
                         ["MAX_MOVES", 1500],
                         ["MAX_MOVES_EXCEEDED_PENALTY", -300.0],
                         ["LEARNING_RATE", 0.0001],
                         ["ERROR_CLIP", 1],
                         ["GAMMA", 0.99],
                         ["ESTIMATED_VARIANCE", 300.0],
                         ["REPLAY_BUFFER_CAPACITY", 10000],
                         ["REPLAY_BUFFER_PADDING", 20],
                         ["BATCH_SIZE", 32]   # ["BATCH_SIZE", 8] ["BATCH_SIZE", 1]
                       ]

      self.func_registry = [
                            # TABLETOP ATOMIC NNs
                            "PT_COLLISION_AVOIDANCE",
                            "COLLISION_AVOIDANCE",
                            "PARK_ARM_HIGH",
                            "PARK_ARM_HIGH_WITH_CUBE",
                            "X_PARK_ARM_UP",
                            "PARK_ARM_RETRACTED",
                            "PARK_ARM_RETRACTED_WITH_CUBE",
                            "QUICK_SEARCH",
                            "QUICK_SEARCH_AND_RELOCATE",
                            "QUICK_SEARCH_FOR_CUBE",
                            "QUICK_SEARCH_FOR_BOX_WITH_CUBE",
                            "RELOCATE",
                            "GOTO_CUBE",
                            "PICK_UP_CUBE",
                            "HIGH_SLOW_SEARCH_FOR_CUBE",
                            "HIGH_SLOW_SEARCH_FOR_BOX_WITH_CUBE",
                            "GOTO_BOX_WITH_CUBE",
                            "DROP_CUBE_IN_BOX",
                             # TABLETOP ATOMIC NNs FOR SUBSUMPTION
                            "STAY_ON_TABLE",
                            "MOVEMENT_CHECK",
                            ## JETBOT ATOMIC NNs 
                            # "LINE_FOLLOWING", "FACE_RECOGNITION", "OBJECT_FOLLOWING", 
                            # "GOTO_OBJECT",
                            # "OBJECT_AVOIDANCE", 
                            ## OTHER ATOMIC NNs FOR SUBSUMPTION
                            # "GOTO_DARK_PLACE", "GOTO_BRIGHT_PLACE",
                            # "HAPPY_DANCE", "YES", "NO", "I_WANT", "IM_THINKING"
                            # "LOOK_AT_FACE", "TRACK_FACE",
                            # "FOLLOW_AND_REACH_FOR"

                            ## FIDUCIAL
                            # "QUICK_SEARCH_FOR_FIDUCIAL"
                            # "DRIVE_TO_FIDUCIAL"
                            # "PICK UP FIDUCIAL"     -> aim at 
                            # "PUSH"
                            # "PUSH WITH GRIPPER"

                            # "Drive to BOUNDING BOX" -> approach angle must be known for pickup
                            # "PICK UP BOUNDING_BOX" -> aim at center?

                            # Wrist rotation is finicky, requiring a NN. It has minimal strength.
                            # In general, keep horizontal.
                            # "ROTATE_WRIST_HORIZ"
                            # "ROTATE_WRIST_VERT"
                            ]

      # self.func_key_val will add key-value attributes to the functions via add_to_func_key_value
      self.func_key_val = self.init_func_key_value(self.func_registry)

      self.func_comments = [
                    ["PT_COLLISION_AVOIDANCE", "PreTrained Collision Avoidance for Jetbot from NVidia"],
                    ["COLLISION_AVOIDANCE", "Collision Avoidance trained via teleop"],
		    ["PARK_ARM_HIGH", "Arm parked with upper arm vertical and lower arm pointing down, with base of robot slightly visible between grippers."],
		    ["X_PARK_ARM_UP", "Model X: move arm / shovel up and as far out as way as possible"],
                    ["PARK_ARM_RETRACTED_WITH_CUBE","PARK_ARM_HIGH with cube in gripper"],
		    ["PARK_ARM_RETRACTED", "Arm parked with upper arm flat backwards and lower arm flat pointing forward. Gripper is open and ground seen a few inches in front of robot.", "ALEXNET"],
		    ["QUICK_SEARCH_FOR_CUBE", "Rotate Left in place, searching for cube. If cube not found, relocate to a different spot and search again."],
		    ["QUICK_SEARCH_FOR_BOX_WITH_CUBE", "Rotate Left in place while gripping a cube while searching for the box. If box not found, relocate to a different spot and search again.", "ALEXNET"],
		    # "GOTO_OBJECT",
		    ["PICK_UP_CUBE", "Position Upper and Lower Arm so that gripper can be closed on cube. Cube is picked up slightly off ground. Minor repositioning of base may be necessary."],
		    ["HIGH_SLOW_SEARCH_FOR_CUBE", "Rotate Left in place while the lower arm scans up and down."],
		    ["HIGH_SLOW_SEARCH_FOR_BOX_WITH_CUBE", "Rotate Left in place while the lower arm scans up and down. Gripper is holding cube while searching for box."],
		    ["GOTO_BOX", "Drive to box while keeping box in clear view."],
		    ["DROP_CUBE_IN_BOX", "Position Upper and Lower arm so that gripper can be openned, dropping cube in box."],
		    # TABLETOP ATOMIC NNs FOR SUBSUMPTION
		    ["STAY_ON_TABLE", "An automated function that tries to rotate left to avoid going off the table. If run stand-alone, will drive around the edges of the table. Can be a background check that encourages staying on the table."],
		    ["MOVEMENT_CHECK", "An automated function that runs in the background. It checks that the arm is moving as expected and prevents the arm from being forced beyond its limits."]
                  ]

      self.func_subsumption= ["STAY_ON_TABLE", "MOVEMENT_CHECK"]
      # self.func_subsumption = ["STAY_ON_TABLE", "FIDUCIAL", "OBJECT_FOLLOWING", "OBJECT_AVOIDANCE", "OBJECT_PICKUP", ["TIMER", secs, ACTION]  ]

      # [Specialized trained versions of general automated_function]
      # Get name of automated function to run and different datasets associated with the function.
      self.func_automated = [["HIGH_SLOW_SEARCH_FOR_CUBE", "HIGH_SLOW_SEARCH"],  
                             ["HIGH_SLOW_SEARCH_FOR_BOX_WITH_CUBE", "HIGH_SLOW_SEARCH"],
                             ["QUICK_SEARCH_FOR_CUBE","QUICK_SEARCH"],
                             ["QUICK_SEARCH_FOR_BOX_WITH_CUBE", "QUICK_SEARCH_AND_RELOCATE"],
                             ["QUICK_SEARCH_AND_RELOCATE", "QUICK_SEARCH_AND_RELOCATE"],
                             ["X_PARK_ARM_UP", "X_PARK_ARM_UP"],
                             ["PARK_ARM_RETRACTED", "PARK_ARM_RETRACTED"],
                             ["PARK_ARM_RETRACTED_WITH_CUBE", "PARK_ARM_RETRACTED"],
                             ["CLOSE_GRIPPER", "CLOSE_GRIPPER"], 
                             ["MOVEMENT_CHECK", "MOVEMENT_CHECK"]
                            ]

      self.arm_actions_park_arm_retracted  = ["UPPER_ARM_UP", "GRIPPER_OPEN", "GRIPPER_CLOSE", "LOWER_ARM_DOWN"]
      self.func_movement_restrictions = [
                             ["PARK_ARM_RETRACTED", self.arm_actions_park_arm_retracted],
                             ["QUICK_SEARCH_FOR_CUBE", self.base_actions],
                             ["GOTO_CUBE", self.base_actions],
                             ["PICK_UP_CUBE", self.robot_actions],
                             ["QUICK_SEARCH_FOR_BOX_WITH_CUBE", self.base_actions],
                             ["GOTO_BOX_WITH_CUBE", self.base_actions],
                             ["DROP_CUBE_IN_BOX", self.robot_actions],
                             ["STAY_ON_TABLE", ["LEFT"]]
                           ]

      # use CNN instead of DQN to train.  May allow more classification app logic in future.
      self.func_pt_col_avoid_func_flow_model = [
            [[],["START", 0]],
            [[0], ["IF", "BLOCKED", "LEFT" ]],
            [[0], ["IF", "FREE", "FORWARD" ]],
            [[0], ["IF", "PENALTY1", "STOP" ]],
            ]
      self.func_col_avoid_func_flow_model = [
            [[],["START", 0]],
            [[0], ["IF", "REWARD1", "LEFT"] ],
            [[0], ["IF", "PENALTY", "STOP" ]],
            [[0], ["IF", "BLOCKED", "LEFT" ]],
            [[0], ["IF", "FREE", "FORWARD" ]],
            ]

      # for composite apps, key-value pirs
      self.func_classifier = [ 
        ["PT_COLLISION_AVOIDANCE", [["BLOCKED", "FREE"], self.func_pt_col_avoid_func_flow_model]],
        ["COLLISION_AVOIDANCE", [["LEFT", "FORWARD"], self.func_col_avoid_func_flow_model]]
       ]
      # TODO: self.classifier = [FIDUCIAL, OBJECT, FACE, DARK_LIGHT]

      # TODO:
      # self.automated_app = ["OBJECT_AVOIDANCE", "OBJECT_FOLLOWING", 
      #                       "SEARCH_FOR_OBJECT" , "DRIVE_TO_OBJECT", "PICKUP_OBJECT", 
      #                       "FIDUCIAL_AVOIDANCE", "FIDUCIAL_FOLLOWING", 
      #                       "SEARCH_FOR_FUDUCIAL", "DRIVE_TO_FIDUCIAL", "PICK_UP_FIDUCIAL",
      #                       "TIMER", "ROTATE_ONCE"]



      # TODO: train across different specializations to make a more general catagory for
      # pretrained transfer learning
      self.func_specializations = [# ["PICK_UP", ["PICK_UP_CUBE"]], ["DROP", ["DROP_CUBE"]],
		  # [QUICK_SEARCH_FOR_OBJECT, [QUICK_SEARCH_FOR_CUBE, QUICK_SEARCH_FOR_FIDUCIAL]],
		  # ["QUICK_SEARCH", ["QUICK_SEARCH_FOR_CUBE", "QUICK_SEARCH_FOR_BOX_WITH_CUBE"]],
		  # ["SEARCH", ["SEARCH_FOR_CUBE", "SEARCH_FOR_BOX_WITH_CUBE"]]
		  # ["PARK", ["PARK_ARM_LOW", "PARK_ARM_HIGH", "PARK_ARM_SHORT"]]
		  # ["GOTO", ["GOTO_DARK_PLACE", "GOTO_BRIGHT_PLACE", "GOTO_CUBE", 
                  #           "GOTO_BOX_WITH_CUBE"]]
		  ]

      # optical flow thresholds for detected movement; a nested k-v pair
      # self.func_attributes = [["MOVEMENT_CHECK",[["OPTFLOWTHRESH", 0.8], ["MAX_NON_MOVEMENT",2]]]]
      # self.OPTFLOWTHRESH = 0.8
      self.OPTFLOWTHRESH = 0.5
      self.MAX_NON_MOVEMENT = 2

      # TODO: use for verification before gathering data
      # ["NON_BASE_ONLY", "BASE_ONLY", ["ONLY", ...], ["START_POSITION", ...]
      # "MODEL": "ALEXNET", "OBJECT_DETECTION"
      # self.func_attributes = ["PARK_ARM_RETRACTED", ["MODEL", "ALEXNET"]]
      #                    "HIGH_SLOW_SEARCH"][],
      #                    ["PARK_ARM_RETRACTED", ["QUICK_SEARCH"]] ]
      # self.func_attributes = ["PARK_ARM_RETRACTED", ["ARM_ONLY"], ["MODEL", "ALEXNET"]]

      # TODO/FUTURE: external sensor-based reward/penalty system shared among a swarm
      # case statement: use for bounding box of classification

      # denormalizes each
      self.add_to_func_key_value("COMMENT", self.func_comments)
      self.add_to_func_key_value("SUBSUMPTION", self.func_subsumption)
      self.add_to_func_key_value("AUTOMATED", self.func_automated)
      self.add_to_func_key_value("MOVEMENT_RESTRICTIONS", self.func_movement_restrictions)
      self.add_to_func_key_value("CLASSIFIER", self.func_classifier)


      #######################
      # DEFINE TABLE_TOP_APP
      #######################
      self.TT_name        = ["TT"]
      self.TT_func        = ["PARK_ARM_RETRACTED",                    #  0
                             "QUICK_SEARCH_FOR_CUBE",                 #  1
                             "GOTO_CUBE",                             #  2
                             "PICK_UP_CUBE",                          #  3
                             "PARK_ARM_RETRACTED_WITH_CUBE",          #  4
                             "QUICK_SEARCH_FOR_BOX_WITH_CUBE",        #  5
                             "GOTO_BOX_WITH_CUBE",                    #  6
                             "DROP_CUBE_IN_BOX",                      #  7
                             # "STAY_ON_TABLE",                         #  8
                             # "MOVEMENT_CHECK"                         #  9
                            ]

      # TODO: A new App reset the TT challenge. Eventually use a cooperating robot with obstacle avoidance.
      # Goto random place on TT.  Search for box.  Push or Pickup box. Grab on far left or far right top.
      # Move to random new location on TT.  Dump the cube (Lower arm straight up).   
      # Move to random new location on TT.  Place the box.   
      # Move to random new location on TT, avoid box and cube.  (verify box. verify cube. (reward))
      # Start TT pick and place again.


      # define the simple process flow model
      # TODO: Walk user through an Q&A to define the process flow, producing the model automatically
      # TODO: add support for:
      #   [RANDOM, [function#],[function#]] : choose random function in list
      #   [[CLASSIFIER, function#] ["class_name", function#],["class_name", function#]...] : 
      #                            ["OTHER_CLASSES", "WAIT"]
      #   ["TIMER", secs, APP_NAME]
      #   ["TIMER", secs, [FUNC #]
      #   ["IF", "DONE", [CLASSIFIER ...]
      #   [["IF","IDLE"], [DO_ANY, [FUNC #]]]  Idle in ["None","Done","Wait","STOP", "Idle"]
      # Note: currently only supports 2 levels of rewards and penalties
      #       If eventually supports > 2 levels, add level as a parameter: ["NEXT_WITH_REWARD",[1]]
      self.TT_func_flow_model = [
            [[],["START", 0]],
            [[0,1,2,4,5,6], ["IF", "REWARD1", "NEXT"] ], 
            [[3], ["IF", "REWARD1", "NEXT_WITH_REWARD1"]], 
            # TODO: [[3], ["IF", "REWARD1", ["GOTO_WITH_REWARD1", [4]]]], 
            [[7], ["IF", "REWARD1", "STOP_WITH_REWARD1" ]], 
            ["ALL", ["IF", "REWARD2", "STOP_WITH_REWARD2"]],
            [[2, 3, 4], ["IF", "PENALTY1", [0]]],
            [[1, 5], ["IF", "PENALTY1", "IGNORE"]],
            [[6, 7], ["IF", "PENALTY1", [4]]],
            ["ALL", ["IF", "PENALTY2", "STOP_WITH_PENALTY2"]],
            ["BACKGROUND_CHECK","STAY_ON_TABLE"],
            ["BACKGROUND_CHECK","MOVEMENT_CHECK"]
            ]

      #          +-r2,p2-+-------+-------+-------+--r2,p2-+-------+-------+-----> STOP
      #          |       |       |       |       |        |       |       |                   
      # START -> 0 -r1-> 1 -r1-> 2 -r1-> 3 -r1-> 4 ->r1-> 5 -r1-> 6 -r1-> 7 -r1-> STOP
      #          ^               |       |       ^                |       |                   
      #          +----p1---------+-------+       +-----p1---------+-------+                      
      # BACKGROUND: STAY_ON_TABLE, MOVEMENT_CHECK

      # for composite apps, key-value pirs
      self.app_registry = [["TT",[self.TT_func, self.TT_func_flow_model]]]

      ######################
      # define TT DQN policy
      self.TT_DQN_policy = [
                         # reward_ph0 = 125 + max((ALLOCATED_MOVES_CUBE_PICKUP - frame_num),0)*MOVE_BONUS
                         # reward_ph1 = 275 + max((ALLOCATED_MOVES_CUBE_DROP_IN_BOX - frame_num),0)*MOVE_BONUS
                         # The reward phases map to the "WITH_REWARD1" actions in the function_flow_model
                         # more phases allowed
                         ["DQN_REWARD_PHASES", [[125,    300],   [275,   400]]],
                         ["DQN_MOVE_BONUS", 1],
                         ["PER_MOVE_PENALTY", -1],
                         ["REWARD2", 50],  # "CUBE_OFF_TABLE_REWARD"
                         ["PENALTY2", -250],  # "ROBOT_OFF_TABLE_PENALTY"
                         ["MAX_MOVES", 1500],
                         ["MAX_MOVES_EXCEEDED_PENALTY", -300.0],
                         ["LEARNING_RATE", 0.0001],
                         ["ERROR_CLIP", 1],
                         ["GAMMA", 0.99],
                         ["ESTIMATED_VARIANCE", 300.0],
                         ["REPLAY_BUFFER_CAPACITY", 10000],
                         ["REPLAY_BUFFER_PADDING", 20],
                         ["BATCH_SIZE", 32]   # ["BATCH_SIZE", 8] ["BATCH_SIZE", 1]
                       ]

      # a key-value pair, Same key should be in app_registry
      self.DQN_registry = [["TT",[self.TT_DQN_policy]]]

      ###########################
      # DEFINE TEST TABLE_TOP_APP
      ###########################
      self.TTT_name        = ["TTT"]
      self.TTT_func        = [
                              # "PARK_ARM_RETRACTED",           #  0
                              "PARK_ARM_HIGH",           #  0
                              "QUICK_SEARCH_FOR_CUBE"         #  1
                             ]

      self.TTT_func_flow_model = [
            [[],["START", 0]],
            [[0], ["IF", "REWARD1", "NEXT"] ],
            [[1], ["IF", "REWARD1", "STOP_WITH_REWARD1" ]],
            [[1], ["IF", "PENALTY1", [0]]],
            ]

      self.TTT_DQN_policy = copy.deepcopy(self.TT_DQN_policy)
      self.TTT_DQN_policy[0] = ["DQN_REWARD_PHASES", [[100,   400]]]  # replace

      # for composite apps, key-value pirs
      self.app_registry.append(["TTT",[self.TTT_func, self.TTT_func_flow_model]])
      self.DQN_registry.append(["TTT",[self.TTT_DQN_policy]])


      ############################
      # DEFINE PRETRAINED PT_COLLISION_AVOIDANCE
      ############################
      self.pt_col_avoid_name        = ["PT_AVOID_COLLISION"]
      self.pt_col_avoid_func        = [
                                      "PT_COLLISION_AVOIDANCE"        #  0
                                      ]

      self.app_pt_col_avoid_func_flow_model = [
            [[],["START", 0]],
            [[0], ["IF", "REWARD1", "LEFT"] ],
            [[0], ["IF", "PENALTY", "STOP" ]],
            [[0], ["IF", "BLOCKED", "LEFT" ]],
            [[0], ["IF", "FREE", "FORWARD" ]],
            ]

      # for composite apps, key-value pirs
      self.app_registry.append(["PT_AVOID_COLLISION",[self.pt_col_avoid_func, self.app_pt_col_avoid_func_flow_model]])

      ############################
      # DEFINE COLLISION_AVOIDANCE
      ############################
      # if we do regular teleop training from scratch instead of pre-trained classifier, 
      # we would define following
      self.col_avoid_name        = ["AVOID_COLLISION"]
      
      self.col_avoid_func     = [
                                "COLLISION_AVOIDANCE"        #  0
                                ]

      # LEFT
      self.col_avoid_func_flow_model = [
            [[],["START", 0]],
            [[0], ["IF", "LEFT", "LEFT"] ],
            [[0], ["IF", "FORWARD", "FORWARD"] ],
            [[0], ["IF", "REWARD1", "STOP" ]],
            [[0], ["IF", "PENALTY1", "STOP" ]],
            ]


      # for composite apps, key-value pirs
      self.app_registry.append(["AVOID_COLLISION",[self.col_avoid_func, self.col_avoid_func_flow_model]])

      self.col_avoid_restrict = ["FORWARD", "LEFT"]
      self.func_movement_restrictions.append(["COLLISION_AVOIDANCE", self.col_avoid_restrict])

      ########################
      # DEFINE PERSONALITY APP
      ########################

      # Wakeup, Happy Dance, 
      # random:
      #   Look for object
      #   Find dark place
      #       -> place to store things
      #   Find bright place
      #       -> place to hang out
      #   Look for face

      # "HAPPY_DANCE", "YES", "NO", 

      # RECORD SCRIPT
      # SHAKE HANDS
    
      # if find object and above ground, do "I_WANT" (Gripper open/close/open/close until on table)
      #    look at object
      # if object on table, pick up object
      # if object too big, push to dark place

      # Fire Engine: Look for fire_hydrant, drive to left of it.
      # IM_THINKING == ROTATE ONCE

      # Facial recognition: https://gist.github.com/ageitgey/e60d74a0afa3e8c801cff3f98c2a64d3
      #    stores known faces
      #    TT: rotates and points arm at known face.  No FORWARD/REVERSE.
      #    Non-TT: Follow known person

      # https://github.com/NVIDIA-AI-IOT/face-mask-detection/blob/master/face-mask-detection.ipynb
      # Faces with Mask:
          # Kaggle Medical Mask Dataset Download Link
          # MAFA - MAsked FAces Download Link
      # Faces without Mask:
          # FDDB Dataset Download Link
          # WiderFace Dataset Download Link
      # Stick figure pose: trt_pose  (made for higher end jetsons)
      # Hand Pose: hand_pose_resnet18_baseline_att_224x224_A
      # https://github.com/holli/hands_ai
      # open_hand, finger_point, finger_gun, fist, pinch, one, two, three, four,
      # thumbs_down, thumbs_up
      # TT: Point to place on Table in front of robot for robot to go to or object to pick up.
      #     Left/Right

      # Note: COCO is for OBJECT_DETECTION and known objects include: 
      #         bird dog cat sheep zebra giraffe bear cow horse teddy bear person
      #         bus train truck motorcycle bicycle airplane car
      #         stop_sign traffic_light fire_hydrant
      #         chair dining_table couch bed book teddy bear
      #         sports_ball cup bowl

      ##########################
      # DEFINE TABLE SETTING APP
      ##########################
      # cafeteria classes: https://github.com/NVIDIA-AI-IOT/GreenMachine
      #   Set a table: knife,fork, plate chobsticks, cups, plates, bottle
      # cup (cup/soup bowls); rutensil (silverware); tutensil (plastic utensils);
      # container (paper containers and to-go boxes); plate (bowls and plates); paper (napkins);
      # stick (chopsticks and coffee stirrers); bottle (water and drink Bottles);
      # wrapper (food and candy wrappers)

      # YOLO9000 classes: https://github.com/pjreddie/darknet/blob/1e729804f61c8627eb257fba8b83f74e04945db7/data/9k.names

      # motion to detect pickup
      # distance to bounding box? size of object ?
      # case distance_to_bounding_box
      # bounding_box_width
      # bounding_box_center
      # GET WRIST ROTATION WORKING
      #    -> wrist rotation isn't simple MCP command.  Need to use a sequence of movements
      #       to ensure it moved.  Might need some back&forth.  Only to modes:
      #       -> It's own automated NN with rewards for when vertical / horizontal
      #       -> wrist_horizontal (camera on right)
      #       -> wrist_vertical (camera on top)



      ###################################
      # Path and File Names
      ###################################
      # used by dataset_utils
      self.APP_DIR               = "./apps/"
      self.DATASET_PATH          = "/dataset/"
      self.DATASET_IDX_DIR       = "/dataset_indexes/"
      self.DATASET_IDX_PROCESSED = "dataset_idx_processed.txt"
      self.DATASET_IDX_POSTFIX   = "_idx.pth"
      self.DATE_FMT              = "_%y_%m_%d"        # + letter[a-zA-Z] + ".txt"
      self.MODEL_POST_FIX        = "_MODEL.pth"
      self.REPLAY_BUFFER         = "_replay_buffer.data" 
      self.DQN_NN_COMBOS         = "_DQN_NN_TRAINING_COMBOS.txt"

      self.NUM_EPOCHS            = 30
      ###################################
      # TODO: CONTROLLER MAPPINGS: 
      ###################################
      # map actions to button/axis
      # map actions to web controller images
      self.DEFAULT_WEBCAM_IP     = self.IP_ADDR 
      self.DEFAULT_WEBCAM_PORT   = 8080
      
    ###################
    # TODO: FIDUCIALS
    ###################

    ###################
    # FUNCTIONS
    ###################

    # func_key_val is a nested key-value structure of [[func, [key, [value]]],[func2, [key2, [value2]]]]
    def init_func_key_value(self, func_registry):
        func_key_val = []
        for func in func_registry:
          func_key_val.append([func,[]])
        return func_key_val

    # take a list in form of:
    #    self.key_name1k = [[func_name1, values1a],[func_name2, values2a]]
    # or self.key_name1k = [[func_name1, [values1a]],[func_name2, [values2a]]]
    # or self.key_name1k = [func_name1, func_name2]
    # and store it in a 2-level nested kv list
    #    self.func_key_val = [[func_name1, [[key_name1k, value1a], [key_name2k value1b]]],
    #                         [func_name2, [[key_name1k, value2a], [key_name3k, value2b]]]]
    def add_to_func_key_value(self, key, func_list):
        k = 0    # key offset
        v = 1    # value offset
        if type(key) != str:
          print("key in key-value pair must be a string:", key)
          exit()
        # Function + key 
        # if func_list is just a list, set attribute to True (None implies False)
        for new_func in func_list:
          if type(new_func) == str:
            func_name = new_func
            value = [True]
            if self.debug:
              print("Func_val0", func_name, value)
          elif type(new_func) == list:
            func_name = new_func[k]
            if type(new_func[v]) == str:
              value = [new_func[v]]
              if self.debug:
                print("Func_val1", func_name, value)
            elif type(new_func) == list:
              value = new_func[v]
              if self.debug:
                print("Func_val2", func_name, value)
          func_kv = self.get_value(self.func_key_val, func_name)
          self.set_value(func_kv, key, value)
    
    def get_func_value(self, func, key):
        k = 0    # key offset
        v = 1    # value offset
        func_kv = self.get_value(self.func_key_val, func)
        func_val = self.get_value(func_kv, key)
        if self.debug:
          print("get_func_val", func, key, func_kv, func_val, self.func_key_val)
        return func_val

      # for key-value sets like self.TT_DQN_policy
    def get_value(self, kv_lst, key):
        k = 0    # key offset
        v = 1    # value offset
        if kv_lst is not None:
          for kv in kv_lst:
            if kv[k] == key:
              if self.debug:
                print("get_value:", kv[k], kv[v]) 
              return kv[v]
        return None

      # for key-value sets like self.TT_DQN_policy
    def set_value(self, kv_lst, key, value):
        k = 0    # key offset
        v = 1    # value offset
        if kv_lst is not None:
          for kv in kv_lst:
            if kv[k] == key:
              if self.debug:
                print("b4",kv_lst, key, value)
              kv[v] = value
              if self.debug:
                print("af",kv_lst, key, value)
              return
        else:
          kv_lst = []
        kv_lst.append([key, value])


    def set_debug(self, TF):
        self.debug = TF
