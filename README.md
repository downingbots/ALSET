# SIR_jetbot

This is SIRjetbot1. The SIR stands for Sharper Image Robot, which I purchased on
 clearance for less than $20. The 1 is because we bought 3 of them that worked.
 
This robot is an inexpensive platform to run DDQN reinforcement learning. However, 
the software is not specific to the Sharper Image Robot. In theory, it could easily
be generalized to many inexpensive RC toys with tracked or differential-drive wheels and 
an arm or crane or dozer or shovel. Just add a Jetson Nano, battery, and camera mounted
near the end of the arm/excavator/bulldozer/crane/shovel as described below. Contact me if interested.

<p align="center">
  <img src="https://github.com/downingbots/SIR_jetbot/blob/master/sharper_image_robot.jpg" width="350" title="Sharper Image Robot">
  <img src="https://github.com/downingbots/SIR_jetbot/blob/master/sharper_image_robot2.jpg" width="350" alt="accessibility text">
</p>
Other potential platforms that code might work for (with jetson nano and minor mods): Remote Control Bulldozer + Top Race 15 Channel RC Excavator
<p align="center">
  <img src="https://github.com/downingbots/SIR_jetbot/blob/master/excavator.jpg" width="200" title="RC Excavator">
  <img src="https://github.com/downingbots/SIR_jetbot/blob/master/bulldozer.jpg" width="200" alt="accessibility text">
</p>

The Sharper Image robot was hacked as followed:

    - Jetson Nano Developer Board with wifi
    - raspberry pi camera v2 with 2 ft cable mounted on a cookie wheel case near the end of the arm.
    - A hacked RC:
        - the inside of the low-end RC control that came with the robot. 
        - You can see/use the buttons here control the robot. 
        - the other side of the board was soldered to wire it up with the IO expansion board.
    - The IO expansion board was required :
        - the RC control uses Tri-state logic on 6 pins.
        - The expansion board uses I2C to communicate with the Jetson Development board
        - The expansion board is connected to the RC control via wires
    - It is powered by a mongo BONAI 5.6A battery pack
    - Logitech gamepad joystick

The code started with the Jetson Notebook tutorial on Obstacle avoidance, but was changed 
significantly. The Notebook tutorials barely ran on my Jetson with the wifi connectivity from my working area. The tutorials were replaced with:

    - The Logitech joystick controls the robot directly
    - The camera is streamed using much lower overhead video webserver for teleoperation.
    - The images are saved directly to the Robot SD card.
    - a poor-man's pwm changes the commands to start/stop up to every tenth of a second or so.
        - takes the picture when stopped.  
        - records the command sent by the joystick (lower or upper arm up/down, open/close gripper, etc.) along with the picture in the directory associated with the NN's command.
    
      This data is then gathered and used to train the robot. Note: a tenth of a second of start/stop proved too much for one of the boards and the robot will stop moving after a few minutes of continuous use. Eventually changed the pwm rate to two-moves-per-second.

The robot runs in 4 modes: RC telepresence, data-capture, and using trained neural net(s) including a single CNN, a multi-part sequence of CNNs, and DDQN reinforcement learning.  The data capture and neural net can be for a single alexnet NN, or a sequence of alexnet NNs with function-specific knowledge. For example, the 8-parts are for an HBRC phase-3 tabletop bot is:

    - get the arm to a known position
    - scans for object on the floor (or table)
    - goes to object, keeping object in center of vision
    - pick up the object
    - get the arm to a known position, while gripping object
    - scans for the box
    - goes to the box (keeping box in center of vision)
    - drops object in box
    - backs up and park arm (back to phase 1)

The default app is the HBRC tabletop challenge phase 3:

  - https://www.hbrobotics.org/robot-challanges/
  
Basically, the robot is on a table, searches for a cube, drives
to the cube, picks the cube up, searches for a box, drives to
the box and drops the cube in the box.  As dropping off a table
is dangerous for a robot's health, I have a setup so that the
tabletop is only inches from the floor during training.

A single NN is sufficient for phase-1 of the tabletop challenge.
Phase 1 is just go end-to-end and back on a table without falling
off the table.  The vanilla Jetbot can do that. However, using
an arm to pick up a cube and having multi-phase objectives is
much harder resulting in using DQN. DQN is used as the robot isn't
continuously running in realtime. The robot stops and
figures out the next move, executes the next move, and repeat.
In addition, picking up a cube is a sparse-reward (taking hundreds
of unrewarding moves), as is dropping
the cube. There's no continuous scoreboard accumulating tons of
rewards/penaties as you might have on a video game.

The same training can be used for both a sequence of functional
NNs (i.e. TT_func ) and for DDQN RL. A potential goal is to train NNs to 
different functions (like above) and then combine the functions together 
in different ways to perform different tasks. Then use DDQN to get
optimized end-to-end functionality.

It's relatively easy to add other apps because the training
is done via teleop.  It's a matter of defining and hooking together
the different functions to train the NNs on and defining the DQN
compute_reward() policies.  You can also add "automatic mode" NNs
to assist in training, as is done for the "scan for cube" and 
"scan for box" NNs.

HOW TO RUN
----------
First, follow the NVIDIA instructions to install the jebot onto the
Jetson Nano. Use a huge SD card as images for the datasets will be
stored there and training will (by default) be done on the Nano.
It is possible to follow the Jetson instructions so that training 
could be done on a laptop instead.

cd to the jetson source directory (or a new directory) and put the
contents of this github repository there. Add an apps subdirectory, 
and further subdirectories for:

  - apps/TT_DQN/dataset
  - apps/TT_func/dataset
  - apps/NN/dataset

Other subdirectories will be automatically created during execution.

If you are doing active development, I have a script hack that copies the 
python files to their system directories:

  - ./copy_to_python 

The robot's camera is streamed while the robot is running. Goto your
web browser and look at the 8080 port on the robot's ip address like:

  - http://10.0.0.31:8080/

The jetson doesn't seem to always properly cleanup the webcam upon
killing the robot executable. If so, run the following script:

  - ./restart_camera

To execute the teleop app, run:

  - python3 ./sir_robot_teleop.py --app_name TT_func
    - You can also run with app_name "TT_DQN" and "NN".

To gather data in teleop, hit the top left button to go into "gather data" mode.
You can toggle off gather_data mode to teleop reposition the robot.

For DQN, gather_data mode executes the NN and gathers data for training.
Training should automatically be done at the end of every run.

To train, run:

  - python3 ./sir_robot_train.py --app_name TT_func
    - You can also train with app_name "TT_DQN" and "NN".

The joystick commands on the logitech controller are:

  - TELEOP LEFT_TRACK               => "y" axis
  - TELEOP RIGHT_TRACK              => "ry" axis
  - TELEOP/DQN REWARD               => "z" axis
  - TELEOP/DQN PENALTY              => "rz" axisNN/TELEOP
  - TELEOP/DQN GATHER_DATA TOGGLE   => "tl" button
  - NN/TELEOP TOGGLE                => "tr" button
  - TELEOP LOWER_ARM_DOWN           => "a" button
  - TELEOP LOWER_ARM_UP             => "y" button
  - DQN ROBOT_OFF_TABLE_PENALTY     => "y" button
  - TELEOP UPPER_ARM_DOWN           => "hat0y" axis
  - TELEOP UPPER_ARM_UP             => "hat0y" axis
  - DQN CUBE_OFF_TABLE_REWARD       => "y" button 
  - TELEOP GRIPPER OPEN             => "x" buttons
  - TELEOP GRIPPER ClOSE            => "b" button
  - TELEOP WRIST ROTATE LEFT/RIGHT  => "hat0x" axis 
     - (note: WRIST is not used by any of the NNs)

The human uses the joystick to define the REWARD and PENALTY.
For TT_func, these will result in moving on to the next NN to
train.
For DQN, these will end the run. Different terminal penalties
and rewards can be awarded by ROBOT_OFF_TABLE_PENALTY and
CUBE_OFF_TABLE_PENALTY.

The DQN tabletop reward policies are defined in compute_reward().
Basically, the rewards are for picking up the cube, dropping the
cube into the box, or a smaller reward for dropping/pushing the 
cube off the table.  The penalties are a small per-move penalty,
the robot going off the table (or other bad state), or taking too
many moves.

REINFORCEMENT LEARNING ON REAL ROBOTS: Lessons Learned
------------------------------------------------------
I want to do Reinforcement Learning (RL) on real robots (specifically mobile manipulators with high-end hobbyist-grade hardware such as dynamixel MX-64 servos.) These robots cost less than a few thousand dollars. Such robots would be considered very low-end by university research robotics labs.

ROS is a good place to start with real robots, but you'll eventually hit the limits of what custom software can achieve.  Robot perception is still not solved and the best human-designed algorithms leave a lot to be desired.  My hope is that RL can adapt to handle low-end hardware and fill some of the intelligence void in robotics. Unfortunately, RL presents its own set of challenges.  I want to learn these challenges and try to solve subsets of these open-end research problems.

SIR_jetbot_the_first addresses several lessons learned the hard way.
  - Over time, I've become convinced that Robot Arms should have camera attached directly to the arm and use RL for digital servoing. SIR_jetbot1 does this with its only sensor - the RPi camera on its gripper (just below the "wrist").
  - SIR_jetbot1 does discrete moves to avoid realtime processing and also to handle low-end hardware limations (mcp23017 communication rate).
  - on-board Jetson is the most expensive component. Total price of the whole robot is a few hundred dollars.
  - Use imitation-learning to reduce amount of RL episodes that you have to run.
  
  There are many obstacles of doing RL on real robots:
 - number of episodes to tune RL (1000-100,000 on model-free). Solving problems like Open AI's DOTA, Deep Mind's Alpha-GO, Open AI's GTP3, etc. requires hundreds of thousands of dollars of computing power. We want to do realtime tuning of RL on a small scale. We want to be order of 10 training sessions or You-Only-Demo-Once.
 - Getting access to state information. Simulations provide the internal state of objects such as block locations so you can compute distance to block.
 - low-end hardware adds more complexity for repeatability

To solve the problem of getting the state of the environment that simulations can provide for free, you need other external mechanisms to evaluate real-world state:
  - OpenCV implementations to identify state (e.g., distance from line, location of block)
  - Separate cameras
  - add fiducials to objects (e.g., block) in environment
  - add sensors to objects in environment
  - Tons of human-interactons to reset the environment

Some of my previous robots: PR2Lite (with Maxwell, which ROSwell cloned), REPLab, RLDonkeycar
<p align="center">
  <img src="https://github.com/downingbots/SIR_jetbot/blob/master/pr2lite_handshake_chess.jpg" width="150" title="RC Excavator">
  <img src="https://github.com/downingbots/SIR_jetbot/blob/master/replab.jpg" width="150" alt="accessibility text">
  <img src="https://github.com/downingbots/RLDonkeycar/blob/master/RLDonkeycar.jpg" width="150" alt="accessibility text">
 
</p>

Lessons from PR2-Lite:
  - PR2-Lite isn't light. ROSwell is a much better size. PR2-Lite was an excellent platform for the team of advanced high-school kids who largely designed, coded and built it though (with lots of mentoring.)
  - ROS has a huge learning curve, but I've been using it since C-turtle days.
  
Lessons from ROSwell:
  - ROS is good. SLAM and complex arm planning (through MoveIt) are now doable by hobbyists. However, doing complex things are really hard to do.  Human perception algorithms just aren't good enough.  This has led to my interest in RL.
  - Human-sized robots are good. But so is transportability. ROSwell / Maxwell is just the right size and pricepoint for a serious robot hobbiest. The current robot's most complex application (chess by Mike Ferguson) is literally out-of-reach for ROSwell's arm. The robot would need to physically move locations to reach the other side of a standard-sized board.
  - Honestly, the picking up of a chess piece or similarly light object just isn't inspiring.  I'd suggest going to a SCARA arm with real lift capabilities.
  - I'd love a robot that can go up stairs. Houses often have too many stairs. I'd love a quadruped like Boston Dynamic's Spot.  Lower-end clones of Spot are on their way, but it's doubtful that they can go up stairs and have a weight-bearing arm.
  - In an attempt to use ROSwell for RL, I tried to get ROSwell working within a physics-simulator so that I didn't risk wearing out or damaging the robot in humongous numbers of random trials. Unfortuantely, the simulations don't match reality at all!  Spent a ton of time trying to get Gazebo physics engine to realistically model a light, top-heavy robot with low-end servos (e.g., dynamixel MX-64).
    - Problems encountered include incompatible upgrades of components
    - Difficulty tuning of weights, inertia, friction, pids, transmissions
    - Physics of torsional friction in gazebo is missing or unrealistic (depending on physics engine release)
    - Top-heavy robot might flip 10 feet into the air!
    - Lots of papers on needing different lighting, coloring, etc.
    - Might as well use no physics engine and assume perfect performance.

Lessons from Donkey-car:
  - Donkey car perfomance changed as it used up batteries. The RL doesn't adapt for this.
  - Continuous realtime RL is hard. On-board processing needs better performance for continuous realtime RL. On the other hand, Off-loaded processing to a laptop needs better communication performance for continuous realtime RL.
  - You do a lot of training but still overfits to environment, (fails at big DIY robocar events due to the addition of spectators or change in location.)  You need to train on many tracks, in many lighting conditions, with and without spectators, etc.

Lessons from REPLab:
  - Intel 3D Realsense camera gave poor results for any single 3D snapshot. Needed to accumulate and integrate results.  Worked around this by using OctoMaps, but this greatly reduces performance. Most RL implementations just use 2D camera, ignoring 3D camera capabilities.
  - Used ROS moveit to assume away much of the RL problem, only using RL for planning the final stage of grasping (e.g., lower straight down from above so only choosing final x/y and theta).  The numerous RL papers on grasping often try to show that they are slightly faster than the previous state of the art. Really, who cares if one algorithm is slightly faster in computing x/y/theta when other real-world concerns like the time of moving the arm dominates?  Provide more complex functionality instead.
  - Frequent calibration between robot camera and arm. Simple calibration didn't do very well across robots or across runs on same robot due to overheating or stressing of motors (e.g., pushing down too hard on tray).
  - Pretrained imagenet models provide some transfer learning for regular cameras, but this doesn't help for 3D cameras.
  - Using OpenCV to evaluate state needed for RL is almost a difficult as solving the problem itself.  For example,identifying the blocks and stacking them can be made easier by adding fiducials or sensors to blocks (blah... feels like cheating.)
  - Need to park arm so that it was away from tray so that state of objects on tray could be accurately assessed.

Lesson from using the Jetson nano / Jetbot:
  - The Jetson "notebook" is a cool idea for tutorials, but in practice needs very fast wifi - better than my house has and very fast SSD - faster than I bought.  But putting the gamepad/logitech on the robot and using lower overhead video webstreaming worked fine.
  - I immediately outgrew the functionality provided by the Jebot. I was able to start with the jetbot code and tutorials to incremental build an armed jetbot (sir_jetbot1).
  
Reinforcement learning in general:
  - Let's look at some more recent major impressive results in deep learning:
    - OpenAI published a paper on GPT-3.  A decent summary of it is:
      - https://in.springboard.com/blog/openai-gpt-3/
    - OpenAI's blog describes the same model applied to images:
      - https://openai.com/blog/image-gpt/
    - Deep Mind's latest breakthrough:
      - https://deepmind.com/blog/article/muzero-mastering-go-chess-shogi-and-atari-without-rules
  - These all confirm the trend in deep learning (that has been known for a while) scales to incredible degrees. Basically, if you have a problem that you need to solve, recursively add more data and then add more levels to the NN (now attention-based), until you achieve good results (near-human or better). 
  - To get anything that seems to be impressive, you need huge datasets and tons of training to such a scale that only big companies with hundreds of thousands of dollars to spend can afford.  Individual developers / researchers can do minor incremental contributions to the area, but to get to the point of gaining common-sense intelligence is out of the reach of all but the biggest, richest companies with the biggest datasets (think Google and Facebook.)

