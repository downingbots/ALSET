

# Q: "What does your robot do?"  

====

# A: "Whatever you train the robot to do."


https://github.com/downingbots/SIR_jetbot

downingbots @ gmail.com

=========================

# Sharper Image Robot 

 - RC Armed Mobile Manipulator 
    - $20 on clearance

<p align="center">
  <img src="https://github.com/downingbots/SIR_jetbot/blob/master/ReadMeImages/sharper_image_robot.jpg" width="200 title="Sharper Image Robot">
  <img src="https://github.com/downingbots/SIR_jetbot/blob/master/ReadMeImages/sharper_image_robot2.jpg" width="200" alt="accessibility text">
</p>
                                                                                                                                             
=========================

# SIR_JETBOT

Converted to Autonomous robot:
  - Jetson Nano Developer Board with wifi
  - raspberry pi camera v2 with 2 ft cable mounted on a cookie wheel case near the end of the arm.
  - A hacked RC:
     - the circuit board from inside of the low-end RC control that came with the robot.
     - You can see/use the buttons from the RC ciruit board to control the robot.
     - the other side of the board was soldered to wire it up with the IO expansion board.
  - The IO expansion board was required :
     - the RC control uses Tri-state logic on 6 pins.
     - The expansion board uses I2C to communicate with the Jetson Development board
     - The expansion board is connected to the RC control via wires
  - Large BONAI 5.6A battery pack.
  - Logitech gamepad joystick communicates with the jetson
     - Will eventually have Smart Phone control via browser

<p align="center">
  <img src="https://github.com/downingbots/SIR_jetbot/blob/master/ReadMeImages/Sir_jetbot1.jpg" width="200" alt="accessibility text">
  <img src="https://github.com/downingbots/SIR_jetbot/blob/master/ReadMeImages/Sir_jetbot2.jpg" width="200" alt="accessibility text">
</p>

=========================

# SIR_JETBOT Operating Modes

1. RC telepresence
2. Telepresence data-capture 
3. Autonomous:
    - a single CNN or pretrained function
    - App defined as a behavior tree of CNNs
    - DDQN reinforcement learning. 
4. End-to-End Pre-Training of DQN Model based on:
    - incrementally trained functions
    - outputs from random runs of each function
    - incremental outputs from end-to-end run of App
    - Reinforcement Learning via DQN

=========================

# DQN Reinforcement Learning

 - Single end-to-end model with reinforcement learning
 - Discrete actions (vs. PPO's continuous actions)
    1. Robot moves
    2. Robot stops and take a picture
    3. Robot determines next move
- Train NN to estimate quality of each move (Q-Val)
  - in a board game, only one reward when winner is decided
  - penultimate move is 99% of final reward
  - 2nd to last move is .99 * .99 of final reward
- Variations: Replay history, Target Network
 
=========================

# SIR_JETBOT: HBRC TABLETOP CHALLENGE

- A single CNN is sufficient to stay on table

https://youtu.be/QVFHAMyEyaI

=========================

# SIR_JETBOT: HBRC TABLETOP CHALLENGE

8 parts for an HBRC phase-3 tabletop bot is:
 - get the arm to a known position
 - scans for object on the floor (or table)
 - goes to object, keeping object in center of vision
 - pick up the object
 - get the arm to a known position, while gripping object
 - scans for the box
 - goes to the box (keeping box in center of vision)
 - drops object in box

https://youtu.be/....

=========================

# Simple Functions Combined to Define App Behavior

 - Parking the robot arm in known positions
 - Different searches for objects (pretrained, newly trained, faces, ficucials)
 - Drive to objects
 - Pick up or push objects
 - Drop or place objects
 - Line, face or object following
 - Object avoidance
 - Stay on Table 
 - Movement verification
 - Scripts (e.g., dance)
 - Train your own function

=========================

# Convert Other RC Toys to Autonomous Vehicles

Same software and analogous RC/HW mods to support: 
 - Excavator
 - BullDozer
 - Dump Truck
 - Firetrucks with RC ladder/firehose
 - Forklifts
 - Utility Bucket Trucks
 - Tanks

Have the robots work together!

<p align="center">
  <img src="https://github.com/downingbots/SIR_jetbot/blob/master/ReadMeImages/excavator.jpg" width="200" title="RC Excavator">
  <img src="https://github.com/downingbots/SIR_jetbot/blob/master/ReadMeImages/bulldozer.jpg" width="200" alt="accessibility text">
</p>
