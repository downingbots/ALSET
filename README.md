# SIR_jetbot
This is SIRjetbot1. The SIR stands for Sharper Image Robot, which I purchased on
 clearance for less than $20. The 1 is because we bought 3 of them that worked.

The Sharper Image robot was hacked as followed:

    - Jetson Nano Developer Board with wifi
    - raspberry pi camera v2 with 2 ft cable
    - mounted on a cookie wheel case
    - A hacked RC:
        - the inside of the low-end RC control that came with the robot. 
        - You can see/use the buttons here control the robot. 
        - the other side of the board was soldered to wire it up with the IO exp
ansion board.
    - The IO expansion board was required because the RC control uses Tri-state 
logic on 6 pins. It uses I2C to communicate with the Jetson Development board
    - It is powered by a mongo BONAI 5.6A battery pack
    - Logitech gamepad

Started with the Jetson Notebook tutorial on Obstacle avoidance, but changed it 
significantly. The Notebook tutorials barely ran on my Jetson with the wifi conn
ectivity from my working area. The tutorials were replaced with:

    - The Logitech joystick controls the robot directly
    - The camera is streamed using much lower overhead video webserver for teleo
peration.
    - The images are saved directly to the Robot SD card.
    - a poor-man's pwm changes the commands to start/stop up to every tenth of a second or so. Currently, takes the picture when stopped.  It records the command sent by the joystick (lower or upper arm up/down, open/close gripper, etc.) along with the picture in the directory associated with the NN's command.
    
      This data is then gathered and used to train the robot. Note: a tenth of a second of start/stop proved too much for one of the boards and the robot will stop moving after a few minutes of continuous use. Eventually changed the pwm rate to two-moves-per-second.

The robot runs in 3 modes: RC telepresence, data-capture, and using the trained neural net.  The data capture and neural net can be for a singe alexnet NN, or a sequence of alexnet NN with function-specific knowledge. For example, the 8-parts are for an HBRC phase-3 tabletop bot is:
    - get the arm to a known position
    - scans for object on the floor (or table)
    - goes to object, keeping object in center of vision
    - pick up the object
    - get the arm to a known position, while gripping object
    - scans for the box
    - goes to the box (keeping box in center of vision)
    - drops object in box
    - backs up and park arm (back to phase 1)

The #1 rule is no falling off the table (or going out-of-bounds)

The Jetson "notebook" is a cool idea for tutorials, but in practice needs very fast wifi - better than my house has and very fast SSD - faster than I bought.  But putting the gamepad/logitech on the robot and using lower overhead video webstreaming worked fine.


