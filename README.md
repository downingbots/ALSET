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
ectivity from my working area. I replaced the tutorials with:

    - The Logitech joystick controls the robot directly
    - The camera is streamed using much lower overhead video webserver for teleo
peration.
    - The images are saved directly to the Robot SD card.
    - I implemented a poor-man's pwm which can change the commands to start/stop up to every tenth of a second or so. Currently, takes the picture when stopped.  It records the command sent by the joystick (lower or upper arm up/down, open/close gripper, etc.)
      This data is then gathered and used to train the robot. Note: a tenth of a second of start/stop proved too much for one of the boards and the robot will stop moving after a few minutes o fcontinuous use. Eventually moved the pwm rate to two-moves-per-second.

The robot runs in 3 modes: RC telepresence, data-capture, and using the trained neural net.

I just got it working a few hours ago. So, I only did phase 1 tabletop challenge and have it on video.

The Jetson "notebook" is a cool idea for tutorials, but in practice needs very f
ast wifi - better than my house has and very fast SSD - faster than I bought.  B
ut putting the gamepad/logitech on the robot and using lower overhead video webs
treaming worked fine.


