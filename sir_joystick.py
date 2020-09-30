# Released by rdb under the Unlicense (unlicense.org)
# Based on information from:
# https://www.kernel.org/doc/Documentation/input/joystick-api.txt

import os, struct, array
from fcntl import ioctl
import threading

class SIR_joystick:
    # We'll store the states here.
    axis_states = {}
    button_states = {}
    pressed_button = None
    pressed_button_value = None
    gripper_active = False
    wrist_active = False
    lower_arm_active = False
    upper_arm_active = False
    
    # These constants were borrowed from linux/input.h
    axis_names = {
        0x00 : 'x',
        0x01 : 'y',
        0x02 : 'z',
        0x03 : 'rx',
        0x04 : 'ry',
        0x05 : 'rz',
        0x06 : 'trottle',
        0x07 : 'rudder',
        0x08 : 'wheel',
        0x09 : 'gas',
        0x0a : 'brake',
        0x10 : 'hat0x',
        0x11 : 'hat0y',
        0x12 : 'hat1x',
        0x13 : 'hat1y',
        0x14 : 'hat2x',
        0x15 : 'hat2y',
        0x16 : 'hat3x',
        0x17 : 'hat3y',
        0x18 : 'pressure',
        0x19 : 'distance',
        0x1a : 'tilt_x',
        0x1b : 'tilt_y',
        0x1c : 'tool_width',
        0x20 : 'volume',
        0x28 : 'misc',
    }
    
    button_names = {
        0x120 : 'trigger',
        0x121 : 'thumb',
        0x122 : 'thumb2',
        0x123 : 'top',
        0x124 : 'top2',
        0x125 : 'pinkie',
        0x126 : 'base',
        0x127 : 'base2',
        0x128 : 'base3',
        0x129 : 'base4',
        0x12a : 'base5',
        0x12b : 'base6',
        0x12f : 'dead',
        0x130 : 'a',
        0x131 : 'b',
        0x132 : 'c',
        0x133 : 'x',
        0x134 : 'y',
        0x135 : 'z',
        0x136 : 'tl',
        0x137 : 'tr',
        0x138 : 'tl2',
        0x139 : 'tr2',
        0x13a : 'select',
        0x13b : 'start',
        0x13c : 'mode',
        0x13d : 'thumbl',
        0x13e : 'thumbr',
    
        0x220 : 'dpad_up',
        0x221 : 'dpad_down',
        0x222 : 'dpad_left',
        0x223 : 'dpad_right',
    
        # XBox 360 controller uses these codes.
        0x2c0 : 'dpad_left',
        0x2c1 : 'dpad_right',
        0x2c2 : 'dpad_up',
        0x2c3 : 'dpad_down',
    }
    
    axis_map = []
    button_map = []
    connected = False
    jsdev = None
    
    def __init__(self, robot_driver):
        self._robot_driver = robot_driver
        self.left_speed = None
        self.right_speed = None
        # Iterate over the joystick devices.
        print('Available devices:')
    
        for fn in os.listdir('/dev/input'):
            if fn.startswith('js'):
                print('  /dev/input/%s' % (fn))
    
        # Open the joystick device.
        fn = '/dev/input/js0'
        print('Opening %s...' % fn)
        self.jsdev = open(fn, 'rb')
        
        # Get the device name.
        #buf = bytearray(63)
        buf = array.array('B', [0] * 64)
        ioctl(self.jsdev, 0x80006a13 + (0x10000 * len(buf)), buf) # JSIOCGNAME(len)
        js_name = buf.tostring().rstrip(b'\x00').decode('utf-8')
        print('Device name: %s' % js_name)
        
        # Get number of axes and buttons.
        buf = array.array('B', [0])
        ioctl(self.jsdev, 0x80016a11, buf) # JSIOCGAXES
        num_axes = buf[0]
        
        buf = array.array('B', [0])
        ioctl(self.jsdev, 0x80016a12, buf) # JSIOCGBUTTONS
        num_buttons = buf[0]
        
        # Get the axis map.
        buf = array.array('B', [0] * 0x40)
        ioctl(self.jsdev, 0x80406a32, buf) # JSIOCGAXMAP
        
        for axis in buf[:num_axes]:
            axis_name = self.axis_names.get(axis, 'unknown(0x%02x)' % axis)
            self.axis_map.append(axis_name)
            self.axis_states[axis_name] = 0.0
        
        # Get the button map.
        buf = array.array('H', [0] * 200)
        ioctl(self.jsdev, 0x80406a34, buf) # JSIOCGBTNMAP
        
        for btn in buf[:num_buttons]:
            btn_name = self.button_names.get(btn, 'unknown(0x%03x)' % btn)
            self.button_map.append(btn_name)
            self.button_states[btn_name] = 0
        
        print('%d axes found: %s' % (num_axes, ', '.join(self.axis_map)))
        print('%d buttons found: %s' % (num_buttons, ', '.join(self.button_map)))
        # 8 axes found: x, y, z, rx, ry, rz, hat0x, hat0y
        # 11 buttons found: a, b, x, y, tl, tr, select, start, mode, thumbl, thumbr
        if num_axes == 8 and num_buttons == 11:
            self.connected = True
        else:
            self.connected = False
        
    def teleop(self):
        evbuf = self.jsdev.read(8)
        if evbuf:
            time, value, type, number = struct.unpack('IhBB', evbuf)
    
            if type & 0x80:
                 # print("(initial)", end="")
                 # print("(initial)")
                 pass
    
            # print("type: %d" % type)
            button = None
            if type & 0x01:
                button = self.button_map[number]
                if button:
                    self.button_states[button] = value
                    if value:
                        # print("%s pressed" % (button))
                        self.pressed_button = button
                        self.pressed_button_value = value
                    else:
                        # print("%s released" % (button))
                        self.pressed_button = None
                        self.pressed_button_value = None
    
            axis = None
            if type & 0x02:
                axis = self.axis_map[number]
                if axis:
                    fvalue = value / 32767.0
                    self.axis_states[axis] = fvalue
                    # print("%s: %.3f" % (axis, fvalue))

            command = []
            arg = []
            if axis == "y":
                command.append("LEFT_TRACK")
                arg.append(fvalue)
                self.left_speed  = fvalue
                # right_speed = None
                self._robot_driver.set_motors(self.left_speed, self.right_speed)
            if axis == "ry":
                command.append("RIGHT_TRACK")
                arg.append(-fvalue)
                self.right_speed  = -fvalue
                # left_speed  = None
                self._robot_driver.set_motors(self.left_speed, self.right_speed)
            if self.pressed_button == "tl":
                mode = self._robot_driver.get_gather_data_mode()
                # toggle current mode
                mode = (not mode)
                self._robot_driver.set_gather_data_mode(mode)
                print("gather data", mode) 
            if self.pressed_button == "tr":
                nn_mode = self._robot_driver.get_NN_mode()
                # toggle current mode
                nn_mode = (not nn_mode)
                self._robot_driver.set_NN_mode(nn_mode)
                print("NN", nn_mode)

            if self.pressed_button == "a":
                command.append("LOWER_ARM")
                arg.append("DOWN")
                self._robot_driver.lower_arm("DOWN")
                self.lower_arm_active = True
            elif self.pressed_button == "y":
                command.append("LOWER_ARM")
                arg.append("UP")
                self._robot_driver.lower_arm("UP")
                self.lower_arm_active = True
            elif self.lower_arm_active:
                command.append("LOWER_ARM")
                arg.append("STOP")
                self._robot_driver.lower_arm("STOP")
                self.lower_arm_active = False
            if axis == "hat0y" and fvalue > .9:
                command.append("UPPER_ARM")
                arg.append("DOWN")
                self._robot_driver.upper_arm("DOWN")
                self.upper_arm_active = True
            elif axis == "hat0y" and fvalue < -.9:
                command.append("UPPER_ARM")
                arg.append("UP")
                self._robot_driver.upper_arm("UP")
                self.upper_arm_active = True
            elif self.upper_arm_active:
                command.append("UPPER_ARM")
                arg.append("STOP")
                self._robot_driver.upper_arm("STOP")
                self.upper_arm_active = False
            if axis == "hat0x" and fvalue == 1:
                command.append("WRIST")
                arg.append("ROTATE_RIGHT")
                self._robot_driver.wrist("ROTATE_RIGHT")
                self.wrist_active = True
            elif axis == "hat0x" and fvalue == -1:
                command.append("WRIST")
                arg.append("ROTATE_LEFT")
                self._robot_driver.wrist("ROTATE_LEFT")
                self.wrist_active = True
            elif self.wrist_active:
                command.append("WRIST")
                arg.append("STOP")
                self._robot_driver.wrist("STOP")
                self.wrist_active = False
            if self.pressed_button == "x":
                command.append("GRIPPER")
                arg.append("CLOSE")
                self._robot_driver.gripper("CLOSE")
                self.gripper_active = True
            elif self.pressed_button == "b":
                command.append("GRIPPER")
                arg.append("OPEN")
                self._robot_driver.gripper("OPEN")
                self.gripper_active = True
            elif self.gripper_active:
                command.append("GRIPPER")
                arg.append("STOP")
                self._robot_driver.gripper("STOP")
                self.gripper_active = False
            for i, c in enumerate(command):
                print(c, arg[i])
            return command, arg
        else:
            return [], []

# j =  SIR_joystick()
# while True:
#     j.read()

class sir_joystick_daemon():
    def __init__(self, robot_driver):
        self._robot_driver = robot_driver
        self.running = False
        self.start()

    def _run(self):
        joy = SIR_joystick(self._robot_driver)
        if not joy.connected:
            print("joystick not connected")
            return
        print("joystick connected")
        while True:
            if not self.running:
                break
            command, arg = joy.teleop()

    def start(self):
        if self.running:
            return
        self.running = True
        print("before thread")
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
