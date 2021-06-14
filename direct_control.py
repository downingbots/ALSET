import time
import traceback
import RPi.GPIO as GPIO
import time

class direct_control:


  def gpio_init(self):
    # set pin as an output pin with optional initial state of HIGH
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(output_pin, GPIO.OUT, initial=GPIO.HIGH)

  def switch_pin_on(self, pin):
    pin_offset = self.all_pin_numbers.index(pin)
    self.all_pin_vals[pin_offset] = True

  def switch_pin_off(self, pin):
    pin_offset = self.all_pin_numbers.index(pin)
    self.all_pin_vals[pin_offset] = False

  def switch_on(self, part_num):
      part_lst = self.PART_SPECS[part_num]
      [part_name, direction, pin] = part_lst[0]
      if self.PARTS[part_num] == part_name and direction in ["FORWARD", "UP", "TOGGLE"]:
         self.switch_pin_on(pin)
      else:
         print("ERR: switch_on", part_num, part_lst)
      if direction != "TOGGLE":
         [part_name, direction, pin] = part_lst[1]
         if self.PARTS[part_num] == part_name and direction in ["BACK", "DOWN"]:
            self.switch_pin_off(pin)
         else:
            print("ERR: switch_on", part_num, part_lst)

  def switch_up(self, part_num):
      self.switch_on(part_num)

  def switch_off(self, part_num):
      part_lst = self.PART_SPECS[part_num]
      [part_name, direction, pin] = part_lst[0]
      if self.PARTS[part_num] == part_name and direction in ["FORWARD", "UP", "TOGGLE"]:
         self.switch_pin_off(pin)
      else:
         print("ERR: switch_off", part_num, part_lst)
      if direction != "TOGGLE":
         [part_name, direction, pin] = part_lst[1]
         if self.PARTS[part_num] == part_name and direction in ["BACK", "DOWN"]:
            self.switch_pin_off(pin)
         else:
            print("ERR: switch_off", part_num, part_lst)

  def switch_down(self, part_num):
      part_lst = self.PART_SPECS[part_num]
      [part_name, direction, pin] = part_lst[0]
      if self.PARTS[part_num] == part_name and direction in ["FORWARD", "UP", "TOGGLE"]:
         self.switch_pin_off(pin)
      else:
         print("ERR: switch_down", part_num, part_lst)
      if direction != "TOGGLE":
         [part_name, direction, pin] = part_lst[1]
         if self.PARTS[part_num] == part_name and direction in ["BACK", "DOWN"]:
            self.switch_pin_on(pin)
         else:
            print("ERR: switch_down", part_num, part_lst)


  def gpio_cleanup(self):
        GPIO.cleanup()

  switch_exec_next_pulse = False
  timeslice = 0.1
  
  def switch_exec(self, exec_next_pulse=True):
      if exec_next_pulse:
          # execute during next pulse processing
          self.switch_exec_next_pulse = True
      else:
          self.switch_exec_next_pulse = False
          for i, TF in enumerate(self.all_pin_vals):
              if TF:
                GPIO.output(self.all_pin_numbers[i], GPIO.HIGH)
              else:
                GPIO.output(self.all_pin_numbers[i], GPIO.LOW)
          self.curr_timeout[self.LEFT_TRACK] = self.compute_pwm_timeout(self.LEFT_TRACK)
          self.curr_timeout[self.RIGHT_TRACK] = self.compute_pwm_timeout(self.RIGHT_TRACK)
  
  def convert_speed(self, speed):
    pulse_speed = max(speed, -1)
    pulse_speed = min(pulse_speed, 1)
    pulse_speed = 10 * round(pulse_speed, 1)
    return pulse_speed

  def get_pulse_speed(self, part):
    return self.curr_speed[pin]

  def get_speed(self, part):
    return (1.0*self.curr_speed[pin]) / 10

  def set_speed(self, part, speed):
    pulse_speed = self.convert_speed(speed)
    if pulse_speed > 0 and self.curr_speed[part] <= 0:
      if part == self.RIGHT_TRACK:
        self.switch_on(part)
      else:
        self.switch_on(part)
      self.switch_exec()
    elif pulse_speed == 0 and self.curr_speed[part] != 0:
      self.switch_off(part)
      self.switch_exec()
    elif pulse_speed < 0 and self.curr_speed[part] >= 0:
      if part == self.RIGHT_TRACK:
        self.switch_down(part)
      else:
        self.switch_down(part)
      self.switch_exec()
    self.curr_speed[part] = pulse_speed
    print("new speed = %d" % pulse_speed)


  def stop_all(self, execute_immediate = True):
    #Configure the register to default value
    for pin in self.all_pin_numbers:
      self.switch_pin_off(pin)

  def drive_forward(self, speed):
          print("F")
          self.switch_down(self.LEFT_TRACK)
          self.curr_speed[self.LEFT_TRACK] = self.convert_speed(speed)
          self.switch_on(self.RIGHT_TRACK)
          self.curr_speed[self.RIGHT_TRACK] = self.convert_speed(speed)
          self.switch_exec()
  
  def drive_reverse(self, speed):
          print("B")
          self.switch_on(self.LEFT_TRACK)
          self.curr_speed[self.LEFT_TRACK] = -self.convert_speed(speed)
          self.switch_down(self.RIGHT_TRACK)
          self.curr_speed[self.RIGHT_TRACK] = -self.convert_speed(speed)
          self.switch_exec()
  
  def drive_rotate_left(self, speed):
          # print("L")
          self.switch_on(self.LEFT_TRACK)
          self.curr_speed[self.LEFT_TRACK] = -self.convert_speed(speed)
          self.switch_on(self.RIGHT_TRACK)
          self.curr_speed[self.RIGHT_TRACK] = self.convert_speed(speed)
          self.switch_exec()
  
  def drive_rotate_right(self, speed):
          print("R")
          self.switch_down(self.LEFT_TRACK)
          self.curr_speed[self.LEFT_TRACK] = self.convert_speed(speed)
          self.switch_down(self.RIGHT_TRACK)
          self.curr_speed[self.RIGHT_TRACK] = -self.convert_speed(speed)
          self.switch_exec()
  
  def drive_stop(self):
          print("S")
          self.switch_off(self.LEFT_TRACK)
          self.curr_speed[self.LEFT_TRACK] = 0
          self.switch_off(self.RIGHT_TRACK)
          self.curr_speed[self.RIGHT_TRACK] = 0
          self.switch_exec()
  
  def compute_pwm_timeout(self, side):
    if self.curr_speed[side] == 0 or self.curr_speed[side] == 10:
      return -1
    speed = abs(self.curr_speed[side])
    if self.curr_mode[side] == "STOP":
      if speed < 5:
        timeout_period = 5 - speed
      elif speed >= 5:
        timeout_period = 1
    elif self.curr_mode[side] == "GO":
      if speed <= 5:
        timeout_period = 1
      elif speed > 5:
        timeout_period = speed - 5
    timeout = (self.curr_pwm_pulse + timeout_period) % 10
    return timeout

  def set_motors(self, left_speed, right_speed):
          if left_speed != None:
            self.set_speed(self.LEFT_TRACK, left_speed)
          if right_speed != None:
            self.set_speed(self.RIGHT_TRACK, right_speed)

  def pwm_stop(self, take_picture = True, process_image = False):
           # stop, no snapshot
           # save_pin_val = self.curr_pin_val
           # save_pin_io_val = self.curr_pin_io_val
           # stop all motors
           self.curr_pin_val = GPIO.LOW
           # self.curr_pin_io_val = self.ALL_FUNC
           self.switch_exec(exec_next_pulse=False)
           # restore state for next pulse
           if self._driver.gather_data.action_name != None or process_image:
             if take_picture:
               # take picture and collect data
               action = self._driver.gather_data.save_snapshot(process_image)
             # self.curr_pin_val = save_pin_val
             # self.curr_pin_io_val = save_pin_io_val
             if process_image:
               print("command: ", action)
               self.execute_command(action)
             self.switch_exec(exec_next_pulse=True)
           elif self._driver.gather_data.action_name == None:
             print("None action_name", self._driver.gather_data.nn_name)

  def handle_pulse(self, pulse_num, process_image):
        # all_on = self.ALL_FUNC
        # functions_not_stopped = all_on ^ self.curr_pin_io_val
        # print("active pins, pulse_num:", self._driver.gather_data.action_name,
        #       bin(functions_not_stopped)[2:].zfill(8), pulse_num)
        if self._driver.gather_data.is_on():
            # next pulse: essentially everything is half speed during data collection
            divisor = 10
        else:
            divisor = 5
        if (pulse_num+1) % divisor == 2:
            print("take snapshot, process image, store image:", process_image)
            self.pwm_stop(take_picture = True, process_image = process_image)
            # action(state) -> next state
            return
        elif (pulse_num+1) % divisor == 0:
            self.switch_exec(exec_next_pulse=False)
        return

  def handle_pwm(self, pulse_num):
    timeout = False
    self.curr_pwm_pulse = pulse_num
    if self.switch_exec_next_pulse:
      timeout = True

    #########################################
    # if gather data mode and TT_DQN reinforcement learning
    # if NN mode, run NN 
    #########################################
    # Some type of automatic execution: DQN, NN or Automatic Function
    #  - Capture, process and store image to determine next move
    #  - Joystick provides reward / penalty
    if ((self._driver.NN_apps.app_type == "DQN" and self._driver.gather_data.is_on()) or
       # Reinforcement Learning or
       (self._driver.get_NN_mode() == "NN") or         # if running neural net
       # if gather data mode and automatic Function 
       (self._driver.gather_data.is_on() and self._driver.NN_apps.nn_automatic_mode())):
        print("auto or nn mode")
        if self._driver.gather_data.action_name in ["REWARD1", "PENALTY1", "REWARD2", "PENALTY2"]:
           print("func:",self._driver.gather_data.action_name)
           self.stop_all(execute_immediate=True)
           self._driver.gather_data.save_snapshot(process_image = True)
           return
        self.handle_pulse(pulse_num, process_image=True)
        return

    #########################################
    # if gather data mode, do simplified non-pwm processing
    #########################################
    elif self._driver.gather_data.is_on() and self._driver.gather_data.action_name != None:
        print("teleop gather mode")
        # gather data, not automatic mode, action provided by joystick
        if self._driver.gather_data.action_name in ["REWARD1", "REWARD2"]:
           print("nn_upon_reward")
           self.stop_all(execute_immediate=True)
           # self._driver.gather_data.set_action("REWARD")
           self._driver.gather_data.save_snapshot()
           self._driver.NN_apps.nn_upon_reward(self._driver.gather_data.action_name)
           return
        elif self._driver.gather_data.action_name in ["PENALTY1", "PENALTY2"]:
           print("nn_upon_penalty")
           self.stop_all(execute_immediate=True)
           # self._driver.gather_data.set_action("PENALTY")
           self._driver.gather_data.save_snapshot()
           self._driver.NN_apps.nn_upon_penalty(self._driver.gather_data.action_name)
           return
        self.handle_pulse(pulse_num, process_image=False)
        return

    #########################################
    # gather data, not automatic mode, no function name from joystick
    #########################################
    elif self._driver.gather_data.is_on():
        # print("idle gather mode")
        self.handle_pulse(pulse_num, process_image=False)
        return

    #########################################
    # if not gather data mode or NN mode, 
    # support pseudo-pwm processing for LEFT/RIGHT TRACK
    #########################################
    # print("left track : ", self.curr_timeout[self.LEFT_TRACK], self.curr_mode[self.LEFT_TRACK], self.curr_speed[self.LEFT_TRACK])
    # print("right track: ", self.curr_timeout[self.RIGHT_TRACK], self.curr_mode[self.RIGHT_TRACK], self.curr_speed[self.RIGHT_TRACK])
    # print("teleop-only mode")
    if self.curr_timeout[self.LEFT_TRACK] == -1:
      if self.curr_mode[self.LEFT_TRACK] == "STOP":
        timeout = True
        if self.curr_speed[self.LEFT_TRACK] > 0:
          self.switch_up(self.LEFT_TRACK)
        elif self.curr_speed[self.LEFT_TRACK] == 0:
          self.switch_off(self.LEFT_TRACK)
        elif self.curr_speed[self.LEFT_TRACK] < 0:
          self.switch_down(self.LEFT_TRACK)
        self.curr_mode[self.LEFT_TRACK] = "GO"
    elif pulse_num == self.curr_timeout[self.LEFT_TRACK]:
      timeout = True
      if self.curr_mode[self.LEFT_TRACK] == "STOP":
        if self.curr_speed[self.LEFT_TRACK] > 0:
            self.switch_up(self.LEFT_TRACK)
        elif self.curr_speed[self.LEFT_TRACK] == 0:
            self.switch_off(self.LEFT_TRACK)
        elif self.curr_speed[self.LEFT_TRACK] < 0:
            self.switch_down(self.LEFT_TRACK)
        self.curr_mode[self.LEFT_TRACK] = "GO"
      elif self.curr_mode[self.LEFT_TRACK] == "GO":
        self.switch_off(self.LEFT_TRACK)
        self.curr_mode[self.LEFT_TRACK] = "STOP"
      self.curr_timeout[self.LEFT_TRACK] = self.compute_pwm_timeout(self.LEFT_TRACK)
    if self.curr_timeout[self.RIGHT_TRACK] == -1:
      if self.curr_mode[self.RIGHT_TRACK] == "STOP":
        timeout = True
        if self.curr_speed[self.RIGHT_TRACK] > 0:
          self.switch_up(self.RIGHT_TRACK)
        elif self.curr_speed[self.RIGHT_TRACK] == 0:
          self.switch_off(self.RIGHT_TRACK)
        elif self.curr_speed[self.RIGHT_TRACK] < 0:
          self.switch_down(self.RIGHT_TRACK)
        self.curr_mode[self.RIGHT_TRACK] = "GO"
    elif pulse_num == self.curr_timeout[self.RIGHT_TRACK]:
      timeout = True
      if self.curr_mode[self.RIGHT_TRACK] == "STOP":
        if self.curr_speed[self.RIGHT_TRACK] > 0:
            self.switch_up(self.RIGHT_TRACK)
        elif self.curr_speed[self.RIGHT_TRACK] == 0:
            self.switch_off(self.RIGHT_TRACK)
        elif self.curr_speed[self.RIGHT_TRACK] < 0:
            self.switch_down(self.RIGHT_TRACK)
        self.curr_mode[self.RIGHT_TRACK] = "GO"
      elif self.curr_mode[self.RIGHT_TRACK] == "GO":
        self.switch_off(self.RIGHT_TRACK)
        self.curr_mode[self.RIGHT_TRACK] = "STOP"
      self.curr_timeout[self.RIGHT_TRACK] = self.compute_pwm_timeout(self.RIGHT_TRACK)
    if timeout:
      # execute immediate
      self.switch_exec(exec_next_pulse=False)

  def upper_arm(self, dir):
    if dir == "UP":
        self.switch_up(self.UPPER_ARM)
        self.switch_exec()
    elif dir == "DOWN":
        self.switch_down(self.UPPER_ARM)
        self.switch_exec()
    elif dir == "STOP":
        self.switch_off(self.UPPER_ARM)
        self.switch_exec()
    else:
        print("upper_arm: invalid input (%s)" % dir)

  def lower_arm(self, dir):
    if dir == "UP":
        self.switch_up(self.LOWER_ARM)
        self.switch_exec()
    elif dir == "DOWN":
        self.switch_down(self.LOWER_ARM)
        self.switch_exec()
    elif dir == "STOP":
        self.switch_off(self.LOWER_ARM)
        self.switch_exec()
    else:
        print("lower_arm: invalid input (%s)" % dir)

  def wrist(self, dir):
    if dir == "ROTATE_LEFT":
        self.switch_up(self.WRIST)
        self.switch_exec()
    elif dir == "ROTATE_RIGHT":
        self.switch_down(self.WRIST)
        self.switch_exec()
    elif dir == "STOP":
        self.switch_off(self.WRIST)
        self.switch_exec()
    else:
        print("wrist: invalid input (%s)" % dir)

  def shovel(self, dir):
    if dir == "UP":
        self.switch_up(self.SHOVEL)
        self.switch_exec()
    elif dir == "OPEN":
        self.switch_down(self.SHOVEL)
        self.switch_exec()
    elif dir == "STOP":
        self.switch_off(self.SHOVEL)
        self.switch_exec()
    else:
        print("gripper: invalid input (%s)" % dir)

  def execute_command(self, command):
      self.stop_all(execute_immediate=False)
      speed = 1
      if command == "UPPER_ARM_UP":
          self.upper_arm("UP")
      elif command == "UPPER_ARM_DOWN":
          self.upper_arm("DOWN")
      elif command == "LOWER_ARM_UP":
          self.lower_arm("UP")
      elif command == "LOWER_ARM_DOWN":
          self.lower_arm("DOWN")
      elif command == "SHOVEL_UP":
          self.shovel("UP")
      elif command == "SHOVEL_DOWN":
          self.shovel("DOWN")
      elif command == "SWIVEL_LEFT":
          self.chassis("LEFT")
      elif command == "SWIVEL_RIGHT":
          self.chassis("RIGHT")
      elif command == "FORWARD":
          self.drive_forward(speed)
      elif command == "REVERSE":
          self.drive_reverse(speed)
      elif command == "LEFT":
          self.drive_rotate_left(speed)
      elif command == "RIGHT":
          self.drive_rotate_right(speed)
      elif command == "REWARD1":
          self.stop_all()
      elif command == "PENALTY1":
          self.stop_all()
      elif command == "REWARD2":
          self.stop_all()
      elif command == "PENALTY2":
          self.stop_all()
      else:
          print("execute_command: command unknown(%s)" % command)

  def test_arm(self):
          for pin in (self.LOWER_ARM, self.UPPER_ARM, self.SHOVEL, self.CHASSIS):
              try:
                  self.switch_up(pin)
                  self.switch_exec()
                  time.sleep(3)
                  self.switch_down(pin)
                  self.switch_exec()
                  time.sleep(3)
                  self.switch_off(pin)
                  self.switch_exec()
              except KeyboardInterrupt:
                  self.stop_all()
  
  def test_drive(self):
          try:
            self.drive_forward()
            time.sleep(3)

            for i in range(1, 100):
              self.drive_forward()
              time.sleep(.08)
              self.drive_stop()
              time.sleep(.08)

            for i in range(1, 100):
              self.drive_forward()
              time.sleep(.1)
              self.drive_stop()
              time.sleep(.2)

            self.drive_stop()
            time.sleep(1)
            self.drive_reverse()
            time.sleep(3)
            self.drive_stop()
            time.sleep(1)
            self.drive_rotate_left()
            time.sleep(3)
            self.drive_stop()
            time.sleep(1)
            self.drive_rotate_right()
            time.sleep(3)
            self.drive_stop()
            time.sleep(1)
          except KeyboardInterrupt:
            self.drive_stop()

  def test_pins(self):
    for pin in self.all_pin_numbers:
       print("pin number: ", pin)
       self.switch_on(pin)
       time.sleep(1)
       self.switch_off(pin)

  def __init__(self, robot_driver=None):
    self._driver = robot_driver

    # From Frank's documentation:
    # 7, 11, 12, 13, 15, 16, 18, 19, 21, 22(not used)
    # 23, 24, 26, 29, 32, 36
    # 39 is ground
    # 15 pins used
    #
    # A1, [1 4], left bck, 24
    # B1, [5,9], Swivel chassis left, 19
    # C1, [5,6], Left fwd, 21
    # D1, [3,1], shovel down, 26
    #
    # A2, [6,7], Demo Key 32
    # B2, [9,4], Program mode 23
    # C2, [6,4], Sound 29
    # D2, [5,9], shovel up, 36
    #
    # A3, [10,3], lower arm up 12
    # B3, [9,7], upper arm up 7
    # C3, [4,10], upper arm down 11
    # D3, [1, 2], lower arm down 16
    #
    # A4, [9,2], right bck, 18
    # B4, [11,2] Swivel chassis right, 13
    # C4, [8,2], right fwd 15
    # D4, NA
    #
    # self.all_pin_spec = [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]
    # self.all_pin_spec = [ 7, 11, 12, 13, 15, 16, 18, 19, 21, 23, 24, 26, 29, 32, 36]
    # self.all_pin_spec = [b3, c3, a3, b4, c4, d3, a4, b1, c1, b2, a1, d1, c2, a2, d2]
    #
    #      [["UPPER_ARM", "UP", 7],["UPPER_ARM", "DOWN", 11], ["LOWER_ARM",  "UP", 12],
    #       ["CHASIS", "LEFT", 13], ["RIGHT_TRACK", "FORWARD", 15], ["LOWER_ARM", "DOWN", 16],
    #       ["RIGHT_TRACK", "BACK", 18], ["CHASIS", "RIGHT", 19], ["LEFT_TRACK", "FORWARD", 21],
    #       ["PROGRAM_MODE", "TOGGLE", 23], ["LEFT_TRACK", "BACK", 24], ["SHOVEL", "DOWN", 26],
    #       ["SOUND", "TOGGLE", 29], ["DEMO", "TOGGLE", 32], ["SHOVEL", "UP", 36]]

    # keeping consistent with mcp code 
    self.LEFT_TRACK = 0
    self.RIGHT_TRACK = 1
    self.CHASIS = 2
    self.SHOVEL = 3
    self.UPPER_ARM = 4
    self.LOWER_ARM = 5
    self.OTHER = 6
    # PARTS can only move one direction, speed, etc despite using 2 pins
    self.PARTS = ["LEFT_TRACK", "RIGHT_TRACK", "CHASIS", "SHOVEL", "UPPER_ARM", "LOWER_ARM", "PROGRAM_MODE", "SOUND", "DEMO"]
    self.PART_DIRECTIONS = [["FORWARD", "BACK"], ["UP", "DOWN"], ["LEFT","RIGHT"], ["ON","OFF"]]
    self.PART_SPECS   = [[["LEFT_TRACK", "FORWARD", 21], ["LEFT_TRACK", "BACK", 24]],
                         [["RIGHT_TRACK", "FORWARD", 15], ["RIGHT_TRACK", "BACK", 18]],
                         [["CHASIS", "LEFT", 13], ["CHASIS", "RIGHT", 19]],
                         [["SHOVEL", "UP", 36], ["SHOVEL", "DOWN", 26]],
                         [["UPPER_ARM", "UP", 7], ["UPPER_ARM", "DOWN", 11]],
                         [["LOWER_ARM",  "UP", 12], ["LOWER_ARM", "DOWN", 16]],
                         [["PROGRAM_MODE", "TOGGLE", 23]],
                         [["SOUND", "TOGGLE", 29]], 
                         [["DEMO", "TOGGLE", 32]] 
                         ]
    self.all_pin_numbers = [ 7, 11, 12, 13, 15, 16, 18, 19, 21, 22, 23, 24, 26, 29, 32, 36]
    self.all_pin_vals =    [ False for i in self.all_pin_numbers]

    self.curr_pwm_pulse = 0
    self.curr_speed = [0 for i in self.PARTS]
    self.curr_mode = [ "STOP" for i in self.PARTS]
    self.curr_timeout = [ 0 for i in self.PARTS]

    GPIO.setmode(GPIO.BOARD)
    for channel in self.all_pin_numbers:
      GPIO.setup(channel, GPIO.OUT)

    try:
      # print("stop and sleep")
      self.stop_all()
      time.sleep(5)
      # self.test_arm()
      # self.test_drive()
    except Exception as e:
      track = traceback.format_exc()
      print(track)
    self.stop_all()
  
  
# alset1 = AlSET_control()
