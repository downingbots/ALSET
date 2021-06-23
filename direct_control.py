import time
import traceback
import RPi.GPIO as GPIO
import copy


class direct_control:


  def gpio_init(self):
    # set pin as an output pin with optional initial state of HIGH
    GPIO.setwarnings(False)
    # GPIO.setmode(GPIO.BCM)
    GPIO.setmode(GPIO.BOARD)
    for output_pin in self.all_pin_numbers:
      GPIO.setup(output_pin, GPIO.OUT, initial=GPIO.LOW)

  def print_pin_name(self, strng, pin):
    for part_lst in self.PART_SPECS:
      [part_name, part_direction, part_pin] = part_lst[0]
      if part_pin == pin:
          pin_offset = self.all_pin_numbers.index(pin)
          print(strng, pin, pin_offset, part_lst[0], self.all_pin_numbers[pin_offset], self.all_pin_vals[pin_offset], self.all_pin_changed[pin_offset])
          return
      if part_direction != "TOGGLE":
        [part_name, part_direction, part_pin] = part_lst[1]
        if part_pin == pin:
          pin_offset = self.all_pin_numbers.index(pin)
          print(strng, pin, pin_offset, part_lst[1], self.all_pin_numbers[pin_offset], self.all_pin_vals[pin_offset], self.all_pin_changed[pin_offset])
          return

  def switch_pin_on(self, pin):
    pin_offset = self.all_pin_numbers.index(pin)
    self.all_pin_vals[pin_offset] = True
    self.all_pin_changed[pin_offset] = True
    # self.print_pin_name("switch_pin_on:", pin_offset)

  def switch_pin_off(self, pin):
    pin_offset = self.all_pin_numbers.index(pin)
    self.all_pin_vals[pin_offset] = False
    self.all_pin_changed[pin_offset] = True
    # self.print_pin_name("switch_pin_off:", pin)

  def switch_on(self, part_num):
      # print("switch_on call")
      self.switch_up(part_num)

  def switch_up(self, part_num):
      part_lst = self.PART_SPECS[part_num]
      [part_name, direction, pin] = part_lst[0]
      if self.PARTS[part_num] == part_name and direction in ["FORWARD", "UP", "TOGGLE", "LEFT"]:
         self.switch_pin_on(pin)
      else:
         print("ERR: switch_off", part_num, part_lst)
      if direction != "TOGGLE":
         [part_name, direction, pin] = part_lst[1]
         if self.PARTS[part_num] == part_name and direction in ["BACK", "DOWN", "RIGHT"]:
            self.switch_pin_off(pin)
         else:
            print("ERR: switch_off", part_num, part_lst)


  def switch_off(self, part_num):
      part_lst = self.PART_SPECS[part_num]
      [part_name, direction, pin] = part_lst[0]
      if self.PARTS[part_num] == part_name and direction in ["FORWARD", "UP", "TOGGLE", "LEFT"]:
         self.switch_pin_off(pin)
      else:
         print("ERR: switch_off", part_num, part_lst)
      if direction != "TOGGLE":
         [part_name, direction, pin] = part_lst[1]
         if self.PARTS[part_num] == part_name and direction in ["BACK", "DOWN", "RIGHT"]:
            self.switch_pin_off(pin)
         else:
            print("ERR: switch_off", part_num, part_lst)

  def switch_down(self, part_num):
      part_lst = self.PART_SPECS[part_num]
      [part_name, direction, pin] = part_lst[0]
      # print("part_lst[0]", part_lst[0])
      if self.PARTS[part_num] == part_name and direction in ["FORWARD", "UP", "TOGGLE", "LEFT"]:
         self.switch_pin_off(pin)
      else:
         print("ERR: switch_down", part_num, part_lst)
      if direction != "TOGGLE":
         [part_name, direction, pin] = part_lst[1]
         # print("part_lst[1]", part_lst[1])
         if self.PARTS[part_num] == part_name and direction in ["BACK", "DOWN", "RIGHT"]:
            self.switch_pin_on(pin)
         else:
            print("ERR: switch_down", part_num, part_lst)

  def gpio_cleanup(self):
        GPIO.cleanup()

  switch_exec_next_pulse = False
  timeslice = 0.1
  
  def switch_exec(self, exec_next_pulse=True):
      on_cnt = 0
      off_cnt = 0
      if exec_next_pulse:
          # execute during next pulse processing
          self.switch_exec_next_pulse = True
      else:
          self.switch_exec_next_pulse = False
          # print("all_pin_num :", self.all_pin_numbers)
          # print("all_pin_vals:", self.all_pin_vals)
          # print("all_pin_chng:", self.all_pin_changed)
          for i, TF in enumerate(self.all_pin_vals):
            if self.all_pin_changed[i]:
              if self.all_pin_numbers[i] in self.all_pin_disallowed:
                # self.print_pin_name("exec disallowed:", self.all_pin_numbers[i])
                self.all_pin_changed[i] = False
                continue
              if TF:
                # self.print_pin_name("exec on:", self.all_pin_numbers[i])
                GPIO.output(self.all_pin_numbers[i], GPIO.HIGH)
                on_cnt += 1
              else:
                # self.print_pin_name("exec off:", self.all_pin_numbers[i])
                GPIO.output(self.all_pin_numbers[i], GPIO.LOW)
                off_cnt += 1
              self.all_pin_changed[i] = False
          self.curr_timeout[self.LEFT_TRACK] = self.compute_pwm_timeout(self.LEFT_TRACK)
          self.curr_timeout[self.RIGHT_TRACK] = self.compute_pwm_timeout(self.RIGHT_TRACK)
          if not(on_cnt == 0 and off_cnt == 14) and (on_cnt > 4 or off_cnt > 4):
            print("WARNING: switch on/off count", on_cnt, off_cnt)
  
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
      self.switch_exec(False)
    elif pulse_speed == 0 and self.curr_speed[part] != 0:
      self.switch_off(part)
      self.switch_exec(False)
    elif pulse_speed < 0 and self.curr_speed[part] >= 0:
      if part == self.RIGHT_TRACK:
        self.switch_down(part)
      else:
        self.switch_down(part)
      self.switch_exec(False)
    self.curr_speed[part] = pulse_speed
    print("new speed = %d" % pulse_speed)


  def save_all(self, execute_immediate = True):
    self.saved_pin_vals = copy.deepcopy(self.all_pin_vals)
    print("saved:", self.saved_pin_vals)

  def restore_all(self, execute_immediate = True):
    self.all_pin_vals = self.saved_pin_vals
    print("restored:", self.all_pin_vals)
    for pin_offset, pin in enumerate(self.all_pin_numbers):
      self.all_pin_changed[pin_offset] = True

  def stop_all(self, execute_immediate = True):
    #Configure the register to default value
    for pin in self.all_pin_numbers:
      self.switch_pin_off(pin)
    if execute_immediate:
      self.switch_exec(False)
    else:
      self.switch_exec(True)
    self.sound("OFF")

  def drive_forward(self, speed=1):
          print("F")
          self.switch_down(self.LEFT_TRACK)
          self.curr_speed[self.LEFT_TRACK] = self.convert_speed(speed)
          self.switch_on(self.RIGHT_TRACK)
          self.curr_speed[self.RIGHT_TRACK] = self.convert_speed(speed)
          self.switch_exec(False)
          self.PARTS_VAL[self.LEFT_TRACK] = "FORWARD"
          self.PARTS_VAL[self.RIGHT_TRACK] = "FORWARD"
  
  def drive_reverse(self, speed=1):
          print("B")
          self.switch_on(self.LEFT_TRACK)
          self.curr_speed[self.LEFT_TRACK] = -self.convert_speed(speed)
          self.switch_down(self.RIGHT_TRACK)
          self.curr_speed[self.RIGHT_TRACK] = -self.convert_speed(speed)
          self.switch_exec(False)
          self.PARTS_VAL[self.LEFT_TRACK] = "BACK"
          self.PARTS_VAL[self.RIGHT_TRACK] = "BACK"
  
  def drive_rotate_left(self, speed=1):
          # print("L")
          self.switch_on(self.LEFT_TRACK)
          self.curr_speed[self.LEFT_TRACK] = -self.convert_speed(speed)
          self.switch_on(self.RIGHT_TRACK)
          self.curr_speed[self.RIGHT_TRACK] = self.convert_speed(speed)
          self.switch_exec(False)
          self.PARTS_VAL[self.LEFT_TRACK] = "BACK"
          self.PARTS_VAL[self.RIGHT_TRACK] = "FORWARD"
  
  def drive_rotate_right(self, speed=1):
          print("R")
          self.switch_down(self.LEFT_TRACK)
          self.curr_speed[self.LEFT_TRACK] = self.convert_speed(speed)
          self.switch_down(self.RIGHT_TRACK)
          self.curr_speed[self.RIGHT_TRACK] = -self.convert_speed(speed)
          self.switch_exec(False)
          self.PARTS_VAL[self.LEFT_TRACK] = "FORWARD"
          self.PARTS_VAL[self.RIGHT_TRACK] = "BACK"
  
  def drive_stop(self):
          print("S")
          self.switch_off(self.LEFT_TRACK)
          self.curr_speed[self.LEFT_TRACK] = 0
          self.switch_off(self.RIGHT_TRACK)
          self.curr_speed[self.RIGHT_TRACK] = 0
          self.switch_exec(False)
          self.PARTS_VAL[self.LEFT_TRACK] = "STOPPED"
          self.PARTS_VAL[self.RIGHT_TRACK] = "STOPPED"
  
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
           # self.curr_pin_val = GPIO.LOW
           # self.curr_pin_io_val = self.ALL_FUNC
           # save
           self.save_all()
           self.stop_all(execute_immediate=True)
           # restore state for next pulse
           if self._driver.gather_data.action_name != None or process_image:
             if take_picture:
               # take picture and collect data
               action = self._driver.gather_data.save_snapshot(process_image)
             # self.curr_pin_val = save_pin_val
             # self.curr_pin_io_val = save_pin_io_val
             self.restore_all()
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
            # divisor = 10
            divisor = 5
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
        self.PARTS_VAL[self.UPPER_ARM] = "UP"
    elif dir == "DOWN":
        self.switch_down(self.UPPER_ARM)
        self.switch_exec()
        self.PARTS_VAL[self.UPPER_ARM] = "DOWN"
    elif dir == "STOP":
        self.switch_off(self.UPPER_ARM)
        self.switch_exec()
        self.PARTS_VAL[self.UPPER_ARM] = "STOPPED"
    else:
        print("upper_arm: invalid input (%s)" % dir)

  def lower_arm(self, dir):
    if dir == "UP":
        self.switch_up(self.LOWER_ARM)
        self.switch_exec()
        self.PARTS_VAL[self.LOWER_ARM] = "UP"
    elif dir == "DOWN":
        self.switch_down(self.LOWER_ARM)
        self.switch_exec()
        self.PARTS_VAL[self.LOWER_ARM] = "DOWN"
    elif dir == "STOP":
        self.switch_off(self.LOWER_ARM)
        self.switch_exec()
        self.PARTS_VAL[self.LOWER_ARM] = "STOPPED"
    else:
        print("lower_arm: invalid input (%s)" % dir)

  def wrist(self, dir):
    if dir == "ROTATE_LEFT":
        self.switch_up(self.WRIST)
        self.switch_exec()
        self.PARTS_VAL[self.WRIST] = "LEFT"
    elif dir == "ROTATE_RIGHT":
        self.switch_down(self.WRIST)
        self.switch_exec()
        self.PARTS_VAL[self.WRIST] = "RIGHT"
    elif dir == "STOP":
        self.switch_off(self.WRIST)
        self.switch_exec()
        self.PARTS_VAL[self.WRIST] = "STOPPED"
    else:
        print("wrist: invalid input (%s)" % dir)

  def shovel(self, dir):
    if dir == "UP":
        self.switch_up(self.SHOVEL)
        self.switch_exec()
        self.PARTS_VAL[self.SHOVEL] = "UP"
    elif dir == "DOWN":
        self.switch_down(self.SHOVEL)
        self.switch_exec()
        self.PARTS_VAL[self.SHOVEL] = "DOWN"
    elif dir == "STOP":
        self.switch_off(self.SHOVEL)
        self.switch_exec()
        self.PARTS_VAL[self.SHOVEL] = "STOPPED"
    else:
        print("gripper: invalid input (%s)" % dir)

  def chassis(self, dir):
    if dir == "LEFT":
        self.switch_up(self.CHASSIS)
        self.switch_exec()
        self.PARTS_VAL[self.CHASSIS] = "LEFT"
    elif dir == "RIGHT":
        self.switch_down(self.CHASSIS)
        self.switch_exec()
        self.PARTS_VAL[self.CHASSIS] = "RIGHT"
    elif dir == "STOP":
        self.switch_off(self.CHASSIS)
        self.switch_exec()
        self.PARTS_VAL[self.CHASSIS] = "STOPPED"
    else:
        print("gripper: invalid input (%s)" % dir)

  def sound(self, on_off="OFF"):
    while on_off is None or self.PARTS_VAL[self.SOUND] != on_off:
      self.switch_up(self.SOUND)
      self.switch_exec(True)
      if self.PARTS_VAL[self.SOUND] == "ON":
        self.PARTS_VAL[self.SOUND] = "OFF"
        print("SOUND_OFF")
      elif self.PARTS_VAL[self.SOUND] == "OFF":
        self.PARTS_VAL[self.SOUND] = "ON"
        print("SOUND_ON")
      time.sleep(1.5)
      self.switch_off(self.SOUND)
      self.switch_exec(True)
      if on_off is None:
        break

  def demo(self, on_off=None):
    while on_off is None or self.PARTS_VAL[self.DEMO] != on_off:
      self.switch_up(self.DEMO)
      self.switch_exec(True)
      if self.PARTS_VAL[self.DEMO] == "ON":
        self.PARTS_VAL[self.DEMO] = "OFF"
      elif self.PARTS_VAL[self.DEMO] == "OFF":
        self.PARTS_VAL[self.DEMO] = "ON"
      self.switch_down(self.DEMO)
      self.switch_exec(False)
      if on_off is None:
        break

  def program_mode(self, on_off=None):
    while on_off is None or self.PARTS_VAL[self.PROGRAM_MODE] != on_off:
      self.switch_up(self.PROGRAM_MODE)
      self.switch_exec(True)
      if self.PARTS_VAL[self.PROGRAM_MODE] == "ON":
         self.PARTS_VAL[self.PROGRAM_MODE] = "OFF"
      elif self.PARTS_VAL[self.PROGRAM_MODE] == "OFF":
         self.PARTS_VAL[self.PROGRAM_MODE] = "ON"
      self.switch_down(self.PROGRAM_MODE)
      self.switch_exec(False)
      if on_off is None:
        break



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
      elif command == "CHASSIS_LEFT":
          self.chassis("LEFT")
      elif command == "CHASSIS_RIGHT":
          self.chassis("RIGHT")
      elif command == "FORWARD":
          self.drive_forward(speed)
      elif command == "REVERSE":
          self.drive_reverse(speed)
      elif command == "LEFT":
          self.drive_rotate_left(speed)
      elif command == "RIGHT":
          self.drive_rotate_right(speed)
      elif command == "PROGRAM_MODE_ON":
          self.program_mode(True)
      elif command == "PROGRAM_MODE_OFF":
          self.program_mode(False)
      elif command == "DEMO_ON":
          self.demo("ON")
      elif command == "DEMO_OFF":
          self.demo("OFF")
      elif command == "SOUND_ON":
          self.sound("ON")
      elif command == "SOUND_OFF":
          self.sound("OFF")
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
          print("*********** TEST_ARM ************")
          for part in (self.LOWER_ARM, self.UPPER_ARM, self.SHOVEL, self.CHASSIS):
              try:
                  self.switch_up(part)
                  self.switch_exec(False)
                  time.sleep(3)
                  self.switch_down(part)
                  self.switch_exec(False)
                  time.sleep(3)
                  self.switch_off(part)
                  self.switch_exec(False)
              except KeyboardInterrupt:
                  self.stop_all()
                  self.switch_exec(False)
  
  def test_drive(self):
          print("*********** TEST_DRIVE **********")
          try:
            self.drive_forward()
            self.switch_exec(False)
            time.sleep(3)

            for i in range(1, 100):
              self.drive_forward()
              self.switch_exec(False)
              time.sleep(.08)
              self.drive_stop()
              self.switch_exec(False)
              time.sleep(.08)

            for i in range(1, 100):
              self.drive_forward()
              self.switch_exec(False)
              time.sleep(.1)
              self.drive_stop()
              self.switch_exec(False)
              time.sleep(.2)

            self.drive_stop()
            self.switch_exec(False)
            time.sleep(1)
            self.drive_reverse()
            self.switch_exec(False)
            time.sleep(3)
            self.drive_stop()
            self.switch_exec(False)
            time.sleep(1)
            self.drive_rotate_left()
            self.switch_exec(False)
            time.sleep(3)
            self.drive_stop()
            self.switch_exec(False)
            time.sleep(1)
            self.drive_rotate_right()
            self.switch_exec(False)
            time.sleep(3)
            self.drive_stop()
            self.switch_exec(False)
            time.sleep(1)
          except KeyboardInterrupt:
            self.drive_stop()
            self.switch_exec(False)

  def test_parts(self):
    print("*********** TEST_PARTS **********")
    for part_num, part in enumerate(self.PARTS):
       print("Test part number: ", part)
       self.switch_up(part_num)
       self.switch_exec(False)
       time.sleep(1)
       self.switch_down(part_num)
       self.switch_exec(False)
       time.sleep(1)
       self.switch_off(part_num)
       self.switch_exec(False)

  def __init__(self, robot_driver=None):
    self._driver = robot_driver

    # From Frank's documentation:
    # 7, 11, 12, 13, 15 ( not used), 16, 18, 19, 21, 22
    # 23, 24, 26, 33, 32, 36
    # 39 is ground
    # 15 pins used
    #
    # B1, [1 4], right bck, 19
    # A1, [5,9], Swivel chassis right, 24
    # A2, [5,6], right fwd, 32
    # B4, [3,1], shovel down, 22
    #
    # D2, [6,7], Demo Key 26
    # A3, [9,4], Program mode 7
    # B3, [6,4], Sound 21
    # A4, [5,9], shovel up, 33
    #
    # D1, [10,3], lower arm up 12
    # C1, [9,7], upper arm up 11
    # B2, [4,10], upper arm down 18
    # B4, [1, 2], lower arm down 13
    #
    # C3, [9,2], left bck, 7
    # D3, [11,2] Swivel chassis left, 16
    # D4, [8,2], left fwd 22
    #
    # self.all_pin_spec    = [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]
    # self.all_pin_numbers = [ 7, 11, 12, 13, 15, 16, 18, 19, 21, 22, 23, 24, 26, 32, 33, 36]
    # self.all_pin_spec    = [c3, b3, a3, b4, c4, d3, a4, b1, c1, d4, b2, a1, d1, a2, c2]
    #
    # [["LEFT_TRACK", "FORWARD", 22], ["LEFT_TRACK", "BACK", 7]],
    # [["RIGHT_TRACK", "FORWARD", 32], ["RIGHT_TRACK", "BACK", 19]],
    # [["CHASSIS", "LEFT", 16], ["CHASSIS", "RIGHT", 24]],
    # [["SHOVEL", "UP", 33], ["SHOVEL", "DOWN", 22 ]],
    # [["UPPER_ARM", "UP", 11], ["UPPER_ARM", "DOWN", 18]],
    # [["LOWER_ARM",  "UP", 12], ["LOWER_ARM", "DOWN", 13]],
    # [["PROGRAM_MODE", "TOGGLE", 12]],
    # [["SOUND", "TOGGLE", 21]],
    # [["DEMO", "TOGGLE", 26]]


    # keeping consistent with mcp code 
    self.LEFT_TRACK = 0
    self.RIGHT_TRACK = 1
    self.CHASSIS = 2
    self.SHOVEL = 3
    self.UPPER_ARM = 4
    self.LOWER_ARM = 5
    self.PROGRAM_MODE = 6
    self.SOUND = 7
    self.DEMO = 8
    # PARTS can only move one direction, speed, etc despite using 2 pins
    self.PARTS = ["LEFT_TRACK", "RIGHT_TRACK", "CHASSIS", "SHOVEL", "UPPER_ARM", "LOWER_ARM", "PROGRAM_MODE", "SOUND", "DEMO"]
    self.PARTS_VAL = ["STOPPED", "STOPPED", "STOPPED", "STOPPED", "STOPPED", "STOPPED", "OFF", "ON", "OFF"]
    self.PART_DIRECTIONS = [["FORWARD", "BACK"], ["UP", "DOWN"], ["LEFT","RIGHT"], ["ON","OFF"]]
    self.PART_SPECS   = [
                         [["LEFT_TRACK", "FORWARD", 7], ["LEFT_TRACK", "BACK", 22]],
                         [["RIGHT_TRACK", "FORWARD", 19], ["RIGHT_TRACK", "BACK", 32]],
                         [["CHASSIS", "LEFT", 16], ["CHASSIS", "RIGHT", 24]],
                         [["SHOVEL", "UP", 18], ["SHOVEL", "DOWN", 13 ]],
                         [["UPPER_ARM", "UP", 26], ["UPPER_ARM", "DOWN", 21]],
                         [["LOWER_ARM",  "UP", 33], ["LOWER_ARM", "DOWN", 23]],
                         [["PROGRAM_MODE", "TOGGLE", 12]],
                         [["SOUND", "TOGGLE", 11]],
                         [["DEMO", "TOGGLE", 36]]
                        ]
    # offset           =   [ 0,  1,  2,  3,  4,  5   6   7   8   9  10  11  12  13  14  15]
    self.all_pin_numbers = [ 7, 11, 12, 13, 15, 16, 18, 19, 21, 22, 23, 24, 26, 32, 33, 36]
    self.all_pin_changed = []
    self.all_pin_vals = []
    self.saved_pin_vals = []
    for i in range(len(self.all_pin_numbers)):
      self.all_pin_vals.append(False)
      self.saved_pin_vals.append(False)
      self.all_pin_changed.append(False)
    self.all_pin_disallowed = [12, 36]  # disallow DEMO mode and PROGRAM mode

    self.curr_pwm_pulse = 0
    self.curr_speed = []
    self.curr_mode = []
    self.curr_timeout = []
    for i in range(len(self.PARTS)):
      self.curr_speed.append(0.0)
      self.curr_mode.append("STOP")
      self.curr_timeout.append(0)

    self.gpio_init()
    try:
      # print("stop and sleep")
      # self.stop_all()
      # time.sleep(5)
      # self.test_parts()
      # self.test_arm()
      # self.test_drive()
      # print("*********** TEST_DONE **********")
      pass
    except Exception as e:
      track = traceback.format_exc()
      print(track)
    self.stop_all()
  
  
# alset1 = AlSET_control()
