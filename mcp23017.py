import smbus
import time
import traceback

# TODO: direct_control and MCP23017 share a lot of code, with minor changes spread
# throughout. The structure and logic of the code is mostly the same. A bug in one
# file may implicate a bug in the other. Eventually, should cleanly split shared functions.
class MCP23017:
  IODIRA = 0x00
  IPOLA  = 0x02
  GPINTENA = 0x04
  DEFVALA = 0x06
  INTCONA = 0x08
  IOCONA = 0x0A
  GPPUA = 0x0C
  INTFA = 0x0E
  INTCAPA = 0x10
  GPIOA = 0x12
  OLATA = 0x14
  
  IODIRB = 0x01
  IPOLB = 0x03
  GPINTENB = 0x05
  DEFVALB = 0x07
  INTCONB = 0x09
  IOCONB = 0x0B
  GPPUB = 0x0D
  INTFB = 0x0F
  INTCAPB = 0x11
  GPIOB = 0x13
  OLATB = 0x15
   
  
  #   Addr(BIN)      Addr(hex)
  #XXX X  A2 A1 A0
  #010 0  1  1  1      0x27 
  #010 0  1  1  0      0x26 
  #010 0  1  0  1      0x25 
  #010 0  1  0  0      0x24 
  #010 0  0  1  1      0x23 
  #010 0  0  1  0      0x22
  #010 0  0  0  1      0x21
  #010 0  0  0  0      0x20
  
  ADDRESS         = 0x27
  LOW             = 0x00
  HIGH            = 0xFF
  PULLUP_DISABLED = 0x00

class MCP_control:
  bus = smbus.SMBus(1)
  curr_pin_val        = MCP23017.LOW
  curr_pin_pullup_val = MCP23017.LOW
  curr_pin_io_val     = MCP23017.HIGH
  switch_exec_next_pulse = False
  timeslice = 0.1

  BLUE   = 0x01 # 0000001
  GREEN  = 0x02 # 0000010
  YELLOW = 0x04 # 0000100
  ORANGE = 0x08 # 0001000
  RED    = 0x10 # 0010000
  BROWN  = 0x20 # 0100000
  
  LEFT_TRACK     = RED
  RIGHT_TRACK    = GREEN
  # LOWER_ARM      = BROWN
  # UPPER_ARM      = ORANGE
  LOWER_ARM      = ORANGE
  UPPER_ARM      = BROWN
  WRIST          = YELLOW
  GRIPPER        = BLUE
  ALL_FUNC = LEFT_TRACK | RIGHT_TRACK | LOWER_ARM | UPPER_ARM | WRIST | GRIPPER
  
  def switch_up(self,pin):
      # print("switch up   %d" % pin)
      # set pin to be in Output mode (low)
      # all_on = MCP23017.HIGH
      all_on = self.ALL_FUNC
      bit_off = all_on ^ pin
      self.curr_pin_io_val &= bit_off
      # set bit to low
      self.curr_pin_val &= bit_off
  
  def switch_off(self,pin):
      # print("switch off  %d" % pin)
      # set pin to be in Input mode (high)
      self.curr_pin_io_val |= pin
      # set High/Low bits to low as default
      all_on = self.ALL_FUNC
      bit_off = all_on ^ pin
      self.curr_pin_val &= bit_off

  def switch_down(self,pin):
      # print("switch down %d" % pin)
      # set pin to be in Output mode (low)
      all_on = self.ALL_FUNC
      bit_off = all_on ^ pin
      self.curr_pin_io_val &= bit_off
      # set High bit
      self.curr_pin_val |= pin
  
  def switch_exec(self, exec_next_pulse=True):
      if exec_next_pulse:
          # execute during next pulse processing
          self.switch_exec_next_pulse = True
      else:
          # print("IO/PIN", bin(self.curr_pin_io_val)[2:].zfill(8),
          #       bin(self.curr_pin_val)[2:].zfill(8))
          # execute immediate (probably called during pulse processing)
          self.switch_exec_next_pulse = False
          self.bus.write_byte_data(MCP23017.ADDRESS,MCP23017.IODIRB,self.curr_pin_io_val)
          self.bus.write_byte_data(MCP23017.ADDRESS,MCP23017.GPIOB,self.curr_pin_val)
          self.curr_timeout[self.LEFT_TRACK] = self.compute_pwm_timeout(self.LEFT_TRACK)
          self.curr_timeout[self.RIGHT_TRACK] = self.compute_pwm_timeout(self.RIGHT_TRACK)
          # self._driver.gather_data.set_action(None)
          # self.stop_all(execute_immediate = False)
          # if self._driver.gather_data.is_on():
          #   self.set_speed(self.LEFT_TRACK, 0)
          #   self.set_speed(self.RIGHT_TRACK, 0)
  
  def convert_speed(self, speed):
    pulse_speed = max(speed, -1)
    pulse_speed = min(pulse_speed, 1)
    pulse_speed = 10 * round(pulse_speed, 1)
    return pulse_speed

  def get_pulse_speed(self, pin):
    return self.curr_speed[pin]

  def get_speed(self, pin):
    return (1.0*self.curr_speed[pin]) / 10

  def set_speed(self, pin, speed):
    pulse_speed = self.convert_speed(speed)
    if pulse_speed > 0 and self.curr_speed[pin] <= 0:
      if pin == self.RIGHT_TRACK:
        self.switch_up(pin)
      else:
        self.switch_up(pin)
      self.switch_exec()
    elif pulse_speed == 0 and self.curr_speed[pin] != 0:
      self.switch_off(pin)
      self.switch_exec()
    elif pulse_speed < 0 and self.curr_speed[pin] >= 0:
      if pin == self.RIGHT_TRACK:
        self.switch_down(pin)
      else:
        self.switch_down(pin)
      self.switch_exec()
    self.curr_speed[pin] = pulse_speed
    print("new speed = %d" % pulse_speed)


  def stop_all(self, execute_immediate = True):
    #Configure the register to default value
    if execute_immediate:
      for addr in range(22):
        if (addr == 0) or (addr == 1):
          self.bus.write_byte_data(MCP23017.ADDRESS, addr, 0xFF)
        else:
          self.bus.write_byte_data(MCP23017.ADDRESS, addr, 0x00)
    #configure all PinB as input
    self.curr_pin_io_val     = MCP23017.HIGH
    # Disable all PinB pullUP
    self.curr_pin_pullup_val = MCP23017.LOW
    self.curr_pin_val        = MCP23017.LOW
    if execute_immediate:
        self.bus.write_byte_data(MCP23017.ADDRESS,MCP23017.IODIRB,self.curr_pin_io_val)
        self.bus.write_byte_data(MCP23017.ADDRESS,MCP23017.GPPUB,self.curr_pin_pullup_val)
        self.bus.write_byte_data(MCP23017.ADDRESS,MCP23017.GPIOB,self.curr_pin_val)

  def drive_forward(self, speed):
          print("F")
          self.switch_down(self.LEFT_TRACK)
          self.curr_speed[self.LEFT_TRACK] = self.convert_speed(speed)
          self.switch_up(self.RIGHT_TRACK)
          self.curr_speed[self.RIGHT_TRACK] = self.convert_speed(speed)
          self.switch_exec()
  
  def drive_reverse(self, speed):
          print("B")
          self.switch_up(self.LEFT_TRACK)
          self.curr_speed[self.LEFT_TRACK] = -self.convert_speed(speed)
          self.switch_down(self.RIGHT_TRACK)
          self.curr_speed[self.RIGHT_TRACK] = -self.convert_speed(speed)
          self.switch_exec()
  
  def drive_rotate_left(self, speed):
          # print("L")
          self.switch_up(self.LEFT_TRACK)
          self.curr_speed[self.LEFT_TRACK] = -self.convert_speed(speed)
          self.switch_up(self.RIGHT_TRACK)
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
           save_pin_val = self.curr_pin_val
           save_pin_io_val = self.curr_pin_io_val
           # stop all motors
           self.curr_pin_val = MCP23017.LOW
           self.curr_pin_io_val = self.ALL_FUNC
           self.switch_exec(exec_next_pulse=False)
           # restore state for next pulse
           if self._driver.gather_data.action_name != None or process_image:
             if take_picture:
               # take picture and collect data
               action = self._driver.gather_data.save_snapshot(process_image)
             self.curr_pin_val = save_pin_val
             self.curr_pin_io_val = save_pin_io_val
             #
             # Execute set_action here!  Does stop_all first in execute_command.
             # So, check for None actions should let Teleop command execute.
             # But save sanapshot already done..
             #
#             if process_image and action is not None:
             if process_image:
               print("command: ", action)
               self.execute_command(action)
               self._driver.gather_data.clear_process_image_action()
#             else:
#               action = self.pin_to_command()
#               if process_image and action is not None:
#                 print("pin to command: ", action)
#                 self.execute_command(action)
             self.switch_exec(exec_next_pulse=True)
           elif self._driver.gather_data.action_name == None:
             print("None action_name", self._driver.gather_data.nn_name)

  def handle_pulse(self, pulse_num, process_image):
        all_on = self.ALL_FUNC
        functions_not_stopped = all_on ^ self.curr_pin_io_val
        # print("active pins, pulse_num:", self._driver.gather_data.action_name,
        #       bin(functions_not_stopped)[2:].zfill(8), pulse_num)
        if self._driver.gather_data.is_on():
            # next pulse: essentially everything is half speed during data collection
            # if self._driver.get_NN_mode() == "NN", divisor of 5 seems sufficient
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

  #
  # webcam->robot->gather_data->nn_apps -> [automatic mode, NN, DQN]
  #                   ^
  #                   | gather_data shared variables.
  #                   v
  # joystick->robot->gather_data->[mcp, direct_control] -> robot manipulation
  #
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

  def gripper(self, dir):
    if dir == "CLOSE":
        self.switch_up(self.GRIPPER)
        self.switch_exec()
    elif dir == "OPEN":
        self.switch_down(self.GRIPPER)
        self.switch_exec()
    elif dir == "STOP":
        self.switch_off(self.GRIPPER)
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
      elif command == "GRIPPER_OPEN":
          self.gripper("OPEN")
      elif command == "GRIPPER_CLOSE":
          self.gripper("CLOSE")
      elif command == "FORWARD":
          self.drive_forward(speed)
      elif command == "REVERSE":
          self.drive_reverse(speed)
      elif command == "LEFT":
          self.drive_rotate_left(speed)
      elif command == "RIGHT":
          self.drive_rotate_right(speed)
      # REWARD/PENALTIES NEEDED FOR DDQN only
      elif command == "ROBOT_OFF_TABLE_PENALTY":
          self.stop_all()
      elif command == "CUBE_OFF_TABLE_REWARD":
          self.stop_all()
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

  def is_switch_up(self,pin):
      all_on = self.ALL_FUNC
      bit_off = all_on ^ pin
      pin_io_val = bit_off & self.curr_pin_io_val
      pin_val = bit_off & self.curr_pin_val
      if (self.curr_pin_io_val == pin_io_val and self.curr_pin_val == pin_val):
        return True
      return False
  
  def is_switch_off(self,pin):
      pin_io_val = self.curr_pin_io_val | pin
      all_on = self.ALL_FUNC
      bit_off = all_on ^ pin
      if self.curr_pin_val == bit_off:
        return True
      return False

  def is_switch_down(self,pin):
      all_on = self.ALL_FUNC
      bit_off = all_on ^ pin
      pin_io_val = self.curr_pin_io_val & bit_off
      pin_io_val |= pin
      if self.curr_pin_val == pin_io_val:
        return True
      return False
  
  def pin_to_command(self): 
      if self.is_switch_down(self.LEFT_TRACK) and self.is_switch_up(self.RIGHT_TRACK):
        return("FORWARD")
      if self.is_switch_up(self.LEFT_TRACK) and self.is_switch_down(self.RIGHT_TRACK):
        return("REVERSE")
      if self.is_switch_up(self.LEFT_TRACK) and self.is_switch_up(self.RIGHT_TRACK):
        return("LEFT")
      if self.is_switch_down(self.LEFT_TRACK) and self.is_switch_down(self.RIGHT_TRACK):
        return("RIGHT")
      if self.is_switch_off(self.LEFT_TRACK) and self.is_switch_off(self.RIGHT_TRACK):
        return("STOP")
      if self.is_switch_up(self.UPPER_ARM):
        return("UPPER_ARM_UP")
      if self.is_switch_down(self.UPPER_ARM):
        return("UPPER_ARM_DOWN")
      if self.is_switch_off(self.UPPER_ARM):
        return("UPPER_ARM_STOP")
      if self.is_switch_up(self.LOWER_ARM):
        return("LOWER_ARM_UP")
      if self.is_switch_down(self.LOWER_ARM):
        return("LOWER_ARM_DOWN")
      if self.is_switch_off(self.LOWER_ARM):
        return("LOWER_ARM_STOP")
      if self.is_switch_up(self.WRIST):
        return("WRIST_ROTATE_LEFT")
      if self.is_switch_down(self.WRIST):
        return("WRIST_ROSTAT_RIGHT")
      if self.is_switch_off(self.WRIST):
        return("WRIST_STOP")
      if self.is_switch_up(self.GRIPPER):
        return("GRIPPER_CLOSE")
      if self.is_switch_down(self.GRIPPER):
        return("GRIPPER_OPEN")
      if self.is_switch_off(self.GRIPPER):
        return("GRIPPER_STOP")

  def test_arm(self):
          for pin in (self.LOWER_ARM, self.UPPER_ARM, self.WRIST, self.GRIPPER):
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
  
  def __init__(self, robot_driver=None):
    self._driver = robot_driver

    self.curr_pwm_pulse = 0
    self.curr_speed = {}
    self.curr_speed[self.LEFT_TRACK]     = 0.0
    self.curr_speed[self.RIGHT_TRACK]    = 0.0
    self.curr_speed[self.LOWER_ARM]      = 0.0
    self.curr_speed[self.UPPER_ARM]      = 0.0
    self.curr_speed[self.WRIST]          = 0.0
    self.curr_speed[self.GRIPPER]        = 0.0

    self.curr_mode = {}
    self.curr_mode[self.LEFT_TRACK]      = "STOP"
    self.curr_mode[self.RIGHT_TRACK]     = "STOP"

    self.curr_timeout = {}
    self.curr_timeout[self.LEFT_TRACK]    = 0
    self.curr_timeout[self.RIGHT_TRACK]   = 0

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
  
  
# alset1 = ALSET_control()
