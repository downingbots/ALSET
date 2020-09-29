import atexit
import traitlets
from traitlets.config.configurable import Configurable
from .pseudopwm import *

class Motor(Configurable):

    value = traitlets.Float()
    pwm = PseudoPWM()
    
    def __init__(self, driver, side, *args, **kwargs):
        super(Motor, self).__init__(*args, **kwargs)  # initializes traitlets

        self._driver = driver
        if side == "LEFT":
          self._side = self._driver.LEFT_TRACK
        elif side == "RIGHT":
          self._side = self._driver.RIGHT_TRACK
        else:
          print("motor: no such side (%s)" % side)
        self.pwm.observe(self.handle_pwm, names='pulse_enum')
        atexit.register(self.pwm.stop)

    def get_speed(self):
        print("get_speed", self._side, self._driver.get_speed(self._side))
        return self._driver.get_speed(self._side)

    def handle_pwm(self,change):
        pulse_enum = change['new']
        pulse_num = pulse_enum.value
        self._driver.handle_pwm(pulse_num)

    @traitlets.observe('value')
    def _observe_value(self, change):
        self._write_value(change['new'])

    def _write_value(self, value):
        """Sets motor value between [-1, 1]"""
        # mapped_value = int(255.0 * (self.alpha * value + self.beta))
        # speed = min(max(abs(mapped_value), 0), 255)
        self._driver.set_speed(self._side, value)

    def _release(self):
        """Stops motor by releasing control"""
        self._driver.set_speed(self._side, 0)
