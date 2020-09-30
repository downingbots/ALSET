import enum
import traitlets
from traitlets.config.configurable import Configurable
import ipywidgets.widgets as widgets
import time
import threading
from .heartbeat import Heartbeat

# class PseudoPWM(Configurable):
class PseudoPWM(Heartbeat):
    class PWMpulse(enum.Enum):
        pulse0 = 0
        pulse1 = 1
        pulse2 = 2
        pulse3 = 3
        pulse4 = 4
        pulse5 = 5
        pulse6 = 6
        pulse7 = 7
        pulse8 = 8
        pulse9 = 9
        pulse10 = 10 #unused

    pulse_enum = traitlets.UseEnum(PWMpulse, default_value=PWMpulse.pulse0)
    
    # config
    # period = traitlets.Float(default_value=0.1).tag(config=True)
    period = traitlets.Float(default_value=0.085).tag(config=True)

    def __init__(self, *args, **kwargs):
        super(Heartbeat, self).__init__(*args,
                                        **kwargs)  # initializes traitlets

        self.start()

    def _run(self):
        while True:
            if not self.running:
                break
            pulse_num = self.pulse_enum.value
            pulse_num += 1
            if pulse_num >= 10:
              pulse_num = 0
            self.pulse_enum = PseudoPWM.PWMpulse(pulse_num)
            time.sleep(self.period)

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
