import time
import math
class StopWatch:
    def __init__(self):
        self.start()
    def start(self):
        self._startTime = time.time()
    def getStartTime(self):
        return self._startTime
    def elapsed(self, prec=3):
        diff= time.time() - self._startTime
        return round(diff, prec)
def round(n, p=0):
    m = 10 ** p
    return math.floor(n * m + 0.5) / m