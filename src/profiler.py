import time

class Profiler:
    "Stores times for latency?"
    def __init__(self):
        self.times = {}

    def tic(self, key):
        self.times[key] = time.time()

    def toc(self, key):
        return time.time() - self.times.get(key, time.time())

