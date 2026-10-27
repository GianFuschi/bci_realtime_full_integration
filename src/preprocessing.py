import numpy as np
from scipy.signal import butter, lfilter

class OnlineBandpass:
    def __init__(self, fs, low, high, n_channels, order=4):
        self.b, self.a = butter(order, [low, high], btype="band", fs=fs)
        self.zi = np.zeros((max(len(self.a), len(self.b)) - 1, n_channels))

    def process(self, chunk):
        y, self.zi = lfilter(self.b, self.a, chunk, axis=0, zi=self.zi)
        return y
