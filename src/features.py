import numpy as np
from scipy.signal import welch
import pylsl as lsl

# class SlidingWindow:
#     def __init__(self, size, step):
#         self.size = size
#         self.step = step
#         self.buffer = []
#         self.counter = 0

#     def update(self, chunk):
#         "Bug: self.counter should only increment when the buffer is full. Currently it is also no quasi-causal. Take a predetrmined delay into account. Also need to make it work with pull chunk but make step size okay."
#         self.counter += 1
#         if len(self.buffer) > self.size:
#             self.buffer.pop(0)
#         if len(self.buffer) == self.size and self.counter % self.step == 0:
#             return np.array(self.buffer)
#         return None


class SlidingWindow:
    def __init__(self, size, step, channels):
        self.size = size
        self.step = step
        self.channels = channels
        self.buffer = np.empty((0, channels))
        self.counter = 0
        self.is_full = False
        self.start_time = None

    # clock to measure how long to fill up the first window
    def start_clock(self):
        self.start_time = lsl.local_clock()

    def update(self, sample):
            
        sample = np.array(sample)
        self.buffer = np.vstack((self.buffer, sample))

        if self.is_full:
            self.counter += len(sample)

        # sliding window: keep the last 'window size' samples
        if len(self.buffer) > self.size:
            self.buffer = self.buffer[-self.size:]      

        # return window only when the 'sliding' reaches the step size 
        if len(self.buffer) == self.size and self.counter % self.step == 0:

            if not self.is_full:
                self.is_full = True
                end_time = lsl.local_clock()
                time_span = end_time - self.start_time
                print(f"Time taken to full up buffer: {time_span:.2f} seconds")
                print(f"Size of window {self.buffer.shape}")

            
            if len(self.buffer.shape) > 1:
                # removed the transpose to keep the shape as (samples, channels) for the Welch
                return self.buffer
            else:
                return self.buffer
            

        return None



class BandPower:
    def __init__(self, fs, band):
        self.fs = fs
        self.band = band

    def compute(self, window):
        freqs, psd = welch(window, fs=self.fs, axis=0)
        idx = (freqs >= self.band[0]) & (freqs <= self.band[1])
        return psd[idx].mean(axis=0)
