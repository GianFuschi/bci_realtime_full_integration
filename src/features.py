import numpy as np
from scipy.signal import welch
import pylsl as lsl

class SlidingWindow:
    def __init__(self, size, step, channels):
        self.size = size
        self.step = step
        self.channels = channels
        self.buffer = np.empty((0, channels))
        self.counter = 0
        self.is_full = False

    def slide_window(self):
        if len(self.buffer) > self.size:
            self.buffer = self.buffer[-self.size:]

    def resolve_overflow(self, chunk, overflow):
        remainder = None
        if overflow > 0:
            overflow_index = len(chunk) - overflow
            remainder = chunk[overflow_index:]
            chunk = chunk[:overflow_index]
        self.buffer = np.vstack((self.buffer, chunk))

        return remainder

    def update(self, chunk):
            
        chunk = np.array(chunk)
        remainder = None

        if self.is_full:
            self.counter += len(chunk)
            #First I only want to add a maximum extra of the step size
            overflow = self.counter - self.step #We do not want to add more than the step size to the buffer, but rather exactly return with every step size.
            remainder = self.resolve_overflow(chunk, overflow)
            self.slide_window()
        else:
            overflow = len(chunk) + len(self.buffer) - self.size
            remainder = self.resolve_overflow(chunk, overflow)

        # return window only when the 'sliding' reaches or exceeds the step size 
        if len(self.buffer) >= self.size and (self.counter >= self.step or self.counter == 0):

            self.is_full = True

            current_window = self.buffer

            if remainder is not None:
                self.buffer = np.vstack((self.buffer, remainder))
                self.slide_window()
                self.counter = overflow
            else:
                self.counter = 0

            return current_window

        return None



class BandPower:
    def __init__(self, fs, band):
        self.fs = fs
        self.band = band

    def compute(self, window):
        freqs, psd = welch(window, fs=self.fs, axis=0)
        idx = (freqs >= self.band[0]) & (freqs <= self.band[1])
        return psd[idx].mean(axis=0)
