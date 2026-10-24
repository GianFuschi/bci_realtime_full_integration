"""
Simple EEG LSL stream simulator for testing
"""

import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet

info = StreamInfo("SimEEG", "EEG", 9, 250, "float32")
outlet = StreamOutlet(info)

while True:
    sample = np.random.randn(9).tolist()
    outlet.push_sample(sample)
    time.sleep(1 / 250)
