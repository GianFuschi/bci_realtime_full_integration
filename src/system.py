import yaml
from src.acquisition import EEGStream
from src.preprocessing import OnlineBandpass
from src.features import SlidingWindow, BandPower
from src.classifier import OnlineLDA
from src.feedback import VisualBar
from src.profiler import Profiler
import numpy as np

CHANNEL_IDX = [1, 3]

class RealTimeBCI:
    def __init__(self, config_path, classifier):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.eeg = EEGStream()
        self.filter = OnlineBandpass(
            cfg["sampling_rate"],
            cfg["filter"]["low"],
            cfg["filter"]["high"],
            cfg["channels"]
        )
        self.window = SlidingWindow(
            cfg["window"]["size"],
            cfg["window"]["step"]
        )
        self.features = BandPower(
            cfg["sampling_rate"],
            cfg["features"]["band"]
        )
        self.classifier = classifier
        self.feedback = VisualBar(cfg["feedback"]["smoothing"])
        self.profiler = Profiler()

    def run(self):
        #check efficiency of this loop
        #make a train function with data extraction?
        while True:
            self.profiler.tic("loop")

            chunk, timestamps = self.eeg.pull_chunk()
            if chunk is None or len(chunk) == 0:
                # print("no data received, skipping iteration")
                continue

            # chunk = chunk[:, CHANNEL_IDX] #chunk is a 2D array with shape (n_samples, n_channels)
            chunk = np.array(chunk)
            print(chunk.shape) #shape is (samples, channels)

            chunk = self.filter.process(chunk) #check if works with pull chunk

            print(chunk.shape) #shape is (samples, channels)

            chunk = chunk[:, CHANNEL_IDX] #chunk is a 2D array with shape (n_samples, n_channels)
            print(chunk.shape) #shape is (samples, channels)



            #ch
            win = self.window.update(chunk)

            if win is not None:
                feats = self.features.compute(win)
                pred = self.classifier.predict(feats)
                if pred is not None:
                    self.feedback.update(float(pred))

            loop_time = self.profiler.toc("loop")
            print(f"Loop time: {loop_time:.4f}s")

if __name__ == "__main__":
    classifier = OnlineLDA()
    bci = RealTimeBCI("config/config.yaml", classifier)
    bci.run()