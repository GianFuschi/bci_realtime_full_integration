import yaml
from src.acquisition import EEGStream
from src.preprocessing import OnlineBandpass
from src.features import SlidingWindow, BandPower
from src.classifier import OnlineLDA
from src.feedback import VisualBar
from src.profiler import Profiler
import numpy as np

from threading import Thread
from queue import Queue



CHANNEL_IDX = [1, 3]

class RealTimeBCI:
    def __init__(self, config_path, classifier):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.eeg = EEGStream(step_size=cfg["window"]["step"]) #To prevent issues with the windowing
        self.filter = OnlineBandpass(
            cfg["sampling_rate"],
            cfg["filter"]["low"],
            cfg["filter"]["high"],
            cfg["channels"]
        )
        self.window = SlidingWindow(
            cfg["window"]["size"],
            cfg["window"]["step"],
            cfg["channels"]

        )
        self.features = BandPower(
            cfg["sampling_rate"],
            cfg["features"]["band"]
        )
        self.classifier = classifier
        self.feedback = VisualBar(cfg["feedback"]["smoothing"])
        self.profiler = Profiler()

        self.window_queue = Queue(maxsize=10)
        self.prediction_queue = Queue(maxsize=10)


    def data_worker(self):
        while True:
            chunk, _ = self.eeg.pull_chunk()
            if chunk is None or len(chunk) == 0:
                continue

            chunk = np.array(chunk)
            chunk = self.filter.process(chunk)
            chunk = chunk[:, CHANNEL_IDX]

            win = self.window.update(chunk)
            if win is not None:
                self.window_queue.put(win)

    def prediction_worker(self):
        while True:
            win = self.window_queue.get()
            try:
                feats = self.features.compute(win) #We can move this to the data loop if there is too much of a time difference.
                pred = self.classifier.predict(feats)
                if pred is not None:
                    #Do we wanna transfer to numbers here?
                    self.prediction_queue.put(pred)    
            finally:
                self.window_queue.task_done()

    def visualization_worker(self):
        while True:
            pred = self.prediction_queue.get()
            try:
                self.feedback.update(float(pred))
            finally:
                self.prediction_queue.task_done()

    def run(self):
        data_thread = Thread(target=self.data_worker, daemon=True)
        prediction_thread = Thread(target=self.prediction_worker, daemon=True)
        visualization_thread = Thread(target=self.visualization_worker, daemon=True)

        data_thread.start()
        prediction_thread.start()
        visualization_thread.start()

        data_thread.join()
        prediction_thread.join()
        visualization_thread.join()
        
    # def run(self):
    #     #check efficiency of this loop
    #     #make a train function with data extraction?
    #     while True:
    #         self.profiler.tic("loop")

    #         chunk, timestamps = self.eeg.pull_chunk()
    #         if chunk is None or len(chunk) == 0:
    #             # print("no data received, skipping iteration")
    #             continue

    #         # chunk = chunk[:, CHANNEL_IDX] #chunk is a 2D array with shape (n_samples, n_channels)
    #         chunk = np.array(chunk)
    #         # print(chunk.shape) #shape is (samples, channels)

    #         chunk = self.filter.process(chunk) #check if works with pull chunk

    #         # print(chunk.shape) #shape is (samples, channels)

    #         chunk = chunk[:, CHANNEL_IDX] #chunk is a 2D array with shape (n_samples, n_channels)
    #         # print(chunk.shape) #shape is (samples, channels)



    #         #ch
    #         win = self.window.update(chunk)

    #         if win is not None:
    #             feats = self.features.compute(win)
    #             pred = self.classifier.predict(feats)
    #             if pred is not None:
    #                 self.feedback.update(float(pred))

    #         loop_time = self.profiler.toc("loop")
    #         # print(f"Loop time: {loop_time:.4f}s")

        

if __name__ == "__main__":
    classifier = OnlineLDA()
    bci = RealTimeBCI("config/config.yaml", classifier)
    bci.run()