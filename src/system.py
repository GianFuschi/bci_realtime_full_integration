import yaml
from src.acquisition import EEGStream
from src.preprocessing import OnlineBandpass
from src.features import SlidingWindow, BandPower
from src.classifier import OnlineLDA
from src.feedback import VisualBar
from src.profiler import Profiler, WindowMessage, PredictionMessage, LatencyMessage
import numpy as np

from threading import Thread, Event
from queue import Queue
import pylsl as lsl


class RealTimeBCI:
    def __init__(self, config_path):
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
            channels=len(cfg["features"]["channel_indeces"])

        )
        self.features = BandPower(
            cfg["sampling_rate"],
            cfg["features"]["band"]
        )
        self.classifier = OnlineLDA()
        self.classifier.load(cfg["classifier"]["path"])

        self.feedback = VisualBar(cfg["feedback"]["smoothing"])
        self.profiler = Profiler()

        self.cfg = cfg        

        self.window_queue = Queue(maxsize=10)
        self.prediction_queue = Queue(maxsize=10)
        self.latency_queue = Queue(maxsize=10)

        self.stop_event = Event()

    def data_worker(self):
        #measure latency between incoming chunks
        #measure latency between windows published
        #measure latency time to fill up window

        channel_idx = self.cfg["features"]["channel_indeces"]
        win_id = 0

        start_stamp = lsl.local_clock()

        while not self.stop_event.is_set():
            chunk, timestamp = self.eeg.pull_chunk()
            if chunk is None or len(chunk) == 0:
                continue

            chunk = np.array(chunk)
            chunk = self.filter.process(chunk)
            chunk = chunk[:, channel_idx]

            win = self.window.update(chunk)
            if win is not None:

                stamps = {"start_stamp": start_stamp,
                          "win_stamp": lsl.local_clock()}
                #If win_id is 0 then the difference is the time it takes to fill up the buffer

                message = WindowMessage(window=win,
                                        window_id=win_id,
                                        stamps=stamps)
                self.window_queue.put(message)
                print("Extracted first window")

                start_stamp = timestamp[0]
                win_id += 1

    def prediction_worker(self):
        while not self.stop_event.is_set():
            win_message = self.window_queue.get(timeout=10)
            try:

                feats = self.features.compute(win_message.window)
                feats = np.log10(feats) # added log
                
                feature_stamp = lsl.local_clock()

                feats = np.expand_dims(feats, axis=0) #add batch dim
                pred = self.classifier.predict(feats) # (1, n_classes)
                pred = np.squeeze(pred, axis=0) #remove batch dim
   
                if pred is not None:
                    pred = pred[1] #Nindex 0 is left hand, 1 is right hand.

                    stamps = win_message.stamps
                    stamps["feature_stamp"] = feature_stamp
                    stamps["prediction_stamp"] = lsl.local_clock()

                    message = PredictionMessage(prediction=pred,
                                                window_id=win_message.window_id,
                                                stamps=stamps
                                                )
                    self.prediction_queue.put(message)    
                    print("Made prediction")
            finally:
                self.window_queue.task_done()

    def visualization_worker(self):
        while not self.stop_event.is_set():
            pred_message = self.prediction_queue.get(timeout=10)
            try:
                self.feedback.update(float(pred_message.prediction))
                stamps = pred_message.stamps
                stamps["feedback_stamp"] = lsl.local_clock()
                lat_message = LatencyMessage(window_id=pred_message.window_id,
                                             stamps=stamps)
                self.latency_queue.put(lat_message)
            finally:
                self.prediction_queue.task_done()

    def profiler_worker(self):
        while not self.stop_event.is_set():
            lat_message = self.latency_queue.get(timeout=10)
            try:
                self.profiler.update(lat_message)
            finally:
                self.latency_queue.task_done()

    def run(self):
        threads = [
            Thread(target=self.data_worker),
            Thread(target=self.prediction_worker),
            Thread(target=self.visualization_worker),
            Thread(target=self.profiler_worker)
        ]
        try:
            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()
        except KeyboardInterrupt:
            print("Stopping pipeline")
            #Send message to the profiler to log final value?

        finally:
            self.stop_event.set()

            for thread in threads:
                if thread.is_alive():
                    thread.join(timeout=2)

            self.profiler.log()

            print("Cleaned up threads and finished shutdown")
        

if __name__ == "__main__":
    bci = RealTimeBCI("config/config.yaml")
    bci.run()