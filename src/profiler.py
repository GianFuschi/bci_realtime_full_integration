import numpy as np
from dataclasses import dataclass
from river import stats
import logging
import time

@dataclass
class WindowMessage:
    window: np.ndarray
    window_id: int
    stamps: dict

@dataclass
class PredictionMessage:
    prediction: float
    window_id: int
    stamps: dict

@dataclass
class LatencyMessage:
    window_id: int
    stamps: dict

class Profiler():
    def __init__(self, log_file = "runningstats.log"):
        #Tuple for tunning mean and std
        self.statistics = {"system": (stats.Mean(), stats.Var()),
                           "window": (stats.Mean(), stats.Var()),
                           "initial_window": 0,
                           "PSD_feature": (stats.Mean(), stats.Var()),
                           "Predictor": (stats.Mean(), stats.Var()),
                           "Feedback": (stats.Mean(), stats.Var())}
        self.mapping = {
                "system": ("start_stamp", "feedback_stamp"),
                "window": ("start_stamp", "win_stamp"),
                "PSD_feature":("win_stamp", "feature_stamp"),
                "Predictor": ("feature_stamp", "prediction_stamp"),
                "Feedback": ("prediction_stamp", "feedback_stamp"),
            }
        
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        self._logger = logging.getLogger(__name__)

        self._last_log_time = time.monotonic()

    def log(self):
        message = {}

        for key, item in self.statistics.items():
            if key == "initial_window":
                message[key] = item
            else:
                message[key] = {"mean": item[0].get(),
                                "var": item[1].get(),
                                "std": np.sqrt(max(0.0, item[1].get()))}

        self._logger.info(f"Current running means: {message}")

    def running_update(self, latency, label):
        mean, std = self.statistics[label]
        mean.update(latency)
        std.update(latency)

    def update(self, message, log_interval=60):
        stamps = message.stamps
        id = message.window_id

        if id == 0:
            #First round we do not take into acocunt as this only fills the window buffer (predetermined delay)
            self.statistics["initial_window"] = stamps["win_stamp"] - stamps["start_stamp"]
        else:
            for key, value in self.mapping.items():
                try:
                    latency = stamps[value[1]] - stamps[value[0]]
                    self.running_update(latency, key)
                except:
                    print(f"Went wrong at {key}")
                    print(value)
                    print(stamps[value[0]], stamps[value[1]])

        now = time.monotonic()

        if now - self._last_log_time >= log_interval:
            self.log()
            self._last_log_time = now

