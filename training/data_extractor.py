import numpy as np
import pyxdf
import pickle
import yaml

from src.preprocessing import OnlineBandpass

# LOAD XDF FILE AND SEPARATE STREAMS
XDF_PATH = "/mnt/c/Users/Gianluca/Downloads/sub-P003/ses-S001/eeg/sub-P003_ses-S001_task-Default_run-001_eeg.xdf"  # Update with your .xdf file path

# eeg_stream = None
# marker_stream = None
# CONFIG_PATH = "config/config.yaml"

class Extractor():
    def __init__(self, config_path ="config/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg
        self.filter = OnlineBandpass(
            cfg["sampling_rate"],
            cfg["filter"]["low"],
            cfg["filter"]["high"],
            cfg["channels"]
        )
        self.streams, _ = pyxdf.load_xdf(XDF_PATH)

    def get_stream_data(self):
        # Identify streams by type or name
        for stream in self.streams:
            s_type = stream['info']['type'][0]
            s_name = stream['info']['name'][0]
            
            if s_type == 'EEG' or 'Unicorn' in s_name:
                eeg_stream = stream
            elif s_type == 'Markers' or s_name == 'MotorImageryMarkers':
                marker_stream = stream

        # Extract data matrices and timestamps
        eeg_data = np.array(eeg_stream['time_series'])       # Shape: (samples, channels)
        print("CHECK THAT FILTERING IS DONE OVER THE RIGHT DIMENSION HERE!!")
        print(eeg_data.shape)
        filtered_eeg_data = self.filter.process(eeg_data)  # Apply bandpass filter to EEG data

        eeg_timestamps = np.array(eeg_stream['time_stamps'])  # Shape: (samples,)
        fs = float(eeg_stream['info']['nominal_srate'][0])     # Nominal sampling rate (e.g., 250 Hz)

        if self.cfg["sampling_rate"] != fs:
            raise Warning(
                f"Detected sampling rate {fs} is not the same as in config file {self.cfg["sampling_rate"]}"
                )

        marker_series = [m[0] for m in marker_stream['time_series']]
        marker_timestamps = np.array(marker_stream['time_stamps'])

        print(f"Loaded EEG Stream: {
            eeg_data.shape[1]} channels @ {fs:.1f} Hz ({eeg_data.shape[0]} samples)")
        print(f"Loaded Marker Stream: {len(marker_series)} markers found")

        data = {"eeg_data": filtered_eeg_data,
                "eeg_timestamps": eeg_timestamps,
                "marker_series": marker_series,
                "marker_timestamps": marker_timestamps}

        return data

    def check_data(self, epochs):
         #make shapes the same for all epochs if that is not the case already
        min_length = min([len(epoch) for epoch in epochs])
        max_length = max([len(epoch) for epoch in epochs])
        counter = 0
        counter_2 = 0
        for epoch in epochs:
            if len(epoch) != min_length:
                counter += 1
            else:
                counter_2 += 1
        print(f"Number of epochs with length {min_length}: {counter_2}")
        print(f"Number of epochs with length different from {min_length}: {counter}")
        print(f"Max epoch length: {max_length}, Min epoch length: {min_length}")
            
        new_epochs = [epoch[:min_length] for epoch in epochs]
        print(f"Epochs shape: {len(new_epochs)} epochs, each with {min_length} samples and {eeg_data.shape[1]} channels")
        del epochs
        return np.array(new_epochs)

    def epoch_eeg_data(self, data, epoch_onset=0, duration=1):

        epochs = []
        labels = []

        for marker_time, marker_label in zip(data["marker_timestamps"], data["marker_series"]):
            # Find the closest EEG timestamp to the marker
            start_idx = np.argmin(np.abs(data["eeg_timestamps"] - (marker_time + epoch_onset)))

            end = data["eeg_timestamps"][start_idx] + duration
            # print(eeg_timestamps[start_idx] , end)
            end_idx = np.argmin(np.abs(data["eeg_timestamps"] - end))
            
            # Extract the epoch data
            epoch_data = data["eeg_data"][start_idx:end_idx]
            epochs.append(epoch_data)
            labels.append(marker_label)

        epochs = self.check_data(epochs)
       

        return (epochs, np.array(labels))

    def save_epochs(self, data):
        print(f"Extracted {len(data[0])} epochs from EEG data.")

        with open(self.cfg["features"]["raw_epochs"], "wb") as f:
            pickle.dump(data, f)

    def extract(self):
        data = self.get_stream_data()
        data = self.epoch_eeg_data(data)
        print(f"Extracted {len(data[0])} epochs from EEG data.")
        self.save_epochs(data) 


extractor = Extractor()
extractor.extract()
