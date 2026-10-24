import numpy as np
import pyxdf
import pickle
import yaml

from src.preprocessing import OnlineBandpass

# UPDATE: Provide a list of XDF files instead of a single path
XDF_PATHS = [
    "raw_signal/sub-P003/ses-S001/eeg/sub-P003_ses-S001_task-Default_run-001_eeg.xdf",
    "raw_signal/sub-P002/ses-S001/eeg/sub-P002_ses-S001_task-Default_run-001_eeg.xdf" # Add your second file here
]

class Extractor():
    def __init__(self, xdf_paths, config_path="config/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg
        self.xdf_paths = xdf_paths  # Store the list of paths
        
        self.filter = OnlineBandpass(
            cfg["sampling_rate"],
            cfg["filter"]["low"],
            cfg["filter"]["high"],
            cfg["channels"]
        )
        # REMOVED: pyxdf.load_xdf from __init__ so we can load multiple files dynamically

    def get_stream_data(self, file_path):
        print(f"\n--- Loading {file_path} ---")
        streams, _ = pyxdf.load_xdf(file_path)
        
        eeg_stream = None
        marker_stream = None
        
        # Identify streams by type or name
        for stream in streams:
            s_type = stream['info']['type'][0]
            s_name = stream['info']['name'][0]
            
            if s_type == 'EEG' or 'Unicorn' in s_name:
                eeg_stream = stream
            elif s_type == 'Markers' or s_name == 'MotorImageryMarkers':
                marker_stream = stream

        # Extract data matrices and timestamps
        eeg_data = np.array(eeg_stream['time_series'])       # Shape: (samples, channels)
        filtered_eeg_data = self.filter.process(eeg_data)  # Apply bandpass filter to EEG data

        eeg_timestamps = np.array(eeg_stream['time_stamps'])  # Shape: (samples,)
        fs = float(eeg_stream['info']['nominal_srate'][0])     # Nominal sampling rate (e.g., 250 Hz)

        if self.cfg["sampling_rate"] != fs:
            raise Warning(
                f"Detected sampling rate {fs} is not the same as in config file {self.cfg['sampling_rate']}"
            )

        marker_series = [m[0] for m in marker_stream['time_series']]
        marker_timestamps = np.array(marker_stream['time_stamps'])

        print(f"Loaded EEG Stream: {eeg_data.shape[1]} channels @ {fs:.1f} Hz ({eeg_data.shape[0]} samples)")
        print(f"Loaded Marker Stream: {len(marker_series)} markers found")

        data = {
            "eeg_data": filtered_eeg_data,
            "eeg_timestamps": eeg_timestamps,
            "marker_series": marker_series,
            "marker_timestamps": marker_timestamps
        }

        return data

    def check_data(self, epochs):
        # make shapes the same for all epochs across ALL files
        min_length = min([len(epoch) for epoch in epochs])
        max_length = max([len(epoch) for epoch in epochs])
        
        counter = 0
        counter_2 = 0
        for epoch in epochs:
            if len(epoch) != min_length:
                counter += 1
            else:
                counter_2 += 1
                
        print("\n--- Final Epoch Consistency Check ---")
        print(f"Number of epochs with length {min_length}: {counter_2}")
        print(f"Number of epochs with length different from {min_length}: {counter}")
        print(f"Max epoch length: {max_length}, Min epoch length: {min_length}")
            
        new_epochs = [epoch[:min_length] for epoch in epochs]
        return np.array(new_epochs)

    def epoch_eeg_data(self, data, epoch_onset=0, duration=1):
        epochs = []
        labels = []

        for marker_time, marker_label in zip(data["marker_timestamps"], data["marker_series"]):
            # Find the closest EEG timestamp to the marker
            start_idx = np.argmin(np.abs(data["eeg_timestamps"] - (marker_time + epoch_onset)))
            end = data["eeg_timestamps"][start_idx] + duration
            end_idx = np.argmin(np.abs(data["eeg_timestamps"] - end))
            
            # Extract the epoch data
            epoch_data = data["eeg_data"][start_idx:end_idx]
            epochs.append(epoch_data)
            labels.append(marker_label)

        # UPDATE: Just return the raw lists, we will check them globally later
        return epochs, labels

    def save_epochs(self, data):
        print(f"\nSaved {len(data[0])} TOTAL epochs from all EEG files.")
        with open(self.cfg["features"]["raw_epochs"], "wb") as f:
            pickle.dump(data, f)

    def extract(self):
        all_epochs = []
        all_labels = []

        # UPDATE: Loop over all provided XDF paths
        for path in self.xdf_paths:
            data = self.get_stream_data(path)
            file_epochs, file_labels = self.epoch_eeg_data(data)
            
            # Append this file's epochs and labels to the global pool
            all_epochs.extend(file_epochs)
            all_labels.extend(file_labels)

        # Ensure all epochs across all files are perfectly aligned to the min_length
        final_epochs_array = self.check_data(all_epochs)
        final_labels_array = np.array(all_labels)

        final_data = (final_epochs_array, final_labels_array)
        self.save_epochs(final_data)


# Run the updated extractor
extractor = Extractor(xdf_paths=XDF_PATHS)
extractor.extract()