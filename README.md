# bci_realtime_full_integration

### Running the Realtime Pipeline

An LSL streamer must be running in the background before starting the realtime BCI system.

To launch the full pipeline, execute `system.py` from the `src` folder. Depending on your Python path configuration, you can run it as a module from the project root:

```bash
python3 -m src.system
```

To shut down the system, press **`Ctrl+C`** in the terminal. This raises a `KeyboardInterrupt`, which the script handles to ensure that all threads are properly stopped and joined. **Do not use `Ctrl+Z`**, as this suspends the process instead of triggering the intended shutdown procedure.

### Project Structure

The `src` folder contains the following components:

- **`acquisition.py`** - Acquires EEG data from the LSL stream.
- **`classifier.py`** - Initializes, trains, saves, loads, and runs predictions using an LDA classifier.
- **`features.py`** - Handles windowing and power spectral density (PSD) feature extraction.
- **`feedback.py`** - Displays the feedback bar based on the predicted class probability.
- **`preprocessing.py`** - Implements causal filtering using `lfilter` and the filter state `zi`.
- **`profiler.py`** - Defines the message data classes and the profiler used to measure system (and individual component) latency ussing running means and variance.
- **`system.py`** - Integrates the individual components into the complete realtime pipeline.

### Training and Data Extraction

The scripts for data extraction and model training are located in the `training` folder. The extracted epochs, features and model weights are stored in the root of the folder such that they can be easily accessed. 

- **`beep.py`** - Creates a beep sound (linux specific).
- **`data_extractor.py`** - Extracts, bandpass filters, and epochs the raw eeg data using the provided marker stream.
- **`markerhelper.py`** - Defines a motor imagery paradigm, where at specific times in a trail a marker is provided to indicate which movement the user should imaginate.
- **`train.py`** - A training pipeline that uses the extracted data, calcultes balanced accuracy following a stratisfied 5 fold cross validation stream, and saves and stores a final LDA model on the entire provided dataset.


### Configuration

All hyperparameters can be configured in `config/config.yaml`. This file also contains paths to the epoched EEG data and the saved model weights.

### Simulated Stream

The `example` folder contains the simulated stream used to test the realtime pipeline.

