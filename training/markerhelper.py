import time
import random
from pylsl import StreamInfo, StreamOutlet
# use this instead of winsound because winsound only works on Windows. 
# on Linux:
# sudo apt install sox
# sudo apt install libsox-fmt-all
from beep import Beep

def create_marker_outlet():
    info = StreamInfo(
        name='MotorImageryMarkers',
        type='Markers',
        channel_count=1,
        nominal_srate=0,  # Irregular sampling rate for event-based triggers
        channel_format='string',
    )
    return StreamOutlet(info)

def run_motor_imagery_paradigm(outlet, total_trials=1):

    conditions = ['LEFT', 'RIGHT']
    
    for trial_num in range(1, total_trials + 1):
        # Randomly select task condition
        condition = random.choice(conditions)
        # use perf_counter for precision
        start_time = time.perf_counter()

        # 1. FIXATION 
        print("X")
        time.sleep(2.000)

        #2. WARNING BEEP
        Beep(freq=1000, ms= 500)
        time.sleep(0.50)

        # 3. CUE + Marker
        while time.perf_counter() - start_time < 3.000:
            time.sleep(0.0001)  # wait untilt time is exactly 3 s
        print(f"CUE_{condition}")
        # outlet.push_sample([f"CUE_{condition}"])
        time.sleep(0.250) # wait till 3.25 s

        # 4. record Motor Imagery + Marker
        # print("time before recording is", time.perf_counter() - start_time)
        outlet.push_sample([f"IMAGERY_{condition}"])
        while time.perf_counter() - start_time < 4.250:
            time.sleep(0.001)
        # print("time after recording is", time.perf_counter() - start_time)

        # 5. keep until 8 s
        while time.perf_counter() - start_time < 8.000:
            time.sleep(0.0001)  # wait until time is exactly 8 s

        print(f"end of trial {trial_num}")

        # 6. INTER-TRIAL INTERVAL
        interval = random.uniform(0.5, 2.5)
        time.sleep(interval)
       
    print("end of experiment")


if __name__ == '__main__':
    # Initialize the marker stream first
    marker_outlet = create_marker_outlet()
    
    print("LSL Marker Outlet initialized: 'MotorImageryMarkers'")
    
    # Linking to LabRecorder
    print("0. Start Unicorn Recorder and run the paradigm code.")
    print("\n1. Open LSL LabRecorder.")
    print("2. Click 'Update' and ensure both 'Unicorn' and 'MotorImageryMarkers' are visible.")
    print("3. Select both streams and click 'Start' to begin recording.")
    
    input("\n--> Press ENTER in this terminal once LabRecorder is recording to start the task...")

    # Start Experiment
    run_motor_imagery_paradigm(marker_outlet, total_trials=30)