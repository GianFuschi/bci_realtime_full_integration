# bci_realtime_full_integration

End-to-end real-time BCI system:

For the realtime system an lsl streamer needs to be running in the background. Then the system.py file can be run to run the full realtime pipeline. When closing this script use ctrl c in the terminal. This creates a keyboard interrupt error which the script recognizes and then properly closes and joins all threads. This does not happen when using ctrl z. 
The scripts for data extraction and training can be found in the training folder.
