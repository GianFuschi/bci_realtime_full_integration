'''beep sound for Linux'''

import subprocess

def Beep(freq, ms):
    subprocess.run(["play", "-nq", "synth", str(ms / 1000), "sine", str(freq)])

Beep(freq=1000, ms=500)