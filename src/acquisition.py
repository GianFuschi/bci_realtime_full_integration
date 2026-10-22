from pylsl import StreamInlet, resolve_streams

class EEGStream:
    def __init__(self, step_size=1):
        # streams = resolve_streams("type", "EEG")
        streams = resolve_streams(1)
        self.inlet = StreamInlet(streams[0])
        self.step_size = step_size

    def pull(self):
        sample, _ = self.inlet.pull_sample()
        return sample

    def pull_chunk(self):
        samples, timestamps = self.inlet.pull_chunk(timeout=0.001, max_samples=self.step_size) #Time out to not return too many empty chunks
        return samples, timestamps
