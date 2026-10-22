from pylsl import StreamInlet, resolve_streams

class EEGStream:
    def __init__(self):
        # streams = resolve_streams("type", "EEG")
        streams = resolve_streams(1)
        self.inlet = StreamInlet(streams[0])

    def pull(self):
        sample, _ = self.inlet.pull_sample()
        return sample

    def pull_chunk(self):
        samples, timestamps = self.inlet.pull_chunk()
        return samples, timestamps
