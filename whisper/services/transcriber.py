import sounddevice as sd
import time

class Transcriber:
    def __init__(self, model):
        self.model = model

    def catch_audio(self, buffer):
        print(buffer)
        print(int(buffer.shape[0]) / 16000)
        sd.play(buffer, samplerate=16000)
        time.sleep(3.5)