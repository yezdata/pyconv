import soundfile as sf
import sounddevice as sd
import time
import math
import numpy as np
from vad import EnergyVAD
from collections.abc import Generator
from scipy import signal
import numpy as np


class AudioStreamer:
    def __init__(self, silence_limit: int, load_chunk_sec: float, max_buffer_size_sec: float, overlap_sec: float, target_sample_rate: int = 16000) -> None:
        self.target_sample_rate = target_sample_rate
        self.load_chunk_sec = load_chunk_sec
        self.max_buffer_size_sec = max_buffer_size_sec
        self.silence_limit = silence_limit

        self.overlap_chunks = int(math.ceil(overlap_sec / load_chunk_sec))

        self.vad_iterator = EnergyVAD(
            sample_rate=target_sample_rate,
            frame_length=int(self.load_chunk_sec * 1000),
            frame_shift=int(self.load_chunk_sec * 1000),
            energy_threshold=0.05,
            pre_emphasis=0.95,
        )


    def _resample(self, data):
        return signal.resample_poly(data, self.target_sample_rate, self.file_sample_rate)

    def _process_to_mono(self, indata):
        if indata.ndim == 1:
            return indata
        elif indata.ndim == 2:
            return np.mean(indata, axis=1)
        else:
            raise ValueError("Unsupported audio data shape: {}".format(indata.shape))
        
    def _stream_file(self, file_path,) -> Generator[np.ndarray, None, None]:
        with sf.SoundFile(file_path) as f:
            self.file_sample_rate = f.samplerate

            samples_to_read = int(self.load_chunk_sec * self.file_sample_rate)

            while f.tell() < len(f):
                data = f.read(samples_to_read, dtype='float32')
                mono = self._process_to_mono(data)
                resampled = self._resample(mono)

                time.sleep(self.load_chunk_sec)
                yield resampled


    def chunk_generator(self, file_path) -> Generator[np.ndarray, None, None]:
        buffer = []
        is_speaking = False

        silence_count = 0


        for chunk in self._stream_file(file_path):
            speech_detected = self.vad_iterator(chunk).any()
            print(f"Speech Detected: {speech_detected}, Buffer Size: {len(buffer)}, Silence Count: {silence_count}")

            if speech_detected and not is_speaking:
                silence_count = 0
                buffer.append(chunk)
                is_speaking = True

            elif speech_detected and is_speaking:
                silence_count = 0
                buffer.append(chunk)

            elif not speech_detected and is_speaking:
                silence_count += 1
                buffer.append(chunk)

                # how many silent chunks before cutting and yielding the buffer (TODO - make this in seconds)
                if silence_count >= 8:
                    yield np.concatenate(buffer)
                    buffer = buffer[-self.overlap_chunks:]
                    is_speaking = False

            if (len(buffer) * self.load_chunk_sec) >= self.max_buffer_size_sec:
                yield np.concatenate(buffer)
                buffer = buffer[-self.overlap_chunks:]
                is_speaking = False
