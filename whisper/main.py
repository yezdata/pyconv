from pathlib import Path

from services.transcriber import Transcriber
from services.loader import AudioStreamer


def main():
    audio_streamer = AudioStreamer(load_chunk_sec=0.032, max_buffer_size_sec=5.0, overlap_sec=1.0, silence_limit=8)
    transcriber = Transcriber(model=None)

    for audio_buffer in audio_streamer.chunk_generator(file_path="data/harvard.wav"):
        transcriber.catch_audio(audio_buffer)



if __name__ == "__main__":    
    main()