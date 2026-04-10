import math
import asyncio
import numpy as np
from typing import AsyncGenerator

from config import AudioConfig
from services.format_segments import create_payload

from my_dev_utils.time_process import convert_ms_timestamp


async def get_speech_segments(
    audio_iterator: AsyncGenerator[tuple[bytes, float], None],
    vad_iterator,
    cfg: AudioConfig
) -> AsyncGenerator[dict, None]:
    overlap_chunk_count = int(math.ceil(cfg.overlap_sec / cfg.load_chunk_sec))
    
    buffer = []
    is_speaking = False
    segment_start_ts = 0.0

    async for chunk, load_chunk_timestamp_start in audio_iterator:
        chunk_array = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        loop = asyncio.get_event_loop()
        vad_out = await loop.run_in_executor(None, vad_iterator, chunk_array)

        print(f"load_chunk_timestamp_start: {convert_ms_timestamp(load_chunk_timestamp_start)}, is_speaking: {is_speaking}, buffer_length: {len(buffer)}")

        if vad_out and "start" in vad_out:
            is_speaking = True
            segment_start_ts = load_chunk_timestamp_start
            buffer = []
            print(vad_out)
        
        if is_speaking:
            buffer.append(chunk)

        if vad_out and "end" in vad_out:
            duration_ms = len(buffer) * cfg.load_chunk_sec * 1000
            timestamp_end = segment_start_ts + duration_ms  

            yield create_payload(buffer, segment_start_ts, timestamp_end, duration_ms, cfg, vad_cut=True)

            # NO OVERLAP FOR END OF SPEECH, ONLY FOR MAX SEGMENT LENGTH
            buffer = []
            segment_start_ts = timestamp_end
            is_speaking = False


        elif len(buffer) * cfg.load_chunk_sec >= cfg.max_segment_length_sec:
            if buffer:
                duration_ms = len(buffer) * cfg.load_chunk_sec * 1000
                timestamp_end = segment_start_ts + duration_ms

                yield create_payload(buffer, segment_start_ts, timestamp_end, duration_ms, cfg, vad_cut=False)

                if overlap_chunk_count > 0:
                    buffer = buffer[-overlap_chunk_count:]
                    segment_start_ts = timestamp_end - (overlap_chunk_count * cfg.load_chunk_sec * 1000)
                else:
                    buffer = []
                    segment_start_ts = timestamp_end