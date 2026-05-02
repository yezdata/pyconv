import math
import asyncio
from loguru import logger
import numpy as np
from typing import AsyncGenerator
from base64 import b64encode
from uuid import uuid4

from audio_cfg import AudioConfig
from services.models import AudioChunk


async def get_speech_segments(
    audio_iterator: AsyncGenerator[tuple[bytes, float], None],
    vad_iterator,
    session_id: str,
    cfg: AudioConfig,
) -> AsyncGenerator[AudioChunk, None]:
    overlap_chunk_count = int(math.ceil(cfg.overlap_sec / cfg.load_chunk_sec))

    buffer = []
    is_speaking = False
    segment_start_ts = 0.0

    async for chunk, load_chunk_timestamp_start in audio_iterator:
        chunk_array = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        loop = asyncio.get_event_loop()
        vad_out = await loop.run_in_executor(None, vad_iterator, chunk_array)

        logger.debug(
            f"load_chunk_timestamp_start: {load_chunk_timestamp_start}, is_speaking: {is_speaking}, buffer_length: {len(buffer)}"
        )

        if vad_out and "start" in vad_out:
            is_speaking = True
            segment_start_ts = load_chunk_timestamp_start
            buffer = []

        if is_speaking:
            buffer.append(chunk)

        if vad_out and "end" in vad_out:
            duration_ms = len(buffer) * cfg.load_chunk_sec * 1000
            timestamp_end = segment_start_ts + duration_ms

            flatten_buffer = b"".join(buffer)
            encoded_string = b64encode(flatten_buffer).decode("utf-8")

            logger.info(
                f"Emitting segment: start={segment_start_ts}, end={timestamp_end}, duration_ms={duration_ms}, buffer_length={len(buffer)}"
            )

            yield AudioChunk(
                chunk_id=uuid4(),
                session_id=session_id,
                timestamp_start=segment_start_ts,
                timestamp_end=timestamp_end,
                sample_rate=cfg.target_sample_rate,
                duration_ms=duration_ms,
                audio_data=encoded_string,
                vad_cut=True,
            )

            # NO OVERLAP FOR END OF SPEECH, ONLY FOR MAX SEGMENT LENGTH
            buffer = []
            segment_start_ts = timestamp_end
            is_speaking = False

        elif len(buffer) * cfg.load_chunk_sec >= cfg.max_segment_length_sec:
            if buffer:
                duration_ms = len(buffer) * cfg.load_chunk_sec * 1000
                timestamp_end = segment_start_ts + duration_ms

                flatten_buffer = b"".join(buffer)
                encoded_string = b64encode(flatten_buffer).decode("utf-8")

                logger.info(
                    f"Emitting segment (max length): start={segment_start_ts}, end={timestamp_end}, duration_ms={duration_ms}, buffer_length={len(buffer)}"
                )

                yield AudioChunk(
                    chunk_id=uuid4(),
                    session_id=session_id,
                    timestamp_start=segment_start_ts,
                    timestamp_end=timestamp_end,
                    sample_rate=cfg.target_sample_rate,
                    duration_ms=duration_ms,
                    audio_data=encoded_string,
                    vad_cut=False,
                )

                if overlap_chunk_count > 0:
                    buffer = buffer[-overlap_chunk_count:]
                    segment_start_ts = timestamp_end - (
                        overlap_chunk_count * cfg.load_chunk_sec * 1000
                    )
                else:
                    buffer = []
                    segment_start_ts = timestamp_end
