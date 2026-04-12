import asyncio
import subprocess
import time
from typing import AsyncGenerator
from loguru import logger

from audio_cfg import AudioConfig


async def load_and_normalize(
    input_source: str, cfg: AudioConfig
) -> AsyncGenerator[tuple[bytes, float], None]:
    command = [
        "ffmpeg",
        "-i",
        input_source,
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(cfg.target_sample_rate),
        "-ac",
        "1",
        "-loglevel",
        "error",
        "-",
    ]
    process = await asyncio.create_subprocess_exec(
        *command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    chunk_size_bytes = int(
        cfg.target_sample_rate * cfg.bytes_per_sample * cfg.load_chunk_sec
    )

    loop = asyncio.get_event_loop()
    chunks_loaded = 0
    start_process_time = loop.time()

    # Simulate streaming input
    start_stream_timestamp = time.time() * 1000
    try:
        while True:
            raw_chunk = await process.stdout.read(chunk_size_bytes)

            if not raw_chunk:
                await process.wait()
                exit_code = process.returncode

                if exit_code != 0:
                    err_data = await process.stderr.read(4096)
                    logger.error(
                        f"FFmpeg exited with code {exit_code}: {err_data.decode().strip()}"
                    )
                break

            actual_len = len(raw_chunk)
            if actual_len == 0:
                break

            is_last = False
            if actual_len < chunk_size_bytes:
                padding_size = chunk_size_bytes - actual_len
                raw_chunk += b"\x00" * padding_size

                is_last = True

            # accurate time negating any delays in processing
            current_chunk_start_ts = start_stream_timestamp + (
                chunks_loaded * cfg.load_chunk_sec * 1000
            )

            chunks_loaded += 1

            # simulate real-time recording
            target_time = start_process_time + (chunks_loaded * cfg.load_chunk_sec)
            sleep_delay = target_time - loop.time()
            if sleep_delay > 0:
                await asyncio.sleep(sleep_delay)

            yield raw_chunk, current_chunk_start_ts

            if is_last:
                break

    except Exception:
        logger.exception(f"Error in load_and_normalize")

    finally:
        if process.returncode is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
        await process.wait()
