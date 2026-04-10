import asyncio
import subprocess
import time
from typing import AsyncGenerator

from config import AudioConfig


async def load_and_normalize(input_source: str, cfg: AudioConfig) -> AsyncGenerator[tuple[bytes, float], None]:
    command = [
        'ffmpeg',
        '-i', input_source,
        '-f', 's16le',
        '-acodec', 'pcm_s16le',
        '-ar', str(cfg.target_sample_rate),
        '-ac', '1',
        '-loglevel', 'quiet',
        '-'
    ]
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    chunk_size_bytes = int(cfg.target_sample_rate * cfg.bytes_per_sample * cfg.load_chunk_sec)

    loop = asyncio.get_event_loop()
    chunks_loaded = 0
    start_process_time = loop.time()

    # Simulate streaming input
    start_stream_timestamp = time.time() * 1000
    try:
        while True:
            raw_chunk = await process.stdout.read(chunk_size_bytes)
            if not raw_chunk and len(raw_chunk) != chunk_size_bytes:
                exit_code = process.returncode
                if exit_code is not None and exit_code != 0:
                    err_data = await process.stderr.read(4096)
                    print(f"FFmpeg exited with code {exit_code}: {err_data.decode().strip()}")
                break

            chunks_loaded += 1

            # accurate time negating any delays in processing 
            current_chunk_start_ts = start_stream_timestamp + (chunks_loaded * cfg.load_chunk_sec * 1000)

            # simulate real-time recording
            target_time = start_process_time + (chunks_loaded * cfg.load_chunk_sec)
            sleep_delay = target_time - loop.time()
            if sleep_delay > 0:
                await asyncio.sleep(sleep_delay)

            # TODO: add Sent 320000 bytes (10.0s) to jsonl logs
            yield raw_chunk, current_chunk_start_ts

            
    except Exception as e:
        print(f"Error in load_and_normalize: {e}")

    finally:
        if process.returncode is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
        await process.wait()