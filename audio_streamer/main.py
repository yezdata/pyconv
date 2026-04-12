import asyncio
from loguru import logger
import aiofiles
import httpx
import torch

from config import LOG_PATH, LOG_LEVEL, setup_logging
from audio_cfg import AudioConfig
from services.load_audio import load_and_normalize
from services.process_audio import get_speech_segments

setup_logging()
logger.info(f"Log level set to: {LOG_LEVEL}")


async def sender_worker(queue: asyncio.Queue, client: httpx.AsyncClient):
    while True:
        chunk = await queue.get()
        try:
            await client.post(
                "http://localhost:8000/ingest", json=chunk.model_dump(mode="json")
            )
            pass
        except Exception:
            logger.exception(f"Network error")
        finally:
            queue.task_done()


async def main(audio_file: str):
    audio_cfg = AudioConfig(
        target_sample_rate=16000,
        load_chunk_sec=0.032,
        max_segment_length_sec=3.0,
        silence_limit_sec=0.256,
        overlap_sec=0.5,
    )

    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
    )
    vad_iterator = utils[3](
        model=model,
        sampling_rate=audio_cfg.target_sample_rate,
        min_silence_duration_ms=int(audio_cfg.silence_limit_sec * 1000),
    )

    queue = asyncio.Queue(maxsize=100)

    audio_gen = load_and_normalize(audio_file, audio_cfg)
    async with httpx.AsyncClient() as client:
        sender_task = asyncio.create_task(sender_worker(queue, client))

        async for audio_chunk in get_speech_segments(
            audio_gen, vad_iterator, audio_cfg
        ):
            async with aiofiles.open(LOG_PATH, "a", encoding="utf-8") as f:
                await f.write(audio_chunk.model_dump_json() + "\n")

            await queue.put(audio_chunk)

        await queue.join()
        sender_task.cancel()


if __name__ == "__main__":
    asyncio.run(main("data/private_02.mp3"))
