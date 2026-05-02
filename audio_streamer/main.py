import asyncio
import argparse
import sys
from loguru import logger
import aiofiles
import httpx
import torch

from config import LOG_PATH, LOG_LEVEL, TORCH_HOME, INGEST_URL, setup_logging
from audio_cfg import AudioConfig
from services.load_audio import load_and_normalize
from services.process_audio import get_speech_segments



async def sender_worker(queue: asyncio.Queue, client: httpx.AsyncClient):
    while True:
        chunk = await queue.get()
        if chunk is None:
            queue.task_done()
            break
        try:
            await client.post(
                INGEST_URL, json=chunk.model_dump(mode="json")
            )
        except Exception:
            logger.exception(f"Network error")
        finally:
            queue.task_done()


async def main():
    parser = argparse.ArgumentParser(description="Audio processing CLI tool")
    parser.add_argument(
        "--file", 
        type=str, 
        required=True, 
        help="Path to the source audio file"
    )
    parser.add_argument(
        "--session", 
        type=str, 
        required=True, 
        help="Unique session identifier"
    )
    args = parser.parse_args()


    setup_logging()
    logger.info(f"Log level set to: {LOG_LEVEL}")


    audio_cfg = AudioConfig(
        target_sample_rate=16000,
        load_chunk_sec=0.032,
        max_segment_length_sec=3.0,
        silence_limit_sec=0.256,
        overlap_sec=0.5,
    )

    try:
        torch.hub.set_dir(TORCH_HOME)
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
        )
        vad_iterator = utils[3](
            model=model,
            sampling_rate=audio_cfg.target_sample_rate,
            min_silence_duration_ms=int(audio_cfg.silence_limit_sec * 1000),
        )
    except Exception as e:
        logger.critical(f"Failed to load VAD model {e}")
        sys.exit(1)

    queue = asyncio.Queue(maxsize=100)

    try:
        audio_gen = load_and_normalize(args.file, audio_cfg)
    except FileNotFoundError:
        logger.exception(f"Audio file not found: {args.file}")
        return

    async with httpx.AsyncClient() as client:
        sender_task = asyncio.create_task(sender_worker(queue, client))

        try:
            async with aiofiles.open(LOG_PATH, "a", encoding="utf-8") as f:
                async for audio_chunk in get_speech_segments(
                    audio_gen, vad_iterator, args.session, audio_cfg
                ):
                    await f.write(audio_chunk.model_dump_json() + "\n")

                    await queue.put(audio_chunk)
        except Exception as e:
            logger.exception(f"Error during processing: {e}")
        finally:
            await queue.put(None)
            await queue.join()
            await sender_task



if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
