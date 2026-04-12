import asyncio

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from loguru import logger
import aiofiles
import redis.asyncio as redis

from services.transcriber import Transcriber
from services.models import AudioChunk
from config import REDIS_URL, LOG_LEVEL, REDIS_LIST_NAME, HF_HOME, LOG_PATH, setup_logging


setup_logging()
logger.info(f"Log level set to: {LOG_LEVEL}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    audio_queue = asyncio.Queue(maxsize=150)

    app.state.redis_client = redis_client
    app.state.audio_queue = audio_queue

    worker_task = asyncio.create_task(transcription_worker(audio_queue, redis_client))

    yield

    # --- SHUTDOWN ---
    logger.info("Shutting down...")
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        logger.info("Worker task cancelled successfully")
    await redis_client.aclose()


app = FastAPI(lifespan=lifespan)


async def transcription_worker(
    audio_queue: asyncio.Queue, redis_client: redis.Redis
) -> None:
    try:
        transcriber = Transcriber(
            diarization=False,
            whisper_model_name="large-v3-turbo",
            # speechbrain/spkrec-ecapa-voxceleb
            diarization_model_name="pyannote/speaker-diarization-3.1",
            max_context_len=50,
            download_root=HF_HOME,
        )
    except Exception:
        logger.exception(f"FATAL: Transcriber failed to initialize")
        return

    loop = asyncio.get_running_loop()

    while True:
        chunk = await audio_queue.get()

        try:
            transcribed_chunk = await loop.run_in_executor(
                None, transcriber.speech_to_text, chunk
            )

            if transcribed_chunk:
                logger.info(transcribed_chunk)

                data = transcribed_chunk.model_dump_json()

                async with aiofiles.open(LOG_PATH, "a", encoding="utf-8") as f:
                    await f.write(data + "\n")

                while await redis_client.llen(REDIS_LIST_NAME) >= 100:
                    logger.warning(
                        "Redis queue is full, waiting for consumer to catch up..."
                    )
                    await asyncio.sleep(0.5)

                await redis_client.rpush(REDIS_LIST_NAME, data)

        except Exception:
            logger.exception("Error occurred while handling transcribed chunk")
        finally:
            audio_queue.task_done()


@app.post("/ingest")
async def ingest(chunk: AudioChunk, request: Request):
    audio_queue: asyncio.Queue = request.app.state.audio_queue

    try:
        await audio_queue.put(chunk)
        return {"status": "accepted"}
    except Exception as e:
        logger.error(f"Ingest failed: {e}")
        return {"status": "error"}, 500


@app.get("/health")
async def health_check(request: Request):
    r = request.app.state.redis_client
    q = request.app.state.audio_queue

    try:
        redis_len = await r.llen(REDIS_LIST_NAME)
        return {
            "status": "ok",
            "internal_asyncio_queue_size": q.qsize(),
            "redis_queue_size": redis_len,
            "redis_connected": await r.ping(timeout=1),
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}, 500


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
