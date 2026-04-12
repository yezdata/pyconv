import asyncio

from loguru import logger
from ollama import RequestError
import aiofiles
import httpx
import redis.asyncio as redis

from services.ollama_classifier import OllamaClassifier
from services.models import ClassifiedSegment, TranscribedChunk
from config import (
    LOG_LEVEL,
    OLLAMA_URL,
    OLLAMA_MODEL_NAME,
    REDIS_URL,
    REDIS_LIST_NAME,
    LOG_PATH,
    setup_logging,
)


setup_logging()
logger.info(f"Log level set to: {LOG_LEVEL}")


async def chunk_producer(chunk_queue: asyncio.Queue, redis_client: redis.Redis) -> None:
    while True:
        try:
            result = await redis_client.blpop(REDIS_LIST_NAME, timeout=15)
            if result is None:
                logger.debug("No new chunks in Redis, waiting...")
                continue

            data_raw = result[1]
            chunk_data = TranscribedChunk.model_validate_json(data_raw)
            await chunk_queue.put(chunk_data)
        except Exception:
            logger.exception("Error occurred while fetching chunk from Redis")
            await asyncio.sleep(1)


async def classification_worker(
    chunk_queue: asyncio.Queue, chunk_batch_size: int
) -> None:
    classifier = OllamaClassifier(
        host_url=OLLAMA_URL, model_name=OLLAMA_MODEL_NAME, max_context_batch_count=5
    )

    while True:
        chunk = await chunk_queue.get()
        batch = [chunk]

        while len(batch) < chunk_batch_size:
            try:
                batch.append(chunk_queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        while True:
            try:
                combined_text = " ".join([c.text for c in batch])
                logger.debug(f"Classifying window with {len(batch)} new chunks.")

                ollama_data = await classifier.classify_text(combined_text)

                if ollama_data:
                    logger.info(ollama_data)

                    classification_segment = ClassifiedSegment(
                        **ollama_data.model_dump(),
                        session_id=batch[-1].session_id,
                        total_duration_s=(
                            batch[-1].timestamp_end - batch[0].timestamp_start
                        )
                        / 1000,
                        models_used=batch[-1].models_used
                        + [f"ollama-{OLLAMA_MODEL_NAME}"],
                    )

                    async with aiofiles.open(LOG_PATH, "a", encoding="utf-8") as f:
                        await f.write(classification_segment.model_dump_json() + "\n")

                    classifier.update_context_history(combined_text)

                break

            except (
                RequestError,
                httpx.ConnectError,
                httpx.RemoteProtocolError,
                ConnectionError,
                TimeoutError,
            ) as e:
                logger.warning(
                    f"Connection to Ollama failed, retrying in 5 seconds... Error: {e}"
                )
                await asyncio.sleep(5)
            except Exception:
                logger.exception(f"Fatal error occurred while classifying chunk")
                break

        for _ in range(len(batch)):
            chunk_queue.task_done()


async def main() -> None:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    chunk_queue = asyncio.Queue(maxsize=150)

    producer_task = asyncio.create_task(chunk_producer(chunk_queue, redis_client))
    worker_task = asyncio.create_task(
        classification_worker(chunk_queue, chunk_batch_size=5)
    )

    try:
        await asyncio.gather(producer_task, worker_task)
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Shutting down consumer service...")
    finally:
        producer_task.cancel()
        worker_task.cancel()
        await asyncio.gather(producer_task, worker_task, return_exceptions=True)

        await redis_client.aclose()
        logger.info("Connections closed.")


if __name__ == "__main__":
    asyncio.run(main())
