from loguru import logger
import sys
import os


LOG_PATH = "../.logs/audio_streamer/audio_stream.jsonl"
TORCH_HOME = os.getenv("TORCH_HOME", "../.models_cache")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def setup_logging() -> None:
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=LOG_LEVEL,
        enqueue=True,
    )