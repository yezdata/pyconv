from loguru import logger
import sys
import os
from dotenv import load_dotenv


load_dotenv()


REDIS_URL = os.getenv("REDIS_URL")
if not REDIS_URL:
    logger.critical("REDIS_URL is not set in environment variables")
    sys.exit(1)
REDIS_LIST_NAME = os.getenv("REDIS_LIST_NAME", "transcribed_chunks")

HF_HOME = os.getenv("HF_HOME", "../.models_cache")

LOG_PATH = os.getenv("LOG_PATH", "../.logs/whisper_pipe/transcriptions.jsonl")


# LOGGING
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def setup_logging() -> None:
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=LOG_LEVEL,
        enqueue=True,
    )
