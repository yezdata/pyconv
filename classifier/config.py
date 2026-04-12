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

OLLAMA_URL = os.getenv("OLLAMA_URL")
if not OLLAMA_URL:
    logger.critical("OLLAMA_URL is not set in environment variables")
    sys.exit(1)

OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "qwen3.5:9b-q4_K_M")

LOG_PATH = os.getenv("LOG_PATH", "../.logs/classifier/classifications.jsonl")

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
