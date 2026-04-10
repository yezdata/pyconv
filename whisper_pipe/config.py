from loguru import logger
import sys
import os
from dotenv import load_dotenv


load_dotenv()


# LOGGING
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logger.info(f"Log level set to: {LOG_LEVEL}")


def setup_logging() -> None:
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=LOG_LEVEL,
        enqueue=True,
    )
