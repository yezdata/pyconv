import asyncio
from fastapi import FastAPI
import numpy as np
from loguru import logger

from services.transcriber import Transcriber
from services.models import AudioChunkPayload
from config import setup_logging

from my_dev_utils.time_process import convert_ms_timestamp

setup_logging()
app = FastAPI()
audio_queue = asyncio.Queue(maxsize=50)


import base64

def process_audio_payload(base64_audio: str) -> np.ndarray:
    audio_bytes = base64.b64decode(base64_audio)
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    return audio_int16.astype(np.float32) / 32768.0


async def transcription_worker() -> None:
    try:
        transcriber = Transcriber(model_name="large-v3-turbo")
    except Exception:
        logger.exception(f"FATAL: Transcriber failed to initialize")
        return

    loop = asyncio.get_event_loop()

    while True:
        chunk = await audio_queue.get()
        logger.debug(f"Processing Chunk: start:{convert_ms_timestamp(chunk.timestamp_start)} end:{convert_ms_timestamp(chunk.timestamp_end)}, duration: {chunk.duration_ms} ms")
        audio_data = process_audio_payload(chunk.audio_data)

        transcribed_chunk = await loop.run_in_executor(
            None, 
            transcriber.speech_to_text, 
            audio_data, 
            chunk.timestamp_start,
            chunk.vad_cut
        )

        if transcribed_chunk:
            logger.info(f"[{convert_ms_timestamp(chunk.timestamp_start)}] {transcribed_chunk['text']}")

        audio_queue.task_done()


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(transcription_worker())

@app.post("/ingest")
async def ingest(chunk: AudioChunkPayload):
    await audio_queue.put(chunk)
    return {"status": "accepted"}


if __name__ == "__main__":    
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
