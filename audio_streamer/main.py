import asyncio
import httpx
import torch

from config import AudioConfig
from services.load_audio import load_and_normalize
from services.process_audio import get_speech_segments

from my_dev_utils.time_process import convert_ms_timestamp

async def sender_worker(queue: asyncio.Queue, client: httpx.AsyncClient):
    while True:
        payload = await queue.get()
        try:
            await client.post("http://localhost:8000/ingest", json=payload)
            pass
        except Exception as e:
            print(f"Network error: {e}")
        finally:
            queue.task_done()

async def main():
    audio_cfg = AudioConfig(target_sample_rate=16000, load_chunk_sec=0.032, max_segment_length_sec=2.0, silence_limit_sec=0.256, overlap_sec=0.5)

    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True)
    vad_iterator = utils[3](
        model=model,
        sampling_rate=audio_cfg.target_sample_rate,
        min_silence_duration_ms=int(audio_cfg.silence_limit_sec * 1000),
    )
    # vad_iterator = EnergyVAD(
    #     sample_rate=target_sample_rate,
    #     frame_length=int(self.load_chunk_sec * 1000),
    #     frame_shift=int(self.load_chunk_sec * 1000),
    #     energy_threshold=0.05,
    #     pre_emphasis=0.95,
    # )
    queue = asyncio.Queue(maxsize=100)

    audio_gen = load_and_normalize("data/harvard.wav", audio_cfg)
    async with httpx.AsyncClient() as client:
        sender_task = asyncio.create_task(sender_worker(queue, client))

        async for chunk_payload in get_speech_segments(audio_gen, vad_iterator, audio_cfg):
            print(convert_ms_timestamp(chunk_payload["timestamp_start"]), convert_ms_timestamp(chunk_payload["timestamp_end"]), chunk_payload["duration_ms"])
            await queue.put(chunk_payload)

        await queue.join()
        sender_task.cancel()



if __name__ == "__main__":    
    asyncio.run(main())