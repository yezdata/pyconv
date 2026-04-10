from base64 import b64encode
from uuid import uuid4


from config import AudioConfig


def create_payload(buffer: list[bytes], timestamp_start: float, timestamp_end: float, duration_ms: float, cfg: AudioConfig, vad_cut: bool) -> dict:
        flatten_buffer = b''.join(buffer)
        encoded_string = b64encode(flatten_buffer).decode('utf-8')

        out = {
            "chunk_id": str(uuid4()),
            "session_id": "manual-session-001",
            "timestamp_start": timestamp_start,
            "timestamp_end": timestamp_end,
            "sample_rate": cfg.target_sample_rate,
            "duration_ms": duration_ms,
            "audio_data": encoded_string,
            "vad_cut": vad_cut
        }
        return out
        
