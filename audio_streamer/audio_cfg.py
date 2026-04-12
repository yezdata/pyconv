from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AudioConfig(BaseSettings):
    target_sample_rate: int = Field(default=16000, ge=8000, le=48000)

    load_chunk_sec: float = Field(
        default=0.032, description="Length of each audio chunk to load in seconds"
    )
    max_segment_length_sec: float = 5.0
    silence_limit_sec: float = 0.5
    overlap_sec: float = 1.0
    bytes_per_sample: int = 2

    session_id: str = "manual-session-001"

    model_config = SettingsConfigDict(env_file=".env", env_prefix="AUDIO_")