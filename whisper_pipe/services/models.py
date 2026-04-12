from uuid import UUID
from pydantic import BaseModel, Field, field_validator
import base64


class AudioChunk(BaseModel):
    chunk_id: UUID
    session_id: str = Field(..., min_length=1)
    timestamp_start: float = Field(..., description="Starting timestamp in ms")
    timestamp_end: float = Field(..., description="Ending timestamp in ms")
    sample_rate: int = Field(..., gt=0)
    duration_ms: float = Field(..., gt=0)
    audio_data: str = Field(..., description="Base64 encoded audio data")
    vad_cut: bool = Field(
        False,
        description="Indicates if the chunk was cut by VAD or by max segment length",
    )

    @field_validator("audio_data")
    @classmethod
    def validate_base64_audio(cls, v: str) -> str:
        try:
            base64.b64decode(v[:16], validate=True)
            return v
        except Exception:
            raise ValueError("audio_data must be a valid Base64 encoded string")

    @field_validator("timestamp_end")
    @classmethod
    def validate_timestamps(cls, v: float, info) -> float:
        if "timestamp_start" in info.data and v < info.data["timestamp_start"]:
            raise ValueError("timestamp_end cannot be earlier than timestamp_start")
        return v


class TranscribedChunk(BaseModel):
    record_id: UUID
    chunk_id: UUID
    session_id: str = Field(..., min_length=1)
    speaker_id: str | None = Field(
        None, description="Optional speaker ID if diarization is performed"
    )
    text: str = Field(..., description="Transcribed text for the audio chunk")
    language: str = Field(..., description="Detected language of the transcribed text")
    timestamp_start: float = Field(..., description="Starting timestamp in ms")
    timestamp_end: float = Field(..., description="Ending timestamp in ms")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score of the transcription"
    )
    words: list[dict] = Field(
        default_factory=list,
        description="List of word-level details including word text, start time, end time, and probability",
    )
    models_used: list[str] = Field(
        default_factory=list,
        description="List of models used for transcription and diarization",
    )
