from typing import Literal

from pydantic import BaseModel, Field
from uuid import UUID


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


class OllamaResponse(BaseModel):
    classification: Literal["private", "topic_based"]
    confidence: float = Field(0.0, ge=0, le=1)
    topics: list[str] = Field(default_factory=list)
    dominant_topic: str | None = None
    privacy_signals: list[str] = Field(default_factory=list)
    sentiment: Literal["neutral", "positive", "negative"] = "neutral"
    participants_count: int | None = None


class ClassifiedSegment(OllamaResponse):
    session_id: str
    total_duration_s: float
    models_used: list[str] = Field(default_factory=list)
