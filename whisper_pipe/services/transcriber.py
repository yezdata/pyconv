from faster_whisper import WhisperModel
from uuid import uuid4
import numpy as np
from pyannote.audio import Pipeline
import torch
from loguru import logger
import base64

from services.process_text import get_diff_text
from services.models import AudioChunk, TranscribedChunk


class Transcriber:
    def __init__(
        self,
        whisper_model_name: str,
        max_context_len: int,
        download_root: str | None = None,
    ):
        self.history_list = []
        self.max_history_len = max_context_len

        self.transcriber_model_name = whisper_model_name

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = WhisperModel(
            whisper_model_name,
            device=self.device,
            compute_type="int8" if self.device == "cpu" else "float16",
            download_root=download_root,
        )


    def _process_audio_payload(self, base64_audio: str) -> np.ndarray:
        audio_bytes = base64.b64decode(base64_audio)
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        return audio_int16.astype(np.float32) / 32768.0


    def speech_to_text(self, chunk: AudioChunk) -> TranscribedChunk | None:
        options = dict(
            # beam_size=5,
            word_timestamps=True,
            # language_detection_segments=1,
            task="transcribe",
            initial_prompt=" ".join(self.history_list) if self.history_list else None,
        )
        audio_data = self._process_audio_payload(chunk.audio_data)

        try:
            segments, info = self.model.transcribe(audio_data, **options)
        except Exception:
            logger.exception(f"Transcription error")
            return None

        if not segments:
            return None

        all_whisper_words = []

        for s in segments:
            if s.words:
                all_whisper_words.extend(s.words)

        new_text = " ".join([w.word for w in all_whisper_words])
        if not new_text:
            return None

        # DEDUPLICATION
        diff_text, drop_count = get_diff_text(self.history_list, new_text)

        if not diff_text or not diff_text.strip():
            return None

        if not chunk.vad_cut:
            diff_text = diff_text.rstrip(". ")

        final_words_data = all_whisper_words[drop_count:]
        confidence = sum(w.probability for w in final_words_data) / len(
            final_words_data
        )


        result = TranscribedChunk(
            record_id=uuid4(),
            chunk_id=chunk.chunk_id,
            session_id=chunk.session_id,
            speaker_id=None,  # Placeholder, to be filled if diarization is implemented
            text=diff_text,
            language=info.language,
            timestamp_start=chunk.timestamp_start + (final_words_data[0].start * 1000),
            timestamp_end=chunk.timestamp_start + (final_words_data[-1].end * 1000),
            confidence=confidence,
            words=[
                {
                    "word": w.word,
                    "start": float(w.start),
                    "end": float(w.end),
                    "probability": w.probability,
                }
                for w in final_words_data
            ],
            models_used=[f"faster-whisper-{self.transcriber_model_name}"]
        )

        self.history_list.extend(diff_text.split())
        self.history_list = self.history_list[-self.max_history_len :]

        return result
