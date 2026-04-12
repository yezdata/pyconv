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
        diarization: bool,
        whisper_model_name: str,
        diarization_model_name: str,
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

        if diarization:
            self.diarization_model_name = diarization_model_name

            self.speaker_registry = {}

            self.diarization_model = Pipeline.from_pretrained(
                diarization_model_name,
                # use_auth_token="YOUR HF TOKEN",
            ).to(torch.device(self.device))

    def _process_audio_payload(self, base64_audio: str) -> np.ndarray:
        audio_bytes = base64.b64decode(base64_audio)
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        return audio_int16.astype(np.float32) / 32768.0

    def diarize(self, audio_buffer, sample_rate):
        # TODO threshold for cos similarity
        # TODO improve precision for already stroed embeddings -> mean
        diarization = self.diarization_model(audio_buffer, sample_rate=sample_rate)
        print(diarization)

        diarize_segments = []
        diarization_list = list(diarization.itertracks(yield_label=True))

        for turn, _, speaker in diarization_list:
            diarize_segments.append(
                {"start": turn.start, "end": turn.end, "speaker": speaker}
            )
        unique_speakers = {speaker for _, _, speaker in diarization_list}
        detected_num_speakers = len(unique_speakers)

        if len(self.speaker_registry) == 0:
            # self.speaker_registry["SPEAKER_00"] =
            return "SPEAKER_00"

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

        # OPTIONAL DIARIZATION
        if hasattr(self, "diarization_model"):
            diarization_result = self.diarize(audio_data, sample_rate=chunk.sample_rate)
        else:
            diarization_result = None

        result = TranscribedChunk(
            record_id=uuid4(),
            chunk_id=chunk.chunk_id,
            session_id=chunk.session_id,
            speaker_id=diarization_result,
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
            + (
                [f"pyannote-{self.diarization_model_name}"]
                if diarization_result
                else []
            ),
        )

        self.history_list.extend(diff_text.split())
        self.history_list = self.history_list[-self.max_history_len :]

        return result
