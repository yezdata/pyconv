from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions
from uuid import uuid4
import re
import numpy as np
from pyannote.audio import Pipeline
import torch
from difflib import SequenceMatcher
from loguru import logger



class Transcriber:
    def __init__(self, model_name: str = "large-v3-turbo"):
        self.history_list = []
        self.max_history_len = 50

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = WhisperModel(
            model_name,
            device=self.device,
            compute_type="int8" if self.device == "cpu" else "float16"
        )

        # self.diarization_model = Pipeline.from_pretrained(
        #     "pyannote/speaker-diarization-3.1",
        #     use_auth_token="YOUR HF TOKEN",
        # ).to(torch.device(self.device))


    def _get_diff_text(self, new_text: str, lookback_words: int = 30, min_match_len: int = 2) -> tuple[str, int]:
        def clean_token(t):
            return re.sub(r'[^\w]', '', t.lower())

        history_comp = [clean_token(w) for w in self.history_list[-lookback_words:]]

        original_words = new_text.strip().split()
        candidate_comp = [clean_token(w) for w in original_words]
        
        if not history_comp:
            return " ".join(original_words), 0

        matcher = SequenceMatcher(None, history_comp, candidate_comp)
        best_match = None

        logger.debug(f"History: '{history_comp}' | Candidate: '{candidate_comp}'")

        for m in matcher.get_matching_blocks():
            logger.debug(f"Match: a={m.a}, b={m.b}, size={m.size}, text='{candidate_comp[m.b:m.b+m.size]}'")
            
            if m.size >= min_match_len and m.b <= 2: 
                if best_match is None or m.size > best_match.size:
                    best_match = m

        if best_match:
            drop_index = best_match.b + best_match.size
            diff_words = original_words[drop_index:]
            return " ".join(diff_words), drop_index
        
        return " ".join(original_words), 0



    def speech_to_text(self, audio_data: np.ndarray, chunk_start_ms: float, vad_cut: bool) -> dict | None:
        options = dict(
            language="en",
            # beam_size=5,
            # vad_filter=True,
            # vad_parameters=VadOptions(
            #     max_speech_duration_s=self.model.feature_extractor.chunk_length,
            #     min_speech_duration_ms=100,
            #     speech_pad_ms=100,
            #     threshold=0.25,
            #     neg_threshold=0.2,
            # ),
            word_timestamps=True,
            # language_detection_segments=1,
            task = "transcribe",
            initial_prompt=" ".join(self.history_list) if self.history_list else None
        )
        try:
            segments, info = self.model.transcribe(audio_data, **options)
        except Exception as e:
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
        diff_text, drop_count = self._get_diff_text(new_text)

        if not diff_text or not diff_text.strip():
            return None

        if not vad_cut:
            diff_text = diff_text.rstrip('. ')

        final_words_data = all_whisper_words[drop_count:]
        confidence = sum(w.probability for w in final_words_data) / len(final_words_data)

        result = {
            "record_id": str(uuid4()),
            "text": diff_text,
            "confidence": confidence,
            "timestamp_start": chunk_start_ms + (final_words_data[0].start * 1000),
            "timestamp_end": chunk_start_ms + (final_words_data[-1].end * 1000),
            "words": [
                {
                    "word": w.word,
                    "start": float(w.start),
                    "end": float(w.end),
                    "probability": w.probability,
                }
                for w in final_words_data
            ]
        }

        self.history_list.extend(diff_text.split())
        self.history_list = self.history_list[-self.max_history_len:]

        return result



    # def diarize(self, audio_buffer, sample_rate):
    #     diarization = self.diarization_model(audio_buffer, sample_rate=sample_rate)
    #     print(diarization)

    #     diarize_segments = []
    #     diarization_list = list(diarization.itertracks(yield_label=True))

    #     for turn, _, speaker in diarization_list:
    #         diarize_segments.append(
    #             {"start": turn.start, "end": turn.end, "speaker": speaker}
    #         )
    #     unique_speakers = {speaker for _, _, speaker in diarization_list}
    #     detected_num_speakers = len(unique_speakers)

