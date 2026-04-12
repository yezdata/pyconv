import re
from difflib import SequenceMatcher
from loguru import logger


def clean_token(t):
    return re.sub(r"[^\w]", "", t.lower())


def get_diff_text(
    history_list: list, new_text: str, lookback_words: int = 8, min_match_len: int = 2
) -> tuple[str, int]:

    history_comp = [clean_token(w) for w in history_list[-lookback_words:]]

    original_words = new_text.strip().split()
    candidate_comp = [clean_token(w) for w in original_words]

    if not history_comp:
        return " ".join(original_words), 0

    matcher = SequenceMatcher(None, history_comp, candidate_comp)
    best_match = None

    logger.debug(f"History: '{history_comp}' | Candidate: '{candidate_comp}'")

    for m in matcher.get_matching_blocks():
        logger.debug(
            f"Match: a={m.a}, b={m.b}, size={m.size}, text='{candidate_comp[m.b : m.b + m.size]}'"
        )

        if m.size >= min_match_len and m.b <= 2:
            if best_match is None or m.size > best_match.size:
                best_match = m

    if best_match:
        drop_index = best_match.b + best_match.size
        diff_words = original_words[drop_index:]
        return " ".join(diff_words), drop_index

    return " ".join(original_words), 0
