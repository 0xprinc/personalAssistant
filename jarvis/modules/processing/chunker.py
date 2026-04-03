"""Chunker Module — ~50-word memory chunk splitter with carry-over buffer.

Rules:
  - Utterances < 3 words are discarded entirely (do not add to carry-over).
  - Fragments < 10 words are held in a carry-over buffer and prepended to the
    next incoming transcript (not emitted as a standalone chunk).
  - A chunk is emitted when the accumulated word count >= 50.
  - On segment end, remaining accumulated text is emitted only if >= 10 words.
"""
import re
from jarvis.infra.logger import Logger


class Chunker:
    DISCARD_THRESHOLD = 3   # utterings under this word count are silently dropped
    CARRY_THRESHOLD = 10    # fragments under this count are carried over
    TARGET_WORDS = 50       # target minimum words per emitted chunk

    def __init__(self):
        self._carry: str = ""   # carry-over buffer across split() calls

    def split(self, clean_text: str, start_ms: int, end_ms: int) -> list[dict]:
        """Split a clean transcript into memory chunks.

        Args:
            clean_text:  The cleaned transcript text for this speech segment.
            start_ms:    Segment start timestamp (unix ms).
            end_ms:      Segment end timestamp (unix ms).

        Returns:
            List of chunk dicts with keys: chunk_text, timestamp_start,
            timestamp_end.  May be empty if the text is too short.
        """
        if not clean_text or not clean_text.strip():
            return []

        # Count words in the raw utterance (before prepending carry-over)
        raw_words = len(clean_text.split())

        # 1. Discard utterances under 3 words entirely
        if raw_words < self.DISCARD_THRESHOLD:
            Logger.log(
                "DEBUG", "chunker",
                f"Discarding short utterance ({raw_words} words): '{clean_text}'"
            )
            return []

        # Prepend any carry-over from previous call
        combined = (self._carry + " " + clean_text).strip() if self._carry else clean_text
        self._carry = ""

        # 2. Split into sentences for natural boundaries
        sentences = re.split(r'(?<=[.!?]) +', combined)

        chunks: list[dict] = []
        current_sentences: list[str] = []
        current_word_count = 0

        for sentence in sentences:
            words = len(sentence.split())
            if current_word_count + words >= self.TARGET_WORDS and current_sentences:
                # Emit the accumulated chunk
                chunk_text = " ".join(current_sentences)
                chunks.append({
                    "chunk_text": chunk_text,
                    "timestamp_start": start_ms,
                    "timestamp_end": end_ms,
                })
                current_sentences = [sentence]
                current_word_count = words
            else:
                current_sentences.append(sentence)
                current_word_count += words

        # Handle remaining accumulated text
        if current_sentences:
            remaining_text = " ".join(current_sentences)
            remaining_words = len(remaining_text.split())

            if remaining_words >= self.TARGET_WORDS:
                # Full chunk — emit it
                chunks.append({
                    "chunk_text": remaining_text,
                    "timestamp_start": start_ms,
                    "timestamp_end": end_ms,
                })
            elif remaining_words >= self.CARRY_THRESHOLD:
                # Segment ended and we have enough words — emit as final chunk
                chunks.append({
                    "chunk_text": remaining_text,
                    "timestamp_start": start_ms,
                    "timestamp_end": end_ms,
                })
            else:
                # Too short — carry over to next call
                self._carry = remaining_text
                Logger.log(
                    "DEBUG", "chunker",
                    f"Carrying over {remaining_words}-word fragment: '{remaining_text}'"
                )

        Logger.log(
            "INFO", "chunker",
            f"Split transcript into {len(chunks)} memory chunks "
            f"(carry-over: {len(self._carry.split()) if self._carry else 0} words)"
        )
        return chunks
