"""Release complete, speakable phrases from a streamed model response.

The chunker is deliberately more conservative than a text UI.  A telephony TTS
continuation that starts with two or three words frequently sounds like the
model is pausing between every word when a later token is delayed.  We prefer a
short clause or sentence, which gives the synthesizer enough audio runway to
bridge ordinary model-token jitter.
"""
import re
import time


class SafeSpeechChunker:
    _NEGATION = re.compile(r"\b(?:not|never|don't|cannot|can't|नहीं|मत)\s*$", re.IGNORECASE)
    _UNSAFE_TAIL = re.compile(r"(?:[$₹€£]\s*\d*|\b\d{1,2}[/-]\d{0,2}|\b\d+[,.]?\d*)$", re.UNICODE)

    def __init__(self, min_chars: int = 48, min_words: int = 7, max_wait_ms: int = 240) -> None:
        self.min_chars, self.min_words, self.max_wait_ms = min_chars, min_words, max_wait_ms
        self.buffer = ""; self.first_token_at: float | None = None

    def push(self, text_delta: str, now: float | None = None) -> list[str]:
        now = time.perf_counter() if now is None else now
        if self.first_token_at is None: self.first_token_at = now
        self.buffer += text_delta
        if self._unsafe_tail() or self._ends_in_negation(): return []
        enough = len(self.buffer) >= self.min_chars or len(self.buffer.split()) >= self.min_words
        boundary = self._last_boundary()
        if boundary and enough: return [self._pop(boundary)]
        # The timer is a latency backstop, not permission to synthesize a
        # token-sized context. It can release only an already phrase-sized
        # prefix, and therefore cannot cause the one-word-at-a-time cadence
        # that prompted this component.
        if enough and (now - self.first_token_at) * 1000 >= self.max_wait_ms:
            word_boundary = self._last_word_boundary()
            if word_boundary and not self._unsafe_tail(word_boundary):
                # Guard: the TEXT we are about to release (not the whole buffer)
                # must itself meet the minimum size threshold.  If we only have a
                # tiny prefix before the word boundary the timer must keep waiting.
                candidate = self.buffer[:word_boundary].strip()
                if len(candidate) >= self.min_chars or len(candidate.split()) >= self.min_words:
                    return [self._pop(word_boundary)]
        return []

    def flush(self) -> list[str]:
        text, self.buffer, self.first_token_at = self.buffer.strip(), "", None
        return [text] if text else []

    def _last_boundary(self) -> int:
        # A provider commonly ends a streamed delta immediately after the
        # period. Treat it as a valid boundary even before the following
        # whitespace arrives; the old expression withheld it until another
        # delta or final flush.
        matches = list(re.finditer(r"[,;:.!?](?:\s+|$)", self.buffer))
        return matches[-1].end() if matches else 0
    def _last_word_boundary(self) -> int:
        matches = list(re.finditer(r"\s+", self.buffer))
        return matches[-1].end() if matches else 0
    def _unsafe_tail(self, end: int | None = None) -> bool: return bool(self._UNSAFE_TAIL.search(self.buffer[:end]))
    def _ends_in_negation(self) -> bool: return bool(self._NEGATION.search(self.buffer))
    def _pop(self, end: int) -> str:
        result = self.buffer[:end].strip()
        self.buffer = self.buffer[end:]
        # Always reset the timer so subsequent chunks each get a full max_wait_ms
        # grace period from the NEXT token that arrives.  Setting it to the pop
        # timestamp (old behaviour) caused every slow-arriving token to trigger an
        # immediate release because the timer appeared to have already expired.
        self.first_token_at = None
        return result
