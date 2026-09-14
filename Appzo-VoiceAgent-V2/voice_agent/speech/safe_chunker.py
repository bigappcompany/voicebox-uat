"""Language-aware-enough conservative boundaries for streamed speech text."""
import re
import time


class SafeSpeechChunker:
    _NEGATION = re.compile(r"\b(?:not|never|don't|cannot|can't|नहीं|मत)\s*$", re.IGNORECASE)
    _UNSAFE_TAIL = re.compile(r"(?:[$₹€£]\s*\d*|\b\d{1,2}[/-]\d{0,2}|\b\d+[,.]?\d*)$", re.UNICODE)

    def __init__(self, min_chars: int = 40, min_words: int = 6, max_wait_ms: int = 120) -> None:
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
        if (now - self.first_token_at) * 1000 >= self.max_wait_ms:
            word_boundary = self._last_word_boundary()
            if word_boundary and not self._unsafe_tail(word_boundary): return [self._pop(word_boundary)]
        return []

    def flush(self) -> list[str]:
        text, self.buffer, self.first_token_at = self.buffer.strip(), "", None
        return [text] if text else []

    def _last_boundary(self) -> int:
        matches = list(re.finditer(r"[,;:.!?]\s+", self.buffer))
        return matches[-1].end() if matches else 0
    def _last_word_boundary(self) -> int:
        matches = list(re.finditer(r"\s+", self.buffer))
        return matches[-1].end() if matches else 0
    def _unsafe_tail(self, end: int | None = None) -> bool: return bool(self._UNSAFE_TAIL.search(self.buffer[:end]))
    def _ends_in_negation(self) -> bool: return bool(self._NEGATION.search(self.buffer))
    def _pop(self, end: int) -> str:
        result = self.buffer[:end].strip(); self.buffer = self.buffer[end:]; self.first_token_at = time.perf_counter() if self.buffer else None; return result
