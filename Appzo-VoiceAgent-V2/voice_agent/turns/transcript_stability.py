from dataclasses import dataclass
import re
import time


def _words(text: str) -> list[str]:
    return re.findall(r"[\w\u0900-\u097F]+", text.lower(), flags=re.UNICODE)


@dataclass
class TranscriptHypothesis:
    text: str = ""; stable_prefix: str = ""; unstable_suffix: str = ""; stable_since: float | None = None
    intent_guess: str | None = None; slot_guess: dict[str, str] | None = None; semantic_hash: str | None = None


class TranscriptStabilityAnalyzer:
    def __init__(self) -> None: self._previous: list[str] = []; self.hypothesis = TranscriptHypothesis(slot_guess={})

    def update(self, text: str, *, intent_guess: str | None = None, slots: dict[str, str] | None = None) -> TranscriptHypothesis:
        current = _words(text); common = 0
        for old, new in zip(self._previous, current):
            if old != new: break
            common += 1
        stable = " ".join(current[:common])
        if stable != self.hypothesis.stable_prefix:
            stable_since = time.perf_counter() if stable else None
        else: stable_since = self.hypothesis.stable_since
        self._previous = current
        self.hypothesis = TranscriptHypothesis(" ".join(current), stable, " ".join(current[common:]), stable_since, intent_guess, slots or {}, None)
        return self.hypothesis
