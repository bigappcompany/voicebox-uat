#!/usr/bin/env python3
"""Deterministic SafeSpeechChunker profile comparison.

This measures buffering policy only; provider/network TTFT belongs to live
latency records. Feed representative model deltas here before changing the
production profile.
"""

from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voice_agent.speech.safe_chunker import SafeSpeechChunker


@dataclass(frozen=True)
class Profile:
    name: str
    min_chars: int
    min_words: int
    max_wait_ms: int


PROFILES = (
    Profile("fast", 32, 5, 160),
    Profile("balanced", 40, 5, 200),
    Profile("natural", 48, 7, 240),
)

SAMPLES = (
    ("short_answer", ("Got it. ", "How many people ", "do you need?")),
    ("requirements", ("Three to four months noted. ", "Which roles ", "are you hiring?")),
    ("negation", ("I can not ", "confirm a booking. ", "Our team will follow up.")),
)


def benchmark(profile: Profile, fragments: tuple[str, ...], cadence_ms: int = 60):
    chunker = SafeSpeechChunker(profile.min_chars, profile.min_words, profile.max_wait_ms)
    emitted: list[tuple[int, str]] = []
    for index, fragment in enumerate(fragments):
        at_ms = index * cadence_ms
        for phrase in chunker.push(fragment, now=at_ms / 1000):
            emitted.append((at_ms, phrase))
    for phrase in chunker.flush():
        emitted.append(((len(fragments) - 1) * cadence_ms, phrase))
    return emitted


def main() -> None:
    print("profile\tsample\tfirst_safe_ms\tfirst_words\tchunks")
    for profile in PROFILES:
        for sample, fragments in SAMPLES:
            emitted = benchmark(profile, fragments)
            first_ms, first = emitted[0]
            print(
                f"{profile.name}\t{sample}\t{first_ms}\t"
                f"{len(first.split())}\t{len(emitted)}"
            )


if __name__ == "__main__":
    main()
