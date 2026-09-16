"""Conservative first-audio trimming and cadence telemetry helpers."""

from __future__ import annotations

import math
import os

from loguru import logger
from pipecat.frames.frames import (
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class InitialSilenceTrimmer(FrameProcessor):
    """Remove only confirmed leading PCM silence from each TTS utterance.

    The processor retains a short pre-roll to avoid clipping initial
    consonants and stops trimming after a bounded interval. It never removes
    pauses after speech has begun.
    """

    def __init__(self) -> None:
        super().__init__(name="InitialSilenceTrimmer")
        self.threshold = int(os.getenv("V2_TTS_TRIM_RMS", "160"))
        self.preroll_ms = int(os.getenv("V2_TTS_TRIM_PREROLL_MS", "20"))
        self.max_trim_ms = int(os.getenv("V2_TTS_MAX_LEADING_SILENCE_MS", "250"))
        self._waiting = True
        self._trimmed_ms = 0.0

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, (TTSStartedFrame, InterruptionFrame)):
            self._waiting = True
            self._trimmed_ms = 0.0
        elif isinstance(frame, TTSStoppedFrame):
            if self._trimmed_ms:
                logger.debug("V2 TTS leading silence trimmed_ms={:.1f}", self._trimmed_ms)
            self._waiting = True
            self._trimmed_ms = 0.0
        elif isinstance(frame, TTSAudioRawFrame) and self._waiting:
            audio, trimmed_ms, audible = self._trim(frame.audio, frame.sample_rate)
            self._trimmed_ms += trimmed_ms
            if not audible and self._trimmed_ms < self.max_trim_ms:
                return
            self._waiting = False
            if audio:
                frame.audio = audio
            elif self._trimmed_ms < self.max_trim_ms:
                return
        await self.push_frame(frame, direction)

    def _trim(self, audio: bytes, sample_rate: int) -> tuple[bytes, float, bool]:
        aligned = audio[: len(audio) - len(audio) % 2]
        if not aligned or sample_rate <= 0:
            return audio, 0.0, False
        frame_samples = max(1, sample_rate // 100)  # 10 ms
        frame_bytes = frame_samples * 2
        blocks = [aligned[i:i + frame_bytes] for i in range(0, len(aligned), frame_bytes)]
        first_audible = next((i for i, block in enumerate(blocks) if self._rms(block) >= self.threshold), None)
        if first_audible is None:
            duration_ms = len(aligned) / (sample_rate * 2) * 1000
            return b"", duration_ms, False
        preroll_blocks = max(0, self.preroll_ms // 10)
        first = max(0, first_audible - preroll_blocks)
        trimmed = first * 10.0
        return b"".join(blocks[first:]), trimmed, True

    @staticmethod
    def _rms(audio: bytes) -> float:
        samples = memoryview(audio).cast("h") if len(audio) >= 2 else ()
        return math.sqrt(sum(int(v) * int(v) for v in samples) / len(samples)) if samples else 0.0
