"""Persistent, versioned greeting audio for the Plivo call opening."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from pipecat.frames.frames import (
    InterruptionFrame,
    OutputTransportMessageUrgentFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


@dataclass(frozen=True)
class GreetingCacheKey:
    tenant_id: str
    agent_id: str
    agent_version: str
    voice_id: str
    tts_model: str
    speed: float
    intro_text: str

    def digest(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class CachedGreeting:
    ulaw: bytes
    sample_rate: int = 8000

    @property
    def packets(self) -> tuple[bytes, ...]:
        size = self.sample_rate // 50  # 20 ms of 8-bit mu-law.
        return tuple(self.ulaw[index:index + size] for index in range(0, len(self.ulaw), size))


class GreetingCache:
    def __init__(self, root: str | Path | None = None, *, silence_threshold: int = 200) -> None:
        self.root = Path(root or os.getenv("V2_GREETING_CACHE_DIR", ".runtime-cache/greetings"))
        self.silence_threshold = silence_threshold

    def get(self, key: GreetingCacheKey) -> CachedGreeting | None:
        path = self.root / f"{key.digest()}.ulaw"
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            return None
        return CachedGreeting(data) if data else None

    async def put_pcm(self, key: GreetingCacheKey, pcm: bytes, *, sample_rate: int) -> CachedGreeting | None:
        if not pcm:
            return None
        ulaw = await asyncio.to_thread(self._convert, pcm, sample_rate)
        if not ulaw:
            return None
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{key.digest()}.ulaw"
        temporary = self.root / f".{key.digest()}.{os.getpid()}.tmp"
        temporary.write_bytes(ulaw)
        temporary.replace(path)
        return CachedGreeting(ulaw)

    def _convert(self, pcm: bytes, sample_rate: int) -> bytes:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            import audioop

        aligned = pcm[: len(pcm) - len(pcm) % 2]
        if not aligned:
            return b""
        frame_bytes = max(2, sample_rate // 100 * 2)  # 10 ms PCM16 mono.
        frames = [aligned[index:index + frame_bytes] for index in range(0, len(aligned), frame_bytes)]
        audible = [index for index, frame in enumerate(frames) if frame and audioop.rms(frame, 2) >= self.silence_threshold]
        if audible:
            # Preserve 20 ms around the voice boundary so consonants are not clipped.
            first = max(0, audible[0] - 2)
            last = min(len(frames), audible[-1] + 3)
            aligned = b"".join(frames[first:last])
        resampled, _ = audioop.ratecv(aligned, 2, 1, sample_rate, 8000, None)
        return audioop.lin2ulaw(resampled, 2)


class GreetingCaptureProcessor(FrameProcessor):
    """Capture only the first TTS utterance and persist it as the greeting."""

    def __init__(self, cache: GreetingCache, key: GreetingCacheKey) -> None:
        super().__init__(name="GreetingCaptureProcessor")
        self.cache = cache
        self.key = key
        self.frames: list[bytes] = []
        self.sample_rate = 24000
        self.complete = False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and not self.complete:
            if isinstance(frame, InterruptionFrame):
                # Never persist a greeting that the caller interrupted. Mark
                # this capture terminal so later response audio cannot be
                # mistaken for the call opening.
                self.abort_capture()
            elif isinstance(frame, TTSAudioRawFrame):
                self.frames.append(frame.audio)
                self.sample_rate = frame.sample_rate
            elif isinstance(frame, TTSStoppedFrame) and self.frames:
                self.complete = True
                audio = b"".join(self.frames)
                self.frames.clear()
                cached = await self.cache.put_pcm(self.key, audio, sample_rate=self.sample_rate)
                if cached:
                    logger.info("V2 greeting cached key={} audio_ms={}", self.key.digest()[:12], len(cached.ulaw) // 8)
        await self.push_frame(frame, direction)

    def abort_capture(self) -> None:
        self.complete = True
        self.frames.clear()


class CachedGreetingPlayer:
    """Paced Plivo-native playback with explicit interruption semantics."""

    def __init__(
        self,
        output_transport,
        *,
        stream_id: str,
        greeting: CachedGreeting,
        call_origin_at: float | None = None,
    ) -> None:
        self.output_transport = output_transport
        self.stream_id = stream_id
        self.greeting = greeting
        self.call_origin_at = call_origin_at
        self.task: asyncio.Task | None = None
        self.started_at: float | None = None
        self.first_packet_at: float | None = None
        self.first_audible_at: float | None = None
        self._cleared = False
        self.finished = asyncio.Event()

    def start(self) -> None:
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._play(), name="v2-cached-greeting")

    async def interrupt(self) -> None:
        if self.task and not self.task.done():
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        # Plivo can still have the last paced packets queued after the local
        # task completes. Clear exactly once on the first caller turn even in
        # that narrow post-task window.
        if self.started_at is not None and not self._cleared:
            await self.output_transport.send_message(
                OutputTransportMessageUrgentFrame({"event": "clearAudio", "streamId": self.stream_id})
            )
            self._cleared = True
            self.finished.set()

    async def close(self) -> None:
        if self.task and not self.task.done():
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        self.finished.set()

    async def _play(self) -> None:
        self.started_at = time.perf_counter()
        try:
            for packet in self.greeting.packets:
                message = {
                    "event": "playAudio",
                    "media": {
                        "contentType": "audio/x-mulaw",
                        "sampleRate": 8000,
                        "payload": base64.b64encode(packet).decode("ascii"),
                    },
                    "streamId": self.stream_id,
                }
                await self.output_transport.send_message(OutputTransportMessageUrgentFrame(message))
                if self.first_packet_at is None:
                    self.first_packet_at = time.perf_counter()
                    logger.info(
                        "V2 CACHED GREETING | start->first-packet={} ms plivo-connect->first-packet={} ms",
                        round((self.first_packet_at - self.started_at) * 1000),
                        round((self.first_packet_at - self.call_origin_at) * 1000) if self.call_origin_at else None,
                    )
                if self.first_audible_at is None and self._is_audible(packet):
                    self.first_audible_at = time.perf_counter()
                    logger.info(
                        "V2 CACHED GREETING | start->first-audible={} ms plivo-connect->first-audible={} ms",
                        round((self.first_audible_at - self.started_at) * 1000),
                        round((self.first_audible_at - self.call_origin_at) * 1000) if self.call_origin_at else None,
                    )
                await asyncio.sleep(0.02)
        except asyncio.CancelledError:
            raise
        finally:
            self.finished.set()

    @staticmethod
    def _is_audible(packet: bytes) -> bool:
        if not packet:
            return False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            import audioop
        pcm = audioop.ulaw2lin(packet, 2)
        return audioop.rms(pcm, 2) >= 200
