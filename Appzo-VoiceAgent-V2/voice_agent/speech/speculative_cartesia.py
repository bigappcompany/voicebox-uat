"""Private Cartesia synthesis used by the V2 soft-EOT path.

The normal Pipecat Cartesia service remains the only public synthesis path.
This helper owns a second, per-call WebSocket and receives a *small* amount of
PCM for a speculative candidate.  Nothing leaves this object until the caller's
hard EOT passes the response-fingerprint commit check in ``live_v2``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import socket
import time
import uuid
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

from loguru import logger
from websockets.asyncio.client import connect

from .audio_commit import SpeculativeAudioCandidate


@dataclass
class PreparedSpeculativeAudio:
    """A private PCM prefix and the state required to safely commit it."""

    candidate: SpeculativeAudioCandidate
    context_id: str
    text: str
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    completed: bool = False
    failed: bool = False
    first_audio_at: float | None = None


class SpeculativeCartesiaBuffer:
    """Warm one private Cartesia socket and retain capped speculative PCM.

    A failed private connection must never interfere with the normal Cartesia
    TTS service.  The caller can therefore safely enable this optimisation on
    low-risk turns while retaining the regular WebSocket stream as fallback.
    """

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        model: str,
        speed: float,
        sample_rate: int = 24000,
        max_audio_ms: int = 600,
        connect_timeout_secs: float = 4.0,
    ) -> None:
        self._api_key = api_key
        self._voice_id = voice_id
        self._model = model
        self._speed = speed
        self._sample_rate = sample_rate
        self._max_audio_ms = max_audio_ms
        self._connect_timeout_secs = connect_timeout_secs
        self._websocket: Any | None = None
        self._reader_task: asyncio.Task | None = None
        self._warm_task: asyncio.Task | None = None
        self._connected = asyncio.Event()
        self._prepared: dict[str, PreparedSpeculativeAudio] = {}
        self._send_lock = asyncio.Lock()
        self._closed = False

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set() and self._websocket is not None

    def warm(self) -> None:
        """Start connecting without delaying the media pipeline."""
        if self._closed or self.is_connected or (self._warm_task and not self._warm_task.done()):
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._warm_task = loop.create_task(self._connect(), name="v2-spec-cartesia-warm")

    async def prepare(
        self, *, fingerprint: str, transcript_basis: str, text: str
    ) -> PreparedSpeculativeAudio | None:
        """Request a single safe text prefix without making it audible."""
        clean = text.strip()
        if self._closed or not clean:
            return None
        self.warm()
        if not self.is_connected:
            # Do not wait in the LLM stream for a cold/failed private socket.
            return None

        context_id = str(uuid.uuid4())
        prepared = PreparedSpeculativeAudio(
            candidate=SpeculativeAudioCandidate(
                fingerprint=fingerprint,
                transcript_basis=transcript_basis,
                text=clean,
                sample_rate=self._sample_rate,
            ),
            context_id=context_id,
            text=clean,
        )
        self._prepared[context_id] = prepared
        message = {
            "transcript": clean,
            "continue": False,
            "context_id": context_id,
            "model_id": self._model,
            "voice": {"mode": "id", "id": self._voice_id},
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": self._sample_rate,
            },
            "add_timestamps": False,
            "max_buffer_delay_ms": 0,
            "generation_config": {"speed": self._speed},
        }
        try:
            async with self._send_lock:
                if not self.is_connected:
                    self._prepared.pop(context_id, None)
                    return None
                await self._websocket.send(json.dumps(message))
            return prepared
        except Exception as exc:
            self._prepared.pop(context_id, None)
            prepared.failed = True
            prepared.ready.set()
            logger.warning("V2 speculative Cartesia send failed: {}", type(exc).__name__)
            return None

    async def commit(
        self, prepared: PreparedSpeculativeAudio | None, *, fingerprint: str
    ) -> list[bytes] | None:
        """Return private PCM only when the final fingerprint matches exactly."""
        if prepared is None:
            return None
        candidate = prepared.candidate
        if candidate.invalidated or candidate.fingerprint != fingerprint:
            await self.abort(prepared)
            return None
        # Hard EOT never waits for speculative synthesis. Prepared PCM is an
        # optimization only; the public WebSocket is the immediate fallback.
        if not candidate.pcm_chunks:
            await self.abort(prepared)
            return None
        candidate.committed = True
        self._prepared.pop(prepared.context_id, None)
        return list(candidate.pcm_chunks)

    async def abort(self, prepared: PreparedSpeculativeAudio | None) -> None:
        if prepared is None:
            return
        prepared.candidate.invalidated = True
        prepared.candidate.pcm_chunks.clear()
        self._prepared.pop(prepared.context_id, None)
        if not self.is_connected:
            return
        try:
            async with self._send_lock:
                if self.is_connected:
                    await self._websocket.send(json.dumps({"context_id": prepared.context_id, "cancel": True}))
        except Exception:
            # The normal public Cartesia connection is intentionally unaffected.
            pass

    async def close(self) -> None:
        self._closed = True
        for prepared in list(self._prepared.values()):
            prepared.candidate.invalidated = True
            prepared.candidate.pcm_chunks.clear()
            prepared.ready.set()
        self._prepared.clear()
        for task in (self._warm_task, self._reader_task):
            if task and not task.done():
                task.cancel()
        await asyncio.gather(
            *(task for task in (self._warm_task, self._reader_task) if task),
            return_exceptions=True,
        )
        if self._websocket is not None:
            try:
                await self._websocket.close()
            except Exception:
                pass
        self._websocket = None
        self._connected.clear()

    async def _connect(self) -> None:
        if self._closed or self.is_connected:
            return
        try:
            # Keep credentials out of the URI and never log either form.
            # Pinning provider WSS to IPv4 avoids this host's stalled IPv6
            # CloudFront route; the normal public TTS uses the same setting.
            url = "wss://api.cartesia.ai/tts/websocket?" + urlencode(
                {"cartesia_version": "2026-03-01"}
            )
            self._websocket = await connect(
                url,
                additional_headers={"X-API-Key": self._api_key},
                proxy=None,
                family=socket.AF_INET,
                open_timeout=self._connect_timeout_secs,
                max_size=None,
            )
            self._connected.set()
            self._reader_task = asyncio.create_task(self._reader(), name="v2-spec-cartesia-reader")
            logger.info("V2 speculative Cartesia buffer is warm")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._websocket = None
            self._connected.clear()
            logger.warning("V2 speculative Cartesia warmup failed: {}", type(exc).__name__)

    async def _reader(self) -> None:
        try:
            async for raw in self._websocket:
                message = json.loads(raw)
                context_id = message.get("context_id")
                prepared = self._prepared.get(context_id)
                if prepared is None:
                    continue
                kind = message.get("type")
                if kind == "chunk":
                    try:
                        pcm = base64.b64decode(message["data"])
                    except (KeyError, ValueError):
                        prepared.failed = True
                        prepared.ready.set()
                        continue
                    if prepared.candidate.append(pcm, self._max_audio_ms):
                        if prepared.first_audio_at is None:
                            prepared.first_audio_at = time.perf_counter()
                            prepared.ready.set()
                    else:
                        # The cap is a successful partial preparation, not an
                        # invalidation. Stop provider generation while keeping
                        # already-buffered PCM eligible for hard-EOT commit.
                        self._prepared.pop(prepared.context_id, None)
                        prepared.completed = True
                        prepared.ready.set()
                        try:
                            async with self._send_lock:
                                if self.is_connected:
                                    await self._websocket.send(json.dumps({
                                        "context_id": prepared.context_id, "cancel": True,
                                    }))
                        except Exception:
                            pass
                elif kind in {"done", "flush_done"}:
                    prepared.completed = True
                    prepared.ready.set()
                elif kind == "error":
                    prepared.failed = True
                    prepared.ready.set()
                    logger.warning("V2 speculative Cartesia returned an error")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("V2 speculative Cartesia reader stopped: {}", type(exc).__name__)
        finally:
            self._connected.clear()
            self._websocket = None
            for prepared in self._prepared.values():
                prepared.failed = True
                prepared.ready.set()
