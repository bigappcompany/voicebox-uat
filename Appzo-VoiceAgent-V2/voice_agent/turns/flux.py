import asyncio
import json
import math
import os
import time
from dataclasses import dataclass

from loguru import logger
from voice_agent.runtime.telemetry import TelemetrySpeechStartedFrame, TelemetrySpeechStoppedFrame

from pipecat.frames.frames import DataFrame
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.turns.user_stop import ExternalUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies


@dataclass
class FluxResumeFrame(DataFrame):
    pass


class OrderedFluxSTTService(DeepgramFluxSTTService):
    def __init__(self, *args, telemetry_enabled=False, **kwargs):
        super().__init__(*args, **kwargs)
        # Enabled only when paired with TelemetryBoundaryFilter by main.py.
        self._telemetry_enabled = telemetry_enabled
        self._telemetry_last_voice_at = None
        self._telemetry_voice_level = int(os.getenv("V2_INPUT_VOICE_RMS", "200"))
        self._input_gap_threshold_secs = float(os.getenv("V2_INPUT_PACKET_GAP_MS", "100")) / 1000
        self._last_input_at = None
        self._input_gap_count = 0
        self._input_gap_max_ms = 0.0
        self._turn_resumed_count = 0

    async def _watchdog_task_handler(self):
        """Prevent dangling turns and keep connection alive during silence."""
        while self._transport_is_active():
            now = time.monotonic()
            threshold = max(self._last_audio_chunk_duration * 2, self._watchdog_min_timeout)
            if (
                self._user_is_speaking
                and self._last_stt_time
                and now - self._last_stt_time > threshold
            ):
                try:
                    await self._send_silence()
                except Exception:
                    pass
                self._last_stt_time = time.monotonic()
            elif (
                not self._user_is_speaking
                and self._last_stt_time
                and now - self._last_stt_time > 5.0
                and self._websocket is not None
            ):
                try:
                    await self.send_with_retry(json.dumps({"type": "KeepAlive"}), self._report_error)
                except Exception:
                    pass
                self._last_stt_time = time.monotonic()
            await asyncio.sleep(0.1)

    async def run_stt(self, audio):
        now = time.perf_counter()
        if self._telemetry_enabled and len(audio) >= 2:
            samples = memoryview(audio[:len(audio) - len(audio) % 2]).cast("h")
            rms = math.sqrt(sum(int(s) * int(s) for s in samples) / len(samples))
            if rms >= self._telemetry_voice_level:
                self._telemetry_last_voice_at = now
        if self._last_input_at is not None:
            gap = max(0.0, now - self._last_input_at)
            if gap >= self._input_gap_threshold_secs:
                self._input_gap_count += 1
                self._input_gap_max_ms = max(self._input_gap_max_ms, gap * 1000)
        self._last_input_at = now
        async for frame in super().run_stt(audio):
            yield frame

    async def _handle_start_of_turn(self, transcript):
        if self._telemetry_enabled:
            await self._emit_speech_telemetry(start=True)
        await super()._handle_start_of_turn(transcript)

    async def _emit_speech_telemetry(self, *, start=False, ended_at=None):
        # Do not let instrumentation failure abort a real user turn.
        try:
            if start:
                await self.push_frame(TelemetrySpeechStartedFrame())
                return
            last_voice = self._telemetry_last_voice_at
            if last_voice is None or last_voice > ended_at:
                logger.debug("FLUX TELEMETRY | missing valid PCM speech-end timestamp")
                return
            delay = ended_at - last_voice
            epoch_end = time.time()
            await self.push_frame(TelemetrySpeechStoppedFrame(timestamp=epoch_end, stop_secs=delay))
            # Pipecat displays this under STT TTFB. Its precise meaning here is
            # last voiced PCM received -> Flux final EOT received, including
            # endpointing. It is NOT eager-EOT -> hard-EOT or pure inference.
            await self.start_ttfb_metrics(start_time=epoch_end - delay)
            await self.stop_ttfb_metrics(end_time=epoch_end)
        except Exception as exc:
            logger.warning("FLUX TELEMETRY FAILED | {}", type(exc).__name__)

    async def _handle_end_of_turn(self, transcript, data):
        ended_at = time.perf_counter()
        data = dict(
            data,
            runtime_eot_at=ended_at,
            runtime_input_gap_count=self._input_gap_count,
            runtime_input_gap_max_ms=round(self._input_gap_max_ms, 1),
            runtime_turn_resumed_count=self._turn_resumed_count,
        )
        self._input_gap_count = 0
        self._input_gap_max_ms = 0.0
        self._turn_resumed_count = 0
        if self._telemetry_enabled:
            await self._emit_speech_telemetry(ended_at=ended_at)
            self._telemetry_last_voice_at = None
        await super()._handle_end_of_turn(transcript, data)

    async def _handle_eager_end_of_turn(self, transcript, data):
        await super()._handle_eager_end_of_turn(
            transcript, dict(data, runtime_eager_at=time.perf_counter())
        )

    async def _handle_turn_resumed(self, event):
        self._turn_resumed_count += 1
        await self.push_frame(FluxResumeFrame())
        await super()._handle_turn_resumed(event)

    async def configure_endpoint(self, profile, *, keyterms=None, language_hints=None) -> None:
        """Apply a state-aware Flux profile over the existing WebSocket."""
        eager = max(0.30, float(profile.eager_eot_threshold))
        eot = max(0.50, float(profile.eot_threshold))
        await self._update_settings(
            self.Settings(
                eager_eot_threshold=eager,
                eot_threshold=eot,
                eot_timeout_ms=profile.eot_timeout_ms,
                keyterm=keyterms,
                language_hints=language_hints,
            )
        )


class FluxStopStrategy(ExternalUserTurnStopStrategy):
    async def _handle_transcription(self, frame):
        await super()._handle_transcription(frame)
        # The real UserStoppedSpeakingFrame updates BOTH the strategy and
        # UserTurnController. If it arrived before queued text, finish now
        # instead of waiting for the inherited 500 ms polling timer. If text
        # arrived first, the normal stop handler finishes the turn later.
        await self._maybe_trigger_user_turn_stopped()


def flux_turn_strategies():
    strategies = ExternalUserTurnStrategies()
    strategies.stop = [FluxStopStrategy()]
    return strategies
