"""Preserve native Flux stop signals and finalize as soon as text arrives."""
import os
import time
from dataclasses import dataclass

from pipecat.frames.frames import DataFrame
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.turns.user_stop import ExternalUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies


@dataclass
class FluxResumeFrame(DataFrame):
    pass


class OrderedFluxSTTService(DeepgramFluxSTTService):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._input_gap_threshold_secs = float(os.getenv("V2_INPUT_PACKET_GAP_MS", "100")) / 1000
        self._last_input_at = None
        self._input_gap_count = 0
        self._input_gap_max_ms = 0.0
        self._turn_resumed_count = 0

    async def run_stt(self, audio):
        now = time.perf_counter()
        if self._last_input_at is not None:
            gap = max(0.0, now - self._last_input_at)
            if gap >= self._input_gap_threshold_secs:
                self._input_gap_count += 1
                self._input_gap_max_ms = max(self._input_gap_max_ms, gap * 1000)
        self._last_input_at = now
        async for frame in super().run_stt(audio):
            yield frame

    async def _handle_end_of_turn(self, transcript, data):
        data = dict(
            data,
            runtime_eot_at=time.perf_counter(),
            runtime_input_gap_count=self._input_gap_count,
            runtime_input_gap_max_ms=round(self._input_gap_max_ms, 1),
            runtime_turn_resumed_count=self._turn_resumed_count,
        )
        self._input_gap_count = 0
        self._input_gap_max_ms = 0.0
        self._turn_resumed_count = 0
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
        await self._update_settings(
            self.Settings(
                eager_eot_threshold=profile.eager_eot_threshold,
                eot_threshold=profile.eot_threshold,
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
