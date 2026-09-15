"""Preserve native Flux stop signals and finalize as soon as text arrives."""
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
    async def _handle_end_of_turn(self, transcript, data):
        data = dict(data, runtime_eot_at=time.perf_counter())
        await super()._handle_end_of_turn(transcript, data)

    async def _handle_turn_resumed(self, event):
        await self.push_frame(FluxResumeFrame())
        await super()._handle_turn_resumed(event)


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
