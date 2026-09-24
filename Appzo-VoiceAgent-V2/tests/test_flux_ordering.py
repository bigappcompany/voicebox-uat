import unittest
import asyncio
from unittest.mock import AsyncMock

from pipecat.frames.frames import TranscriptionFrame, UserStoppedSpeakingFrame, UserStartedSpeakingFrame
from pipecat.turns.user_turn_controller import UserTurnController
from pipecat.utils.asyncio.task_manager import TaskManager
from voice_agent.turns.flux import OrderedFluxSTTService, FluxStopStrategy, flux_turn_strategies
from voice_agent.turns.endpoint_profiles import flux_profile
from voice_agent.speech.booking_guard import BookingClaimGuard


class FluxOrderingTests(unittest.IsolatedAsyncioTestCase):
    async def test_endpoint_profile_can_update_without_reconnect(self):
        stt = OrderedFluxSTTService(api_key="test")
        await stt.configure_endpoint(flux_profile("yes_no"))
        self.assertEqual(stt._settings.eager_eot_threshold, .20)
        self.assertEqual(stt._settings.eot_threshold, .52)
        self.assertEqual(stt._settings.eot_timeout_ms, 500)

    async def test_real_aggregator_delivers_complete_text_before_shutdown(self):
        from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregator, LLMUserAggregatorParams
        from pipecat.processors.aggregators.llm_context import LLMContext
        from pipecat.processors.frame_processor import FrameProcessorSetup, FrameDirection
        from pipecat.clocks.system_clock import SystemClock
        from pipecat.frames.frames import StartFrame, CancelFrame
        aggregator = LLMUserAggregator(LLMContext(), params=LLMUserAggregatorParams(user_turn_strategies=flux_turn_strategies()))
        aggregator.push_frame = AsyncMock()
        messages = []
        async def stopped(_, strategy, message):
            messages.append(message.content)
        aggregator.add_event_handler("on_user_turn_stopped", stopped)
        await aggregator.setup(FrameProcessorSetup(clock=SystemClock(), task_manager=TaskManager(), pipeline_worker=None))
        async def send(frame):
            await aggregator.process_frame(frame, FrameDirection.DOWNSTREAM)
        try:
            await send(StartFrame())
            await send(UserStartedSpeakingFrame())
            await send(TranscriptionFrame("Please call", "u", "", finalized=False))
            await send(TranscriptionFrame("tomorrow at three", "u", "", finalized=True))
            await send(UserStoppedSpeakingFrame())
            await asyncio.sleep(0)
            self.assertEqual(messages, ["Please call tomorrow at three"])
            await send(UserStartedSpeakingFrame())
            await send(UserStoppedSpeakingFrame())
            await send(TranscriptionFrame("I need four engineers next month", "u", "", finalized=True))
            await asyncio.sleep(0)
            self.assertIn("I need four engineers next month", messages)
            await asyncio.sleep(0.55)
            self.assertEqual(len(messages), 2)
        finally:
            await send(CancelFrame())
            await aggregator.cleanup()

    async def test_final_text_precedes_stop_and_stop_has_no_timeout(self):
        stt = OrderedFluxSTTService(api_key="test")
        stt.push_frame = AsyncMock()
        stt.emit_stt_usage_metrics = AsyncMock()
        stt._handle_transcription = AsyncMock()
        stt.stop_processing_metrics = AsyncMock()
        await stt._handle_end_of_turn("complete request", {"turn_index": 7})
        from pipecat.processors.frame_processor import FrameDirection
        downstream = [c.args[0] for c in stt.push_frame.await_args_list
                      if len(c.args) == 1 or c.args[1] == FrameDirection.DOWNSTREAM]
        self.assertIsInstance(downstream[0], TranscriptionFrame)
        self.assertIsInstance(downstream[1], UserStoppedSpeakingFrame)
        self.assertEqual(downstream[0].result["turn_index"], 7)
        self.assertGreater(downstream[0].result["runtime_eot_at"], 0)
        strategy = FluxStopStrategy()
        strategy.trigger_user_turn_stopped = AsyncMock()
        await strategy.process_frame(UserStartedSpeakingFrame())
        await strategy.process_frame(downstream[0])
        strategy.trigger_user_turn_stopped.assert_not_awaited()
        await strategy.process_frame(downstream[1])
        strategy.trigger_user_turn_stopped.assert_awaited_once()

    async def test_real_controller_finalizes_both_arrival_orders_once(self):
        controller = UserTurnController(user_turn_strategies=flux_turn_strategies())
        stopped, inferred = AsyncMock(), AsyncMock()
        controller.add_event_handler("on_user_turn_stopped", stopped)
        controller.add_event_handler("on_user_turn_inference_triggered", inferred)
        await controller.setup(TaskManager())
        try:
            for stop_first in (True, False):
                await controller.process_frame(UserStartedSpeakingFrame())
                final = TranscriptionFrame("three pm tomorrow", "user", "", finalized=True)
                frames = [UserStoppedSpeakingFrame(), final] if stop_first else [final, UserStoppedSpeakingFrame()]
                before = stopped.await_count
                await controller.process_frame(frames[0])
                self.assertEqual(stopped.await_count, before)
                await controller.process_frame(frames[1])
                self.assertEqual(stopped.await_count, before + 1)
            await asyncio.sleep(0.55)
            self.assertEqual(stopped.await_count, 2)
            self.assertEqual(inferred.await_count, 2)
        finally:
            await controller.cleanup()

    def test_booking_claim_split_across_tokens_never_escapes(self):
        guard = BookingClaimGuard()
        self.assertEqual(guard.push("I have sched"), "")
        self.assertEqual(guard.push("uled the follow-up for three pm"), "")
        result = guard.push(". Any questions?")
        self.assertNotIn("scheduled", result)
        self.assertIn("needs confirmation", result)
