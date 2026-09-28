import unittest
from unittest.mock import AsyncMock, patch

from pipecat.frames.frames import (
    LLMFullResponseStartFrame, LLMFullResponseEndFrame, LLMTextFrame,
    MetricsFrame, TTSSpeakFrame, BotStartedSpeakingFrame,
)
from pipecat.metrics.metrics import TTFBMetricsData, LLMUsageMetricsData, LLMTokenUsage
from pipecat.observers.base_observer import FramePushed
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from voice_agent.runtime.telemetry import (
    CloudTelemetryProcessor, TelemetrySpeechStartedFrame,
    TelemetrySpeechStoppedFrame, cloud_rtvi_params,
)


class CloudTelemetryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.processor = CloudTelemetryProcessor()
        self.observer = self.processor.create_rtvi_observer(params=cloud_rtvi_params())
        self.observer.send_rtvi_message = AsyncMock()
        self.source = FrameProcessor()
        self.destination = FrameProcessor()

    async def emit(self, frame, direction=FrameDirection.DOWNSTREAM):
        await self.observer.on_push_frame(FramePushed(
            source=self.source, destination=self.destination, frame=frame,
            direction=direction, timestamp=0,
        ))

    def messages(self):
        return [call.args[0] for call in self.observer.send_rtvi_message.await_args_list]

    async def test_fixed_lifecycle_once_without_synthetic_metrics(self):
        frame = TTSSpeakFrame("Recorded your preference.", append_to_context=True)
        await self.emit(frame)
        await self.emit(frame)
        self.assertEqual([m.type for m in self.messages()], ["bot-llm-started", "bot-llm-stopped"])
        self.assertEqual(frame.text, "Recorded your preference.")
        self.assertTrue(frame.append_to_context)

    async def test_hosted_lifecycle_without_partial_text(self):
        await self.emit(LLMFullResponseStartFrame())
        await self.emit(LLMTextFrame("Partial text"))
        await self.emit(LLMFullResponseEndFrame())
        self.assertEqual([m.type for m in self.messages()], ["bot-llm-started", "bot-llm-stopped"])

    async def test_hosted_metrics_and_usage_forwarded_unchanged(self):
        usage = LLMTokenUsage(prompt_tokens=101, completion_tokens=7, total_tokens=108)
        await self.emit(MetricsFrame(data=[
            TTFBMetricsData(processor="V2RoutingController#0", model="test-model", value=0.32),
            LLMUsageMetricsData(processor="V2RoutingController#0", model="test-model", value=usage),
        ]))
        message, = self.messages()
        self.assertEqual(message.type, "metrics")
        self.assertEqual(message.data["ttfb"][0]["value"], 0.32)
        self.assertEqual(message.data["ttfb"][0]["processor"], "V2RoutingController#0")
        self.assertEqual(message.data["tokens"][0]["total_tokens"], 108)

    async def test_disabled_lifecycle_emits_no_fixed_events(self):
        self.observer._params.bot_llm_enabled = False
        await self.emit(TTSSpeakFrame("Hello"))
        self.assertEqual(self.messages(), [])

    async def test_native_response_latency_does_not_require_llm_frames(self):
        observer = UserBotLatencyObserver()
        observer._call_event_handler = AsyncMock()
        for frame in (
            TelemetrySpeechStartedFrame(),
            TelemetrySpeechStoppedFrame(timestamp=100.4, stop_secs=0.4),
            BotStartedSpeakingFrame(),
        ):
            with patch("pipecat.observers.user_bot_latency_observer.time.time", return_value=100.6):
                await observer.on_push_frame(FramePushed(
                    source=self.source, destination=self.destination, frame=frame,
                    direction=FrameDirection.DOWNSTREAM, timestamp=0,
                ))
        measurements = [c.args[1] for c in observer._call_event_handler.await_args_list
                        if c.args[0] == "on_latency_measured"]
        self.assertEqual(len(measurements), 1)
        self.assertAlmostEqual(measurements[0], 0.6)
