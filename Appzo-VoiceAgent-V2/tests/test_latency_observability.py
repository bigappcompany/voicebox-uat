"""Unit tests for latency observability and metrics."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from main import (
    BotStartedSpeakingFrame,
    FrameDirection,
    LLMFullResponseStartFrame,
    LiveLatencyObserver,
    TTSStoppedFrame,
    TurnMetrics,
)
from voice_agent.llm.context_adapter import ConversationContextAdapter, HostedContext
from voice_agent.runtime.intents import CanonicalIntentModel
from voice_agent.runtime.latency_breakdown import LatencyBreakdown, LatencyContribution
from voice_agent.speech.booking_guard import BookingClaimGuard


def make_observer(controller=None):
    c = controller or Mock()
    c.metrics_by_turn = {}
    c._model = "test-model"
    session = SimpleNamespace(call_id="test-call", tenant_id="test-tenant", agent=SimpleNamespace(version="1.0"))
    return LiveLatencyObserver(
        c, tts_transport="websocket", session=session, call_origin_at=0.0
    )


class TestLatencyObservability(unittest.IsolatedAsyncioTestCase):

    def test_observer_initialization(self):
        observer = make_observer()
        self.assertIsInstance(observer._reported_turn_ids, set)
        self.assertIsInstance(observer._samples, dict)

    def test_turn_metrics_has_context_fields(self):
        metrics = TurnMetrics(turn_id=1)
        self.assertTrue(hasattr(metrics, "pipecat_context_read_ms"))
        self.assertTrue(hasattr(metrics, "context_selection_ms"))
        self.assertTrue(hasattr(metrics, "context_token_estimation_ms"))
        self.assertTrue(hasattr(metrics, "prompt_build_ms"))

    def test_flux_derived_metrics(self):
        metrics = TurnMetrics(turn_id=1)
        metrics.last_voiced_at = 100.0
        metrics.eager_eot_at = 101.5
        metrics.turn_committed_at = 102.0

        observer = make_observer()
        record = observer._record(
            metrics,
            LatencyBreakdown(
                turn_id=1,
                measured_from="last_voiced",
                started_at=100.0,
                ended_at=105.0,
                contributions=(),
            ),
            first_audible=105.0,
        )
        self.assertEqual(record["raw_speech_end_to_eager_eot_ms"], 1500.0)
        self.assertEqual(record["eager_eot_to_hard_eot_ms"], 500.0)
        self.assertEqual(record["raw_speech_end_to_hard_eot_ms"], 2000.0)

    def test_speculation_derived_metrics(self):
        metrics = TurnMetrics(turn_id=1)
        metrics.speculative_started_at = 100.0
        metrics.first_safe_text_at = 101.0
        metrics.spec_tts_started_at = 101.1
        metrics.spec_tts_pcm_ready_at = 101.5
        metrics.turn_committed_at = 102.0

        observer = make_observer()
        record = observer._record(
            metrics,
            LatencyBreakdown(
                turn_id=1,
                measured_from="last_voiced",
                started_at=100.0,
                ended_at=105.0,
                contributions=(),
            ),
            first_audible=105.0,
        )
        self.assertEqual(record["spec_start_to_first_safe_ms"], 1000.0)
        self.assertEqual(record["spec_start_to_pcm_ready_ms"], 400.0)
        self.assertEqual(record["spec_tts_savings_ms"], 500.0)

    def test_boolean_interpreter_yes(self):
        model = CanonicalIntentModel()
        self.assertEqual(model._parse_boolean("that is true yeah"), "yes")
        self.assertEqual(model._parse_boolean("yes please"), "yes")
        self.assertEqual(model._parse_boolean("sure"), "yes")

    def test_boolean_interpreter_no(self):
        model = CanonicalIntentModel()
        self.assertEqual(model._parse_boolean("yeah no dont do that"), "no")
        self.assertEqual(model._parse_boolean("nope"), "no")
        self.assertEqual(model._parse_boolean("not interested"), "no")

    def test_boolean_interpreter_ambiguous(self):
        model = CanonicalIntentModel()
        self.assertEqual(model._parse_boolean("I am not sure"), "ambiguous")

    def test_booking_guard_catches_unsupported(self):
        text = "I will arrange the follow-up for you tomorrow."
        result = BookingClaimGuard.check(text)
        self.assertIn("confirmation from the team", result)

    def test_booking_guard_allows_safe(self):
        text = "Got it, you want a meeting. What day works for you?"
        result = BookingClaimGuard.check(text)
        self.assertNotIn("confirmation from the team", result)

    def test_latency_breakdown_formatting(self):
        breakdown = LatencyBreakdown(
            turn_id=1,
            measured_from="hard_eot",
            started_at=100.0,
            ended_at=102.0,
            contributions=(
                LatencyContribution("test", "test_label", "test_kind", "test_owner", 100.0, 102.0),
            ),
        )
        lines = breakdown.turn_contribution_lines()
        self.assertTrue(any("TOTAL" in line for line in lines))
        self.assertTrue(any("test_label" in line for line in lines))

    @patch("voice_agent.llm.context_adapter.ConversationContextAdapter._dialogue")
    def test_hosted_context_timings(self, mock_dialogue):
        mock_dialogue.return_value = [{"role": "user", "content": "hi"}]
        from voice_agent.agents.bundle import AgentBundle
        from voice_agent.runtime.session import CallSession

        session = CallSession(
            call_id="call-1",
            tenant_id="t1",
            agent=AgentBundle(agent_id="a1", version="1.0", tenant_id="t1"),
        )
        adapter = ConversationContextAdapter(Mock(), session)
        context = adapter.build_hosted_context(current_user_text="hi")
        self.assertGreaterEqual(context.pipecat_read_ms, 0.0)
        self.assertGreaterEqual(context.selection_ms, 0.0)
        self.assertGreaterEqual(context.token_estimation_ms, 0.0)

    def test_report_deduplication(self):
        observer = make_observer()
        observer._reported_turn_ids.add(1)
        metrics = TurnMetrics(turn_id=1, route="hosted")
        with patch("asyncio.create_task") as mock_task:
            observer._schedule_report_if_not_reported(metrics)
            mock_task.assert_not_called()

    async def test_bot_started_is_only_a_fallback_when_pcm_metrics_are_enabled(self):
        state = Mock()
        state.metrics = TurnMetrics(turn_id=1, route="hosted")
        state.metrics.tts_requested_at = 1.0
        controller = Mock()
        controller._state = state
        controller.metrics_by_turn = {}
        latency_observer = make_observer(controller)
        event = SimpleNamespace(
            direction=FrameDirection.DOWNSTREAM, frame=BotStartedSpeakingFrame()
        )
        with patch("asyncio.create_task") as create_task:
            await latency_observer.on_push_frame(event)
            create_task.assert_not_called()

    async def test_final_stop_reports_when_output_pcm_was_not_observed(self):
        state = Mock()
        state.metrics = TurnMetrics(turn_id=1, route="hosted")
        state.metrics.tts_requested_at = 1.0
        controller = Mock()
        controller._state = state
        controller.metrics_by_turn = {}
        latency_observer = make_observer(controller)
        event = SimpleNamespace(
            direction=FrameDirection.DOWNSTREAM, frame=TTSStoppedFrame()
        )
        with patch("asyncio.create_task") as create_task:
            await latency_observer.on_push_frame(event)
            create_task.assert_called_once()

    async def test_late_stop_is_attributed_to_response_owner_not_new_current_turn(self):
        old_state = SimpleNamespace(metrics=TurnMetrics(turn_id=7, route="hosted"))
        old_state.metrics.tts_requested_at = 1.0
        new_state = SimpleNamespace(metrics=TurnMetrics(turn_id=8, route="pending"))
        controller = Mock()
        controller._state = old_state
        controller.metrics_by_turn = {}
        observer = make_observer(controller)

        await observer.on_push_frame(SimpleNamespace(
            direction=FrameDirection.DOWNSTREAM,
            frame=LLMFullResponseStartFrame(),
        ))
        controller._state = new_state

        with patch("asyncio.create_task") as create_task:
            await observer.on_push_frame(SimpleNamespace(
                direction=FrameDirection.DOWNSTREAM,
                frame=TTSStoppedFrame(),
            ))
            create_task.assert_called_once()

        self.assertIn(7, observer._reported_turn_ids | observer._reporting_turn_ids)
        self.assertNotIn(8, observer._reported_turn_ids | observer._reporting_turn_ids)

    def test_booking_guard_releases_safe_comma_clause_without_waiting_for_sentence(self):
        guard = BookingClaimGuard()
        self.assertEqual(guard.push("Certainly, "), "Certainly, ")
        self.assertEqual(guard.push("our team can help."), "our team can help.")

    def test_booking_guard_blocks_claim_ending_at_comma(self):
        guard = BookingClaimGuard()
        self.assertIn("confirmation from the team", guard.push("I will arrange, "))


if __name__ == "__main__":
    unittest.main()
