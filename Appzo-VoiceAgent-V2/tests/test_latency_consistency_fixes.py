"""Unit tests for VoiceAgent V2 latency consistency and routing fixes (Fixes 1-5)."""

import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from main import (
    BotStartedSpeakingFrame,
    FrameDirection,
    InputAudioRawFrame,
    LLMFullResponseStartFrame,
    LiveLatencyObserver,
    StreamingVoiceController,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    TurnMetrics,
    TurnState,
)
from voice_agent.agents.bundle import AgentBundle
from voice_agent.flows.callbacks import CallbackCoordinator
from voice_agent.flows.facts import FactExtractor
from voice_agent.runtime.flags import RuntimeFlags
from voice_agent.runtime.intents import CanonicalIntentModel, IntentMatch
from voice_agent.runtime.latency_breakdown import LatencyBreakdown
from voice_agent.runtime.response_plan import ResponsePlan
from voice_agent.runtime.session import CallSession, PendingQuestion
from voice_agent.routing.deterministic import DeterministicRouter


def make_bundle() -> AgentBundle:
    return AgentBundle(agent_id="test-agent", version="1.0", tenant_id="test-tenant")


class TestFix1BookingClaimGuardGating(unittest.TestCase):
    """Fix 1: BookingClaimGuard gating and metrics."""

    def test_response_plan_does_not_require_guard_for_general_intents_with_callback_slots(self):
        """callback_sensitive must NOT be set just because callback slots exist."""
        from live_v2 import V2RoutingController

        bundle = make_bundle()
        controller = V2RoutingController(
            "dummy_key",
            system_prompt="dummy_prompt",
            session=CallSession(
                call_id="call-1",
                tenant_id=bundle.tenant_id,
                agent=bundle,
                slots={"callback_day": "tomorrow", "callback_time": "3pm", "callback_state": "AWAITING_TIME"},
            ),
        )
        plan, speech = controller._response_plan("do you do blue collar staffing?")
        self.assertFalse(plan.requires_booking_guard)

    def test_response_plan_requires_guard_for_scheduling_intents(self):
        """requires_booking_guard is True for scheduling/booking/callback intents."""
        from live_v2 import V2RoutingController

        bundle = make_bundle()
        controller = V2RoutingController(
            "dummy_key",
            system_prompt="dummy_prompt",
            session=CallSession(
                call_id="call-1",
                tenant_id=bundle.tenant_id,
                agent=bundle,
                slots={},
            ),
        )
        for utterance in [
            "can we schedule a callback?",
            "book an appointment please",
            "let's do a follow-up",
        ]:
            plan, _ = controller._response_plan(utterance)
            self.assertTrue(plan.requires_booking_guard, f"Failed for utterance: {utterance}")

    def test_turn_metrics_has_booking_guard_fields(self):
        metrics = TurnMetrics(turn_id=1)
        self.assertFalse(metrics.booking_guard_enabled)
        self.assertEqual(metrics.booking_guard_wait_ms, 0.0)


class TestFix2FactExtractorRoleFalsePositives(unittest.TestCase):
    """Fix 2: Remove 'it' -> technology role false positive in FactExtractor."""

    def setUp(self):
        self.extractor = FactExtractor()

    def test_pronoun_it_never_extracts_technology_role(self):
        """Ensure 'confirm it', 'leave it', 'forget it', 'do it', 'schedule it', 'that's it' never match."""
        phrases = [
            "confirm it",
            "leave it",
            "forget it",
            "do it",
            "schedule it",
            "that's it",
            "thats it",
            "confirm IT",
            "leave IT",
        ]
        for phrase in phrases:
            update = self.extractor.extract(phrase, {})
            roles = update.values.get("roles", [])
            self.assertNotIn(
                "technology",
                roles,
                f"Phrase '{phrase}' incorrectly extracted technology role: {roles}",
            )

    def test_contextual_it_phrases_extract_technology_role(self):
        """Phrases like 'it department', 'it team', 'it roles', 'it jobs', 'information technology' match."""
        phrases = [
            "we need people for our it department",
            "looking to hire an it team",
            "we have several it roles open",
            "need help with it jobs",
            "we need an it function lead",
            "hiring for information technology",
        ]
        for phrase in phrases:
            update = self.extractor.extract(phrase, {})
            roles = update.values.get("roles", [])
            self.assertIn(
                "technology",
                roles,
                f"Phrase '{phrase}' failed to extract technology role",
            )

    def test_explicit_uppercase_it_in_non_pronoun_context(self):
        """Uppercase IT in non-pronoun context extracts technology role."""
        update = self.extractor.extract("we are hiring for IT", {})
        roles = update.values.get("roles", [])
        self.assertIn("technology", roles)

    def test_pending_slot_roles_with_answer_it(self):
        """When pending_question slot is 'roles', 'IT' or 'it' extracts technology."""
        pending = PendingQuestion("ask_roles", "roles", "role_list")
        update = self.extractor.extract("IT", {}, pending_question=pending)
        roles = update.values.get("roles", [])
        self.assertIn("technology", roles)

        update_lower = self.extractor.extract("it", {}, pending_question=pending)
        roles_lower = update_lower.values.get("roles", [])
        self.assertIn("technology", roles_lower)


class TestFix3DeterministicControlIntents(unittest.TestCase):
    """Fix 3: Keep common conversational control intents out of GPT."""

    def setUp(self):
        self.coordinator = CallbackCoordinator()
        self.router = DeterministicRouter()
        self.bundle = make_bundle()

    def test_request_human_with_existing_preference(self):
        """3A: request_human with callback_preference answers deterministically with preference."""
        slots = {"callback_preference": "Wednesday at 2pm", "callback_state": "PREFERENCE_RECORDED"}
        routed = self.coordinator.route("request_human", slots)
        self.assertIsNotNone(routed)
        plan, speech = routed
        self.assertEqual(plan.route, "callback-preference-acknowledged")
        self.assertIn("Wednesday at 2pm", speech)
        self.assertIn("confirm availability", speech)

    def test_request_human_without_existing_preference(self):
        """3A: request_human without callback_preference asks for day and time."""
        slots = {}
        routed = self.coordinator.route("request_human", slots)
        self.assertIsNotNone(routed)
        plan, speech = routed
        self.assertEqual(plan.next_state, "CALLBACK")
        self.assertIn("What day and time would be convenient?", speech)
        self.assertEqual(plan.pending_question.slot, "callback_preference")

    def test_busy_with_existing_preference(self):
        """3B: busy with callback_preference answers deterministically with preference."""
        slots = {"callback_preference": "Friday at 10am", "callback_state": "PREFERENCE_RECORDED"}
        routed = self.coordinator.route("busy", slots)
        self.assertIsNotNone(routed)
        plan, speech = routed
        self.assertIn("Friday at 10am", speech)
        self.assertIn("Of course.", speech)

    def test_busy_without_existing_preference(self):
        """3B: busy without callback_preference asks for alternative time."""
        slots = {}
        routed = self.coordinator.route("busy", slots)
        self.assertIsNotNone(routed)
        plan, speech = routed
        self.assertEqual(plan.next_state, "CALLBACK")
        self.assertIn("What day and time would work better?", speech)

    def test_callback_confirmation_when_preference_recorded(self):
        """3C: In PREFERENCE_RECORDED, confirmations return callback_preference_acknowledged."""
        slots = {"callback_preference": "Thursday at 4pm", "callback_state": "PREFERENCE_RECORDED"}
        model = CanonicalIntentModel()
        for confirmation in [
            "yes",
            "yeah",
            "that's correct",
            "correct",
            "sounds good",
            "confirm it",
            "that's right",
            "okay that's fine",
        ]:
            match = model.classify(confirmation, fact_values=slots)
            self.assertEqual(match.intent_id, "callback_consent_yes", f"Failed intent for: {confirmation}")
            routed = self.coordinator.route(match, slots)
            self.assertIsNotNone(routed, f"Failed routing for: {confirmation}")
            plan, speech = routed
            self.assertEqual(plan.route, "callback-preference-acknowledged")
            self.assertIn("Thursday at 4pm", speech)

    def test_recall_callback_preference(self):
        """3D: recall_callback_preference returns preference directly from state."""
        slots = {"callback_preference": "Monday at 11am", "callback_state": "PREFERENCE_RECORDED"}
        model = CanonicalIntentModel()
        for recall in [
            "I already told you",
            "I told you already",
            "same time as earlier",
            "what time did I give you?",
            "as I mentioned before",
        ]:
            match = model.classify(recall, fact_values=slots)
            self.assertEqual(match.intent_id, "recall_callback_preference", f"Failed intent for: {recall}")
            routed = self.coordinator.route(match, slots)
            self.assertIsNotNone(routed, f"Failed routing for: {recall}")
            plan, speech = routed
            self.assertIn("Monday at 11am", speech)
            self.assertIn("You mentioned", speech)

    def test_precedence_in_response_plan(self):
        """3E: Deterministic control intent routing runs before falling through to hosted LLM."""
        from live_v2 import V2RoutingController

        bundle = make_bundle()
        controller = V2RoutingController(
            "dummy_key",
            system_prompt="dummy_prompt",
            session=CallSession(
                call_id="call-1",
                tenant_id=bundle.tenant_id,
                agent=bundle,
                slots={"callback_preference": "Tuesday at 3pm", "callback_state": "PREFERENCE_RECORDED"},
            ),
        )
        plan, speech = controller._response_plan("can I talk to a human?")
        self.assertIsNotNone(speech)
        self.assertNotEqual(plan.route, "hosted")
        self.assertEqual(plan.intent_id, "request_human")


class TestFix4LatencyAttributionAndTimestamps(unittest.IsolatedAsyncioTestCase):
    """Fix 4: Latency attribution, reporting locks, and speech timestamps."""

    def test_live_latency_observer_initialization(self):
        controller = Mock()
        observer = LiveLatencyObserver(controller, tts_transport="websocket", call_origin_at=0.0)
        self.assertIsInstance(observer._reporting_turn_ids, set)
        self.assertIsInstance(observer._reported_turn_ids, set)
        self.assertIsNone(observer._latest_voiced_audio_at)

    def test_schedule_report_locking(self):
        controller = Mock()
        observer = LiveLatencyObserver(controller, tts_transport="websocket", call_origin_at=0.0)
        metrics = TurnMetrics(turn_id=42)

        # First call schedules
        with patch("asyncio.create_task") as mock_task:
            observer._schedule_report_if_not_reported(metrics)
            mock_task.assert_called_once()
            self.assertIn(42, observer._reporting_turn_ids)

        # Second call while reporting is blocked
        with patch("asyncio.create_task") as mock_task2:
            observer._schedule_report_if_not_reported(metrics)
            mock_task2.assert_not_called()

    async def test_report_latency_releases_lock_on_none_breakdown(self):
        """If breakdown is None, turn_id is removed from _reporting_turn_ids and NOT in _reported_turn_ids."""
        controller = Mock()
        controller.metrics_by_turn = {}
        controller._model = "test-model"
        observer = LiveLatencyObserver(controller, tts_transport="websocket", call_origin_at=0.0)
        metrics = TurnMetrics(turn_id=42)  # Missing hard_eot and first_audible -> breakdown is None

        observer._reporting_turn_ids.add(42)
        await observer._report_latency(metrics)

        self.assertNotIn(42, observer._reporting_turn_ids)
        self.assertNotIn(42, observer._reported_turn_ids)

    async def test_inbound_audio_updates_latest_voiced_audio_at(self):
        """Inbound audio frame with RMS >= threshold updates _latest_voiced_audio_at."""
        controller = Mock()
        state = TurnState(turn_id=1, metrics=TurnMetrics(turn_id=1))
        controller._state = state
        observer = LiveLatencyObserver(controller, tts_transport="websocket", call_origin_at=0.0)

        # Create 16-bit PCM audio with non-zero amplitude
        import struct
        samples = [5000] * 160  # Loud samples
        pcm_bytes = struct.pack(f"<{len(samples)}h", *samples)
        frame = InputAudioRawFrame(audio=pcm_bytes, sample_rate=16000, num_channels=1)

        event = SimpleNamespace(direction=FrameDirection.DOWNSTREAM, frame=frame, processor=None)
        await observer.on_process_frame(event)

        self.assertIsNotNone(observer._latest_voiced_audio_at)
        self.assertIsNotNone(state.metrics.last_voiced_at)

    async def test_fresh_voice_timestamp_attached_to_new_turn(self):
        """When a new turn is created, fresh _latest_voiced_audio_at is assigned to last_voiced_at."""
        controller = StreamingVoiceController(
            "dummy_key",
            system_prompt="prompt",
        )
        observer = LiveLatencyObserver(controller, tts_transport="websocket", call_origin_at=0.0)
        controller._latency_observer = observer

        now = time.perf_counter()
        observer._latest_voiced_audio_at = now

        await controller._start_turn()
        self.assertEqual(controller._state.metrics.last_voiced_at, now)


class TestFix5TTSLatencyTimestampSemantics(unittest.TestCase):
    """Fix 5: Separate response_release_at, tts_requested_at, tts_first_audio_at."""

    def test_metrics_has_response_release_at(self):
        metrics = TurnMetrics(turn_id=1)
        self.assertIsNone(metrics.response_release_at)
        self.assertIsNone(metrics.tts_requested_at)

    def test_latency_breakdown_separates_release_and_tts_dispatch(self):
        metrics = TurnMetrics(turn_id=1, route="hosted")
        metrics.turn_committed_at = 100.0
        metrics.response_release_at = 100.5
        metrics.tts_requested_at = 100.8
        metrics.tts_first_audio_at = 101.2
        metrics.output_first_non_silent_at = 101.5

        breakdown = LatencyBreakdown.from_turn(
            metrics, model="llama-3.3-70b-versatile", tts_transport="websocket"
        )
        self.assertIsNotNone(breakdown)
        keys = [c.key for c in breakdown.contributions]
        self.assertIn("response.release", keys)
        self.assertIn("response.release_to_tts", keys)
        self.assertIn("tts.first_audio", keys)

        # Check durations
        c_map = {c.key: c.duration_secs for c in breakdown.contributions}
        self.assertAlmostEqual(c_map["response.release_to_tts"], 0.3, places=2)
        self.assertAlmostEqual(c_map["tts.first_audio"], 0.4, places=2)


class TestFix6PoliteClosuresAndNumbersAndKeepAlive(unittest.IsolatedAsyncioTestCase):
    """Fix 6: Polite closures (thank you / goodbye), double-digit numbers, and STT keepalive."""

    def setUp(self):
        self.model = CanonicalIntentModel()
        self.coordinator = CallbackCoordinator()
        self.router = DeterministicRouter()
        self.bundle = make_bundle()
        self.extractor = FactExtractor()

    def test_thank_you_intent_classification(self):
        for phrase in ["thank you", "thanks", "thank you so much", "thanks a lot", "ok thank you", "many thanks"]:
            match = self.model.classify(phrase)
            self.assertEqual(match.intent_id, "thank_you", f"Failed for phrase: {phrase}")

        for negated in ["no thanks", "no thank you", "not interested thanks"]:
            match = self.model.classify(negated)
            self.assertNotEqual(match.intent_id, "thank_you", f"Incorrectly matched thank_you for: {negated}")

    def test_goodbye_compound_phrases(self):
        for phrase in [
            "ok thank you goodbye", "thank you goodbye", "ok goodbye",
            "goodbye thank you", "bye bye", "have a good day", "have a great day",
        ]:
            match = self.model.classify(phrase)
            self.assertEqual(match.intent_id, "goodbye", f"Failed goodbye match for: {phrase}")

        match_neg = self.model.classify("do not end the call")
        self.assertNotEqual(match_neg.intent_id, "goodbye")

    def test_thank_you_routing_in_preference_recorded(self):
        slots = {"callback_state": "PREFERENCE_RECORDED", "callback_preference": "Friday at 6pm"}
        match = self.model.classify("thank you", fact_values=slots)
        self.assertEqual(match.intent_id, "thank_you")
        routed = self.coordinator.route(match, slots)
        self.assertIsNotNone(routed)
        plan, speech = routed
        self.assertEqual(plan.route, "callback-preference-acknowledged")
        self.assertIn("You're welcome!", speech)
        self.assertIn("Friday at 6pm", speech)

    def test_goodbye_falls_through_callback_coordinator_to_router(self):
        slots = {"callback_state": "PREFERENCE_RECORDED", "callback_preference": "Friday at 6pm"}
        match = self.model.classify("ok thank you goodbye", fact_values=slots)
        self.assertEqual(match.intent_id, "goodbye")
        # Coordinator returns None for goodbye
        routed_coord = self.coordinator.route(match, slots)
        self.assertIsNone(routed_coord)
        # DeterministicRouter handles goodbye with end_call
        routed = self.router.route("ok thank you goodbye", self.bundle, intent=match, slots=slots)
        self.assertIsNotNone(routed)
        plan, speech = routed
        self.assertEqual(plan.action, "end_call")
        self.assertIn("Goodbye", speech)

    def test_thank_you_in_general_state_deterministic_response(self):
        slots = {}
        match = self.model.classify("thank you", fact_values=slots)
        self.assertEqual(match.intent_id, "thank_you")
        routed = self.router.route("thank you", self.bundle, intent=match, slots=slots)
        self.assertIsNotNone(routed)
        plan, speech = routed
        self.assertEqual(plan.action, "continue")
        self.assertIn("You're welcome!", speech)
        self.assertNotIn("Could you tell me a little more", speech)

    def test_unknown_short_response_in_preference_recorded_does_not_ask_requirements(self):
        slots = {"callback_state": "PREFERENCE_RECORDED", "callback_preference": "Friday at 6pm"}
        routed = self.router.route("got it", self.bundle, slots=slots)
        self.assertIsNotNone(routed)
        plan, speech = routed
        self.assertEqual(plan.intent_id, "callback_preference_acknowledged")
        self.assertNotIn("Could you tell me a little more about what you need help with?", speech)
        self.assertIn("Friday at 6pm", speech)

    def test_fact_extractor_double_digit_numbers(self):
        # Range of tens
        update_range = self.extractor.extract("we need thirty to forty people", {})
        hc = update_range.values.get("headcount")
        self.assertIsNotNone(hc)
        self.assertEqual(str(hc), "about 30-40 people")

        # Single tens
        update_sixty = self.extractor.extract("hire around sixty engineers", {})
        self.assertEqual(update_sixty.values.get("headcount"), 60)
        self.assertIn("technology", update_sixty.values.get("roles", []))

        # Compound number "twenty five" / "twenty-five"
        update_compound = self.extractor.extract("looking for twenty-five candidates", {})
        self.assertEqual(update_compound.values.get("headcount"), 25)

        # Timeline with tens
        update_timeline = self.extractor.extract("in forty days", {})
        self.assertEqual(update_timeline.values.get("hiring_timeline"), "forty days")

    async def test_ordered_flux_watchdog_keepalive(self):
        from voice_agent.turns.flux import OrderedFluxSTTService
        import json
        stt = OrderedFluxSTTService(api_key="test")
        stt._transport_is_active = Mock(side_effect=[True, False])
        stt._user_is_speaking = False
        stt._last_stt_time = time.monotonic() - 6.0  # Expired > 5.0s
        stt._websocket = Mock()
        stt.send_with_retry = AsyncMock()
        await stt._watchdog_task_handler()
        stt.send_with_retry.assert_awaited_once_with(json.dumps({"type": "KeepAlive"}), stt._report_error)


if __name__ == "__main__":
    unittest.main()

