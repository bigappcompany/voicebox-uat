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
from voice_agent.flows.facts import FactExtractor, NumericRange
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

    def test_headcount_department_and_conversational_filler(self):
        # Spoken department headcount with conversational filler from live call
        text = "above let s say three for department but honey i i wouldn t to have a proper conversation with regards to this"
        update = self.extractor.extract(text, {"hiring_status": "yes"})
        self.assertEqual(update.values.get("headcount"), 3)

        # "twenty people for each department"
        update_dept2 = self.extractor.extract("we need twenty people for each department", {})
        self.assertEqual(update_dept2.values.get("headcount"), 20)

        # "three per team"
        update_team = self.extractor.extract("three per team", {})
        self.assertEqual(update_team.values.get("headcount"), 3)

        # pending_slot="headcount" with conversational trailer
        pending_q = Mock(slot="headcount")
        update_pending = self.extractor.extract("let s say four but we need to check", {}, pending_question=pending_q)
        self.assertEqual(update_pending.values.get("headcount"), 4)

        # pending_slot="headcount" should NOT extract time or timeline
        update_time = self.extractor.extract("call me tomorrow at 5 pm", {}, pending_question=pending_q)
        self.assertNotIn("headcount", update_time.values)

    def test_closing_state_trailing_affirmations_trigger_end_call(self):
        # 1. thank_you in PREFERENCE_RECORDED transitions to CLOSING
        slots = {"callback_state": "PREFERENCE_RECORDED", "callback_preference": "Friday at 2pm"}
        match = self.model.classify("thank you so much", fact_values=slots)
        routed = self.coordinator.route(match, slots)
        self.assertIsNotNone(routed)
        plan, speech = routed
        self.assertEqual(plan.slots_written.get("callback_state"), "CLOSING")
        self.assertIn("Have a great day!", speech)

        # 2. In CLOSING state, caller saying "yeah" triggers end_call
        closing_slots = {"callback_state": "CLOSING", "callback_preference": "Friday at 2pm"}
        match_yeah = self.model.classify("yeah", fact_values=closing_slots)
        routed_yeah = self.coordinator.route(match_yeah, closing_slots)
        self.assertIsNotNone(routed_yeah)
        plan_yeah, speech_yeah = routed_yeah
        self.assertEqual(plan_yeah.action, "end_call")
        self.assertEqual(plan_yeah.slots_written.get("callback_state"), "CLOSED")
        self.assertIn("Goodbye", speech_yeah)

        # 3. In CLOSING state, router handles short affirmative deterministically
        routed_ok = self.router.route("ok", self.bundle, slots=closing_slots)
        self.assertIsNotNone(routed_ok)
        plan_ok, speech_ok = routed_ok
        self.assertEqual(plan_ok.action, "end_call")
        self.assertIn("Goodbye", speech_ok)

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
class TestFixesItems1To7(unittest.IsolatedAsyncioTestCase):
    """Specific regression test cases for production call telemetry fixes 1-7."""

    def setUp(self):
        self.fe = FactExtractor()
        self.im = CanonicalIntentModel()
        self.bundle = make_bundle()

    def test_fix1_standalone_boolean_guard(self):
        """Fix 1: Leading yeah or not right now in long utterances must not hijack boolean pending questions."""
        pq_continue = PendingQuestion("ask_anything_else", "conversation_continue", "boolean", 1)
        pq_callback = PendingQuestion("ask_callback_consent", "followup_consent", "boolean", 1)
        pq_hiring = PendingQuestion("ask_hiring_status", "hiring_status", "boolean", 1)

        # Standalone booleans
        self.assertEqual(self.im.classify("yeah", pending_question=pq_continue).intent_id, "conversation_continue_yes")
        self.assertEqual(self.im.classify("no thanks", pending_question=pq_continue).intent_id, "conversation_continue_no")
        self.assertEqual(self.im.classify("not right now", pending_question=pq_continue).intent_id, "conversation_continue_no")
        self.assertEqual(self.im.classify("not right now", pending_question=pq_callback).intent_id, "callback_consent_no")
        self.assertEqual(self.im.classify("yes", pending_question=pq_hiring).intent_id, "provide_hiring_status")

        # Non-standalone utterances with leading boolean word must NOT trigger boolean intent
        match_marketing = self.im.classify("yeah you missed out on the marketing department", pending_question=pq_continue)
        self.assertNotEqual(match_marketing.intent_id, "conversation_continue_yes")

        match_summary = self.im.classify("not right now can you give me a summary of what you do", pending_question=pq_callback)
        self.assertNotEqual(match_summary.intent_id, "callback_consent_no")

        # Hiring status not erased by 'not right now' when pending question was not hiring_status
        facts = self.fe.extract("not right now can you give me a summary", {"hiring_status": "yes"}, pending_question=pq_continue)
        self.assertNotIn("hiring_status", facts.values)

    def test_fix2_explicit_facts_before_pending_boolean(self):
        """Fix 2: Explicit facts and corrections must be processed before pending boolean."""
        pq_continue = PendingQuestion("ask_anything_else", "conversation_continue", "boolean", 1)

        # Utterance with correction and department
        match = self.im.classify("yeah you missed out on the marketing department", pending_question=pq_continue)
        self.assertEqual(match.intent_id, "provide_role")

        # Utterance with correction and headcount
        match_headcount = self.im.classify("yeah actually six people", pending_question=pq_continue)
        self.assertEqual(match_headcount.intent_id, "provide_headcount")

    def test_fix3_marketing_aliases_and_department_preservation(self):
        """Fix 3: Marketing aliases and department preservation across turns."""
        update1 = self.fe.extract("we need people in the marketing department", {})
        self.assertIn("marketing", update1.values.get("roles", []))
        self.assertIn("marketing", update1.values.get("departments", []))

        # Digital marketing alias
        update_alias = self.fe.extract("looking for growth and digital marketing roles", {})
        self.assertIn("marketing", update_alias.values.get("roles", []))

        # Preservation across turns: turn 1 mentioned technology, turn 2 mentions marketing
        turn1_facts = {"roles": ["technology"], "departments": ["technology"]}
        update2 = self.fe.extract("we also need two people for marketing", turn1_facts)
        self.assertIn("technology", update2.values.get("roles", []))
        self.assertIn("marketing", update2.values.get("roles", []))
        self.assertIn("technology", update2.values.get("departments", []))
        self.assertIn("marketing", update2.values.get("departments", []))

    def test_fix4_reject_ambiguous_digit_sequences(self):
        """Fix 4: Reject ambiguous digit sequences like 'eight two five' instead of inventing ranges."""
        pq_headcount = PendingQuestion("ask_headcount", "headcount", "integer_or_range", 1)

        update_ambiguous = self.fe.extract("eight two five", {}, pending_question=pq_headcount)
        self.assertNotIn("headcount", update_ambiguous.values)

        update_digits = self.fe.extract("1 2 3", {}, pending_question=pq_headcount)
        self.assertNotIn("headcount", update_digits.values)

        # Genuine range without connector before noun
        update_valid = self.fe.extract("three four people in operations", {})
        self.assertEqual(update_valid.values.get("headcount"), NumericRange(3, 4, "people", True))

        # Genuine range with connector
        update_connector = self.fe.extract("about six to eight candidates", {})
        self.assertEqual(update_connector.values.get("headcount"), NumericRange(6, 8, "people", True))

    async def test_fix5_low_information_transcripts_buffered(self):
        """Fix 5: Buffer tiny low-information transcripts (hiria, very) without firing reprompt."""
        from live_v2 import V2RoutingController, _LOW_INFO_FRAGMENTS

        controller = V2RoutingController(
            "dummy_key",
            system_prompt="dummy_prompt",
            session=CallSession("call-1", self.bundle.tenant_id, self.bundle),
        )
        metrics = TurnMetrics(turn_id=1)
        controller._state = TurnState(turn_id=1, metrics=metrics)
        controller._pending_text = "hiria"
        await controller._settle_turn()
        self.assertEqual(controller._prefix_buffer, "hiria")
        self.assertEqual(controller._state.metrics.route, "stt-low-info-buffered")

        # In OPENING state, deterministic router suppresses incomplete_response for 1-2 word unknown utterances
        router = DeterministicRouter()
        plan = router.route("hiria", self.bundle, slots={"state": "OPENING"}, turn_id=1)
        self.assertIsNone(plan)

    def test_fix7_profile_for_prompt_freeform_expansion(self):
        """Fix 7: Open-ended requests get freeform profile; dynamic expansion in _on_interim."""
        from voice_agent.turns.endpoint_profiles import profile_for_prompt

        # Open-ended questions asking for elaboration return freeform
        self.assertEqual(
            profile_for_prompt("Could you tell me a little more about what you need help with?"),
            "freeform",
        )
        self.assertEqual(
            profile_for_prompt("What else can I help you with?"),
            "freeform",
        )
        # Genuinely boolean questions return yes_no
        self.assertEqual(
            profile_for_prompt("Are you looking to hire right now?"),
            "yes_no",
        )

    def test_option2_flux_profiles_wide_gap(self):
        """Option 2: Flux profiles maintain a wide eager-to-hard threshold gap (>= 0.30)."""
        from voice_agent.turns.endpoint_profiles import flux_profile

        for name, expected_eager, expected_eot in [
            ("yes_no", 0.30, 0.55),
            ("short_entity", 0.30, 0.58),
            ("requirements", 0.30, 0.60),
            ("freeform", 0.32, 0.62),
            ("fast", 0.30, 0.60),
        ]:
            prof = flux_profile(name)
            self.assertAlmostEqual(prof.eager_eot_threshold, expected_eager, places=2)
            self.assertAlmostEqual(prof.eot_threshold, expected_eot, places=2)
            self.assertGreaterEqual(round(prof.eot_threshold - prof.eager_eot_threshold, 2), 0.25)

    def test_option2_detected_profile_for_prompt(self):
        """Option 2: Prompt-detected profiles identify targeted entity vs freeform intents."""
        from voice_agent.turns.endpoint_profiles import detected_profile_for_prompt, profile_for_prompt

        self.assertEqual(detected_profile_for_prompt("How may I help you today?"), "freeform")
        self.assertEqual(detected_profile_for_prompt("What time tomorrow would be convenient?"), "short_entity")
        self.assertEqual(detected_profile_for_prompt("What roles are you planning to hire for?"), "requirements")
        self.assertEqual(detected_profile_for_prompt("Would you like a hiring specialist to follow up?"), "yes_no")
        self.assertIsNone(detected_profile_for_prompt("Thank you for your time. Goodbye."))
        self.assertEqual(profile_for_prompt("Thank you for your time. Goodbye."), "freeform")

    async def test_option2_eager_deterministic_staging_and_reuse(self):
        """Option 2: Deterministic plan is staged on eager input and reused on matching final transcript."""
        from live_v2 import V2RoutingController
        from main import TurnState
        from voice_agent.runtime.metrics import TurnMetrics
        from voice_agent.runtime.session import CallSession
        from voice_agent.agents.bundle import AgentBundle

        bundle = AgentBundle(
            agent_id="test_agent",
            version="1.0",
            tenant_id="goodbox",
            fact_profile={"company_name": "Goodbox"},
            stt_profile={"provider": "deepgram"},
        )
        session = CallSession(
            "test-call",
            agent=bundle,
            tenant_id="goodbox",
            slots={"callback_state": "AWAITING_DAY_TIME"},
            pending_question=PendingQuestion("ask_callback_day_time", "callback_preference", "date_and_time", 0),
        )
        controller = V2RoutingController("dummy_key", system_prompt="dummy_prompt", session=session)
        state = TurnState(turn_id=1, metrics=TurnMetrics(turn_id=1))
        controller._state = state

        # Caller provides callback time on eager interim
        await controller._start_candidate(state, "tomorrow at two pm", source="eager")
        self.assertTrue(getattr(state.metrics, "eager_deterministic_staged", False))
        self.assertIsNotNone(getattr(state, "eager_deterministic_speech", None))
        self.assertIn("two pm", state.eager_deterministic_speech)

        # Hard EOT arrives with matching transcript
        state.final_transcript = "tomorrow at two pm"
        await controller._commit_final_turn(state)
        self.assertTrue(getattr(state.metrics, "eager_deterministic_hit", False))
        self.assertEqual(state.metrics.decision_route, "callback-preference")


if __name__ == "__main__":
    unittest.main()
