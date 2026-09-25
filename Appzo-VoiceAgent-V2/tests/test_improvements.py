import asyncio
import json
import os
import struct
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from main import LiveLatencyObserver, TurnMetrics
from pipecat.frames.frames import TTSAudioRawFrame
from pipecat.observers.base_observer import FrameProcessed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_output import BaseOutputTransport
from scripts.record_baseline import (
    evaluate_targets,
    parse_records,
    summarize,
    summarize_contributions,
    summarize_user_perceived,
)
from voice_agent.agents.bundle import AgentBundle
from voice_agent.agents.compiler import AgentCompiler
from voice_agent.agents.registry import AgentBundleRegistry
from voice_agent.flows.callbacks import CallbackCoordinator
from voice_agent.flows.facts import FactExtractor, NumericRange
from voice_agent.llm.prompt_builder import PromptBuilder
from voice_agent.routing.deterministic import DeterministicRouter
from voice_agent.runtime.bootstrap import RuntimeBootstrap
from voice_agent.runtime.flags import RuntimeFlags
from voice_agent.runtime.intents import CanonicalIntentModel
from voice_agent.runtime.latency_breakdown import LatencyBreakdown
from voice_agent.runtime.response_plan import ResponsePlan
from voice_agent.speech.greeting_cache import (
    CachedGreeting,
    CachedGreetingPlayer,
    GreetingCache,
    GreetingCacheKey,
    GreetingCaptureProcessor,
)
from voice_agent.speech.audio_quality import InitialSilenceTrimmer
from voice_agent.speech.safe_chunker import SafeSpeechChunker
from voice_agent.speech.booking_guard import BookingClaimGuard
from voice_agent.turns.endpoint_profiles import flux_profile, profile_for_prompt


def agent(**changes):
    values = {
        "agent_id": "agent",
        "version": "v1",
        "tenant_id": "tenant",
        "invariant_prompt": "Be concise.",
        "flow_graph": {"initial_state": "OPEN", "states": {"OPEN": {"objective": "Collect requirements."}}},
        "compiled_prompt": {"invariant": "Stable policy.", "states": {"OPEN": {"objective": "Ask only for missing facts."}}},
        "cached_utterances": {"repeat": "Please repeat that.", "faq:office hours": "Nine to five."},
    }
    values.update(changes)
    return AgentBundle(**values)


class ImprovementUnitTests(unittest.TestCase):
    def test_low_latency_flux_profiles_and_prompt_classification(self):
        self.assertEqual(flux_profile("fast").eot_timeout_ms, 1200)
        self.assertEqual(profile_for_prompt("Are you hiring right now?"), "yes_no")
        self.assertEqual(profile_for_prompt("Is your company hiring right now?"), "yes_no")
        self.assertEqual(profile_for_prompt("What time should we call?"), "short_entity")
        self.assertEqual(profile_for_prompt("Which roles and how many people?"), "requirements")
        self.assertEqual(profile_for_prompt("Is this a good time to speak?"), "yes_no")
        self.assertEqual(
            profile_for_prompt("Could you share the roles and expected timeline?"),
            "requirements",
        )

    def test_safe_chunker_timer_never_emits_a_single_word(self):
        chunker = SafeSpeechChunker(min_chars=20, min_words=3, max_wait_ms=80)
        self.assertEqual(chunker.push("Hello ", now=1.0), [])
        self.assertEqual(chunker.push("there", now=2.0), [])
        self.assertEqual(chunker.flush(), ["Hello there"])

    def test_safe_chunker_releases_a_complete_punctuated_phrase_without_waiting_for_next_delta(self):
        chunker = SafeSpeechChunker(min_chars=20, min_words=3, max_wait_ms=240)
        self.assertEqual(
            chunker.push("We can discuss your hiring plans.", now=1.0),
            ["We can discuss your hiring plans."],
        )

    def test_safe_chunker_wall_clock_release_does_not_need_another_token(self):
        chunker = SafeSpeechChunker(min_chars=20, min_words=3, max_wait_ms=60)
        self.assertEqual(chunker.push("We can discuss your hiring", now=1.0), [])
        self.assertEqual(chunker.release_due(now=1.061), ["We can discuss your hiring"])

    def test_pricing_intent_wins_over_model_identity(self):
        match = CanonicalIntentModel().classify("What's your pricing model?")
        self.assertEqual(match.intent_id, "faq_pricing")

    def test_role_local_counts_cover_operations_and_transcribed_ranges(self):
        facts = FactExtractor().extract(
            "three four people for my tech department and a couple of people for my operations department",
            {},
        ).values
        self.assertEqual(facts["headcount_by_role"]["technology"], NumericRange(3, 4, "people", True))
        self.assertEqual(facts["headcount_by_role"]["operations"], 2)

    def test_booking_guard_blocks_unverified_connect_and_meeting_claims(self):
        guard = BookingClaimGuard()
        self.assertEqual(
            guard.push("I will connect with you tomorrow at five pm."),
            "Your requested follow-up time needs confirmation from the team. ",
        )
        self.assertEqual(
            guard.push("I have noted the meeting for tomorrow."),
            "Your requested follow-up time needs confirmation from the team. ",
        )
        self.assertEqual(guard.push("Noted, you are hiring five people."), "Noted, you are hiring five people.")

    def test_compiler_creates_runtime_invariant_once_when_authoring_is_large(self):
        source = "You are Riya calling from Example Staffing. " + "General material. " * 600 + "Never claim a callback is confirmed."
        compiled = AgentCompiler().compile_goodbox({"chatbot_id": "a", "tenant_id": "t", "prompt": source})
        self.assertLessEqual(len(compiled.compiled_prompt["invariant"]), 3600)
        self.assertIn("Never claim a callback is confirmed.", compiled.compiled_prompt["invariant"])

    def test_initial_silence_trimmer_preserves_preroll(self):
        trimmer = InitialSilenceTrimmer()
        silence = struct.pack("<240h", *([0] * 240))
        voice = struct.pack("<240h", *([1000] * 240))
        audio, trimmed_ms, audible = trimmer._trim(silence + silence + voice, 24000)
        self.assertTrue(audible)
        self.assertEqual(trimmed_ms, 0.0)  # 20ms pre-roll preserves both silent blocks.
        self.assertEqual(audio, silence + silence + voice)

    def test_prompt_invariant_is_bounded(self):
        with patch.dict("os.environ", {"V2_MAX_INVARIANT_CHARS": "100"}):
            builder = PromptBuilder()
        messages = builder.build(
            agent=agent(invariant_prompt="A" * 500, compiled_prompt={}), state={"name": "OPEN"}, slots={},
            route=ResponsePlan("hosted"), knowledge=[], history=[], user_text="Hello",
        )
        self.assertIn("AUTHORING DETAIL OMITTED", messages[0]["content"])
        self.assertLess(builder.section_token_estimates["invariant"], 50)

    def test_fact_extraction_retains_values_and_accepts_corrections(self):
        extractor = FactExtractor(default_profile="recruitment")
        first = extractor.extract(
            "We are hiring five people for operations and tech in five months.", {}
        )
        self.assertEqual(first.values["hiring_status"], "yes")
        self.assertEqual(first.values["headcount"], 5)
        self.assertEqual(first.values["roles"], ["operations", "technology"])
        self.assertEqual(first.values["departments"], ["operations", "technology"])
        self.assertEqual(first.values["hiring_timeline"], "five months")
        correction = extractor.extract("Actually six, not five.", first.values)
        self.assertEqual(correction.values["headcount"], 6)
        self.assertEqual(correction.corrected, ("headcount",))

    def test_headcount_each_is_retained_per_role(self):
        facts = FactExtractor(default_profile="recruitment").extract(
            "We need five people each for operations and tech.", {}
        ).values
        self.assertEqual(facts["headcount_by_role"], {"operations": 5, "technology": 5})

    def test_timeline_correction_replaces_prior_value(self):
        update = FactExtractor(default_profile="recruitment").extract(
            "Actually two months, not five.", {"hiring_timeline": "five months"}
        )
        self.assertEqual(update.values["hiring_timeline"], "two months")
        self.assertEqual(update.corrected, ("hiring_timeline",))

    def test_fact_role_aliases_are_tenant_scoped(self):
        extractor = FactExtractor({"role_aliases": {"customer-success": ["client happiness"]}})
        self.assertEqual(
            extractor.extract("We need two people for client happiness.", {}).values["roles"],
            ["customer-success"],
        )

    def test_callback_state_machine_records_only_a_preference(self):
        callback = CallbackCoordinator()
        slots = {"callback_state": callback.FOLLOWUP_OFFERED}
        plan, speech = callback.route("yes please", slots)
        slots.update(plan.slots_written)
        self.assertEqual(slots["callback_state"], callback.AWAITING_DAY_TIME)
        plan, speech = callback.route("tomorrow at three p.m.", slots)
        self.assertEqual(plan.route, "callback-preference")
        self.assertIn("not a confirmed booking", speech)
        self.assertNotRegex(speech.casefold(), r"\b(?:scheduled|booked)\b")

    def test_normal_hosted_plan_does_not_require_booking_buffer(self):
        plan = ResponsePlan("hosted")
        self.assertFalse(plan.requires_booking_guard)

    def test_extended_router_can_be_rolled_back_independently(self):
        current = agent()
        self.assertEqual(DeterministicRouter(extended=True).route("repeat that", current)[0].intent_id, "repeat")
        self.assertIsNone(DeterministicRouter(extended=False).route("repeat that", current))
        self.assertEqual(DeterministicRouter(extended=False).route("office hours", current)[0].route, "cache")

    def test_compiled_prompt_is_state_scoped_bounded_and_reports_tokens(self):
        builder = PromptBuilder(max_history_turns=1)
        messages = builder.build(
            agent=agent(), state={"name": "OPEN"}, slots={"headcount": 5},
            route=ResponsePlan("hosted"), knowledge=[],
            history=[
                {"role": "user", "content": "old"}, {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "recent"}, {"role": "assistant", "content": "recent answer"},
            ], user_text="What next?",
        )
        self.assertIn("Stable policy.", messages[0]["content"])
        self.assertIn("Ask only for missing facts.", messages[0]["content"])
        self.assertIn('KNOWN FACTS: {"headcount":5}', messages[0]["content"])
        self.assertEqual([item["content"] for item in messages[-3:]], ["recent", "recent answer", "What next?"])
        self.assertGreater(builder.section_token_estimates["total"], 0)

    def test_prompt_builder_safely_serializes_complex_and_typed_slots(self):
        from voice_agent.flows.facts import NumericRange, FactValue
        builder = PromptBuilder()
        complex_slots = {
            "headcount": NumericRange(5, 10, "people", True),
            "hiring_timeline": NumericRange(2, 4, "months", False),
            "fact_provenance": FactValue(value=NumericRange(1, 3), confidence=0.95, source_turn=2, raw_text="1 to 3"),
            "roles": ["developer", "tester"],
            "departments": {"eng", "qa"},
            "mixed_keys": {2: "second", 1: "first", "alpha": "test"},
            "null_value": None,
        }
        # Verify custom actions on agent as list or None as well
        custom_agent = agent(actions=["transfer", "escalate"])
        messages = builder.build(
            agent=custom_agent,
            state={"name": "FOLLOWUP"},
            slots=complex_slots,
            route=ResponsePlan("hosted"),
            knowledge=[],
            history=[],
            user_text="we are hiring",
        )
        self.assertIn("KNOWN FACTS:", messages[0]["content"])
        self.assertIn('"headcount":"about 5-10 people"', messages[0]["content"])
        self.assertIn('"hiring_timeline":"2-4 months"', messages[0]["content"])
        self.assertIn('"fact_provenance":"1-3"', messages[0]["content"])
        self.assertIn('"1":"first"', messages[0]["content"])
        self.assertIn('"2":"second"', messages[0]["content"])
        self.assertIn("ALLOWED ACTIONS: transfer, escalate", messages[0]["content"])

    def test_master_switch_disables_improvements(self):
        with patch.dict("os.environ", {"ENABLE_V2_IMPROVEMENTS": "false"}, clear=False):
            flags = RuntimeFlags.from_env()
        self.assertFalse(flags.enable_cached_greeting)
        self.assertFalse(flags.enable_spec_tts)
        self.assertFalse(flags.enable_compiled_prompts)

    def test_baseline_summary_groups_and_uses_nearest_rank_percentiles(self):
        records = [
            {"route": "hosted", "endpoint_mode": "fast", "model": "m", "hard_eot_at": 1, "bot_started_at": 1.1},
            {"route": "hosted", "endpoint_mode": "fast", "model": "m", "hard_eot_at": 2, "bot_started_at": 2.3},
            {"route": "v2-error", "endpoint_mode": "fast", "model": "m", "hard_eot_at": 3, "bot_started_at": 9},
        ]
        metrics = summarize(records)["hosted|fast|m"]
        self.assertEqual(metrics, {"n": 2, "p50_ms": 100.0, "p90_ms": 300.0, "p95_ms": 300.0, "p99_ms": 300.0})

    def test_baseline_parser_accepts_server_log_marker(self):
        record = {"route": "hosted", "hard_eot_at": 1, "bot_started_at": 2}
        parsed = parse_records(
            "2026-01-01 | INFO | ordinary line\n"
            "  0.100s speech synthesis [service: Cartesia]\n"
            "2026-01-01 | INFO | LATENCY RECORD | " + json.dumps(record)
        )
        self.assertEqual(parsed, [record])

    def test_baseline_parser_merges_whole_utterance_cadence(self):
        latency = {"call_id": "c", "turn_id": 2, "route": "hosted"}
        cadence = {"call_id": "c", "turn_id": 2, "output_max_packet_gap_ms": 420}
        parsed = parse_records(
            "INFO | LATENCY RECORD | " + json.dumps(latency) + "\n"
            "INFO | AUDIO CADENCE RECORD | " + json.dumps(cadence)
        )
        self.assertEqual(parsed[0]["output_max_packet_gap_ms"], 420)

    def test_baseline_summarizes_structured_layer_contributions(self):
        records = [
            {
                "latency_breakdown": {
                    "total_secs": 1.0,
                    "contributions": [
                        {
                            "key": "tts.first_audio",
                            "owner_kind": "service",
                            "owner": "Cartesia/websocket-stream",
                            "duration_secs": 0.1,
                        },
                        {
                            "key": "response.first_safe_text",
                            "owner_kind": "service",
                            "owner": "gpt-4.1-mini",
                            "duration_secs": 0.9,
                        },
                    ],
                }
            }
        ]
        summary = summarize_contributions(records)
        self.assertEqual(summary["tts.first_audio"]["total_share_pct"], 10.0)
        self.assertEqual(summary["response.first_safe_text"]["p50_ms"], 900.0)

    def test_baseline_separates_user_perceived_cohort_and_gates(self):
        record = {
            "hard_eot_at": 1.2,
            "first_audible_at": 1.5,
            "latency_breakdown": {
                "measured_from": "last_voiced_audio",
                "total_secs": .5,
                "contributions": [
                    {"key": "endpointing.final_transcript", "duration_secs": .2}
                ],
            },
            "turn_resumed_count": 0,
            "output_max_packet_gap_ms": 20,
        }
        self.assertEqual(summarize_user_perceived([record])["p50_ms"], 500.0)
        passed, lines = evaluate_targets([record] * 50)
        self.assertTrue(passed, lines)

    def test_audible_detector_distinguishes_silence_from_voice(self):
        observer = LiveLatencyObserver(SimpleNamespace(_state=None), tts_transport="test")
        silence = struct.pack("<100h", *([0] * 100))
        voice = struct.pack("<100h", *([1000] * 100))
        self.assertFalse(observer._is_audible(silence))
        self.assertTrue(observer._is_audible(voice))

    def test_latency_breakdown_is_additive_and_layer_owned(self):
        metrics = TurnMetrics(
            turn_id=7,
            last_voiced_at=1.0,
            provider_eot_at=1.2,
            aggregator_stop_at=1.21,
            turn_committed_at=1.2,
            commit_at=1.22,
            first_safe_text_at=1.50,
            tts_requested_at=1.52,
            tts_first_audio_at=1.62,
            output_first_packet_at=1.625,
            output_first_non_silent_at=1.64,
            route="v2-hosted",
            endpoint_profile="fast",
        )
        breakdown = LatencyBreakdown.from_turn(
            metrics, model="gpt-4.1-mini", tts_transport="websocket-stream"
        )
        self.assertIsNotNone(breakdown)
        self.assertEqual(breakdown.measured_from, "last_voiced_audio")
        self.assertAlmostEqual(breakdown.total_secs, 0.64)
        self.assertAlmostEqual(
            sum(item.duration_secs for item in breakdown.contributions),
            breakdown.total_secs,
        )
        self.assertEqual(
            [item.key for item in breakdown.contributions],
            [
                "endpointing.final_transcript",
                "turn.provider_to_aggregator",
                "turn.commit",
                "response.first_safe_text",
                "response.release_to_tts",
                "tts.first_audio",
                "output.first_packet",
                "output.first_audible",
            ],
        )
        self.assertIn("[service: gpt-4.1-mini]", "\n".join(breakdown.turn_contribution_lines()))

    def test_latency_breakdown_falls_back_to_hard_eot(self):
        metrics = TurnMetrics(
            turn_id=1,
            turn_committed_at=10.0,
            commit_at=10.01,
            tts_requested_at=10.02,
            tts_first_audio_at=10.12,
            output_first_packet_at=10.125,
            bot_started_at=10.13,
            route="v2-fixed",
        )
        breakdown = LatencyBreakdown.from_turn(metrics, model="m", tts_transport="websocket-stream")
        self.assertEqual(breakdown.measured_from, "hard_eot")
        self.assertAlmostEqual(breakdown.total_secs, 0.13)
        self.assertAlmostEqual(
            sum(item.duration_secs for item in breakdown.contributions),
            breakdown.total_secs,
        )


class ImprovementAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_interrupted_first_greeting_is_not_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = GreetingCache(directory)
            key = GreetingCacheKey("t", "a", "v1", "voice", "model", 1.0, "Hello")
            capture = GreetingCaptureProcessor(cache, key)
            capture.frames = [struct.pack("<100h", *([1000] * 100))]
            capture.abort_capture()
            self.assertTrue(capture.complete)
            self.assertEqual(capture.frames, [])
            self.assertIsNone(cache.get(key))

    async def test_output_pcm_metrics_are_distinct_from_provider_pcm(self):
        metrics = TurnMetrics(1, tts_requested_at=1.0, turn_committed_at=1.0)
        controller = SimpleNamespace(_state=SimpleNamespace(metrics=metrics))
        observer = LiveLatencyObserver(controller, tts_transport="test")
        output = MagicMock(spec=BaseOutputTransport)
        audio = struct.pack("<100h", *([1000] * 100))
        await observer.on_process_frame(
            FrameProcessed(
                processor=output,
                frame=TTSAudioRawFrame(audio=audio, sample_rate=24000, num_channels=1),
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0,
            )
        )
        self.assertIsNotNone(metrics.output_first_packet_at)
        self.assertEqual(metrics.output_first_packet_at, metrics.output_first_non_silent_at)

    async def test_compiled_bundle_is_reused_by_authoring_digest(self):
        class CountingCompiler(AgentCompiler):
            def __init__(self):
                self.calls = 0

            def compile_goodbox(self, payload):
                self.calls += 1
                return super().compile_goodbox(payload)

        compiler = CountingCompiler()
        bootstrap = RuntimeBootstrap(AgentBundleRegistry(), compiler)
        payload = {"chatbot_id": "a", "tenant_id": "t", "prompt": "hello"}
        first = await bootstrap.start_call({"call_id": "one"}, payload)
        second = await bootstrap.start_call({"call_id": "two"}, json.loads(json.dumps(payload)))
        self.assertEqual(compiler.calls, 1)
        self.assertIs(first.session.agent, second.session.agent)

    async def test_greeting_cache_trims_encodes_and_invalidates_by_key(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = GreetingCache(directory, silence_threshold=200)
            key = GreetingCacheKey("t", "a", "v1", "voice", "model", 1.0, "Hello")
            pcm = (
                struct.pack("<240h", *([0] * 240))
                + struct.pack("<480h", *([2000] * 480))
                + struct.pack("<240h", *([0] * 240))
            )
            cached = await cache.put_pcm(key, pcm, sample_rate=24000)
            self.assertIsNotNone(cached)
            self.assertEqual(cache.get(key).ulaw, cached.ulaw)
            changed = GreetingCacheKey("t", "a", "v2", "voice", "model", 1.0, "Hello")
            self.assertIsNone(cache.get(changed))
            self.assertNotEqual(key.digest(), changed.digest())

    async def test_cached_greeting_is_paced_and_interruptible(self):
        output = SimpleNamespace(send_message=AsyncMock())
        greeting = CachedGreeting(bytes(range(256)) * 4)
        player = CachedGreetingPlayer(output, stream_id="stream", greeting=greeting)
        player.start()
        await asyncio.sleep(0.025)
        await player.interrupt()
        events = [call.args[0].message["event"] for call in output.send_message.await_args_list]
        self.assertIn("playAudio", events)
        self.assertEqual(events[-1], "clearAudio")

    async def test_completed_cached_greeting_still_clears_carrier_buffer_once(self):
        output = SimpleNamespace(send_message=AsyncMock())
        player = CachedGreetingPlayer(output, stream_id="stream", greeting=CachedGreeting(b"\xff" * 160))
        player.start()
        await player.finished.wait()
        await player.interrupt()
        await player.interrupt()
        events = [call.args[0].message["event"] for call in output.send_message.await_args_list]
        self.assertEqual(events.count("clearAudio"), 1)

    def test_flux_profile_custom_env_overrides(self):
        from voice_agent.turns.endpoint_profiles import flux_profile
        with patch.dict(os.environ, {"V2_FLUX_FAST_EAGER": "0.33", "V2_FLUX_FAST_TIMEOUT_MS": "950"}):
            prof = flux_profile("fast")
            self.assertEqual(prof.eager_eot_threshold, 0.33)
            self.assertEqual(prof.eot_timeout_ms, 950)

    def test_goodbox_clean_prompt_bypass(self):
        from goodbox_server import _goodbox_prompt
        with patch.dict(os.environ, {"V2_BYPASS_GOODBOX_PROMPT": "true"}):
            prompt = _goodbox_prompt({"system_prompt": "bloated prompt", "prompt": "legacy"})
            self.assertNotIn("OK|", prompt)
            self.assertNotIn("END|", prompt)
            self.assertNotIn("V1 LANGUAGE OVERRIDE", prompt)
            self.assertIn("Riya", prompt)


if __name__ == "__main__":
    unittest.main()

