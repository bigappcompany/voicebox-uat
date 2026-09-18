import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from live_v2 import V2RoutingController
from voice_agent.agents.bundle import AgentBundle
from voice_agent.runtime.session import CallSession
from voice_agent.runtime.session import PendingQuestion
from pipecat.frames.frames import (
    EndFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSAudioRawFrame,
)


class FakeStream:
    def __init__(self, text):
        self.text = text
        self.close = AsyncMock()
    def __aiter__(self):
        return self.chunks()
    async def chunks(self):
        for content in self.text:
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])


class LiveRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._tts_transport = os.environ.get("V2_TTS_TRANSPORT")
        self._stream_speech = os.environ.get("V2_STREAM_SPEECH")
        self._spec_tts = os.environ.get("ENABLE_SPEC_TTS")
        os.environ["V2_TTS_TRANSPORT"] = "websocket"
        os.environ["V2_STREAM_SPEECH"] = "true"
        os.environ["ENABLE_SPEC_TTS"] = "false"
        agent = AgentBundle("a", "1", "t", cached_utterances={"faq:office hours": "Nine to five."})
        self.client = AsyncMock()
        self.controller = V2RoutingController("test", client=self.client, system_prompt="Goodbox prompt",
            session=CallSession("c", "t", agent))
        self.controller.push_frame = AsyncMock()
        await self.controller._start_turn()

    async def asyncTearDown(self):
        await self.controller.cleanup()
        if self._tts_transport is None:
            os.environ.pop("V2_TTS_TRANSPORT", None)
        else:
            os.environ["V2_TTS_TRANSPORT"] = self._tts_transport
        if self._stream_speech is None:
            os.environ.pop("V2_STREAM_SPEECH", None)
        else:
            os.environ["V2_STREAM_SPEECH"] = self._stream_speech
        if self._spec_tts is None:
            os.environ.pop("ENABLE_SPEC_TTS", None)
        else:
            os.environ["ENABLE_SPEC_TTS"] = self._spec_tts

    async def answer(self, text):
        state = self.controller._state
        state.final_transcript = text
        await self.controller._commit_final_turn(state)
        if state.final_request:
            await state.final_request.task
        return state

    async def test_goodbye_speaks_then_ends_without_llm(self):
        state = await self.answer("goodbye")
        self.assertEqual(state.metrics.route, "v2-fixed")
        self.client.chat.completions.create.assert_not_awaited()
        frames = [c.args[0] for c in self.controller.push_frame.await_args_list]
        self.assertIn("Goodbye", frames[0].text)
        self.assertIsInstance(frames[1], EndFrame)

    async def test_exact_faq_reaches_tts(self):
        await self.answer("office hours")
        self.assertEqual(self.controller.push_frame.await_args.args[0].text, "Nine to five.")
        self.client.chat.completions.create.assert_not_awaited()

    async def test_negated_goodbye_uses_hosted_and_strips_split_markers(self):
        stream = FakeStream(["OK|Do not worry. E", "ND", "|"])
        self.client.chat.completions.create.return_value = stream
        state = await self.answer("do not end the call")
        self.assertEqual(state.metrics.route, "v2-hosted")
        frames = [c.args[0] for c in self.controller.push_frame.await_args_list]
        self.assertEqual("".join(f.text for f in frames if isinstance(f, LLMTextFrame)).strip(), "Do not worry.")
        self.assertFalse(any(isinstance(f, EndFrame) for f in frames))
        stream.close.assert_awaited_once()

    async def test_late_stops_keep_the_complete_aggregated_turn(self):
        self.controller._commit_final_turn = AsyncMock()
        await self.controller.handle_native_turn_stopped("we need engineers")
        await self.controller.handle_native_turn_stopped("in December")
        await self.controller._settle_task
        self.controller._commit_final_turn.assert_awaited_once()
        self.assertEqual(self.controller._state.final_transcript, "in december")

    async def test_barge_in_cancels_pending_generation(self):
        task = asyncio.create_task(asyncio.sleep(30))
        from main import LLMRequest
        self.controller._state.committed = True
        self.controller._state.final_request = LLMRequest("old", False, task=task)
        await self.controller.handle_native_turn_started()
        await asyncio.gather(task, return_exceptions=True)
        self.assertTrue(task.cancelled())

    async def test_transcription_start_does_not_cancel_generation(self):
        task = asyncio.create_task(asyncio.sleep(30))
        from main import LLMRequest
        self.controller._state.committed = True
        self.controller._state.final_request = LLMRequest("old", False, task=task)
        await self.controller.handle_native_turn_started(transcription_only=True)
        await asyncio.sleep(0)
        self.assertFalse(task.cancelled())
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def test_confirmed_transcription_barge_in_flushes_queued_response(self):
        """Quiet speech starts from STT only, then becomes a real interruption.

        A single late transcription event remains harmless, but fresh meaningful
        interim text must cancel the old turn and broadcast an interruption so
        Plivo cannot play its buffered audio ahead of the new response.
        """
        task = asyncio.create_task(asyncio.sleep(30))
        from main import LLMRequest
        old_state = self.controller._state
        old_state.committed = True
        old_state.final_request = LLMRequest("old", False, task=task)
        self.controller.session.history = [{"role": "assistant", "content": "Old response."}]
        self.controller.broadcast_interruption = AsyncMock()
        self.controller._spec_min_chars = 10_000  # This test covers barge-in, not speculation.

        await self.controller.handle_native_turn_started(transcription_only=True)
        await self.controller._on_interim("please stop")
        await asyncio.gather(task, return_exceptions=True)

        self.assertTrue(task.cancelled())
        self.controller.broadcast_interruption.assert_awaited_once()
        self.assertNotEqual(self.controller._state.turn_id, old_state.turn_id)
        self.assertTrue(self.controller.session.history[-1]["content"].endswith("[Playback may have been interrupted.]"))

    async def test_final_segment_does_not_replace_complete_aggregate(self):
        self.controller._commit_final_turn = AsyncMock()
        self.controller._settle_seconds = 0.05
        await self.controller.handle_native_turn_stopped("complete request with details")
        await asyncio.sleep(0)
        self.controller.note_stt_final(True)
        await self.controller._on_final_transcript("with details")
        await self.controller._settle_task

        self.assertEqual(self.controller._state.final_transcript, "complete request with details")
        self.controller._commit_final_turn.assert_awaited_once()

    async def test_changed_suffix_rejects_speculation(self):
        self.controller.router.extended = False
        self.client.chat.completions.create.return_value = FakeStream(["Which roles do you need?"])
        await self.controller._on_interim("we need four")
        await self.controller._on_interim("we need four engineers")
        candidate = self.controller._state.candidate
        await candidate.task
        plan, _ = self.controller._response_plan("we need four engineers but not now")
        self.assertFalse(self.controller._candidate_matches(candidate, "we need four engineers but not now", plan))

    async def test_flux_eager_is_private_and_resume_invalidates(self):
        self.controller.router.extended = False
        self.controller._flux_mode = True
        self.client.chat.completions.create.return_value = FakeStream(["Which roles do you need?"])
        await self.controller._on_interim("we need four engineers")
        self.assertIsNone(self.controller._state.candidate)
        self.controller._flux_eager = True
        await self.controller._on_interim("we need four engineers")
        await self.controller._state.candidate.task
        self.controller.push_frame.assert_not_awaited()
        await self.controller._invalidate_candidate()
        self.assertIsNone(self.controller._state.candidate)

    async def test_flux_stable_interim_starts_private_work_before_eager_eot(self):
        self.controller._flux_mode = True
        self.controller._stable_interim_secs = 0.01
        self.client.chat.completions.create.return_value = FakeStream(["Which roles do you need?"])
        await self.controller._on_interim("we need four")
        await self.controller._on_interim("we need four engineers")
        await asyncio.sleep(0.02)
        candidate = self.controller._state.candidate
        self.assertIsNotNone(candidate)
        await candidate.task
        self.assertFalse(self.controller._flux_eager)
        self.controller.push_frame.assert_not_awaited()

    async def test_flux_resume_cancels_stable_interim_debounce(self):
        self.controller._flux_mode = True
        self.controller._stable_interim_secs = 1
        await self.controller._on_interim("we need four")
        await self.controller._on_interim("we need four engineers")
        debounce = self.controller._state.debounce_task
        self.assertIsNotNone(debounce)
        await self.controller._invalidate_candidate()
        await asyncio.gather(debounce, return_exceptions=True)
        self.assertTrue(debounce.cancelled())
        self.assertIsNone(self.controller._state.candidate)

    async def test_punctuated_callback_time_is_local(self):
        from main import normalize
        self.controller.session.slots["callback_state"] = "AWAITING_DAY_TIME"
        self.controller.session.pending_question = PendingQuestion(
            "ask_callback_day_time", "callback_preference", "date_and_time", 0
        )
        state = await self.answer(normalize("Okay, three p.m. tomorrow please."))
        self.assertEqual(state.metrics.route, "v2-callback-preference")
        self.client.chat.completions.create.assert_not_awaited()

    async def test_identity_question_bypasses_the_model(self):
        self.controller._company_name = "The Hiring Company"
        state = await self.answer("who is this")
        self.assertEqual(state.metrics.route, "v2-identity")
        self.client.chat.completions.create.assert_not_awaited()
        self.assertIn("The Hiring Company", self.controller.push_frame.await_args.args[0].text)

    async def test_callback_preference_does_not_call_model_or_claim_booking(self):
        self.controller.session.slots["callback_state"] = "AWAITING_DAY_TIME"
        self.controller.session.pending_question = PendingQuestion(
            "ask_callback_day_time", "callback_preference", "date_and_time", 0
        )
        state = await self.answer("tomorrow two pm afternoon")
        self.assertEqual(state.metrics.route, "v2-callback-preference")
        self.client.chat.completions.create.assert_not_awaited()
        self.assertIn("not a confirmed booking", self.controller.push_frame.await_args.args[0].text)

    async def test_callback_preference_acknowledgement_stays_local_and_unconfirmed(self):
        self.controller.session.slots.update(
            {
                "callback_state": "PREFERENCE_RECORDED",
                "callback_preference": "tomorrow at five pm",
            }
        )
        state = await self.answer("yes")
        self.assertEqual(state.metrics.route, "v2-callback-preference-acknowledged")
        self.client.chat.completions.create.assert_not_awaited()
        speech = self.controller.push_frame.await_args.args[0].text.casefold()
        self.assertIn("requested follow-up time", speech)
        self.assertIn("confirm availability", speech)

    async def test_unrelated_time_is_not_a_callback(self):
        self.client.chat.completions.create.return_value = FakeStream(["What would you like to do then?"])
        state = await self.answer("two pm tomorrow")
        self.assertEqual(state.metrics.route, "v2-hosted")

    async def test_first_text_released_before_completion(self):
        released = asyncio.Event()
        controller = self.controller
        original_push = controller.push_frame
        async def capture(frame):
            if isinstance(frame, LLMTextFrame):
                released.set()
            await original_push(frame)
        controller.push_frame = capture
        class SlowStream(FakeStream):
            async def chunks(self):
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="We provide staffing support. "))])
                await asyncio.wait_for(released.wait(), 1)
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Which roles do you need?"))])
        self.client.chat.completions.create.return_value = SlowStream([])
        await self.answer("tell me about staffing")
        self.assertTrue(released.is_set())

    async def test_stable_interim_response_is_released_only_after_matching_hard_eot(self):
        self.controller.router.extended = False
        self.client.chat.completions.create.return_value = FakeStream(["We can help you hire engineers."])
        await self.controller._on_interim("we need four")
        await self.controller._on_interim("we need four engineers")
        candidate = self.controller._state.candidate
        self.assertIsNotNone(candidate)
        await candidate.task
        self.assertEqual(self.controller.push_frame.await_count, 0)
        self.controller._state.final_transcript = "we need four engineers"
        await self.controller._commit_final_turn(self.controller._state)
        self.assertEqual(self.controller._state.metrics.route, "v2-speculation-hit")
        self.assertTrue(any(isinstance(c.args[0], LLMTextFrame) for c in self.controller.push_frame.await_args_list))

    async def test_hard_eot_promotes_a_safe_interim_phrase_before_llm_completion(self):
        """A hit must keep streaming the original request rather than restart it.

        This is the critical overlap: the first safe phrase is already ready at
        hard EOT, while the remaining LLM tokens are still arriving.
        """
        first_phrase_ready = asyncio.Event()
        release_tail = asyncio.Event()

        class GatedStream(FakeStream):
            async def chunks(self):
                yield SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="We can help you hire engineers. "))]
                )
                first_phrase_ready.set()
                await release_tail.wait()
                yield SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="Which roles do you need?"))]
                )

        self.controller.router.extended = False
        self.client.chat.completions.create.return_value = GatedStream([])
        await self.controller._on_interim("we need four")
        await self.controller._on_interim("we need four engineers")
        candidate = self.controller._state.candidate
        self.assertIsNotNone(candidate)
        await asyncio.wait_for(first_phrase_ready.wait(), 1)
        await asyncio.sleep(0)
        self.assertTrue(candidate.answer_chunks)
        self.assertFalse(candidate.completed)

        self.controller._state.final_transcript = "we need four engineers"
        await self.controller._commit_final_turn(self.controller._state)
        self.assertEqual(self.controller._state.metrics.route, "v2-speculation-hit")
        self.assertFalse(candidate.task.done())
        frames_before_tail = [call.args[0] for call in self.controller.push_frame.await_args_list]
        self.assertTrue(any(isinstance(frame, LLMFullResponseStartFrame) for frame in frames_before_tail))
        self.assertTrue(any(isinstance(frame, LLMTextFrame) for frame in frames_before_tail))
        self.assertEqual(self.client.chat.completions.create.await_count, 1)

        release_tail.set()
        await candidate.task
        frames = [call.args[0] for call in self.controller.push_frame.await_args_list]
        self.assertEqual(sum(isinstance(frame, LLMFullResponseStartFrame) for frame in frames), 1)
        self.assertEqual(sum(isinstance(frame, LLMFullResponseEndFrame) for frame in frames), 1)
        self.assertEqual(
            "".join(frame.text for frame in frames if isinstance(frame, LLMTextFrame)).strip(),
            "We can help you hire engineers. Which roles do you need?",
        )

    async def test_hard_eot_keeps_a_valid_interim_llm_request_before_first_safe_phrase(self):
        """Hard EOT must not restart a valid request merely because it is mid-TTFT."""
        stream_started = asyncio.Event()
        release_first_phrase = asyncio.Event()

        class DelayedFirstPhrase(FakeStream):
            async def chunks(self):
                stream_started.set()
                await release_first_phrase.wait()
                yield SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="We can help with that."))]
                )

        self.controller.router.extended = False
        self.client.chat.completions.create.return_value = DelayedFirstPhrase([])
        await self.controller._on_interim("we need four")
        await self.controller._on_interim("we need four engineers")
        candidate = self.controller._state.candidate
        self.assertIsNotNone(candidate)
        await asyncio.wait_for(stream_started.wait(), 1)
        self.assertFalse(candidate.answer_chunks)

        self.controller._state.final_transcript = "we need four engineers"
        await self.controller._commit_final_turn(self.controller._state)
        self.assertEqual(self.controller._state.metrics.route, "v2-speculation-hit")
        self.assertIs(self.controller._state.final_request, candidate)
        self.assertEqual(self.client.chat.completions.create.await_count, 1)
        self.assertTrue(
            any(isinstance(call.args[0], LLMFullResponseStartFrame) for call in self.controller.push_frame.await_args_list)
        )

        release_first_phrase.set()
        await candidate.task
        self.assertTrue(
            any(isinstance(call.args[0], LLMTextFrame) for call in self.controller.push_frame.await_args_list)
        )

    async def test_matching_hard_eot_can_commit_private_pcm_before_public_tts_tail(self):
        class PrivateCartesia:
            def warm(self):
                pass
            async def prepare(self, **kwargs):
                return SimpleNamespace(text=kwargs["text"])
            async def commit(self, _prepared, *, fingerprint):
                return [b"\x00\x00" * 240]
            async def abort(self, _prepared):
                pass
            async def close(self):
                pass

        self.controller.router.extended = False
        self.controller._spec_audio = PrivateCartesia()
        self.client.chat.completions.create.return_value = FakeStream(["We can help you hire engineers."])
        await self.controller._on_interim("we need four")
        await self.controller._on_interim("we need four engineers")
        candidate = self.controller._state.candidate
        await candidate.task
        self.controller._state.final_transcript = "we need four engineers"
        await self.controller._commit_final_turn(self.controller._state)
        frames = [call.args[0] for call in self.controller.push_frame.await_args_list]
        self.assertTrue(any(isinstance(frame, TTSAudioRawFrame) for frame in frames))
        self.assertEqual(self.controller._state.metrics.speculation, "hit+tts")

    async def test_cartesia_retries_timeout_without_exposing_uri(self):
        from resilient_tts import ResilientCartesiaTTSService
        from pipecat.services.cartesia.tts import CartesiaTTSService
        service = ResilientCartesiaTTSService(api_key="test", settings=CartesiaTTSService.Settings(voice="test"))
        connected = object()
        with patch.object(CartesiaTTSService, "_websocket_connect", new=AsyncMock(side_effect=[TimeoutError(), connected])) as connect:
            result = await service._websocket_connect("wss://example.test?api_key=secret&cartesia_version=2026-03-01")
        self.assertIs(result, connected)
        self.assertEqual(connect.await_count, 2)
        for call in connect.await_args_list:
            self.assertNotIn("api_key", call.args[0])
            self.assertEqual(call.kwargs["additional_headers"], {"X-API-Key": "test"})

    async def test_turn13_followup_with_numeric_range_slots_does_not_raise_typeerror(self):
        from voice_agent.flows.facts import NumericRange
        self.controller.session.state = {"name": "FOLLOWUP"}
        self.controller.session.pending_question = PendingQuestion(
            "ask_callback_consent", "followup_consent", "boolean", 12
        )
        self.controller.session.slots = {
            "hiring_status": "yes",
            "roles": ["developer"],
            "headcount": NumericRange(5, 10, "people", True),
            "hiring_timeline": NumericRange(1, 2, "months", False),
            "callback_state": "FOLLOWUP_OFFERED",
        }
        self.client.chat.completions.create.return_value = FakeStream(["We would be glad to help."])
        # Trigger eager candidate speculation (turn 13)
        await self.controller._start_candidate(self.controller._state, "we are hiring", source="eager")
        candidate = self.controller._state.candidate
        self.assertIsNotNone(candidate)
        await candidate.task
        self.assertTrue(candidate.completed)
        self.assertEqual(candidate.answer_text, "We would be glad to help.")
        # Trigger final turn commitment
        self.controller._state.final_transcript = "we are hiring"
        await self.controller._commit_final_turn(self.controller._state)
        self.assertNotEqual(self.controller._state.metrics.route, "v2-error")
        self.assertIn("We would be glad to help.", [call.args[0].text for call in self.controller.push_frame.await_args_list if isinstance(call.args[0], LLMTextFrame)])

    async def test_turn13_followup_non_speculative_with_numeric_range_slots_does_not_raise_typeerror(self):
        from voice_agent.flows.facts import NumericRange
        self.controller.session.state = {"name": "FOLLOWUP"}
        self.controller.session.pending_question = PendingQuestion(
            "ask_callback_consent", "followup_consent", "boolean", 12
        )
        self.controller.session.slots = {
            "hiring_status": "yes",
            "roles": ["developer"],
            "headcount": NumericRange(5, 10, "people", True),
            "hiring_timeline": NumericRange(1, 2, "months", False),
            "callback_state": "FOLLOWUP_OFFERED",
        }
        self.client.chat.completions.create.return_value = FakeStream(["We would be glad to help."])
        state = await self.answer("we are hiring")
        self.assertNotEqual(state.metrics.route, "v2-error")
        self.assertEqual(state.metrics.route, "v2-hosted")
        self.assertIn("We would be glad to help.", [call.args[0].text for call in self.controller.push_frame.await_args_list if isinstance(call.args[0], LLMTextFrame)])

