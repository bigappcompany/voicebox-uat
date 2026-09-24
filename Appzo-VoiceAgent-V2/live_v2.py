"""Low-latency Goodbox V2 routing on Pipecat's native turn events.

Interim work is private. Only a VAD/Smart-Turn hard EOT can release text to
Cartesia and Plivo. This overlaps hosted LLM work with caller speech without
letting stale or unvalidated speech become audible.
"""

import asyncio
import contextlib
import os
import re
import time
import uuid
from dataclasses import replace

from loguru import logger
from pipecat.frames.frames import (
    EndFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from voice_agent.turns.flux import FluxResumeFrame

from main import LLMRequest, StreamingVoiceController, normalize
from voice_agent.llm.prompt_builder import PromptBuilder
from voice_agent.llm.context_adapter import ConversationContextAdapter
from voice_agent.flows.engine import FlowEngine
from voice_agent.flows.slots import SlotValidator
from voice_agent.flows.facts import FactExtractor
from voice_agent.flows.callbacks import CallbackCoordinator
from voice_agent.routing.deterministic import DeterministicRouter
from voice_agent.runtime.fingerprints import ResponseFingerprint
from voice_agent.runtime.flags import RuntimeFlags
from voice_agent.runtime.intents import CanonicalIntentModel
from voice_agent.runtime.response_plan import ResponsePlan
from voice_agent.runtime.session import AssistantDeliveryState, PendingQuestion
from voice_agent.speech.safe_chunker import SafeSpeechChunker
from voice_agent.speech.speculative_cartesia import SpeculativeCartesiaBuffer
from voice_agent.speech.stream_filter import SpeechStreamFilter
from voice_agent.speech.booking_guard import BookingClaimGuard
from voice_agent.turns.transcript_stability import TranscriptStabilityAnalyzer
from voice_agent.turns.endpoint_profiles import profile_for_prompt, detected_profile_for_prompt

_LOW_INFO_FRAGMENTS = frozenset({
    "hiria", "very", "uh", "um", "ah", "hmm", "ha", "er", "oh", "huh",
})


class V2RoutingController(StreamingVoiceController):
    def __init__(
        self,
        *args,
        session,
        cartesia_api_key: str | None = None,
        cartesia_voice_id: str | None = None,
        cartesia_model: str | None = None,
        cartesia_speed: float = 1.0,
        tts_transport: str = "websocket",
        flux_mode: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.session = session
        self.flags = RuntimeFlags.from_env()
        self._flux_mode = flux_mode
        self._flux_eager = False
        self._closed_flux_turns = set()
        self.router = DeterministicRouter(extended=self.flags.enable_extended_deterministic_routing)
        self.intent_model = CanonicalIntentModel()
        self.flow_engine = FlowEngine()
        self.slot_validator = SlotValidator()
        profile_name = str(session.agent.fact_profile.get("name") or os.getenv("V2_FACT_PROFILE", "recruitment"))
        self.fact_extractor = FactExtractor(session.agent.fact_profile, default_profile=profile_name)
        self.callback_coordinator = CallbackCoordinator()
        self._context_turns = int(os.getenv("V2_PIPECAT_CONTEXT_TURNS", "3"))
        self._context_max_tokens = int(os.getenv("V2_PIPECAT_CONTEXT_MAX_TOKENS", "500"))
        self.prompt_builder = PromptBuilder(
            max_history_turns=self._context_turns,
            compiled=self.flags.enable_compiled_prompts,
        )
        self._context_adapter: ConversationContextAdapter | None = None
        self._use_pipecat_context = os.getenv(
            "V2_USE_PIPECAT_DIALOGUE_CONTEXT", "true"
        ).lower() == "true"
        self._compare_context = os.getenv("V2_ENABLE_CONTEXT_COMPARE", "false").lower() == "true"
        self._log_llm_payloads = os.getenv("V2_LOG_LLM_PAYLOADS", "false").lower() == "true"
        self._request_attempts: dict[int, int] = {}
        self._planned_pending_question: PendingQuestion | None = None
        self._delivery_state = None
        self._delivery_speech = ""
        self._pending_text = ""
        self._latest_finalized_text = ""
        self._settle_task: asyncio.Task | None = None
        self._native_stop_at: float | None = None
        self._company_name = os.getenv("V2_COMPANY_NAME") or session.agent.identity.get("company_name")
        self._streaming = (
            os.getenv("V2_STREAM_SPEECH", "true").lower() == "true"
            and tts_transport == "websocket"
        )
        # Nova can deliver a provisional aggregate immediately before its
        # final transcript.  A tiny settle window prevents that trailing final
        # from becoming a second turn. Flux final frames are marked finalized,
        # so Flux takes the zero-delay branch below.
        self._settle_seconds = float(os.getenv("V2_TURN_SETTLE_SECS", "0.18"))
        self._stt_finalized = False
        self._transcription_barge_in_pending = False
        self._transcription_barge_in_turn_id: int | None = None
        self._transcription_barge_in_min_words = int(
            os.getenv("V2_TRANSCRIPTION_BARGE_IN_MIN_WORDS", "2")
        )
        self._transcription_barge_in_min_chars = int(
            os.getenv("V2_TRANSCRIPTION_BARGE_IN_MIN_CHARS", "8")
        )
        self._stability = TranscriptStabilityAnalyzer()
        self._spec_min_words = int(os.getenv("V2_SPECULATION_MIN_WORDS", "3"))
        self._spec_min_chars = int(os.getenv("V2_SPECULATION_MIN_CHARS", "12"))
        default_commit_wait_ms = "0" if self.flags.enable_zero_delay_spec_promotion else "80"
        self._spec_commit_wait_secs = float(
            os.getenv("V2_SPECULATION_COMMIT_WAIT_MS", default_commit_wait_ms)
        ) / 1000
        self._stable_interim_secs = float(os.getenv("V2_STABLE_INTERIM_MS", "80")) / 1000
        self._spec_max_restarts = int(os.getenv("V2_SPECULATION_MAX_RESTARTS", "2"))
        chunk_profiles = {
            "fast": (32, 5, 160),
            "balanced": (40, 5, 200),
            "natural": (48, 7, 240),
        }
        self._safe_chunk_profile = os.getenv("V2_SAFE_CHUNK_PROFILE", "natural").casefold()
        default_chars, default_words, default_wait = chunk_profiles.get(
            self._safe_chunk_profile, chunk_profiles["natural"]
        )
        self._safe_chunk_chars = int(os.getenv("V2_SAFE_CHUNK_CHARS", str(default_chars)))
        self._safe_chunk_words = int(os.getenv("V2_SAFE_CHUNK_WORDS", str(default_words)))
        self._safe_chunk_max_wait_ms = int(os.getenv("V2_SAFE_CHUNK_MAX_WAIT_MS", str(default_wait)))
        self._min_final_transcript_chars = int(os.getenv("V2_MIN_FINAL_TRANSCRIPT_CHARS", "2"))
        self._spec_restart_min_new_words = int(os.getenv("V2_SPEC_RESTART_MIN_NEW_WORDS", "3"))
        self._min_useful_spec_lead_ms = int(os.getenv("V2_MIN_USEFUL_SPEC_LEAD_MS", "100"))
        # This is an operator budget, not a filler trigger. A hosted model
        # that misses it is visible in the per-turn record; deterministic and
        # promoted speculative candidates have already been considered before
        # this request is allowed to own caller-facing speech.
        self._hosted_ttft_observe_ms = int(os.getenv("V2_HOSTED_TTFT_OBSERVE_MS", "350"))
        self._spec_tts_commit_wait_secs = float(os.getenv("V2_SPEC_TTS_COMMIT_WAIT_MS", "40")) / 1000
        # A post-promotion first phrase is the only point where normal TTS is
        # deliberately held for private PCM. This needs a realistic Cartesia
        # first-audio budget, while the original pre-EOT path stays at 40 ms.
        self._late_spec_tts_wait_secs = float(os.getenv("V2_LATE_SPEC_TTS_WAIT_MS", "400")) / 1000
        self._endpoint_profile = None
        self._endpoint_profile_updater = None
        self._deferred_endpoint_profile: str | None = None
        self._prefix_buffer = ""
        self._enable_spec_tts = self.flags.enable_spec_tts
        self._spec_audio: SpeculativeCartesiaBuffer | None = None
        if self._enable_spec_tts and self._streaming and cartesia_api_key and cartesia_voice_id and cartesia_model:
            requested_spec_audio_ms = int(os.getenv("V2_SPEC_TTS_MAX_AUDIO_MS", "2000"))
            # Older deployments used 600 ms because partial PCM was treated
            # as committable. Complete-phrase commit needs enough room for a
            # short 2-6 word safe chunk; retain the setting as an upper bound
            # knob while migrating unsafe legacy values to a safe minimum.
            spec_audio_ms = max(2000, requested_spec_audio_ms)
            if spec_audio_ms != requested_spec_audio_ms:
                logger.warning(
                    "V2 speculative TTS audio cap raised from {} ms to {} ms for complete-phrase safety",
                    requested_spec_audio_ms,
                    spec_audio_ms,
                )
            self._spec_audio = SpeculativeCartesiaBuffer(
                api_key=cartesia_api_key,
                voice_id=cartesia_voice_id,
                model=cartesia_model,
                speed=cartesia_speed,
                max_audio_ms=spec_audio_ms,
                connect_timeout_secs=float(os.getenv("V2_SPEC_TTS_CONNECT_TIMEOUT_SECS", "4")),
            )
            # Connection establishment is outside the EOT-to-audio path. If a
            # deployment constructs the controller before its event loop is
            # running, the first interim simply retries this warmup instead.
            self._spec_audio.warm()
        elif self._enable_spec_tts and self._streaming:
            logger.warning("V2 speculative TTS is enabled but Cartesia runtime settings are incomplete")
        logger.info(
            "V2 SAFE CHUNK PROFILE | name={} min_chars={} min_words={} max_wait_ms={}",
            self._safe_chunk_profile, self._safe_chunk_chars,
            self._safe_chunk_words, self._safe_chunk_max_wait_ms,
        )

    def set_dialogue_context(self, context) -> None:
        """Attach the call's Pipecat-owned semantic dialogue store."""
        self._context_adapter = ConversationContextAdapter(context, self.session)

    def _hosted_history(self, text: str) -> tuple[list[dict[str, str]], dict[str, int]]:
        if not self._use_pipecat_context or self._context_adapter is None:
            history = list(self.session.history[-self._context_turns * 2:])
            return history, {
                "total": len(self.session.history),
                "selected": len(history),
                "tokens": sum((len(item.get("content", "")) + 3) // 4 for item in history),
                "version": len(self.session.history),
            }
        started = time.perf_counter()
        hosted = self._context_adapter.build_hosted_context(
            current_user_text=text,
            max_turns=self._context_turns,
            max_tokens=self._context_max_tokens,
        )
        history = [dict(item) for item in hosted.recent_dialogue]
        metrics = {
            "total": hosted.dialogue_version,
            "selected": len(history),
            "tokens": hosted.estimated_tokens,
            "version": hosted.dialogue_version,
            "bridge_us": round((time.perf_counter() - started) * 1_000_000),
            "pipecat_context_read_ms": hosted.pipecat_read_ms,
            "context_selection_ms": hosted.selection_ms,
            "context_token_estimation_ms": hosted.token_estimation_ms,
        }
        if self._compare_context:
            shadow = self.session.history[-self._context_turns * 2:]
            logger.debug(
                "V2 CONTEXT COMPARE | pipecat_messages={} session_history_messages={} semantic_match={}",
                len(history), len(shadow), history == shadow,
            )
        return history, metrics

    def _prepare_request(self, state, request: LLMRequest, text: str, *, slots_before: dict) -> None:
        attempt = self._request_attempts.get(state.turn_id, 0) + 1
        self._request_attempts[state.turn_id] = attempt
        history, context_metrics = self._hosted_history(text)
        request.v2_request_id = uuid.uuid4().hex
        request.v2_attempt = attempt
        request.v2_history = history
        request.v2_context_metrics = context_metrics
        request.v2_slots_before = dict(slots_before)

    @staticmethod
    def _pending_payload(pending) -> dict | None:
        if pending is None:
            return None
        return {
            "intent": pending.intent,
            "slot": pending.slot,
            "expected_type": pending.expected_type,
            "asked_turn": pending.asked_turn,
        }

    async def process_frame(self, frame, direction):
        if direction == FrameDirection.DOWNSTREAM and self._flux_mode:
            if isinstance(frame, FluxResumeFrame):
                await self._invalidate_candidate()
            elif isinstance(frame, InterimTranscriptionFrame):
                result = frame.result or {}
                self._flux_eager = result.get("event") == "EagerEndOfTurn"
                if self._flux_eager and self._state:
                    self._state.metrics.eager_eot_at = result.get("runtime_eager_at") or time.perf_counter()
                    self._state.metrics.eager_eot_confidence = result.get("end_of_turn_confidence")
            elif isinstance(frame, TranscriptionFrame):
                result = frame.result or {}
                turn_id = result.get("turn_index")
                if turn_id is not None and turn_id in self._closed_flux_turns:
                    return
                if turn_id is not None:
                    self._closed_flux_turns.add(turn_id)
                if self._state:
                    self._state.metrics.provider_eot_at = result.get("runtime_eot_at")
                    self._state.metrics.provider_turn_id = turn_id
                    self._state.metrics.eot_trigger = result.get("trigger")
                    self._state.metrics.eot_confidence = result.get("end_of_turn_confidence")
                    self._state.metrics.turn_resumed_count = int(result.get("runtime_turn_resumed_count") or 0)
                    self._state.metrics.input_gap_count = int(result.get("runtime_input_gap_count") or 0)
                    self._state.metrics.input_gap_max_ms = float(result.get("runtime_input_gap_max_ms") or 0.0)
        await super().process_frame(frame, direction)

    async def _invalidate_candidate(self):
        state = self._state
        if state:
            self._cancel_task(state.debounce_task)
        if state and not state.committed and state.candidate:
            self._cancel_task(state.candidate.task)
            await self._abort_spec_audio(state.candidate)
            state.candidate = None
            state.metrics.speculative_started_at = None
            state.metrics.llm_request_started_at = None
            state.metrics.llm_stream_opened_at = None
            state.metrics.llm_first_token_at = None
            state.metrics.first_speech_filter_text_at = None
            state.metrics.first_filtered_text_at = None
            state.metrics.first_safe_text_at = None

    async def _on_interim(self, text: str) -> None:
        """Prepare one low-risk response from a stable interim prefix."""
        self._stt_finalized = False
        state = self._state
        interim = normalize(text)
        # A transcription start is a deliberate soft fallback for quiet
        # callers.  Formerly we ignored it forever to protect against a late
        # previous-turn transcript. That left already-buffered bot speech
        # alive, so its audio could be attributed to (and queue behind) the
        # next answer. Require meaningful fresh interim text before treating
        # that fallback as a real barge-in.
        if state is not None and state.committed and self._transcription_barge_in_pending:
            if self._is_meaningful_barge_in(interim):
                await self._confirm_transcription_barge_in()
                state = self._state
        if state is None:
            await self._start_turn()
            state = self._state
        if state is None or state.committed or state.turn_stopped:
            return
        if not interim:
            return
        state.latest_interim = interim
        if getattr(self, "_endpoint_profile", None) == "yes_no" and len(interim.strip().split()) >= 3:
            # Do not reconfigure Flux while this caller turn is active: its
            # async Configure acknowledgement can otherwise race final EOT.
            # Use freeform for the following listening turn instead.
            self._deferred_endpoint_profile = "freeform"
        hypothesis = self._stability.update(interim)
        eager = self._flux_mode and self._flux_eager
        if eager:
            basis = interim
        elif self.flags.enable_stable_interim_speculation:
            basis = hypothesis.stable_prefix
        else:
            return
        if len(basis) < self._spec_min_chars or len(basis.split()) < self._spec_min_words:
            return

        if not self._flux_mode:
            await self._start_candidate(state, interim, source="stable")
            return
        if not eager:
            if state.metrics.stable_interim_at is None:
                state.metrics.stable_interim_at = hypothesis.stable_since or time.perf_counter()
            self._cancel_task(state.debounce_task)
            state.debounce_task = asyncio.create_task(
                self._start_stable_candidate_after_delay(state, basis, hypothesis.stable_since)
            )
            return
        await self._start_candidate(state, basis, source="eager")

    async def _start_stable_candidate_after_delay(self, state, basis: str, stable_since: float | None) -> None:
        try:
            elapsed = max(0.0, time.perf_counter() - (stable_since or time.perf_counter()))
            await asyncio.sleep(max(0.0, self._stable_interim_secs - elapsed))
            if self._state is not state or state.committed or state.turn_stopped:
                return
            current = self._stability.hypothesis
            if current.stable_prefix != basis:
                return
            await self._start_candidate(state, basis, source="stable")
        except asyncio.CancelledError:
            raise

    async def _start_candidate(self, state, basis: str, *, source: str) -> None:
        if state.speculation_restarts >= self._spec_max_restarts:
            return

        if self._spec_audio:
            self._spec_audio.warm()
        provisional = (
            self.fact_extractor.extract(
                basis, self.session.slots,
                pending_question=self.session.pending_question,
                source_turn=state.turn_id,
            )
            if self.flags.enable_structured_facts else None
        )
        provisional_facts = provisional.values if provisional else {}
        provisional_slots = {**self.session.visible_facts(), **provisional_facts}
        plan, speech = self._response_plan(
            basis, slots=provisional_slots, pending_question=self.session.pending_question,
            current_facts=provisional_facts,
        )
        if speech is not None:
            state.eager_deterministic_plan = plan
            state.eager_deterministic_speech = speech
            state.eager_deterministic_basis = basis
            state.eager_deterministic_slots = provisional_slots
            state.metrics.eager_deterministic_staged = True
            return
        if plan.risk_class not in {"LOW_PUBLIC", "LOW_WORKFLOW"} or plan.tool_name:
            return
        fingerprint = self._fingerprint(basis, plan, slots=provisional_slots)
        old = state.candidate
        if old and old.transcript == basis and (
            old.completed or (old.task is not None and not old.task.done())
        ):
            if source == "eager":
                old.v2_source = "eager"
                await self._start_existing_spec_audio(state, old)
            return
        if (
            old and source == "eager" and basis.startswith(old.transcript + " ")
            and getattr(old, "v2_fingerprint", None) == fingerprint
        ):
            old.v2_source = "eager"
            await self._start_existing_spec_audio(state, old)
            return
        if old and old.task is not None and not old.task.done() and basis.startswith(old.transcript + " "):
            new_words = len(basis.split()) - len(old.transcript.split())
            if source != "eager" and new_words < self._spec_restart_min_new_words:
                return
        self._cancel_task(old.task if old else None)
        await self._abort_spec_audio(old)

        candidate = LLMRequest(basis, speculative=True)
        candidate.v2_plan = plan
        candidate.v2_slots = provisional_slots
        candidate.v2_source = source
        candidate.v2_fingerprint = fingerprint
        self._prepare_request(
            state, candidate, basis, slots_before=self.session.visible_facts()
        )
        state.candidate = candidate
        state.speculation_restarts += 1
        state.metrics.speculative_started_at = time.perf_counter()
        state.metrics.stable_candidate_source = source
        if source == "stable" and state.metrics.stable_candidate_started_at is None:
            state.metrics.stable_candidate_started_at = state.metrics.speculative_started_at
        state.metrics.spec_tts_eligible = bool(self._spec_audio and plan.may_prepare_audio())
        state.metrics.spec_tts = (
            "waiting_for_eager_eot" if state.metrics.spec_tts_eligible and source == "stable"
            else "waiting_for_safe_text" if state.metrics.spec_tts_eligible
            else "not_eligible"
        )
        state.metrics.spec_tts_reason = (
            "stable_candidate_waiting_for_eager_eot" if state.metrics.spec_tts == "waiting_for_eager_eot"
            else "safe_text_not_ready" if state.metrics.spec_tts == "waiting_for_safe_text"
            else "plan_or_private_tts_ineligible"
        )
        state.metrics.llm_request_started_at = None
        state.metrics.llm_stream_opened_at = None
        state.metrics.llm_first_token_at = None
        state.metrics.first_speech_filter_text_at = None
        state.metrics.first_filtered_text_at = None
        state.metrics.first_safe_text_at = None
        candidate.task = asyncio.create_task(self._generate(state, basis, plan, request=candidate))
        logger.debug("V2 SPEC START turn={} source={} stable_words={} route={}", state.turn_id, source, len(basis.split()), plan.route)

    async def _on_final_transcript(self, text: str) -> None:
        # Pipecat's universal aggregator normally supplies the complete
        # hard-EOT text. Preserve a just-arrived finalized transcript during
        # the short Nova settle window so it replaces an earlier provisional
        # aggregate rather than opening a duplicate turn.
        # Final frames may be segments. Only the aggregator owns assembly;
        # replacing its complete text with the last frame loses earlier words.
        return

    def note_stt_final(self, finalized: bool) -> None:
        self._stt_finalized = finalized

    async def handle_native_turn_started(self, *, transcription_only: bool = False) -> None:
        # A raw transcription start may be a delayed result from the previous
        # turn. Record it first; a meaningful interim or final transcript will
        # confirm the barge-in and explicitly interrupt queued output.
        if transcription_only:
            if self._state and self._state.committed:
                self._transcription_barge_in_pending = True
                self._transcription_barge_in_turn_id = self._state.turn_id
            return
        greeting_interrupt = getattr(self, "_greeting_interrupt", None)
        if greeting_interrupt is not None:
            await greeting_interrupt()
        self._transcription_barge_in_pending = False
        self._transcription_barge_in_turn_id = None
        self._flux_eager = False
        self._mark_prior_playback_interrupted()
        if self._settle_task and not self._settle_task.done():
            self._settle_task.cancel()
        if self._state and self._state.candidate:
            self._cancel_task(self._state.candidate.task)
            await self._abort_spec_audio(self._state.candidate)
        await super().handle_native_turn_started()
        self._stability = TranscriptStabilityAnalyzer()
        self._latest_finalized_text = ""

    async def handle_native_turn_stopped(self, transcript: str | None = None) -> None:
        self._native_stop_at = time.perf_counter()
        if self._state:
            self._state.metrics.aggregator_stop_at = self._native_stop_at
            self._native_stop_at = self._state.metrics.provider_eot_at or self._native_stop_at
        # A short quiet utterance may never reach the interim threshold. Its
        # final aggregate still confirms that the pending transcription start
        # was genuine, so it must clear any queued bot audio before we commit.
        if self._transcription_barge_in_pending and normalize(transcript or ""):
            await self._confirm_transcription_barge_in()
        if transcript:
            # Pipecat gives an aggregate. Concatenating callbacks fabricated
            # utterances and directly caused the earlier wrong answers.
            self._pending_text = transcript.strip()
        if self._settle_task and not self._settle_task.done():
            self._settle_task.cancel()
        self._settle_task = asyncio.create_task(self._settle_turn())

    async def _settle_turn(self) -> None:
        await asyncio.sleep(0 if self._stt_finalized else self._settle_seconds)
        text, self._pending_text = self._pending_text, ""
        self._latest_finalized_text = ""
        if self._prefix_buffer:
            text = f"{self._prefix_buffer} {text}".strip()
            self._prefix_buffer = ""
        normalized = normalize(text)
        if normalized and len(normalized) < self._min_final_transcript_chars:
            # Flux occasionally finalizes a one-character noise fragment. It
            # must not trigger another full greeting/pitch after the cached
            # greeting has already played.
            if self._state and not self._state.committed:
                self._state.committed = True
                self._state.metrics.route = "stt-noise-ignored"
            logger.info("V2 ignored short final transcript chars={}", len(normalized))
            return
        words = normalized.split()
        if words and all(w in _LOW_INFO_FRAGMENTS for w in words):
            self._prefix_buffer = text
            if self._state and not self._state.committed:
                self._state.committed = True
                self._state.metrics.route = "stt-low-info-buffered"
            logger.info("V2 buffered low-information fragment: {}", text)
            return
        await super().handle_native_turn_stopped(text)

    def _is_meaningful_barge_in(self, text: str) -> bool:
        return bool(
            text
            and len(text) >= self._transcription_barge_in_min_chars
            and len(text.split()) >= self._transcription_barge_in_min_words
        )

    def _mark_prior_playback_interrupted(self) -> None:
        if self._state and self._state.committed and self.session.history:
            last = self.session.history[-1]
            if last["role"] == "assistant" and not last["content"].endswith("[Playback may have been interrupted.]"):
                last["content"] += " [Playback may have been interrupted.]"

    async def _confirm_transcription_barge_in(self) -> None:
        if not self._transcription_barge_in_pending:
            return
        state = self._state
        self._transcription_barge_in_pending = False
        self._transcription_barge_in_turn_id = None
        if state is None or not state.committed:
            return
        self._mark_prior_playback_interrupted()
        # The aggregator emitted an interruption when the fallback started,
        # but it may have done so before a media queue was populated. Sending
        # one more confirmed interruption guarantees Cartesia and Plivo flush
        # the old response before the new turn is created.
        await self.broadcast_interruption()
        if self._settle_task and not self._settle_task.done():
            self._settle_task.cancel()
        if state.candidate:
            self._cancel_task(state.candidate.task)
            await self._abort_spec_audio(state.candidate)
        await super().handle_native_turn_started()
        self._latest_finalized_text = ""
        logger.info("V2 confirmed transcription barge-in; previous turn={} interrupted", state.turn_id)

    async def _commit_final_turn(self, state) -> None:
        if state is not self._state or state.committed:
            return
        self._cancel_task(state.final_wait_task)
        state.committed = True
        state.metrics.commit_at = time.perf_counter()
        if self._native_stop_at is not None:
            state.metrics.turn_committed_at = self._native_stop_at
        self.session.turn_id = state.turn_id
        text = state.final_transcript
        state.v2_slots_before = dict(self.session.visible_facts())
        if self.flags.enable_structured_facts:
            facts = self.fact_extractor.extract(
                text, self.session.visible_facts(),
                pending_question=self.session.pending_question,
                source_turn=state.turn_id,
            )
            self.session.slots.update(facts.values)
            self.session.facts.update(facts.records)
            if facts.corrected:
                logger.info("V2 FACT correction fields={}", ",".join(facts.corrected))
        if self._transcript_callback:
            self._transcript_callback("user", text)

        current_facts = facts.values if self.flags.enable_structured_facts else {}
        if (
            getattr(state, "eager_deterministic_speech", None) is not None
            and text == getattr(state, "eager_deterministic_basis", None)
        ):
            plan = state.eager_deterministic_plan
            speech = state.eager_deterministic_speech
            state.metrics.eager_deterministic_hit = True
            logger.debug(
                "V2 EAGER DETERMINISTIC HIT turn={} route={} speech='{}'",
                state.turn_id, plan.route, speech[:40],
            )
        else:
            plan, speech = self._response_plan(
                text, slots=self.session.visible_facts(), pending_question=self.session.pending_question,
                current_facts=current_facts,
            )
        state.metrics.decision_route = plan.route
        state.metrics.decision_intent = plan.intent_id or "unknown"
        state.metrics.decision_reason = plan.decision_reason
        state.metrics.knowledge_direct_hit = plan.route == "faq-direct"
        state.metrics.retrieval_query = text
        state.metrics.selected_doc_id = plan.knowledge_ids[0] if plan.knowledge_ids else None
        state.metrics.retrieval_confidence = (
            plan.decision_confidence if plan.knowledge_ids else None
        )
        if state.metrics.spec_tts == "none":
            state.metrics.spec_tts_eligible = bool(
                speech is None and self._spec_audio and plan.may_prepare_audio()
            )
            state.metrics.spec_tts = "miss" if state.metrics.spec_tts_eligible else "not_eligible"
            state.metrics.spec_tts_reason = (
                "no_stable_candidate" if state.metrics.spec_tts_eligible
                else "deterministic_or_private_tts_unavailable"
            )
        logger.info(
            "V2 DECISION | turn={} route={} intent={} confidence={} reason={} state={}",
            state.turn_id, plan.route, plan.intent_id or "unknown",
            round(plan.decision_confidence, 3), plan.decision_reason or "fallback",
            self.session.state.get("name"),
        )
        candidate = state.candidate
        if self._candidate_matches(candidate, text, plan):
            state.metrics.candidate_validated_at = time.perf_counter()
            if not self.flags.enable_zero_delay_spec_promotion and self._spec_commit_wait_secs > 0:
                try:
                    if candidate.task and not candidate.task.done() and not candidate.answer_chunks:
                        await asyncio.wait_for(asyncio.shield(candidate.task), timeout=self._spec_commit_wait_secs)
                except TimeoutError:
                    pass
            if self._state is not state:
                return
            # A completed task that produced no safe text failed upstream, so
            # use a fresh final request for the normal error/fallback path.
            if candidate.task is None or (candidate.task.done() and not candidate.answer_chunks and not candidate.completed):
                pass
            else:
                safe_text_at = getattr(candidate, "first_safe_text_at", None) or state.metrics.first_safe_text_at
                commit_time = state.metrics.turn_committed_at or time.perf_counter()
                has_safe_text_at_eot = safe_text_at is not None and safe_text_at <= (commit_time + 0.05)
                lead_ms = (
                    (commit_time - state.metrics.speculative_started_at) * 1000
                    if state.metrics.speculative_started_at is not None
                    else None
                )
                useful_speculation = lead_ms is not None and lead_ms >= self._min_useful_spec_lead_ms
                if has_safe_text_at_eot:
                    route_name = "speculation-hit"
                    state.metrics.speculation = "hit" if useful_speculation else "hit"
                    state.metrics.route = "v2-speculation-hit"
                else:
                    route_name = "late-reuse"
                    state.metrics.speculation = "late-reuse"
                    state.metrics.route = "v2-late-reuse"
                    if not useful_speculation and state.metrics.spec_tts not in {"hit", "pcm_ready"}:
                        state.metrics.spec_tts_eligible = False
                        state.metrics.spec_tts = "not_eligible"
                        state.metrics.spec_tts_reason = "insufficient_speculation_lead"
                state.metrics.semantic_spec_reused = normalize(candidate.transcript) != normalize(text)
                state.metrics.hosted_llm_used = True
                state.v2_candidate_result = "promoted"
                logger.info("V2 ROUTE turn={} route={} action={} state={}", state.turn_id, route_name, plan.action, self.session.state.get("name"))
                # Preserve the validated final fingerprint even when the LLM
                # has not produced a safe phrase yet.  A promoted candidate
                # can then race private Cartesia against the public path when
                # that first phrase arrives, rather than losing speculative
                # TTS merely because hard EOT came first.
                final_fingerprint = self._fingerprint(text, plan)
                candidate.v2_final_fingerprint = final_fingerprint
                pcm = await self._commit_spec_audio(candidate, final_fingerprint)
                if pcm:
                    state.metrics.speculation = "hit+tts"
                state.metrics.candidate_promoted_at = time.perf_counter()
                await self._promote_speculative_candidate(state, candidate, pcm or [], plan)
                return

        if candidate:
            state.v2_candidate_result = "rejected"
            self._cancel_task(candidate.task)
            await self._abort_spec_audio(candidate)
            state.metrics.speculative_started_at = None
            state.metrics.llm_request_started_at = None
            state.metrics.llm_stream_opened_at = None
            state.metrics.llm_first_token_at = None
            state.metrics.first_speech_filter_text_at = None
            state.metrics.first_filtered_text_at = None
            state.metrics.first_safe_text_at = None
        state.candidate = None
        if not hasattr(state, "v2_candidate_result"):
            state.v2_candidate_result = "none"
        state.metrics.speculation = "miss" if candidate else "none"
        state.metrics.route = "v2-" + plan.route
        logger.info("V2 ROUTE turn={} route={} action={} state={}", state.turn_id, plan.route, plan.action, self.session.state.get("name"))
        if speech is not None:
            await self._deliver(state, speech, plan)
            return
        request = LLMRequest(text, speculative=False)
        request.v2_plan = plan
        request.v2_slots = self.session.visible_facts()
        request.v2_source = "final"
        self._prepare_request(
            state, request, text, slots_before=state.v2_slots_before
        )
        state.metrics.hosted_llm_used = True
        state.final_request = request
        request.task = asyncio.create_task(self._generate(state, text, plan, request=request))

    def _response_plan(
        self, text: str, *, slots: dict | None = None, pending_question=None,
        current_facts: dict | None = None,
    ) -> tuple[ResponsePlan, str | None]:
        slots = dict(slots or self.session.visible_facts())
        pending_question = pending_question if pending_question is not None else self.session.pending_question
        callback_state = str(slots.get("callback_state") or CallbackCoordinator.IDLE)
        if pending_question is None:
            callback_pending = {
                CallbackCoordinator.FOLLOWUP_OFFERED: ("ask_callback_consent", "followup_consent", "boolean"),
                CallbackCoordinator.AWAITING_DAY_TIME: ("ask_callback_day_time", "callback_preference", "date_and_time"),
                CallbackCoordinator.AWAITING_DAY: ("ask_callback_day", "callback_day", "date"),
                CallbackCoordinator.AWAITING_TIME: ("ask_callback_time", "callback_time", "time"),
            }.get(callback_state)
            if callback_pending:
                pending_question = PendingQuestion(*callback_pending, self.session.turn_id)
        patterns = self.session.agent.routing_policy.get("intent_patterns") or {}
        intent = self.intent_model.classify(
            text, pending_question=pending_question, fact_values=slots,
            configured_patterns=patterns,
            current_facts=current_facts,
        )
        routed = (
            self.callback_coordinator.route(intent, slots, turn_id=self.session.turn_id)
            if self.flags.enable_callback_state_machine else None
        )
        if routed is None:
            routed = self.router.route(
                text, self.session.agent, intent=intent, slots=slots,
                pending_question=pending_question, turn_id=self.session.turn_id,
            )
        risk = str(self.session.agent.risk_policy.get("class", "LOW_PUBLIC"))
        if routed and routed[0].route in {"cache", "faq-direct"} and risk != "LOW_PUBLIC":
            routed = None
        if routed:
            plan, speech = routed
            if plan.intent_id == "company_identity" and self._company_name:
                speech = f"I'm calling from {self._company_name} about your hiring plans."
        else:
            configured_intent = self._configured_intent(text)
            if configured_intent:
                plan = self.flow_engine.plan(
                    self.session.agent.flow_graph,
                    str(self.session.state.get("name", "OPEN")),
                    configured_intent,
                    slots,
                    risk_class=risk,
                )
                plan = replace(
                    plan, decision_reason=intent.reason,
                    decision_confidence=intent.confidence,
                    material_slots=self._material_slots(intent.intent_id, slots),
                )
                speech = None
            else:
                callback_sensitive = bool(
                    re.search(r"\b(?:book|schedule|appointment|callback|follow.?up)\b", text, re.I)
                    or intent.intent_id in {
                        "callback_consent_yes",
                        "callback_day",
                        "callback_time",
                        "callback_day_time",
                        "callback_preference_acknowledged",
                        "offer_callback",
                    }
                )
                plan, speech = (
                    ResponsePlan(
                        "hosted",
                        intent_id=intent.intent_id,
                        risk_class=risk,
                        allow_speculative_audio=risk in {"LOW_PUBLIC", "LOW_WORKFLOW"},
                        requires_booking_guard=callback_sensitive,
                        material_slots=self._material_slots(intent.intent_id, slots),
                        decision_reason=intent.reason,
                        decision_confidence=intent.confidence,
                    ),
                    None,
                )
        matches = self._knowledge_matches(text, plan)
        if speech is None and matches:
            top = matches[0]
            threshold = float(self.session.agent.routing_policy.get("knowledge_direct_threshold", .78))
            if top.confidence >= threshold and top.record.risk_class == "LOW_PUBLIC":
                plan = replace(
                    plan, route="faq-direct", intent_id=plan.intent_id if plan.intent_id != "unknown" else "faq_services",
                    knowledge_ids=(top.record.document_id,), allow_speculative_audio=True,
                    decision_reason=f"knowledge:{top.reason}", decision_confidence=top.confidence,
                )
                speech = top.record.text
            else:
                plan = replace(plan, knowledge_ids=tuple(item.record.document_id for item in matches))

        if (
            speech is None
            and plan.intent_id == "faq_services"
            and re.search(r"\b(?:blue collar|white collar)\b", normalize(text))
            and not matches
        ):
            plan = replace(
                plan,
                route="fixed",
                allow_speculative_audio=False,
                decision_reason="approved_knowledge_unavailable",
                decision_confidence=.99,
            )
            if "marketing" in set((current_facts or {}).get("roles") or ()):
                speech = (
                    "I've noted the marketing requirement. The available information doesn't confirm "
                    "blue- or white-collar hiring support. Our team can clarify that."
                )
            else:
                speech = (
                    "The available information doesn't confirm blue- or white-collar hiring support. "
                    "Our team can clarify that."
                )

        callback_day = str((current_facts or {}).get("callback_day") or "").strip()
        callback_time = str((current_facts or {}).get("callback_time") or "").strip()
        if callback_day and callback_time and plan.intent_id not in {
            "callback_day", "callback_time", "callback_day_time",
        }:
            preference = f"{callback_day} at {callback_time}"
            plan = replace(
                plan,
                next_state="FOLLOWUP",
                clear_pending_question=True,
                requires_booking_guard=True,
                booking_authority="preference",
                slots_written={
                    **plan.slots_written,
                    "callback_state": CallbackCoordinator.PREFERENCE_RECORDED,
                    "callback_day": callback_day,
                    "callback_time": callback_time,
                    "callback_preference": preference,
                },
                material_slots=tuple(sorted({*plan.material_slots, "callback_day", "callback_time"})),
            )
        return plan, speech

    @staticmethod
    def _material_slots(intent_id: str, slots: dict) -> tuple[str, ...]:
        requirements = (
            "hiring_status", "roles", "departments", "headcount",
            "headcount_by_role", "hiring_timeline",
        )
        mapping = {
            "provide_hiring_status": requirements,
            "provide_role": requirements,
            "provide_headcount": requirements,
            "provide_timeline": requirements,
            "correction": requirements,
            "callback_consent_yes": ("followup_consent",),
            "callback_day": ("callback_day",),
            "callback_time": ("callback_time",),
            "callback_day_time": ("callback_day", "callback_time"),
        }
        names = mapping.get(intent_id, ())
        return tuple(name for name in names if name in slots)

    def _configured_intent(self, text: str) -> str | None:
        """Match only tenant-authored phrases; never infer a flow from prose."""
        patterns = self.session.agent.routing_policy.get("intent_patterns") or {}
        normalized = normalize(text)
        if not isinstance(patterns, dict):
            return None
        for intent, phrases in patterns.items():
            if isinstance(phrases, str):
                phrases = [phrases]
            if any(normalize(str(phrase)) and normalize(str(phrase)) in normalized for phrase in phrases or []):
                return str(intent)
        return None

    def _knowledge_for(self, text: str, plan: ResponsePlan):
        """Use only the call-start tenant index; never retrieve in the cloud hot path."""
        return [match.record for match in self._knowledge_matches(text, plan)]

    def _knowledge_matches(self, text: str, plan: ResponsePlan):
        """Return ranked tenant-local knowledge matches with confidence."""
        index = self.session.knowledge_index
        if (
            not self.flags.enable_local_retrieval or index is None
            or plan.risk_class.startswith("HIGH_")
        ):
            return []
        matches = index.search_matches(
            tenant_id=self.session.tenant_id,
            agent_id=self.session.agent.agent_id,
            knowledge_version=self.session.agent.knowledge_version,
            query=text,
        )
        return [item for item in matches if not item.record.risk_class.startswith("HIGH_")]

    def _fingerprint(self, text: str, plan: ResponsePlan, *, slots: dict | None = None) -> str:
        # The prefix check below ensures generic hosted turns cannot reuse a
        # plan merely because both map to the same broad route.
        return ResponseFingerprint.from_plan(
            tenant_id=self.session.tenant_id,
            agent_version=self.session.agent.version,
            state=str(self.session.state.get("name", "OPEN")),
            intent=plan.intent_id or plan.route,
            knowledge_version=self.session.agent.knowledge_version,
            plan=plan,
            slots=slots if slots is not None else self.session.slots,
        ).digest()

    def _candidate_matches(self, candidate: LLMRequest | None, final_text: str, final_plan: ResponsePlan) -> bool:
        candidate_plan = getattr(candidate, "v2_plan", None) if candidate else None
        if candidate_plan is None:
            return False
        if candidate_plan.route != final_plan.route or candidate_plan.risk_class != final_plan.risk_class:
            return False
        if getattr(candidate, "v2_fingerprint", None) != self._fingerprint(final_text, final_plan):
            return False
        basis_words = normalize(candidate.transcript).split()
        final_words = normalize(final_text).split()
        if not basis_words:
            return False
        if (
            self.flags.enable_semantic_spec_reuse
            and final_plan.intent_id not in {None, "", "unknown"}
            and candidate_plan.intent_id == final_plan.intent_id
        ):
            return True
        return final_words == basis_words

    def _messages(
        self, text: str, plan: ResponsePlan, *, slots: dict | None = None,
        history: list[dict[str, str]] | None = None,
        diagnostics: dict | None = None,
    ) -> list[dict[str, str]]:
        contract = (
            "Runtime output contract: Output spoken text only, without control markers. "
            "The runtime owns call termination. Do not claim a callback was scheduled or an action completed without a verified tool result. "
            "No scheduling tool is connected. Start with the answer immediately. The first phrase must normally be an independently speakable 2-6 word clause ending in a period or comma. "
            "Avoid introductions such as 'Thank you for sharing', 'I understand', and 'Based on the information'. Use short spoken sentences and ask at most one question. "
            "Do not repeat information the caller already provided. Total response must be under 100 characters. Do not repeat the pitch, invent a company name, or treat uncertainty as consent."
        )
        matches = self._knowledge_matches(text, plan)
        if plan.knowledge_ids:
            matches = [m for m in matches if m.record.document_id in plan.knowledge_ids]

        if diagnostics is not None:
            diagnostics["retrieval"] = [
                {
                    "retrieval_query": text,
                    "retrieval_confidence": match.score,
                    "selected_doc_id": match.record.document_id,
                }
                for match in matches
            ]

        documents = [m.record for m in matches]
        messages = self.prompt_builder.build(
            agent=self.session.agent,
            state=self.session.state,
            slots=slots if slots is not None else self.session.slots,
            route=plan,
            knowledge=documents,
            history=self.session.history if history is None else history,
            user_text=text,
        )
        messages[0]["content"] += "\n\n" + contract
        if self._company_name:
            messages[0]["content"] += f"\nApproved company identity: {self._company_name}. Use only this company name."
        return messages

    async def _generate(self, state, text: str, plan: ResponsePlan, *, request: LLMRequest) -> None:
        stream = None
        started = False
        speech_filter = SpeechStreamFilter()
        from voice_agent.speech.booking_guard import BookingClaimGuard
        booking_guard = BookingClaimGuard() if getattr(plan, "requires_booking_guard", False) else None
        state.metrics.booking_guard_enabled = booking_guard is not None
        chunker = SafeSpeechChunker(
            min_chars=self._safe_chunk_chars,
            min_words=self._safe_chunk_words,
            max_wait_ms=self._safe_chunk_max_wait_ms,
        )
        parts: list[str] = []
        speech_chunk_count = 0
        last_safe_chunk_at: float | None = None
        chunk_lock = asyncio.Lock()
        deadline_task: asyncio.Task | None = None

        async def emit_safe(chunk: str) -> None:
            nonlocal started, speech_chunk_count, last_safe_chunk_at
            now = time.perf_counter()
            request.answer_chunks.append(chunk)
            if state.metrics.first_safe_text_at is None:
                state.metrics.first_safe_text_at = now
            speech_chunk_count += 1
            gap_ms = (
                round((now - last_safe_chunk_at) * 1000, 1)
                if last_safe_chunk_at is not None
                else 0.0
            )
            last_safe_chunk_at = now
            logger.debug(
                "V2 SPEECH CHUNK | turn={} n={} chars={} words={} upstream_gap_ms={} speculative={}",
                state.turn_id,
                speech_chunk_count,
                len(chunk),
                len(chunk.split()),
                gap_ms,
                request.speculative,
            )
            # Normally private synthesis begins while the request is still
            # speculative. A zero-delay hard-EOT promotion, however, can
            # precede the LLM's first safe phrase. Keep that *first* promoted
            # phrase eligible for the same private-TTS race; no later public
            # phrase is ever synthesized privately.
            can_late_prepare = bool(
                request.public_response_started
                and not request.normal_tts_text_sent
                and not request.private_pcm_committed
                and getattr(request, "v2_final_fingerprint", "")
            )
            if (
                (request.speculative or can_late_prepare)
                and self._spec_audio
                and request.spec_audio is None
                and request.spec_audio_task is None
                and plan.may_prepare_audio()
                # Stable interims may start private LLM/retrieval work. Flux
                # speculative synthesis waits for the stronger EagerEOT level
                # so provider cost is not spent on every changing prefix.
                and (
                    not self._flux_mode
                    or self._flux_eager
                    or getattr(request, "v2_source", "") == "eager"
                )
            ):
                state.metrics.spec_tts_eligible = True
                state.metrics.spec_tts_started_at = time.perf_counter()
                state.metrics.spec_tts = "started"
                state.metrics.spec_tts_reason = "private_synthesis_in_progress"
                request.spec_audio_text = chunk
                request.spec_audio_task = asyncio.create_task(
                    self._prepare_spec_audio(state, request, chunk),
                    name=f"v2-spec-tts-{state.turn_id}",
                )
            elif request.speculative and state.metrics.spec_tts_eligible:
                state.metrics.spec_tts = (
                    "waiting_for_eager_eot" if self._flux_mode and not self._flux_eager
                    else "waiting_for_safe_text"
                )
            if request.speculative or not self._streaming:
                return
            if not started:
                if request.public_response_started:
                    started = True
                else:
                    started = True
                    request.public_response_started = True
                    if state.metrics.response_release_at is None:
                        state.metrics.response_release_at = time.perf_counter()
                    await self.push_frame(LLMFullResponseStartFrame())
            # For a candidate promoted before it had text, give its freshly
            # started private Cartesia request a small, bounded head start.
            # If it has PCM, emit that directly; otherwise this falls through
            # to normal streaming TTS without delaying the rest of the turn.
            if can_late_prepare and request.spec_audio_task is not None:
                pcm = await self._commit_spec_audio(
                    request,
                    getattr(request, "v2_final_fingerprint"),
                    wait_secs=self._late_spec_tts_wait_secs,
                )
                if pcm:
                    request.private_pcm_committed = True
                    request.last_tts_text = chunk.strip()
                    state.metrics.speculation = "hit+tts"
                    for audio in pcm:
                        await self.push_frame(
                            TTSAudioRawFrame(audio=audio, sample_rate=24000, num_channels=1)
                        )
                    return
            if state.metrics.tts_requested_at is None:
                state.metrics.tts_requested_at = time.perf_counter()
            await self.push_frame(LLMTextFrame(self._tts_delta(request, chunk)))
            request.normal_tts_text_sent = True

        def schedule_chunk_deadline() -> None:
            """Arm a real wall-clock flush for a paused streamed response."""
            nonlocal deadline_task
            if deadline_task is not None and not deadline_task.done():
                return
            token_at = chunker.first_token_at
            if token_at is None:
                return

            async def flush_when_due(expected_token_at: float) -> None:
                try:
                    await asyncio.sleep(self._safe_chunk_max_wait_ms / 1000)
                    async with chunk_lock:
                        if request.terminal or chunker.first_token_at != expected_token_at:
                            return
                        due = chunker.release_due()
                    for safe in due:
                        await emit_safe(safe)
                except asyncio.CancelledError:
                    raise

            deadline_task = asyncio.create_task(
                flush_when_due(token_at), name=f"v2-safe-chunk-deadline-{state.turn_id}"
            )

        async def push_safe_text(clean: str) -> None:
            async with chunk_lock:
                safe_chunks = chunker.push(clean)
                schedule_chunk_deadline()
            for safe in safe_chunks:
                await emit_safe(safe)

        try:
            if state.metrics.speculative_started_at is None:
                state.metrics.speculative_started_at = time.perf_counter()
            state.metrics.llm_request_started_at = time.perf_counter()
            t_build_start = time.perf_counter()
            diagnostics = {}
            messages = self._messages(
                text,
                plan,
                slots=getattr(request, "v2_slots", None),
                history=getattr(request, "v2_history", None),
                diagnostics=diagnostics,
            )

            def _norm(s):
                return " ".join(s.casefold().split())
            user_msg_count = sum(1 for m in messages if m["role"] == "user" and _norm(m["content"]) == _norm(text))
            dedup_success = (user_msg_count == 1)
            logger.debug("V2 CONTEXT DEDUP | turn={} context_current_user_deduplicated={}", state.turn_id, str(dedup_success).lower())
            state.metrics.prompt_build_ms = round((time.perf_counter() - t_build_start) * 1000, 2)

            ctx_metrics = getattr(request, "v2_context_metrics", {})
            state.metrics.pipecat_context_read_ms = ctx_metrics.get("pipecat_context_read_ms", 0.0)
            state.metrics.context_selection_ms = ctx_metrics.get("context_selection_ms", 0.0)
            state.metrics.context_token_estimation_ms = ctx_metrics.get("context_token_estimation_ms", 0.0)
            state.metrics.pipecat_context_messages_total = int(ctx_metrics.get("total", 0))
            state.metrics.recent_context_messages_selected = int(ctx_metrics.get("selected", 0))
            state.metrics.recent_context_tokens = int(ctx_metrics.get("tokens", 0))
            state.metrics.total_prompt_tokens = int(
                self.prompt_builder.section_token_estimates.get("total", 0)
            )

            if self._log_llm_payloads:
                payload = {
                    "turn_id": state.turn_id,
                    "request_id": getattr(request, "v2_request_id", None),
                    "attempt": getattr(request, "v2_attempt", 1),
                    "request_kind": "speculative" if request.speculative else "final",
                    "transcript_source": getattr(request, "v2_source", "final"),
                    "transcript": text,
                    "state": str(self.session.state.get("name", "OPEN")),
                    "pending_question": self._pending_payload(self.session.pending_question),
                    "intent": plan.intent_id or "unknown",
                    "slots_before": getattr(request, "v2_slots_before", {}),
                    "history_sent": getattr(request, "v2_history", []),
                    "messages_sent": messages,
                    "context": ctx_metrics,
                    "diagnostics": diagnostics,
                }
                logger.debug(
                    "V2 LLM REQUEST | {}",
                    self.prompt_builder._safe_json_dumps(payload),
                )
            else:
                logger.debug(
                    "V2 LLM REQUEST | turn={} request_kind={} transcript_source={} intent={} prompt_build_ms={} retrieval={}",
                    state.turn_id,
                    "speculative" if request.speculative else "final",
                    getattr(request, "v2_source", "final"),
                    plan.intent_id or "unknown",
                    state.metrics.prompt_build_ms,
                    self.prompt_builder._safe_json_dumps(diagnostics.get("retrieval", [])),
                )
            async with asyncio.timeout(20):
                stream = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    stream=True,
                    temperature=0,
                    max_completion_tokens=self._llm_max_tokens,
                )
                state.metrics.llm_stream_opened_at = time.perf_counter()
                logger.debug("V2 PROMPT TOKENS estimated={}", self.prompt_builder.section_token_estimates)
                async for item in stream:
                    if self._state is not state or request.terminal:
                        return
                    choice = item.choices[0] if item.choices else None
                    content = getattr(getattr(choice, "delta", None), "content", None)
                    if not content:
                        continue
                    if state.metrics.llm_first_token_at is None:
                        state.metrics.llm_first_token_at = time.perf_counter()
                        ttft_ms = round(
                            (state.metrics.llm_first_token_at - state.metrics.llm_request_started_at) * 1000,
                            1,
                        ) if state.metrics.llm_request_started_at is not None else None
                        if ttft_ms is not None and ttft_ms > self._hosted_ttft_observe_ms:
                            logger.warning(
                                "V2 HOSTED TTFT BUDGET EXCEEDED | turn={} ttft_ms={} budget_ms={} speculative={} "
                                "direct_routes_checked=true hedge_enabled={}",
                                state.turn_id,
                                ttft_ms,
                                self._hosted_ttft_observe_ms,
                                request.speculative,
                                self.flags.enable_hedged_models,
                            )
                    filtered = speech_filter.push(content)
                    if filtered and state.metrics.first_speech_filter_text_at is None:
                        state.metrics.first_speech_filter_text_at = time.perf_counter()
                    clean = booking_guard.push(filtered) if booking_guard else filtered
                    if clean:
                        if state.metrics.first_filtered_text_at is None:
                            state.metrics.first_filtered_text_at = time.perf_counter()
                            if booking_guard and state.metrics.first_speech_filter_text_at is not None:
                                state.metrics.booking_guard_wait_ms = round(
                                    (state.metrics.first_filtered_text_at - state.metrics.first_speech_filter_text_at) * 1000,
                                    2,
                                )
                        parts.append(clean)
                        await push_safe_text(clean)
                filtered_tail = speech_filter.push("", final=True)
                tail = booking_guard.push(filtered_tail, final=True) if booking_guard else filtered_tail
                if tail:
                    if state.metrics.first_filtered_text_at is None:
                        state.metrics.first_filtered_text_at = time.perf_counter()
                        if booking_guard and state.metrics.first_speech_filter_text_at is not None:
                            state.metrics.booking_guard_wait_ms = round(
                                (state.metrics.first_filtered_text_at - state.metrics.first_speech_filter_text_at) * 1000,
                                2,
                            )
                    parts.append(tail)
                    await push_safe_text(tail)
                async with chunk_lock:
                    final_chunks = chunker.flush()
                for safe in final_chunks:
                    await emit_safe(safe)

            speech = "".join(parts).strip()
            if not speech:
                raise ValueError("empty_response")
            request.answer_text = speech
            request.completed = True
            if request.speculative:
                return
            if started or request.public_response_started:
                await self._finish_streamed_response(state, request, speech, plan)
            else:
                await self._deliver(state, speech, plan)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            code = getattr(exc, "code", None)
            logger.opt(exception=True).warning("V2 response failed turn={} speculative={} type={} code={} exc={}", state.turn_id, request.speculative, type(exc).__name__, code, exc)
            if not request.speculative and self._state is state:
                if started:
                    await self.push_frame(LLMFullResponseEndFrame())
                state.metrics.route = "v2-error"
                await self._speak_fixed(state, self._refusal_message if code == "content_filter" else self._operational_error_message)
        finally:
            if deadline_task is not None and not deadline_task.done():
                deadline_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await deadline_task
            if stream is not None:
                await stream.close()

    async def _prepare_spec_audio(self, state, request: LLMRequest, chunk: str) -> None:
        if not self._spec_audio or request.terminal or getattr(request, "spec_audio_invalidated", False):
            return
        prepared = await self._spec_audio.prepare(
            fingerprint=getattr(request, "v2_fingerprint", ""),
            transcript_basis=request.transcript,
            text=chunk,
        )
        if request.terminal or getattr(request, "spec_audio_invalidated", False):
            await self._spec_audio.abort(prepared)
            return
        request.spec_audio = prepared
        if prepared is None:
            if self._state is state:
                state.metrics.spec_tts = "miss"
                state.metrics.spec_tts_reason = "private_socket_unavailable"
            return
        ready = getattr(prepared, "ready", None)
        if ready is not None:
            await ready.wait()
        candidate_audio = getattr(prepared, "candidate", None)
        if (
            self._state is state and not request.spec_audio_invalidated
            and not getattr(prepared, "failed", False)
            and getattr(prepared, "completed", True)
            and (candidate_audio is None or candidate_audio.pcm_chunks)
        ):
            first_audio_at = getattr(prepared, "first_audio_at", None)
            state.metrics.spec_tts_pcm_ready_at = (
                getattr(prepared, "completed_at", None) or time.perf_counter()
            )
            state.metrics.spec_tts_first_audio_at = first_audio_at
            state.metrics.spec_tts = "pcm_ready"
            state.metrics.spec_tts_reason = "private_pcm_buffered"

    async def _start_existing_spec_audio(self, state, request: LLMRequest) -> None:
        """Upgrade an already-running stable candidate when EagerEOT arrives."""
        if (
            not self._spec_audio
            or request.spec_audio is not None
            or request.spec_audio_task is not None
            or not request.answer_chunks
            or not request.v2_plan.may_prepare_audio()
        ):
            return
        chunk = request.answer_chunks[0]
        state.metrics.spec_tts_started_at = time.perf_counter()
        state.metrics.spec_tts_eligible = True
        state.metrics.spec_tts = "started"
        state.metrics.spec_tts_reason = "private_synthesis_in_progress"
        request.spec_audio_text = chunk
        request.spec_audio_task = asyncio.create_task(
            self._prepare_spec_audio(state, request, chunk),
            name=f"v2-spec-tts-{state.turn_id}",
        )

    def _pending_from_speech(self, speech: str, turn_id: int) -> PendingQuestion | None:
        """Attach semantic meaning to a generated question for later replies."""
        if "?" not in speech:
            return None
        # Only the final question clause determines the pending semantic
        # question. A declarative callback acknowledgement followed by "Is
        # there anything else?" is not another callback-consent question.
        question_end = speech.rfind("?")
        question_start = max(
            speech.rfind(".", 0, question_end),
            speech.rfind("!", 0, question_end),
            speech.rfind("?", 0, question_end),
        )
        value = normalize(speech[question_start + 1:question_end + 1])
        patterns = (
            (r"\b(?:what day and time|day and time|preferred day|what time)\b", "ask_callback_day_time", "callback_preference", "date_and_time"),
            (r"\b(?:arrange|follow up|callback|call you back)\b", "ask_callback_consent", "followup_consent", "boolean"),
            (r"\b(?:hiring now|planning to hire|hire soon)\b", "ask_hiring_status", "hiring_status", "boolean"),
            (r"\b(?:what roles|which roles|roles are you)\b", "ask_roles", "roles", "list"),
            (r"\b(?:how many|roughly how many|headcount)\b", "ask_headcount", "headcount", "integer_or_range"),
            (r"\b(?:would you like details|interested|want details)\b", "ask_service_details", "service_details", "boolean"),
            (r"\b(?:anything else|any other questions|help you with anything else)\b", "ask_anything_else", "conversation_continue", "boolean"),
        )
        for pattern, intent, slot, expected in patterns:
            if re.search(pattern, value):
                return PendingQuestion(intent, slot, expected, turn_id)
        return PendingQuestion("ask_conversation_reply", "conversation_reply", "string", turn_id)

    def _stage_delivery(self, state, speech: str, plan: ResponsePlan) -> None:
        request = state.final_request or state.candidate
        response_id = str(getattr(request, "v2_request_id", "") or uuid.uuid4().hex)
        pending = plan.pending_question or self._pending_from_speech(speech, state.turn_id)
        self._planned_pending_question = pending
        self._delivery_state = state
        self._delivery_speech = speech
        if pending is not None or plan.clear_pending_question:
            # The prior question has been answered. A replacement question is
            # activated only by the assistant aggregator after output.
            self.session.pending_question = None
        self.session.assistant_delivery = AssistantDeliveryState(
            response_id=response_id,
            plan_intent=plan.intent_id,
            generated_text=speech,
            question_id=(f"{state.turn_id}:{pending.slot}" if pending else None),
        )

    async def handle_assistant_turn_stopped(self, *, content: str, interrupted: bool) -> None:
        delivery = self.session.assistant_delivery
        if delivery is None:
            return
        delivery.spoken_text = content
        delivery.interrupted = interrupted
        pending = self._planned_pending_question
        if pending is not None and not interrupted and content.strip():
            self.session.pending_question = pending
            logger.debug(
                "V2 QUESTION ACTIVATED | turn={} slot={} response_id={}",
                self.session.turn_id, pending.slot, delivery.response_id,
            )
        elif pending is not None:
            logger.info(
                "V2 QUESTION CANCELLED | turn={} slot={} interrupted={}",
                self.session.turn_id, pending.slot, interrupted,
            )
        self._planned_pending_question = None
        logger.debug(
            "V2 ASSISTANT TURN | turn={} response_id={} interrupted={} content_chars={}",
            self.session.turn_id, delivery.response_id, interrupted, len(content),
        )
        if self._delivery_state is not None:
            self._log_turn_finalized(self._delivery_state, self._delivery_speech)
        self._delivery_state = None
        self._delivery_speech = ""

    def _state_consistency(self, speech: str) -> str:
        """Flag obvious callback text/state disagreement in finalization logs."""
        preference = normalize(str(self.session.slots.get("callback_preference") or ""))
        if not preference:
            return "not_checked"
        days = re.findall(
            r"\b(?:today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            normalize(speech),
        )
        if days and any(day not in preference for day in days):
            return "mismatch"
        return "match" if days else "not_checked"

    def _log_turn_finalized(self, state, speech: str) -> None:
        if getattr(state, "v2_finalization_logged", False):
            return
        state.v2_finalization_logged = True
        request = state.final_request or state.candidate
        payload = {
            "turn_id": state.turn_id,
            "request_id": getattr(request, "v2_request_id", None),
            "final_transcript": state.final_transcript,
            "candidate_result": getattr(state, "v2_candidate_result", "none"),
            "slots_before": getattr(state, "v2_slots_before", {}),
            "slots_after": self.session.visible_facts(),
            "generated_response": speech,
            "spoken_response": (
                self.session.assistant_delivery.spoken_text
                if self.session.assistant_delivery else ""
            ),
            "assistant_interrupted": (
                self.session.assistant_delivery.interrupted
                if self.session.assistant_delivery else False
            ),
            "pending_question_after": self._pending_payload(self.session.pending_question),
            "state_consistency": self._state_consistency(speech),
        }
        logger.info(
            "V2 TURN FINALIZED | {}",
            self.prompt_builder._safe_json_dumps(payload),
        )

    async def _deliver(self, state, speech: str, plan: ResponsePlan) -> None:
        if self._state is not state or state.answer_finished:
            return
        if state.metrics.response_release_at is None:
            state.metrics.response_release_at = time.perf_counter()
        from voice_agent.speech.booking_guard import BookingClaimGuard
        speech = BookingClaimGuard.check(speech)
        self._stage_delivery(state, speech, plan)
        await self._speak_fixed(state, speech)
        self._remember(state, speech)
        self._apply_plan(plan)
        await self._configure_next_endpoint(speech)
        if plan.action == "end_call":
            await self.push_frame(EndFrame())

    async def _configure_next_endpoint(self, speech: str) -> None:
        updater = self._endpoint_profile_updater
        if not updater or not self.flags.enable_dynamic_endpoints:
            return
        state_name = str(self.session.state.get("name", "OPEN"))
        states = self.session.agent.flow_graph.get("states") or {}
        state_config = states.get(state_name, {}) if isinstance(states, dict) else {}
        prompt_profile = detected_profile_for_prompt(speech) if speech else None
        name = str(
            self._deferred_endpoint_profile
            or prompt_profile
            or state_config.get("endpoint_profile")
            or "freeform"
        )
        self._deferred_endpoint_profile = None
        if name == "yes_no" and self.session.pending_question and self.session.pending_question.expected_type not in {"boolean", "yes_no"}:
            name = "freeform"
        stt_profile = self.session.agent.stt_profile or {}
        terms = list(stt_profile.get("keyterms") or [])
        state_terms = stt_profile.get("state_keyterms") or {}
        terms.extend(state_terms.get(state_name, []) if isinstance(state_terms, dict) else [])
        roles = self.session.visible_facts().get("roles") or []
        terms.extend(roles if isinstance(roles, list) else [str(roles)])
        keyterms = list(dict.fromkeys(str(term) for term in terms if str(term).strip()))
        await updater(
            name, keyterms=keyterms,
            language_hints=stt_profile.get("language_hints") or None,
        )
        self._endpoint_profile = name

    async def _deliver_chunks(self, state, chunks: list[str], speech: str, plan: ResponsePlan) -> None:
        if not chunks or not self._streaming:
            await self._deliver(state, speech, plan)
            return
        if state.metrics.response_release_at is None:
            state.metrics.response_release_at = time.perf_counter()
        if state.metrics.tts_requested_at is None:
            state.metrics.tts_requested_at = time.perf_counter()
        state.answer_finished = True
        self._stage_delivery(state, speech, plan)
        await self.push_frame(LLMFullResponseStartFrame())
        for chunk in chunks:
            await self.push_frame(LLMTextFrame(chunk))
        await self.push_frame(LLMFullResponseEndFrame())
        if self._transcript_callback:
            self._transcript_callback("assistant", speech)
        await self._publish_answer_to_conversation(speech)
        self._remember(state, speech)
        self._apply_plan(plan)
        await self._configure_next_endpoint(speech)
        if plan.action == "end_call":
            await self.push_frame(EndFrame())

    async def _commit_spec_audio(
        self, request: LLMRequest, fingerprint: str, *, wait_secs: float | None = None
    ) -> list[bytes] | None:
        def complete_pcm_ready() -> bool:
            prepared = request.spec_audio
            candidate_audio = getattr(prepared, "candidate", None)
            return bool(
                prepared is not None
                and not getattr(prepared, "failed", False)
                and getattr(prepared, "completed", True)
                and (candidate_audio is None or candidate_audio.pcm_chunks)
            )

        prepared_has_pcm = complete_pcm_ready()
        if request.spec_audio_task and not request.spec_audio_task.done() and not prepared_has_pcm:
            commit_wait_secs = self._spec_tts_commit_wait_secs if wait_secs is None else wait_secs
            if commit_wait_secs > 0:
                try:
                    await asyncio.wait_for(
                        asyncio.shield(request.spec_audio_task),
                        timeout=commit_wait_secs,
                    )
                except TimeoutError:
                    pass
            prepared_has_pcm = complete_pcm_ready()
            if not prepared_has_pcm:
                request.spec_audio_invalidated = True
                if self._state:
                    self._state.metrics.spec_tts = "miss"
                    self._state.metrics.spec_tts_reason = "pcm_not_ready_at_commit"
                if self._spec_audio and request.spec_audio is not None:
                    await self._spec_audio.abort(request.spec_audio)
                self._cancel_task(request.spec_audio_task)
                return None
        if request.spec_audio is None and request.spec_audio_task and request.spec_audio_task.done():
            try:
                request.spec_audio_task.result()
            except (asyncio.CancelledError, Exception):
                pass
        if not self._spec_audio or request.spec_audio is None:
            if self._state and request.spec_audio_task is not None:
                self._state.metrics.spec_tts = "not-ready"
            return None
        if not complete_pcm_ready():
            await self._spec_audio.abort(request.spec_audio)
            if self._state:
                self._state.metrics.spec_tts = "miss"
                self._state.metrics.spec_tts_reason = "private_phrase_incomplete"
            return None
        pcm = await self._spec_audio.commit(request.spec_audio, fingerprint=fingerprint)
        if pcm and self._state:
            self._state.metrics.spec_tts_eligible = True
            self._state.metrics.spec_tts = "hit"
            self._state.metrics.spec_tts_reason = "fingerprint_validated"
            self._state.metrics.spec_tts_first_audio_at = getattr(request.spec_audio, "first_audio_at", None)
            self._state.metrics.spec_tts_committed_at = time.perf_counter()
        elif self._state:
            self._state.metrics.spec_tts = "miss"
            self._state.metrics.spec_tts_reason = "fingerprint_or_pcm_mismatch"
        return pcm

    async def _abort_spec_audio(self, request: LLMRequest | None) -> None:
        if request is None:
            return
        request.spec_audio_invalidated = True
        self._cancel_task(request.spec_audio_task)
        if self._spec_audio and request.spec_audio is not None:
            await self._spec_audio.abort(request.spec_audio)
            if self._state and self._state.metrics.spec_tts not in {"hit", "not_eligible"}:
                self._state.metrics.spec_tts = "cancelled"
                self._state.metrics.spec_tts_reason = "candidate_invalidated"

    async def _promote_speculative_candidate(self, state, request: LLMRequest, pcm: list[bytes], plan: ResponsePlan) -> None:
        """Make a validated soft-EOT candidate public without restarting its LLM.

        The candidate may still be generating. Only its first safe phrase is
        eligible for private PCM; subsequent safe chunks stream into the normal
        Cartesia WebSocket while that prefix is playing.
        """
        if self._state is not state or state.answer_finished or request.public_response_started:
            return
        state.metrics.response_release_at = time.perf_counter()
        request.speculative = False
        request.public_response_started = True
        request.private_pcm_committed = bool(pcm)
        state.final_request = request
        await self.push_frame(LLMFullResponseStartFrame())
        for audio in pcm:
            await self.push_frame(TTSAudioRawFrame(audio=audio, sample_rate=24000, num_channels=1))
        # The private socket receives only the first safe phrase. Reuse all
        # other chunks that arrived before hard EOT, then let the same LLM
        # task keep streaming new chunks. This avoids both a dropped phrase
        # and a second hosted request.
        public_chunks = list(request.answer_chunks)
        if pcm and request.spec_audio_text:
            try:
                public_chunks.remove(request.spec_audio_text)
            except ValueError:
                # A custom provider can return PCM without exposing its
                # prepared text. In that case prefer speaking the full text
                # once over silently dropping content.
                pass
            else:
                request.last_tts_text = request.spec_audio_text
        for chunk in public_chunks:
            if state.metrics.tts_requested_at is None:
                state.metrics.tts_requested_at = time.perf_counter()
            await self.push_frame(LLMTextFrame(self._tts_delta(request, chunk)))
            request.normal_tts_text_sent = True
        if request.completed:
            await self._finish_streamed_response(state, request, request.answer_text, plan)

    @staticmethod
    def _tts_delta(request: LLMRequest, chunk: str) -> str:
        """Keep independently emitted safe phrases lexically separated.

        ``SafeSpeechChunker`` deliberately returns trimmed phrase values. The
        provider receives several frames, however, so a missing separator
        would turn ``"engineers."`` + ``"Which roles"`` into
        ``"engineers.Which roles"``. The same rule applies when public text
        follows a committed private PCM phrase.
        """
        text = chunk.strip()
        if not text:
            return ""
        if request.last_tts_text or request.private_pcm_committed:
            text = " " + text
        request.last_tts_text = text
        return text

    async def _finish_streamed_response(self, state, request: LLMRequest, speech: str, plan: ResponsePlan) -> None:
        if self._state is not state or state.answer_finished:
            return
        state.answer_finished = True
        self._stage_delivery(state, speech, plan)
        await self.push_frame(LLMFullResponseEndFrame())
        # A private PCM-only candidate has no public Cartesia context to emit
        # this marker. Without it the output transport would remain speaking.
        if request.private_pcm_committed and not request.normal_tts_text_sent:
            await self.push_frame(TTSStoppedFrame())
        if self._transcript_callback:
            self._transcript_callback("assistant", speech)
        await self._publish_answer_to_conversation(speech)
        self._remember(state, speech)
        self._apply_plan(plan)
        await self._configure_next_endpoint(speech)
        if plan.action == "end_call":
            await self.push_frame(EndFrame())

    def _apply_plan(self, plan: ResponsePlan) -> None:
        """Commit deterministic state only after the validated response is released."""
        if plan.next_state:
            self.session.state["name"] = plan.next_state
        if plan.clear_pending_question and plan.pending_question is None:
            self.session.pending_question = None
        for name in plan.slots_cleared:
            self.session.slots.pop(name, None)
            self.session.facts.pop(name, None)
        for name, value in plan.slots_written.items():
            schema = self.session.agent.slot_schema.get(name, {})
            validation = self.slot_validator.validate(schema, value)
            if validation.valid:
                self.session.slots[name] = validation.value
            else:
                logger.warning("V2 skipped invalid configured slot update name={} reason={}", name, validation.reason)

    def _remember(self, state, speech: str) -> None:
        self.session.history.extend([
            {"role": "user", "content": state.final_transcript},
            {"role": "assistant", "content": speech},
        ])
        self.session.history = self.session.history[-6:]

    async def cleanup(self):
        if self._settle_task:
            self._settle_task.cancel()
            await asyncio.gather(self._settle_task, return_exceptions=True)
        if self._spec_audio:
            await self._spec_audio.close()
        await super().cleanup()
