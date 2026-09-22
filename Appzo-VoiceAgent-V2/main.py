import asyncio
import json
import os
import re
import time
import math
import socket
from urllib.parse import urlencode
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field

from dotenv import load_dotenv
from loguru import logger
from openai import AsyncAzureOpenAI, AsyncOpenAI

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FrameProcessed, FramePushed
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregatorParams,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi.models import BotOutputMessage, BotOutputMessageData
from pipecat.processors.frameworks.rtvi.observer import RTVIObserverParams
from pipecat.processors.frameworks.rtvi.processor import RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.services.cartesia.tts import CartesiaTTSService, GenerationConfig
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.tts_service import TextAggregationMode
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.turns.user_start import TranscriptionUserTurnStartStrategy
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.workers.runner import WorkerRunner
from websockets.asyncio.client import connect as websocket_connect

from voice_agent.runtime.latency_breakdown import LatencyBreakdown


load_dotenv()


DEFAULT_REFUSAL = "Sorry, I can't help with that request."
OPERATIONAL_ERROR = "Sorry, I couldn't process that request just now. Please try again."

LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "180"))
DEEPGRAM_ENDPOINTING_MS = int(os.getenv("DEEPGRAM_ENDPOINTING_MS", "100"))
SPECULATION_ENABLED = os.getenv("SPECULATION_ENABLED", "true").lower() == "true"
SPECULATION_DEBOUNCE_SECS = int(os.getenv("SPECULATION_DEBOUNCE_MS", "80")) / 1000
SPECULATION_MIN_WORDS = int(os.getenv("SPECULATION_MIN_WORDS", "3"))
SPECULATION_MIN_CHARS = int(os.getenv("SPECULATION_MIN_CHARS", "12"))
SPECULATION_MAX_RESTARTS = int(os.getenv("SPECULATION_MAX_RESTARTS", "2"))
CARTESIA_MODEL = os.getenv("CARTESIA_MODEL", "sonic-3.5")


@dataclass(frozen=True)
class AgentRuntimeConfig:
    """Call-specific settings supplied by Goodbox at call start."""

    llm_api_key: str
    llm_model: str
    llm_max_tokens: int
    system_prompt: str
    deepgram_api_key: str
    stt_model: str
    stt_language: str
    deepgram_endpointing_ms: int
    cartesia_api_key: str
    cartesia_voice_id: str
    cartesia_model: str
    cartesia_speed: float
    vad_confidence: float = 0.7
    vad_start_secs: float = 0.2
    vad_stop_secs: float = 0.2
    vad_min_volume: float = 0.6
    intro_message: str | None = None
    llm_client: object | None = None
    owns_llm_client: bool = True
    refusal_message: str = DEFAULT_REFUSAL
    operational_error_message: str = OPERATIONAL_ERROR


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower(), flags=re.UNICODE).split())


@dataclass
class TurnMetrics:
    turn_id: int
    provider_eot_at: float | None = None
    aggregator_stop_at: float | None = None
    provider_turn_id: int | None = None
    last_voiced_at: float | None = None
    turn_committed_at: float | None = None
    final_stt_at: float | None = None
    stable_interim_at: float | None = None
    stable_candidate_started_at: float | None = None
    stable_candidate_source: str | None = None
    eager_eot_at: float | None = None
    eager_eot_confidence: float | None = None
    eot_confidence: float | None = None
    eot_trigger: str | None = None
    turn_resumed_count: int = 0
    input_gap_count: int = 0
    input_gap_max_ms: float = 0.0
    speculative_started_at: float | None = None
    llm_request_started_at: float | None = None
    llm_stream_opened_at: float | None = None
    llm_first_token_at: float | None = None
    first_speech_filter_text_at: float | None = None
    first_filtered_text_at: float | None = None
    first_safe_text_at: float | None = None
    spec_tts_started_at: float | None = None
    spec_tts_first_audio_at: float | None = None
    spec_tts_pcm_ready_at: float | None = None
    spec_tts_committed_at: float | None = None
    spec_tts_eligible: bool = False
    commit_at: float | None = None
    candidate_validated_at: float | None = None
    candidate_promoted_at: float | None = None
    protocol_validated_at: float | None = None
    response_release_at: float | None = None
    booking_guard_enabled: bool = False
    booking_guard_wait_ms: float = 0.0
    tts_requested_at: float | None = None
    tts_first_audio_at: float | None = None
    tts_first_non_silent_at: float | None = None
    output_first_packet_at: float | None = None
    output_first_non_silent_at: float | None = None
    output_last_packet_at: float | None = None
    output_packet_count: int = 0
    output_max_packet_gap_ms: float = 0.0
    output_long_gap_count: int = 0
    output_silent_packet_count: int = 0
    output_audio_ms: float = 0.0
    bot_started_at: float | None = None
    route: str = "pending"
    speculation: str = "none"
    spec_tts: str = "none"
    spec_tts_reason: str = ""
    endpoint_profile: str = ""
    semantic_spec_reused: bool = False
    knowledge_direct_hit: bool = False
    decision_route: str = "pending"
    decision_intent: str = "unknown"
    decision_reason: str = ""
    hosted_llm_used: bool = False
    pipecat_context_read_ms: float = 0.0
    context_selection_ms: float = 0.0
    context_token_estimation_ms: float = 0.0
    prompt_build_ms: float = 0.0
    pipecat_context_messages_total: int = 0
    recent_context_messages_selected: int = 0
    recent_context_tokens: int = 0
    total_prompt_tokens: int = 0
    retrieval_query: str = ""
    retrieval_confidence: float | None = None
    selected_doc_id: str | None = None


@dataclass
class LLMRequest:
    transcript: str
    speculative: bool
    task: asyncio.Task | None = None
    protocol: str | None = None
    prefix_buffer: str = ""
    answer_chunks: list[str] = field(default_factory=list)
    answer_text: str = ""
    emitted: bool = False
    completed: bool = False
    terminal: bool = False
    end_call: bool = False
    # V2 fills these only for a low-risk, private speculative-TTS candidate.
    # Keeping them on the shared request object lets cancellation invalidate
    # both hosted generation and its prepared PCM together.
    spec_audio: object | None = None
    spec_audio_task: asyncio.Task | None = None
    spec_audio_text: str = ""
    spec_audio_invalidated: bool = False
    public_response_started: bool = False
    normal_tts_text_sent: bool = False
    private_pcm_committed: bool = False
    last_tts_text: str = ""


@dataclass
class TurnState:
    turn_id: int
    metrics: TurnMetrics
    latest_interim: str = ""
    final_transcript: str = ""
    turn_stopped: bool = False
    committed: bool = False
    answer_finished: bool = False
    speculation_restarts: int = 0
    candidate: LLMRequest | None = None
    final_request: LLMRequest | None = None
    debounce_task: asyncio.Task | None = None
    final_wait_task: asyncio.Task | None = None


class StreamingVoiceController(FrameProcessor):
    """Goodbox-configured voice controller shared by the V1 rollback and V2 router."""

    _VOICE_LEVEL = int(os.getenv("V2_INPUT_VOICE_RMS", "200"))

    def __init__(
        self,
        api_key: str,
        publish_conversation_answer: Callable[[str], Awaitable[None]] | None = None,
        *,
        client=None,
        model: str = LLM_MODEL,
        max_tokens: int = LLM_MAX_TOKENS,
        system_prompt: str,
        transcript_callback: Callable[[str, str], None] | None = None,
        refusal_message: str = DEFAULT_REFUSAL,
        operational_error_message: str = OPERATIONAL_ERROR,
        owns_llm_client: bool = True,
    ) -> None:
        super().__init__(name="StreamingVoiceController")
        self._client = client or AsyncOpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        self._owns_llm_client = owns_llm_client if client is not None else True
        self._model = model
        self._llm_max_tokens = max_tokens
        self._system_prompt = system_prompt
        self._transcript_callback = transcript_callback
        self._refusal_message = refusal_message
        self._operational_error_message = operational_error_message
        self._publish_conversation_answer = publish_conversation_answer
        self._history: list[dict[str, str]] = []
        self._turn_counter = 0
        self._state: TurnState | None = None
        self._metrics_by_turn: dict[int, TurnMetrics] = {}
        self._latency_observer: Any | None = None
        self._latest_voiced_audio_at: float | None = None

    @property
    def metrics_by_turn(self) -> dict[int, TurnMetrics]:
        return self._metrics_by_turn

    async def handle_native_turn_started(self) -> None:
        """Receive Pipecat's VAD/Smart Turn start signal from the aggregator."""
        await self._start_turn()

    async def handle_native_turn_stopped(self, transcript: str | None = None) -> None:
        """Receive Pipecat's validated turn-stop signal from the aggregator.

        Pipecat can start a turn from transcription when VAD misses quiet
        speech. In that case this controller has not received its usual VAD
        start callback, so reconcile from the aggregator's finalized text
        before committing the turn.
        """
        final_text = normalize(transcript or "")
        state = self._state
        if final_text and (state is None or state.committed):
            await self._start_turn()
            state = self._state
        if final_text and state is not None and not state.committed:
            state.final_transcript = final_text
            state.metrics.final_stt_at = time.perf_counter()
        await self._on_turn_stopped()

    async def cleanup(self):
        await self._cancel_turn_work(self._state)
        if self._owns_llm_client:
            await self._client.close()
        await super().cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, InputAudioRawFrame):
            self._mark_voice(frame.audio)
        elif isinstance(frame, InterimTranscriptionFrame):
            logger.info("STT interim chars={}", len(frame.text or ""))
            await self._on_interim(frame.text)
        elif isinstance(frame, TranscriptionFrame):
            logger.info("STT final chars={} finalized={}", len(frame.text or ""), frame.finalized)
            if hasattr(self, "note_stt_final"):
                self.note_stt_final(frame.finalized)
            await self._on_final_transcript(frame.text)

        # Turn boundaries come exclusively from the user aggregator callbacks
        # below. Deepgram may emit a transcription-start frame after a turn has
        # already stopped. That event also broadcasts an InterruptionFrame, but
        # it is not a real barge-in and must not cancel the final-STT wait.
        # A real VAD start invokes _start_turn() below, which cancels prior
        # work before creating the next turn.

        # The native user aggregator creates LLMContextFrame as part of its
        # Smart Turn lifecycle. This controller owns the single Groq request,
        # so the frame must not continue to a second LLM service.
        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)

    async def _start_turn(self) -> None:
        # Both VAD and transcription can report a start for the same utterance.
        # Ignore the duplicate so it cannot cancel an in-flight final transcript.
        if self._state and not self._state.committed and not self._state.turn_stopped:
            return
        await self._cancel_turn_work(self._state)
        self._turn_counter += 1
        metrics = TurnMetrics(turn_id=self._turn_counter)
        metrics.endpoint_profile = getattr(self, "_endpoint_profile", "")
        # Attach fresh voiced audio timestamp if available
        obs_latest = None
        if getattr(self, "_latency_observer", None) is not None:
            obs_latest = self._latency_observer._latest_voiced_audio_at
        if obs_latest is None:
            obs_latest = self._latest_voiced_audio_at
        if obs_latest is not None:
            freshness_ms = float(os.getenv("V2_VOICE_TIMESTAMP_FRESHNESS_MS", "1000"))
            if (time.perf_counter() - obs_latest) * 1000 <= freshness_ms:
                metrics.last_voiced_at = obs_latest
        self._metrics_by_turn[self._turn_counter] = metrics
        self._state = TurnState(turn_id=self._turn_counter, metrics=metrics)

    def _mark_voice(self, audio: bytes) -> None:
        if len(audio) < 2:
            return
        aligned = audio[: len(audio) - len(audio) % 2]
        samples = memoryview(aligned).cast("h")
        rms = math.sqrt(sum(int(sample) * int(sample) for sample in samples) / len(samples)) if samples else 0.0
        if rms >= self._VOICE_LEVEL:
            now = time.perf_counter()
            self._latest_voiced_audio_at = now
            if getattr(self, "_latency_observer", None) is not None:
                self._latency_observer._latest_voiced_audio_at = now
            state = self._state
            if state is not None and not state.committed:
                state.metrics.last_voiced_at = now

    async def _on_interim(self, text: str) -> None:
        state = self._state
        if state is None:
            # Defensive fallback for transports that yield a transcript before
            # their VAD callback reaches the user-turn aggregator.
            await self._start_turn()
            state = self._state
        if state is None or state.committed:
            return
        normalized = normalize(text)
        if not normalized or normalized == state.latest_interim:
            return
        state.latest_interim = normalized
        if not SPECULATION_ENABLED:
            return
        if len(normalized) < SPECULATION_MIN_CHARS or len(normalized.split()) < SPECULATION_MIN_WORDS:
            return
        self._cancel_task(state.debounce_task)
        state.debounce_task = asyncio.create_task(self._debounce_speculation(state, normalized))

    async def _debounce_speculation(self, state: TurnState, transcript: str) -> None:
        try:
            await asyncio.sleep(SPECULATION_DEBOUNCE_SECS)
            if self._state is not state or state.committed or state.latest_interim != transcript:
                return
            if state.speculation_restarts >= SPECULATION_MAX_RESTARTS:
                return
            if state.candidate and state.candidate.transcript == transcript:
                return
            self._cancel_task(state.candidate.task if state.candidate else None)
            request = LLMRequest(transcript=transcript, speculative=True)
            state.candidate = request
            state.speculation_restarts += 1
            state.metrics.speculative_started_at = time.perf_counter()
            request.task = asyncio.create_task(self._run_llm(state, request))
        except asyncio.CancelledError:
            raise

    async def _on_final_transcript(self, text: str) -> None:
        if not normalize(text):
            return
        state = self._state
        if state is None:
            await self._start_turn()
            state = self._state
        if state is None or state.committed:
            return
        state.final_transcript = normalize(text)
        state.metrics.final_stt_at = time.perf_counter()
        if state.turn_stopped:
            await self._commit_final_turn(state)

    async def _on_turn_stopped(self) -> None:
        state = self._state
        if state is None or state.committed or state.turn_stopped:
            return
        state.turn_stopped = True
        logger.info("TURN stop turn={} final_chars={} interim_chars={}",
                    state.turn_id, len(state.final_transcript), len(state.latest_interim))
        state.metrics.turn_committed_at = time.perf_counter()
        if state.final_transcript:
            await self._commit_final_turn(state)
            return
        state.final_wait_task = asyncio.create_task(self._wait_for_final_transcript(state))

    async def _wait_for_final_transcript(self, state: TurnState) -> None:
        try:
            await asyncio.sleep(0.75)
            if self._state is not state or state.committed:
                return
            state.final_wait_task = None
            if state.latest_interim:
                logger.warning(
                    f"Turn {state.turn_id}: no final Nova-3 transcript; "
                    "recovering from the latest interim transcript"
                )
                state.final_wait_task = None
                state.final_transcript = state.latest_interim
                state.metrics.final_stt_at = time.perf_counter()
                await self._commit_final_turn(state)
            else:
                logger.warning("STT empty turn={}; asking caller to repeat", state.turn_id)
                state.committed = True
                state.metrics.route = "stt-empty-retry"
                await self._speak_fixed(state, "Sorry, I didn't catch that. Could you please repeat?")
        except asyncio.CancelledError:
            raise

    async def _commit_final_turn(self, state: TurnState) -> None:
        if state.committed or self._state is not state:
            return
        self._cancel_task(state.final_wait_task)
        final_text = state.final_transcript
        state.committed = True
        if self._transcript_callback:
            self._transcript_callback("user", final_text)

        candidate = state.candidate
        if candidate and not candidate.terminal and candidate.transcript == final_text:
            state.metrics.route = "llm"
            state.metrics.speculation = "hit"
            await self._release_if_ready(state, candidate)
            return

        if candidate:
            state.metrics.speculation = "miss"
            self._cancel_task(candidate.task)
        state.metrics.route = "llm"
        request = LLMRequest(transcript=final_text, speculative=False)
        state.final_request = request
        request.task = asyncio.create_task(self._run_llm(state, request))

    async def _run_llm(self, state: TurnState, request: LLMRequest) -> None:
        try:
            logger.info("LLM request turn={} model={} speculative={}", state.turn_id, self._model, request.speculative)
            messages = [{"role": "system", "content": self._system_prompt}, *self._history, {"role": "user", "content": request.transcript}]
            request_kwargs = dict(
                model=self._model,
                messages=messages,
                stream=True,
                temperature=0,
                max_completion_tokens=self._llm_max_tokens,
            )
            # Groq accepts these extensions; Azure GPT-4.1 does not.
            if not isinstance(self._client, AsyncAzureOpenAI):
                request_kwargs["extra_body"] = {
                    "reasoning_effort": "low",
                    "include_reasoning": False,
                }
            stream = await asyncio.wait_for(self._client.chat.completions.create(**request_kwargs), timeout=15)
            async for chunk in stream:
                if self._state is not state or request.terminal:
                    return
                # Azure may include content-filter/usage metadata chunks with a
                # Choice but no Delta. They are valid stream events, not an LLM
                # failure, and contain no text for TTS.
                choice = chunk.choices[0] if chunk.choices else None
                delta = getattr(choice, "delta", None)
                text = getattr(delta, "content", None)
                if not text:
                    continue
                if state.metrics.llm_first_token_at is None:
                    logger.info("LLM first token turn={}", state.turn_id)
                    state.metrics.llm_first_token_at = time.perf_counter()
                await self._accept_llm_text(state, request, text)
            request.completed = True
            if not request.answer_text.strip():
                request.terminal = True
                if self._state is state and state.committed and not state.answer_finished:
                    await self._speak_fixed(state, self._operational_error_message)
                return
            await self._release_if_ready(state, request)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(f"Turn {state.turn_id}: LLM request failed: {exc}")
            if self._state is state and state.committed and not state.answer_finished:
                request.terminal = True
                await self._speak_fixed(state, self._operational_error_message)

    async def _accept_llm_text(self, state: TurnState, request: LLMRequest, text: str) -> None:
        if request.protocol is None:
            request.prefix_buffer += text
            stripped = request.prefix_buffer.lstrip()
            if stripped.startswith("OK|"):
                request.protocol = "OK"
                state.metrics.protocol_validated_at = time.perf_counter()
                remainder = stripped[3:]
                request.prefix_buffer = ""
                if remainder:
                    request.answer_chunks.append(remainder)
                    request.answer_text += remainder
            elif stripped.startswith("END|"):
                request.protocol = "END"
                request.end_call = True
                state.metrics.protocol_validated_at = time.perf_counter()
                remainder = stripped[4:]
                request.prefix_buffer = ""
                if remainder:
                    request.answer_chunks.append(remainder)
                    request.answer_text += remainder
            elif stripped.startswith("NO|"):
                request.protocol = "NO"
                request.terminal = True
                state.metrics.protocol_validated_at = time.perf_counter()
                if state.committed:
                    await self._speak_fixed(state, self._refusal_message)
                return
            elif len(stripped) > 12:
                request.terminal = True
                logger.warning(f"Turn {state.turn_id}: LLM violated the OK|/END|/NO| protocol")
                if state.committed:
                    await self._speak_fixed(state, self._operational_error_message)
                return
            else:
                return
        else:
            request.answer_chunks.append(text)
            request.answer_text += text

        await self._release_if_ready(state, request)

    async def _release_if_ready(self, state: TurnState, request: LLMRequest) -> None:
        if self._state is not state or not state.committed or request.terminal:
            return
        if request.protocol == "NO":
            request.terminal = True
            await self._speak_fixed(state, self._refusal_message)
            return
        if request.protocol not in {"OK", "END"}:
            return
        if not request.emitted:
            request.emitted = True
            state.metrics.tts_requested_at = time.perf_counter()
            await self.push_frame(LLMFullResponseStartFrame())
        while request.answer_chunks:
            await self.push_frame(LLMTextFrame(request.answer_chunks.pop(0)))
        if request.completed and not state.answer_finished:
            state.answer_finished = True
            if self._transcript_callback:
                self._transcript_callback("assistant", request.answer_text)
            await self._publish_answer_to_conversation(request.answer_text)
            await self.push_frame(LLMFullResponseEndFrame())
            self._history = [
                {"role": "user", "content": request.transcript},
                {"role": "assistant", "content": request.answer_text.strip()},
            ]
            if request.end_call:
                logger.info(f"Turn {state.turn_id}: closing response completed; ending Plivo call")
                await self.push_frame(EndFrame())

    async def _speak_fixed(self, state: TurnState, text: str) -> None:
        if self._state is not state or state.answer_finished:
            return
        state.answer_finished = True
        if self._transcript_callback:
            self._transcript_callback("assistant", text)
        now = time.perf_counter()
        if state.metrics.response_release_at is None:
            state.metrics.response_release_at = now
        if state.metrics.tts_requested_at is None:
            state.metrics.tts_requested_at = now
        await self._publish_answer_to_conversation(text)
        # The assistant aggregator is the canonical same-call dialogue store.
        # Preserve fixed/deterministic speech in that context just like hosted
        # streamed speech; interruption handling decides what was delivered.
        await self.push_frame(TTSSpeakFrame(text, append_to_context=True))

    async def _publish_answer_to_conversation(self, text: str) -> None:
        """Send the complete response to the UI without changing the TTS stream.

        Cartesia's token-level streaming intentionally suppresses partial
        ``bot-output`` messages in the modern RTVI conversation component.
        Publishing the completed text on the data channel avoids its trailing
        punctuation (often ``.``) being displayed as the entire response.
        """
        answer = text.strip()
        if answer and self._publish_conversation_answer:
            await self._publish_conversation_answer(answer)

    async def _cancel_turn_work(self, state: TurnState | None) -> None:
        if state is None:
            return
        self._cancel_task(state.debounce_task)
        self._cancel_task(state.final_wait_task)
        self._cancel_task(state.candidate.task if state.candidate else None)
        self._cancel_task(state.final_request.task if state.final_request else None)

    @staticmethod
    def _cancel_task(task: asyncio.Task | None) -> None:
        if task and not task.done():
            task.cancel()


class LiveLatencyObserver(BaseObserver):
    """Logs response latency without conflating TTS TTFB with full turn latency."""

    def __init__(
        self,
        controller: StreamingVoiceController,
        *,
        tts_transport: str,
        session=None,
        call_origin_at: float | None = None,
    ) -> None:
        super().__init__()
        # Pipecat 1.7.0 has BaseObserver hooks but no packaged latency
        # observer/event. Expose the requested event contract on our custom
        # observer so applications can consume LatencyBreakdown objects.
        self._register_event_handler("on_latency_breakdown", sync=True)
        self._controller = controller
        self._tts_transport = tts_transport
        self._session = session
        self._call_origin_at = call_origin_at
        self._seen: set[int] = set()
        self._reported_turn_ids: set[int] = set()
        self._reporting_turn_ids: set[int] = set()
        self._cadence_seen: set[int] = set()
        self._latest_voiced_audio_at: float | None = None
        self._voice_freshness_ms = float(os.getenv("V2_VOICE_TIMESTAMP_FRESHNESS_MS", "1000"))
        # Output frames can arrive after the controller has opened the next
        # caller turn. Capture the response owner when speech starts instead
        # of resolving late PCM/stop frames through mutable ``_state``.
        self._active_response_metrics: TurnMetrics | None = None
        self._samples: dict[str, list[int]] = {}
        self._audible_samples: dict[str, list[int]] = {}
        self._audible_threshold = int(os.getenv("V2_AUDIBLE_PCM_RMS", "200"))
        self._audio_gap_alert_ms = max(100, int(os.getenv("V2_AUDIO_GAP_ALERT_MS", "350")))
        self._breakdown_min_secs = max(
            0.0, float(os.getenv("V2_LATENCY_BREAKDOWN_MIN_MS", "1")) / 1000
        )
        from voice_agent.runtime.flags import RuntimeFlags
        self._runtime_flags = RuntimeFlags.from_env()
        self._audible_metrics_enabled = self._runtime_flags.enable_audible_pcm_metrics

    async def on_pipeline_started(self):
        if self._call_origin_at is not None:
            logger.info(
                "V2 PIPELINE READY | plivo-connect->pipeline-ready={} ms",
                round((time.perf_counter() - self._call_origin_at) * 1000),
            )

    async def on_process_frame(self, data: FrameProcessed):
        """Record PCM as it enters the live output transport.

        TTS-service timestamps measure provider output. These separate fields
        expose queue/transport delay before the same audio is accepted by the
        telephony output processor.
        """
        if data.direction != FrameDirection.DOWNSTREAM:
            return
        if isinstance(data.frame, InputAudioRawFrame):
            audio = data.frame.audio
            if len(audio) >= 2:
                aligned = audio[: len(audio) - len(audio) % 2]
                samples = memoryview(aligned).cast("h")
                rms = math.sqrt(sum(int(s) * int(s) for s in samples) / len(samples)) if samples else 0.0
                if rms >= self._audible_threshold:
                    now = time.perf_counter()
                    self._latest_voiced_audio_at = now
                    state = self._controller._state
                    if state is not None and not state.committed:
                        state.metrics.last_voiced_at = now
            return
        if not isinstance(data.processor, BaseOutputTransport) or not isinstance(data.frame, TTSAudioRawFrame):
            return
        metrics = self._active_response_metrics
        if metrics is None:
            state = self._controller._state
            metrics = state.metrics if state is not None else None
        if metrics is None or metrics.tts_requested_at is None:
            return
        now = time.perf_counter()
        if metrics.output_first_packet_at is None:
            metrics.output_first_packet_at = now
        if metrics.output_last_packet_at is not None:
            gap_ms = (now - metrics.output_last_packet_at) * 1000
            metrics.output_max_packet_gap_ms = max(metrics.output_max_packet_gap_ms, gap_ms)
            if gap_ms >= self._audio_gap_alert_ms:
                metrics.output_long_gap_count += 1
                logger.warning(
                    "V2 AUDIO GAP | turn={} gap_ms={} threshold_ms={} packets_before_gap={}",
                    metrics.turn_id,
                    round(gap_ms, 1),
                    self._audio_gap_alert_ms,
                    metrics.output_packet_count,
                )
        metrics.output_last_packet_at = now
        metrics.output_packet_count += 1
        metrics.output_audio_ms += len(data.frame.audio) / max(1, data.frame.sample_rate * 2) * 1000
        audible = self._is_audible(data.frame.audio)
        if not audible:
            metrics.output_silent_packet_count += 1
        if (
            self._audible_metrics_enabled
            and metrics.output_first_non_silent_at is None
            and audible
        ):
            metrics.output_first_non_silent_at = now
            self._schedule_report_if_not_reported(metrics)

    async def on_push_frame(self, data: FramePushed):
        if data.direction != FrameDirection.DOWNSTREAM or data.frame.id in self._seen:
            return
        self._seen.add(data.frame.id)
        if isinstance(data.frame, InputAudioRawFrame):
            audio = data.frame.audio
            if len(audio) >= 2:
                aligned = audio[: len(audio) - len(audio) % 2]
                samples = memoryview(aligned).cast("h")
                rms = math.sqrt(sum(int(s) * int(s) for s in samples) / len(samples)) if samples else 0.0
                if rms >= self._audible_threshold:
                    now = time.perf_counter()
                    self._latest_voiced_audio_at = now
                    state = self._controller._state
                    if state is not None and not state.committed:
                        state.metrics.last_voiced_at = now
            return
        state = self._controller._state
        if isinstance(data.frame, (LLMFullResponseStartFrame, TTSSpeakFrame)):
            if state is not None and state.metrics.tts_requested_at is not None:
                self._active_response_metrics = state.metrics
        metrics = self._active_response_metrics
        if metrics is None and state is not None and state.metrics.tts_requested_at is not None:
            metrics = state.metrics
        if metrics is None:
            return
        # No current response has requested speech yet. Audio here belongs to
        # the greeting or an interrupted older response, not this caller turn.
        if metrics.tts_requested_at is None:
            return
        now = time.perf_counter()
        if isinstance(data.frame, TTSAudioRawFrame) and metrics.tts_first_audio_at is None:
            metrics.tts_first_audio_at = now
        if self._audible_metrics_enabled and isinstance(data.frame, TTSAudioRawFrame) and metrics.tts_first_non_silent_at is None:
            if self._is_audible(data.frame.audio):
                metrics.tts_first_non_silent_at = now
        if isinstance(data.frame, BotStartedSpeakingFrame) and metrics.bot_started_at is None:
            metrics.bot_started_at = now
            # This is only a fallback. When PCM metrics are enabled, wait for
            # actual non-silent output PCM so the report is caller-perceived.
            if not self._audible_metrics_enabled:
                self._schedule_report_if_not_reported(metrics)
        if (
            isinstance(data.frame, (BotStoppedSpeakingFrame, TTSStoppedFrame))
            and metrics.turn_id not in self._cadence_seen
        ):
            # Final safety net: report even if no inspectable output PCM or
            # BotStartedSpeakingFrame was observed.
            self._schedule_report_if_not_reported(metrics)
            self._cadence_seen.add(metrics.turn_id)
            logger.info(
                "AUDIO CADENCE | turn={} packets={} audio_ms={} max_packet_gap_ms={} long_gaps={} silent_packets={}",
                metrics.turn_id,
                metrics.output_packet_count,
                round(metrics.output_audio_ms),
                round(metrics.output_max_packet_gap_ms, 1),
                metrics.output_long_gap_count,
                metrics.output_silent_packet_count,
            )
            logger.info(
                "AUDIO CADENCE RECORD | {}",
                json.dumps(
                    {
                        "call_id": getattr(self._session, "call_id", None),
                        "turn_id": metrics.turn_id,
                        "output_packet_count": metrics.output_packet_count,
                        "output_max_packet_gap_ms": metrics.output_max_packet_gap_ms,
                        "output_long_gap_count": metrics.output_long_gap_count,
                        "output_silent_packet_count": metrics.output_silent_packet_count,
                        "output_audio_ms": metrics.output_audio_ms,
                    },
                    sort_keys=True,
                ),
            )
            self._active_response_metrics = None

    @staticmethod
    def _metrics_for(value) -> TurnMetrics:
        return getattr(value, "metrics", value)

    def _schedule_report_if_not_reported(self, response) -> None:
        metrics = self._metrics_for(response)
        if metrics.turn_id in self._reported_turn_ids or metrics.turn_id in self._reporting_turn_ids:
            return
        self._reporting_turn_ids.add(metrics.turn_id)
        asyncio.create_task(self._report_latency(metrics))

    async def _report_latency(self, response) -> None:
        metrics = self._metrics_for(response)
        try:
            first_audible = (
                metrics.output_first_non_silent_at
                if self._audible_metrics_enabled
                else metrics.bot_started_at
            )
            breakdown = LatencyBreakdown.from_turn(
                metrics,
                model=getattr(self._controller, "_model", "unknown"),
                tts_transport=self._tts_transport,
                require_audible_pcm=self._audible_metrics_enabled,
            )
            native_eot_to_audio = self._ms(metrics.turn_committed_at, metrics.bot_started_at)
            raw_audio_to_eot = self._ordered_ms(metrics.last_voiced_at, metrics.turn_committed_at)
            raw_audio_to_bot = self._ordered_ms(metrics.last_voiced_at, first_audible)
            if native_eot_to_audio is not None and metrics.route not in {"v2-error", "stt-empty-retry"}:
                self._samples.setdefault(metrics.route, []).append(native_eot_to_audio)
            eot_to_audible = self._ms(metrics.turn_committed_at, first_audible)
            if eot_to_audible is not None and metrics.route not in {"v2-error", "stt-empty-retry"}:
                self._audible_samples.setdefault(metrics.route, []).append(eot_to_audible)
            report_lines = [
                "RESPONSE LATENCY | "
                f"tenant={getattr(self._session, 'tenant_id', 'legacy')} bundle={getattr(getattr(self._session, 'agent', None), 'version', 'legacy')} "
                f"state={getattr(self._session, 'state', {}).get('name', 'UNKNOWN') if self._session else 'LEGACY'} "
                f"turn={metrics.turn_id} route={metrics.route} tts={self._tts_transport} speculation={metrics.speculation} spec_tts={metrics.spec_tts} spec_tts_reason={metrics.spec_tts_reason or 'none'} | "
                f"provider-turn={metrics.provider_turn_id} eot-trigger={metrics.eot_trigger} eot-confidence={metrics.eot_confidence} "
                f"provider-EOT->aggregator={self._ms(metrics.provider_eot_at, metrics.aggregator_stop_at)} ms | "
                f"native-EOT->bot-audio={native_eot_to_audio} ms | raw-audio->native-EOT={raw_audio_to_eot} ms | "
                f"raw-audio->first-audible={raw_audio_to_bot} ms | "
                f"LLM-request->stream={self._ms(metrics.llm_request_started_at, metrics.llm_stream_opened_at)} ms | "
                f"LLM-stream->first-token={self._ms(metrics.llm_stream_opened_at, metrics.llm_first_token_at)} ms | "
                f"first-token->speech-filter={self._ms(metrics.llm_first_token_at, metrics.first_speech_filter_text_at)} ms | "
                f"speech-filter->booking-safe={self._ms(metrics.first_speech_filter_text_at, metrics.first_filtered_text_at)} ms | "
                f"booking-safe->safe-chunk={self._ms(metrics.first_filtered_text_at, metrics.first_safe_text_at)} ms | "
                f"EOT->first-safe-text={self._ms(metrics.turn_committed_at, metrics.first_safe_text_at)} ms | "
                f"EOT->first-audible={eot_to_audible} ms | "
                f"input-gaps={metrics.input_gap_count} input-max-gap={metrics.input_gap_max_ms} ms"
            ]
            logger.info("\n".join(report_lines))
            coverage = self._optimization_coverage()
            logger.info(
                "V2 OPTIMIZATION METRICS | turn={} deterministic_coverage={} hosted_llm_coverage={} "
                "stable_spec_lead_eager_ms={} stable_spec_lead_hard_eot_ms={} semantic_spec_hit={} "
                "spec_tts_eligible={} spec_tts_started={} spec_tts_pcm_ready={} spec_tts_committed={} "
                "knowledge_direct_hit={} first_token_to_first_safe_text_ms={} "
                "hard_eot_to_first_audible_ms={} raw_speech_end_to_first_audible_ms={}",
                metrics.turn_id,
                coverage["deterministic_coverage"], coverage["hosted_llm_coverage"],
                self._lead_ms(metrics.stable_candidate_started_at, metrics.eager_eot_at),
                self._lead_ms(metrics.stable_candidate_started_at, metrics.turn_committed_at),
                coverage["semantic_spec_hit"], coverage["spec_tts_eligible"],
                coverage["spec_tts_started"], coverage["spec_tts_pcm_ready"],
                coverage["spec_tts_committed"], coverage["knowledge_direct_hit"],
                self._ms(metrics.llm_first_token_at, metrics.first_safe_text_at),
                eot_to_audible, raw_audio_to_bot,
            )
            if breakdown is not None:
                self._reported_turn_ids.add(metrics.turn_id)
                await self._call_event_handler("on_latency_breakdown", breakdown)
            else:
                logger.warning(
                    "LATENCY BREAKDOWN UNAVAILABLE | turn={} route={} "
                    "hard_eot={} output_first_non_silent={} bot_started={}",
                    metrics.turn_id, metrics.route, metrics.turn_committed_at,
                    metrics.output_first_non_silent_at, metrics.bot_started_at,
                )
            logger.info("LATENCY RECORD | {}", json.dumps(self._record(metrics, breakdown, first_audible), sort_keys=True))
        finally:
            self._reporting_turn_ids.discard(metrics.turn_id)

    def _record(self, metrics, breakdown, first_audible):
        return {
            "call_id": getattr(self._session, "call_id", None),
            "tenant_id": getattr(self._session, "tenant_id", "legacy"),
            "agent_version": getattr(getattr(self._session, "agent", None), "version", "legacy"),
            "turn_id": metrics.turn_id,
            "route": metrics.route,
            "endpoint_mode": metrics.endpoint_profile,
            "model": getattr(self._controller, "_model", "unknown"),
            "tts_transport": self._tts_transport,
            "speculation": metrics.speculation,
            "spec_tts": metrics.spec_tts,
            "spec_tts_reason": metrics.spec_tts_reason,
            "decision_route": metrics.decision_route,
            "decision_intent": metrics.decision_intent,
            "decision_reason": metrics.decision_reason,
            "hosted_llm_used": metrics.hosted_llm_used,
            "knowledge_direct_hit": metrics.knowledge_direct_hit,
            "semantic_spec_reused": metrics.semantic_spec_reused,
            "flags": asdict(self._runtime_flags),
            "last_voiced_at": metrics.last_voiced_at,
            "provider_eot_at": metrics.provider_eot_at,
            "aggregator_stop_at": metrics.aggregator_stop_at,
            "hard_eot_at": metrics.turn_committed_at,
            "commit_at": metrics.commit_at,
            "eager_eot_at": metrics.eager_eot_at,
            "eager_eot_confidence": metrics.eager_eot_confidence,
            "eot_confidence": metrics.eot_confidence,
            "eot_trigger": metrics.eot_trigger,
            "turn_resumed_count": metrics.turn_resumed_count,
            "input_gap_count": metrics.input_gap_count,
            "input_gap_max_ms": metrics.input_gap_max_ms,
            "speculative_started_at": metrics.speculative_started_at,
            "stable_candidate_started_at": metrics.stable_candidate_started_at,
            "stable_candidate_source": metrics.stable_candidate_source,
            "stable_spec_lead_eager_ms": self._lead_ms(metrics.stable_candidate_started_at, metrics.eager_eot_at),
            "stable_spec_lead_hard_eot_ms": self._lead_ms(metrics.stable_candidate_started_at, metrics.turn_committed_at),
            "llm_request_started_at": metrics.llm_request_started_at,
            "llm_stream_opened_at": metrics.llm_stream_opened_at,
            "llm_first_token_at": metrics.llm_first_token_at,
            "first_filtered_text_at": metrics.first_filtered_text_at,
            "first_speech_filter_text_at": metrics.first_speech_filter_text_at,
            "candidate_validated_at": metrics.candidate_validated_at,
            "candidate_promoted_at": metrics.candidate_promoted_at,
            "first_safe_text_at": metrics.first_safe_text_at,
            "first_token_to_first_safe_text_ms": self._ms(metrics.llm_first_token_at, metrics.first_safe_text_at),
            "spec_tts_eligible": metrics.spec_tts_eligible,
            "spec_tts_started_at": metrics.spec_tts_started_at,
            "spec_tts_pcm_ready_at": metrics.spec_tts_pcm_ready_at,
            "spec_tts_committed_at": metrics.spec_tts_committed_at,
            "tts_requested_at": metrics.tts_requested_at,
            "tts_first_audio_at": metrics.tts_first_audio_at,
            "tts_first_non_silent_at": metrics.tts_first_non_silent_at,
            "output_first_packet_at": metrics.output_first_packet_at,
            "output_first_non_silent_at": metrics.output_first_non_silent_at,
            "output_packet_count": metrics.output_packet_count,
            "output_max_packet_gap_ms": metrics.output_max_packet_gap_ms,
            "output_long_gap_count": metrics.output_long_gap_count,
            "output_silent_packet_count": metrics.output_silent_packet_count,
            "output_audio_ms": metrics.output_audio_ms,
            "first_audible_at": first_audible,
            "hard_eot_to_first_audible_ms": self._ms(metrics.turn_committed_at, first_audible),
            "raw_speech_end_to_first_audible_ms": self._ordered_ms(metrics.last_voiced_at, first_audible),
            "first_audible_source": "output_non_silent_pcm" if first_audible is not None else "unavailable",
            "bot_started_at": metrics.bot_started_at,
            "raw_speech_end_to_eager_eot_ms": self._ordered_ms(metrics.last_voiced_at, metrics.eager_eot_at),
            "eager_eot_to_hard_eot_ms": self._ordered_ms(metrics.eager_eot_at, metrics.turn_committed_at),
            "raw_speech_end_to_hard_eot_ms": self._ordered_ms(metrics.last_voiced_at, metrics.turn_committed_at),
            "spec_start_to_first_safe_ms": self._ms(metrics.speculative_started_at, metrics.first_safe_text_at),
            "spec_start_to_pcm_ready_ms": self._ms(metrics.spec_tts_started_at, metrics.spec_tts_pcm_ready_at),
            "spec_tts_savings_ms": self._ordered_ms(metrics.spec_tts_pcm_ready_at, metrics.turn_committed_at),
            "pipecat_context_read_ms": metrics.pipecat_context_read_ms,
            "context_selection_ms": metrics.context_selection_ms,
            "context_token_estimation_ms": metrics.context_token_estimation_ms,
            "prompt_build_ms": metrics.prompt_build_ms,
            "pipecat_context_messages_total": metrics.pipecat_context_messages_total,
            "recent_context_messages_selected": metrics.recent_context_messages_selected,
            "recent_context_tokens": metrics.recent_context_tokens,
            "total_prompt_tokens": metrics.total_prompt_tokens,
            "retrieval_query": metrics.retrieval_query,
            "retrieval_confidence": metrics.retrieval_confidence,
            "selected_doc_id": metrics.selected_doc_id,
            "latency_breakdown": breakdown.as_dict() if breakdown is not None else None,
        }

    async def cleanup(self):
        metrics_by_turn = getattr(self._controller, "metrics_by_turn", {})
        response_turns = [m for m in metrics_by_turn.values() if m.tts_requested_at is not None]
        audible_turns = [m for m in response_turns if m.output_first_non_silent_at is not None]
        missing_reports = [m.turn_id for m in audible_turns if m.turn_id not in self._reported_turn_ids]
        report_without_response = [
            turn_id for turn_id in self._reported_turn_ids
            if turn_id not in metrics_by_turn or metrics_by_turn[turn_id].tts_requested_at is None
        ]
        audible_bot_responses = sum(
            1 for m in response_turns
            if m.output_first_non_silent_at is not None or m.bot_started_at is not None
        )
        valid_latency_reports = len(self._reported_turn_ids)
        missing_speech_end_timestamps = sum(
            1 for m in metrics_by_turn.values()
            if m.tts_requested_at is not None and (m.last_voiced_at is None or m.turn_committed_at is None)
        )
        phantom_pending_reports = sum(
            1 for tid in self._reported_turn_ids
            if tid in metrics_by_turn and (
                metrics_by_turn[tid].route == "pending"
                or metrics_by_turn[tid].tts_requested_at is None
            )
        )
        logger.info(
            "LATENCY INTEGRITY | responses={} audible={} reports={} missing_audible_reports={} reports_without_response={}",
            len(response_turns), len(audible_turns), len(self._reported_turn_ids),
            missing_reports, report_without_response,
        )
        logger.info(
            "LATENCY CALL-END INTEGRITY | audible_bot_responses={} valid_latency_reports={} "
            "missing_speech_end_timestamps={} phantom_pending_reports={}",
            audible_bot_responses,
            valid_latency_reports,
            missing_speech_end_timestamps,
            phantom_pending_reports,
        )
        self._log_call_summary(list(metrics_by_turn.values()))
        if self._reported_turn_ids:
            logger.info(
                "V2 OPTIMIZATION SUMMARY | {}",
                " ".join(f"{key}={value}" for key, value in self._optimization_coverage().items()),
            )
        for route, samples in sorted(self._samples.items()):
            ordered = sorted(samples)
            percentile = lambda p: ordered[max(0, math.ceil(len(ordered) * p) - 1)]
            logger.info(
                f"NATIVE-EOT->BOT-AUDIO SUMMARY | route={route} tts={self._tts_transport} n={len(ordered)} p50={percentile(.50)} ms "
                f"p90={percentile(.90)} ms p95={percentile(.95)} ms max={ordered[-1]} ms"
            )
        for route, samples in sorted(self._audible_samples.items()):
            ordered = sorted(samples)
            percentile = lambda p: ordered[max(0, math.ceil(len(ordered) * p) - 1)]
            logger.info(
                f"EOT->FIRST-AUDIBLE SUMMARY | route={route} tts={self._tts_transport} n={len(ordered)} p50={percentile(.50)} ms "
                f"p90={percentile(.90)} ms p95={percentile(.95)} ms p99={percentile(.99)} ms max={ordered[-1]} ms"
            )
        await super().cleanup()

    def _log_call_summary(self, all_turns: list[TurnMetrics]) -> None:
        turns = [item for item in all_turns if item.tts_requested_at is not None]
        successful = [
            item for item in turns
            if item.output_first_non_silent_at is not None
            and item.route not in {"v2-error", "stt-empty-retry"}
        ]
        deterministic_routes = {
            "fixed", "cache", "identity", "faq-direct", "callback-preference",
            "callback-preference-acknowledged",
        }

        def values(start_name: str, end_name: str, items=successful) -> list[int]:
            result = []
            for item in items:
                value = self._ms(getattr(item, start_name), getattr(item, end_name))
                if value is not None:
                    result.append(value)
            return result

        def stats(items: list[int]) -> str:
            if not items:
                return "n=0 p50=None p90=None p95=None max=None"
            ordered = sorted(items)
            pick = lambda p: ordered[max(0, math.ceil(len(ordered) * p) - 1)]
            return (
                f"n={len(ordered)} p50={pick(.50)} p90={pick(.90)} "
                f"p95={pick(.95)} max={ordered[-1]}"
            )

        hosted = [item for item in successful if item.hosted_llm_used]
        deterministic = [item for item in successful if item.decision_route in deterministic_routes]
        retrieval = [item for item in successful if item.decision_route in {"cache", "faq-direct"}]
        errors = [item for item in turns if item.route in {"v2-error", "stt-empty-retry"}]
        context_turns = [item for item in hosted if item.recent_context_messages_selected or item.recent_context_tokens]
        avg_messages = (
            round(sum(item.recent_context_messages_selected for item in context_turns) / len(context_turns), 1)
            if context_turns else 0.0
        )
        avg_tokens = (
            round(sum(item.recent_context_tokens for item in context_turns) / len(context_turns), 1)
            if context_turns else 0.0
        )
        logger.info(
            "CALL LATENCY SUMMARY | total_turns={} deterministic_turns={} hosted_turns={} retrieval_turns={} error_turns={} "
            "speech_end_to_audible=[{}] speech_end_to_hard_eot=[{}] hosted_llm_ttft=[{}] "
            "tts_first_pcm=[{}] avg_context_messages={} avg_context_tokens={}",
            len(all_turns), len(deterministic), len(hosted), len(retrieval), len(errors),
            stats(values("last_voiced_at", "output_first_non_silent_at")),
            stats(values("last_voiced_at", "turn_committed_at")),
            stats(values("llm_request_started_at", "llm_first_token_at", hosted)),
            stats(values("tts_requested_at", "tts_first_audio_at")),
            avg_messages, avg_tokens,
        )

    def _optimization_coverage(self) -> dict[str, str]:
        metrics_by_turn = getattr(self._controller, "metrics_by_turn", {})
        turns = [
            metrics for turn_id, metrics in metrics_by_turn.items()
            if turn_id in self._reported_turn_ids
        ]
        total = len(turns)
        speculative = [item for item in turns if item.speculation in {"hit", "hit+tts", "miss"}]
        eligible = [item for item in turns if item.spec_tts_eligible]

        def percent(count: int, denominator: int) -> str:
            return f"{(100 * count / denominator):.1f}%({count}/{denominator})" if denominator else "n/a(0/0)"

        deterministic = sum(
            item.decision_route in {"fixed", "cache", "identity", "faq-direct", "callback-preference", "callback-preference-acknowledged"}
            for item in turns
        )
        return {
            "deterministic_coverage": percent(deterministic, total),
            "hosted_llm_coverage": percent(sum(item.hosted_llm_used for item in turns), total),
            "semantic_spec_hit": percent(sum(item.semantic_spec_reused for item in speculative), len(speculative)),
            "spec_tts_eligible": percent(len(eligible), total),
            "spec_tts_started": percent(sum(item.spec_tts_started_at is not None for item in eligible), len(eligible)),
            "spec_tts_pcm_ready": percent(sum(item.spec_tts_pcm_ready_at is not None for item in eligible), len(eligible)),
            "spec_tts_committed": percent(sum(item.spec_tts_committed_at is not None for item in eligible), len(eligible)),
            "knowledge_direct_hit": percent(sum(item.knowledge_direct_hit for item in turns), total),
        }

    def _is_audible(self, audio: bytes) -> bool:
        if len(audio) < 2:
            return False
        samples = memoryview(audio[: len(audio) - len(audio) % 2]).cast("h")
        if not samples:
            return False
        rms = math.sqrt(sum(int(sample) * int(sample) for sample in samples) / len(samples))
        return rms >= self._audible_threshold

    @staticmethod
    def _ms(start: float | None, end: float | None) -> int | None:
        return round((end - start) * 1000) if start is not None and end is not None and end >= start else None

    @staticmethod
    def _lead_ms(start: float | None, end: float | None) -> int | None:
        return round((end - start) * 1000) if start is not None and end is not None and end >= start else None

    @staticmethod
    def _ordered_ms(start: float | None, end: float | None) -> int | None:
        if start is None or end is None or start > end:
            return None
        return round((end - start) * 1000)


async def _cartesia_websocket_available(api_key: str, timeout_secs: float) -> bool:
    """Check the exact Cartesia WSS path without putting a secret in logs."""
    url = "wss://api.cartesia.ai/tts/websocket?" + urlencode(
        {"cartesia_version": "2026-03-01"}
    )
    try:
        async with websocket_connect(
            url,
            additional_headers={"X-API-Key": api_key},
            proxy=None,
            family=socket.AF_INET,
            open_timeout=timeout_secs,
            max_size=None,
        ):
            return True
    except Exception as exc:
        logger.warning("Cartesia WebSocket health check failed: {}", type(exc).__name__)
        return False

async def run_bot(
    transport: BaseTransport,
    runner_args: RunnerArguments | None = None,
    *,
    runtime_config: AgentRuntimeConfig | None = None,
    transcript_callback: Callable[[str, str], None] | None = None,
    v2_session=None,
    telephony_stream_id: str | None = None,
    telephony_connected_at: float | None = None,
) -> None:
    """Run one Goodbox-configured agent over phone media."""
    if runtime_config is None:
        raise ValueError("V2 requires a Goodbox AgentRuntimeConfig at call start")
    runtime = runtime_config
    use_flux = runtime.stt_model.lower().startswith("flux")
    from voice_agent.turns.flux import OrderedFluxSTTService, flux_turn_strategies
    from voice_agent.turns.endpoint_profiles import flux_profile
    from voice_agent.runtime.flags import RuntimeFlags
    runtime_flags = RuntimeFlags.from_env()
    endpoint_profile_name = os.getenv(
        "V2_FLUX_ENDPOINT_PROFILE",
        "fast" if runtime_flags.enable_flux_tuning else "balanced",
    ).casefold()
    endpoint_profile = flux_profile(endpoint_profile_name)
    greeting_capture = None
    greeting_player = None
    cached_greeting = None
    greeting_key = None
    greeting_cache = None
    if (
        v2_session is not None
        and runtime.intro_message
        and runtime_flags.enable_cached_greeting
        and telephony_stream_id
    ):
        from voice_agent.speech.greeting_cache import GreetingCache, GreetingCacheKey
        greeting_cache = GreetingCache()
        greeting_key = GreetingCacheKey(
            tenant_id=v2_session.tenant_id,
            agent_id=v2_session.agent.agent_id,
            agent_version=v2_session.agent.version,
            voice_id=runtime.cartesia_voice_id,
            tts_model=runtime.cartesia_model,
            speed=runtime.cartesia_speed,
            intro_text=runtime.intro_message,
        )
        cached_greeting = greeting_cache.get(greeting_key)
    endpoint_profile_preset = os.getenv("V2_ENDPOINT_PROFILE_PRESET", "current").lower()
    tts_buffer_ms = int(os.getenv("CARTESIA_MAX_BUFFER_DELAY_MS", os.getenv("V2_TTS_BUFFER_DELAY_MS", "75")))
    logger.info(
        "TURN config model={} stt={} endpointing_ms={} vad_stop_secs={} strategy={} flux_preset={} tts_buffer_ms={}",
        runtime.llm_model,
        runtime.stt_model,
        runtime.deepgram_endpointing_ms,
        runtime.vad_stop_secs,
        f"Flux/ExternalTurn/{endpoint_profile_name}" if use_flux else "SmartTurn",
        endpoint_profile_preset,
        tts_buffer_ms,
    )
    rtvi_processor = RTVIProcessor()

    async def publish_conversation_answer(text: str) -> None:
        await rtvi_processor.push_transport_message(
            BotOutputMessage(
                data=BotOutputMessageData(
                    text=text,
                    aggregated_by="sentence",
                    spoken=True,
                    will_be_spoken=True,
                    # RTVI v2 requires an initial "new" event to create the
                    # assistant message. A completed-only event updates the
                    # currently active segment; when none exists, the client
                    # renders its empty typing placeholder ("...").
                    spoken_status="new",
                )
            )
        )
        logger.debug(f"CONVERSATION UI | published {len(text)} characters")

    requested_tts_transport = os.getenv("V2_TTS_TRANSPORT", "websocket").lower()
    resolved_tts_transport = requested_tts_transport
    if (
        v2_session is not None
        and os.getenv("ENABLE_V2_ROUTING", "true").lower() == "true"
        and requested_tts_transport == "auto"
    ):
        available = await _cartesia_websocket_available(
            runtime.cartesia_api_key,
            float(os.getenv("V2_CARTESIA_HEALTH_TIMEOUT_SECS", "2")),
        )
        resolved_tts_transport = "websocket" if available else "http"
        logger.info(
            "V2 Cartesia transport requested=auto resolved={}", resolved_tts_transport
        )

    controller_class = StreamingVoiceController
    controller_options = {}
    if v2_session is not None and os.getenv("ENABLE_V2_ROUTING", "true").lower() == "true":
        from live_v2 import V2RoutingController
        controller_class = V2RoutingController
        controller_options.update(
            session=v2_session,
            cartesia_api_key=runtime.cartesia_api_key,
            cartesia_voice_id=runtime.cartesia_voice_id,
            cartesia_model=runtime.cartesia_model,
            cartesia_speed=runtime.cartesia_speed,
            tts_transport=resolved_tts_transport,
            flux_mode=use_flux,
        )
        logger.info("V2 ROUTING enabled; deterministic/cache/hosted routes active")
    controller = controller_class(
        runtime.llm_api_key,
        publish_conversation_answer=publish_conversation_answer,
        client=runtime.llm_client,
        model=runtime.llm_model,
        max_tokens=runtime.llm_max_tokens,
        system_prompt=runtime.system_prompt,
        transcript_callback=transcript_callback,
        refusal_message=runtime.refusal_message,
        operational_error_message=runtime.operational_error_message,
        owns_llm_client=runtime.owns_llm_client,
        **controller_options,
    )
    controller._endpoint_profile = endpoint_profile_name if use_flux else "nova-smartturn"
    if cached_greeting is not None and telephony_stream_id:
        from voice_agent.speech.greeting_cache import CachedGreetingPlayer
        greeting_player = CachedGreetingPlayer(
            transport.output(),
            stream_id=telephony_stream_id,
            greeting=cached_greeting,
            call_origin_at=telephony_connected_at,
        )
        controller._greeting_interrupt = greeting_player.interrupt
    latency_observer = LiveLatencyObserver(
        controller,
        tts_transport=("http-batch" if controller_options and resolved_tts_transport == "http" else "websocket-stream"),
        session=v2_session if controller_options else None,
        call_origin_at=telephony_connected_at,
    )
    controller._latency_observer = latency_observer

    @latency_observer.event_handler("on_latency_breakdown")
    async def on_latency_breakdown(_observer, breakdown):
        metrics = controller.metrics_by_turn.get(breakdown.turn_id)
        context_lines = ""
        if metrics is not None:
            context_lines = (
                "\n"
                f"context_messages={metrics.recent_context_messages_selected} "
                f"context_tokens={metrics.recent_context_tokens} "
                f"total_prompt_tokens={metrics.total_prompt_tokens}\n"
                f"pipecat_context_read_ms={metrics.pipecat_context_read_ms} "
                f"context_selection_ms={metrics.context_selection_ms} "
                f"context_token_estimation_ms={metrics.context_token_estimation_ms} "
                f"prompt_build_ms={metrics.prompt_build_ms}"
            )
        logger.info(
            "── LATENCY BREAKDOWN | turn={} ──\n{}{}",
            breakdown.turn_id,
            "\n".join(
                f"  {line}"
                for line in breakdown.turn_contribution_lines(
                    latency_observer._breakdown_min_secs
                )
            ),
            context_lines,
        )
    context = LLMContext()
    # Flux owns start, resume and hard-EOT detection.  Giving the aggregator
    # external strategies is essential: pairing Flux with Silero/SmartTurn
    # would double-trigger turn boundaries and reintroduce the duplicate-turn
    # tail latency seen in the Nova logs.
    user_params = LLMUserAggregatorParams(
        user_turn_strategies=flux_turn_strategies() if use_flux else None,
        vad_analyzer=(
            None
            if use_flux
            else SileroVADAnalyzer(
                params=VADParams(
                    confidence=runtime.vad_confidence,
                    start_secs=runtime.vad_start_secs,
                    stop_secs=runtime.vad_stop_secs,
                    min_volume=runtime.vad_min_volume,
                )
            )
        ),
    )
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=user_params,
        assistant_params=LLMAssistantAggregatorParams(
            enable_auto_context_summarization=False,
        ),
    )
    if controller_options:
        controller.set_dialogue_context(context)

    @user_aggregator.event_handler("on_user_turn_started")
    async def on_user_turn_started(_aggregator, strategy):
        # A transcription can arrive after the VAD turn has stopped. It must
        # not start a new controller turn and discard the final transcript.
        if controller_options and isinstance(strategy, TranscriptionUserTurnStartStrategy):
            await controller.handle_native_turn_started(transcription_only=True)
        else:
            await controller.handle_native_turn_started()

    @user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(_aggregator, _strategy, _message):
        await controller.handle_native_turn_stopped(_message.content)

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(_aggregator, message):
        if controller_options:
            await controller.handle_assistant_turn_stopped(
                content=message.content or "",
                interrupted=bool(message.interrupted),
            )
    keyterms = list(v2_session.agent.stt_profile.get("keyterms", [])) if controller_options else []
    if controller_options:
        initial_state = str(v2_session.state.get("name", "OPEN"))
        state_keyterms = v2_session.agent.stt_profile.get("state_keyterms") or {}
        if isinstance(state_keyterms, dict):
            keyterms.extend(state_keyterms.get(initial_state, []) or [])
        keyterms = list(dict.fromkeys(str(term) for term in keyterms if str(term).strip()))
    if use_flux:
        # `flux-general-multi` accepts language hints rather than the Nova
        # `language=multi` query value.  Goodbox currently serves English and
        # Hindi callers, so preserve that multilingual intent explicitly.
        language_hints = [Language.EN, Language.HI] if runtime.stt_language == "multi" else None
        stt = OrderedFluxSTTService(
            api_key=runtime.deepgram_api_key,
            url=os.getenv("DEEPGRAM_FLUX_URL", "wss://api.in.deepgram.com/v2/listen"),
            settings=DeepgramFluxSTTService.Settings(
                model=runtime.stt_model,
                language_hints=language_hints,
                eager_eot_threshold=float(os.getenv("V2_FLUX_EAGER_EOT_THRESHOLD", str(endpoint_profile.eager_eot_threshold))),
                eot_threshold=float(os.getenv("V2_FLUX_EOT_THRESHOLD", str(endpoint_profile.eot_threshold))),
                eot_timeout_ms=int(os.getenv("V2_FLUX_EOT_TIMEOUT_MS", str(endpoint_profile.eot_timeout_ms))),
                keyterm=keyterms,
            ),
        )
        if controller_options and runtime_flags.enable_dynamic_endpoints:
            from voice_agent.turns.endpoint_profiles import profile_for_prompt

            async def update_flux_endpoint(
                name: str, *, keyterms: list[str] | None = None, language_hints=None
            ) -> None:
                profile = flux_profile(name)
                await stt.configure_endpoint(
                    profile, keyterms=keyterms, language_hints=language_hints
                )
                controller._endpoint_profile = name
                logger.info(
                    "V2 FLUX PROFILE | state={} name={} eager={} eot={} timeout_ms={} keyterms={}",
                    getattr(controller, "session", None).state.get("name", "UNKNOWN"),
                    name,
                    profile.eager_eot_threshold,
                    profile.eot_threshold,
                    profile.eot_timeout_ms,
                    len(keyterms or []),
                )

            controller._endpoint_profile_updater = update_flux_endpoint
            if runtime.intro_message:
                initial_name = profile_for_prompt(runtime.intro_message)
                initial_profile = flux_profile(initial_name)
                stt._settings.eager_eot_threshold = initial_profile.eager_eot_threshold
                stt._settings.eot_threshold = initial_profile.eot_threshold
                stt._settings.eot_timeout_ms = initial_profile.eot_timeout_ms
                controller._endpoint_profile = initial_name
    else:
        stt = DeepgramSTTService(
            api_key=runtime.deepgram_api_key,
            settings=DeepgramSTTService.Settings(
                model=runtime.stt_model,
                language=runtime.stt_language,
                interim_results=True,
                endpointing=runtime.deepgram_endpointing_ms,
                punctuate=True,
                smart_format=False,
                keyterm=keyterms,
            ),
        )
    tts_settings = CartesiaTTSService.Settings(
        voice=runtime.cartesia_voice_id,
        model=runtime.cartesia_model,
        generation_config=GenerationConfig(speed=runtime.cartesia_speed),
    )
    if controller_options and resolved_tts_transport == "http":
        # An explicit HTTP setting, or an `auto` WebSocket health-check miss.
        # This preserves audible calls while keeping provider failure visible.
        from pipecat.services.cartesia.tts import CartesiaHttpTTSService
        logger.warning("V2 TTS transport=http; WebSocket streaming is unavailable for this call")
        tts = CartesiaHttpTTSService(
            api_key=runtime.cartesia_api_key,
            settings=tts_settings,
            sample_rate=24000,
            encoding="pcm_s16le",
            container="raw",
        )
    else:
        # Use the same direct Cartesia WebSocket path that the proven
        # low-latency call prototype uses. Retrying an opening handshake in
        # the media path turns a transient failure into seconds of silence.
        tts_class = CartesiaTTSService
        if controller_options:
            from resilient_tts import ResilientCartesiaTTSService
            tts_class = ResilientCartesiaTTSService
        tts = tts_class(
            api_key=runtime.cartesia_api_key,
            settings=tts_settings,
            sample_rate=24000,
            encoding="pcm_s16le",
            container="raw",
            # V2's SafeSpeechChunker emits complete phrase-sized chunks; token
            # mode prevents Pipecat from adding a second sentence-sized wait.
            # A non-zero buffer delay lets Cartesia bridge the gap between
            # successive phrase chunks when the LLM streams tokens slowly, avoiding
            # the elongated-word / dropout artefact on slow-LLM turns.
            text_aggregation_mode=(TextAggregationMode.TOKEN if controller_options else TextAggregationMode.SENTENCE),
            max_buffer_delay_ms=int(os.getenv("CARTESIA_MAX_BUFFER_DELAY_MS", os.getenv("V2_TTS_BUFFER_DELAY_MS", "75"))) if controller_options else None,
        )
    pipeline_processors = [transport.input(), stt, controller, user_aggregator, tts]
    if controller_options and os.getenv("ENABLE_TTS_LEADING_SILENCE_TRIM", "true").lower() == "true":
        from voice_agent.speech.audio_quality import InitialSilenceTrimmer
        pipeline_processors.append(InitialSilenceTrimmer())
    if greeting_cache is not None and greeting_key is not None and cached_greeting is None:
        from voice_agent.speech.greeting_cache import GreetingCaptureProcessor
        greeting_capture = GreetingCaptureProcessor(greeting_cache, greeting_key)
        pipeline_processors.append(greeting_capture)
    # Pipecat dialogue memory must observe the assistant text after the output
    # path. This keeps committed context aligned with what was actually sent
    # and lets interruption events reconcile V2's semantic pending question.
    pipeline_processors.extend([transport.output(), assistant_aggregator])
    pipeline = Pipeline(pipeline_processors)
    runner = WorkerRunner(
        handle_sigint=runner_args.handle_sigint if runner_args else False
    )
    worker = PipelineWorker(
        pipeline,
        name="goodbox-voice-agent",
        rtvi_processor=rtvi_processor,
        # Cartesia's token stream emits a punctuation-only aggregated segment
        # to the stock observer. Completed answers are sent explicitly above.
        rtvi_observer_params=RTVIObserverParams(
            bot_output_enabled=False,
            bot_llm_enabled=False,
            bot_tts_enabled=False,
        ),
        params=PipelineParams(
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
            enable_metrics=True,
            enable_usage_metrics=True,
            send_initial_empty_metrics=False,
        ),
        observers=[latency_observer],
    )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport, _client):
        await runner.cancel()

    if runtime.intro_message:

        @transport.event_handler("on_client_connected")
        async def on_client_connected(_transport, _client):
            if transcript_callback:
                transcript_callback("assistant", runtime.intro_message)
            if greeting_player is not None:
                greeting_player.start()
            else:
                await worker.queue_frame(
                    TTSSpeakFrame(runtime.intro_message, append_to_context=False)
                )

    await runner.add_workers(worker)
    try:
        await runner.run()
    finally:
        if greeting_player is not None:
            await greeting_player.close()


async def bot(runner_args: RunnerArguments):
    del runner_args
    raise RuntimeError("V2 is started through goodbox_server.py; standalone demo mode was removed.")


if __name__ == "__main__":
    raise SystemExit("Run python3 goodbox_server.py to start the Goodbox V2 voice agent.")
