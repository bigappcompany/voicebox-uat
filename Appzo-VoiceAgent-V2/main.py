import asyncio
import os
import re
import time
import math
import socket
from urllib.parse import urlencode
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from dotenv import load_dotenv
from loguru import logger
from openai import AsyncAzureOpenAI, AsyncOpenAI

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
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
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi.models import BotOutputMessage, BotOutputMessageData
from pipecat.processors.frameworks.rtvi.observer import RTVIObserverParams
from pipecat.processors.frameworks.rtvi.processor import RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.services.cartesia.tts import CartesiaTTSService, GenerationConfig
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.tts_service import TextAggregationMode
from pipecat.transports.base_transport import BaseTransport
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.workers.runner import WorkerRunner
from websockets.asyncio.client import connect as websocket_connect


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
    last_voiced_at: float | None = None
    turn_committed_at: float | None = None
    final_stt_at: float | None = None
    speculative_started_at: float | None = None
    llm_first_token_at: float | None = None
    first_safe_text_at: float | None = None
    spec_tts_started_at: float | None = None
    spec_tts_first_audio_at: float | None = None
    commit_at: float | None = None
    protocol_validated_at: float | None = None
    tts_requested_at: float | None = None
    tts_first_audio_at: float | None = None
    bot_started_at: float | None = None
    route: str = "pending"
    speculation: str = "none"
    spec_tts: str = "none"


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

    _VOICE_LEVEL = 450  # telemetry only; native VAD/Smart Turn owns turns.

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
        self._metrics_by_turn[self._turn_counter] = metrics
        self._state = TurnState(turn_id=self._turn_counter, metrics=metrics)

    def _mark_voice(self, audio: bytes) -> None:
        state = self._state
        if state is None or state.committed or len(audio) < 2:
            return
        samples = memoryview(audio).cast("h")
        if samples and sum(abs(sample) for sample in samples) / len(samples) >= self._VOICE_LEVEL:
            state.metrics.last_voiced_at = time.perf_counter()

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
        state.metrics.tts_requested_at = time.perf_counter()
        await self._publish_answer_to_conversation(text)
        await self.push_frame(TTSSpeakFrame(text, append_to_context=False))

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

    def __init__(self, controller: StreamingVoiceController, *, tts_transport: str, session=None) -> None:
        super().__init__()
        self._controller = controller
        self._tts_transport = tts_transport
        self._session = session
        self._seen: set[int] = set()
        self._samples: list[int] = []

    async def on_push_frame(self, data: FramePushed):
        if data.direction != FrameDirection.DOWNSTREAM or data.frame.id in self._seen:
            return
        self._seen.add(data.frame.id)
        state = self._controller._state
        if state is None:
            return
        metrics = state.metrics
        # No current response has requested speech yet. Audio here belongs to
        # the greeting or an interrupted older response, not this caller turn.
        if metrics.tts_requested_at is None:
            return
        now = time.perf_counter()
        if isinstance(data.frame, TTSAudioRawFrame) and metrics.tts_first_audio_at is None:
            metrics.tts_first_audio_at = now
        if isinstance(data.frame, BotStartedSpeakingFrame) and metrics.bot_started_at is None:
            metrics.bot_started_at = now
            native_eot_to_audio = self._ms(metrics.turn_committed_at, metrics.bot_started_at)
            raw_audio_to_eot = self._ordered_ms(metrics.last_voiced_at, metrics.turn_committed_at)
            raw_audio_to_bot = self._ordered_ms(metrics.last_voiced_at, metrics.bot_started_at)
            if native_eot_to_audio is not None and metrics.route not in {"v2-error", "stt-empty-retry"}:
                self._samples.append(native_eot_to_audio)
            logger.info(
                "RESPONSE LATENCY | "
                f"tenant={getattr(self._session, 'tenant_id', 'legacy')} bundle={getattr(getattr(self._session, 'agent', None), 'version', 'legacy')} "
                f"state={getattr(self._session, 'state', {}).get('name', 'UNKNOWN') if self._session else 'LEGACY'} "
                f"turn={metrics.turn_id} route={metrics.route} tts={self._tts_transport} speculation={metrics.speculation} spec_tts={metrics.spec_tts} | "
                f"native-EOT->bot-audio={native_eot_to_audio} ms | "
                f"raw-audio->native-EOT={raw_audio_to_eot} ms | "
                f"raw-audio->bot-audio={raw_audio_to_bot} ms | "
                f"LLM-TTFT={self._ms(metrics.speculative_started_at or metrics.turn_committed_at, metrics.llm_first_token_at)} ms | "
                f"EOT->first-safe-text={self._ms(metrics.turn_committed_at, metrics.first_safe_text_at)} ms | "
                f"EOT->TTS-audio={self._ms(metrics.turn_committed_at, metrics.tts_first_audio_at)} ms | "
                f"EOT->spec-PCM={self._ms(metrics.turn_committed_at, metrics.spec_tts_first_audio_at)} ms"
            )

    async def cleanup(self):
        if self._samples:
            ordered = sorted(self._samples)
            percentile = lambda p: ordered[max(0, math.ceil(len(ordered) * p) - 1)]
            logger.info(
                f"NATIVE-EOT->BOT-AUDIO SUMMARY | tts={self._tts_transport} n={len(ordered)} p50={percentile(.50)} ms "
                f"p90={percentile(.90)} ms p95={percentile(.95)} ms max={ordered[-1]} ms"
            )
        await super().cleanup()

    @staticmethod
    def _ms(start: float | None, end: float | None) -> int | None:
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
) -> None:
    """Run one Goodbox-configured agent over phone media."""
    if runtime_config is None:
        raise ValueError("V2 requires a Goodbox AgentRuntimeConfig at call start")
    runtime = runtime_config
    logger.info("TURN config model={} stt={} endpointing_ms={} vad_stop_secs={} strategy=SmartTurn",
                runtime.llm_model, runtime.stt_model, runtime.deepgram_endpointing_ms, runtime.vad_stop_secs)
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
    latency_observer = LiveLatencyObserver(
        controller,
        tts_transport=("http-batch" if controller_options and resolved_tts_transport == "http" else "websocket-stream"),
        session=v2_session if controller_options else None,
    )
    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    confidence=runtime.vad_confidence,
                    start_secs=runtime.vad_start_secs,
                    stop_secs=runtime.vad_stop_secs,
                    min_volume=runtime.vad_min_volume,
                )
            )
        ),
    )

    @user_aggregator.event_handler("on_user_turn_started")
    async def on_user_turn_started(_aggregator, strategy):
        # A transcription can arrive after the VAD turn has stopped. It must
        # not start a new controller turn and discard the final transcript.
        if controller_options and not isinstance(strategy, VADUserTurnStartStrategy):
            await controller.handle_native_turn_started(transcription_only=True)
        elif isinstance(strategy, VADUserTurnStartStrategy):
            await controller.handle_native_turn_started()

    @user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(_aggregator, _strategy, _message):
        await controller.handle_native_turn_stopped(_message.content)
    stt = DeepgramSTTService(
        api_key=runtime.deepgram_api_key,
        settings=DeepgramSTTService.Settings(
            model=runtime.stt_model,
            language=runtime.stt_language,
            interim_results=True,
            endpointing=runtime.deepgram_endpointing_ms,
            punctuate=True,
            smart_format=False,
            keyterm=(v2_session.agent.stt_profile.get("keyterms", []) if controller_options else []),
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
        text_aggregation_mode=(TextAggregationMode.TOKEN if controller_options else TextAggregationMode.SENTENCE),
        max_buffer_delay_ms=0 if controller_options else None,
        )
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            controller,
            user_aggregator,
            tts,
            assistant_aggregator,
            transport.output(),
        ]
    )
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
            await worker.queue_frame(
                TTSSpeakFrame(runtime.intro_message, append_to_context=False)
            )

    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    del runner_args
    raise RuntimeError("V2 is started through goodbox_server.py; standalone demo mode was removed.")


if __name__ == "__main__":
    raise SystemExit("Run python3 goodbox_server.py to start the Goodbox V2 voice agent.")
