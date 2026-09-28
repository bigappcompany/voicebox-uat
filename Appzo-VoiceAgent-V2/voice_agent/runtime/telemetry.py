"""Passive Cloud telemetry adapters and transcript-free session reports."""

import asyncio
import json
import math
from pathlib import Path
from uuid import uuid4

from loguru import logger
from pipecat.frames.frames import TTSSpeakFrame, VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import models as RTVI
from pipecat.processors.frameworks.rtvi.observer import RTVIObserver, RTVIObserverParams
from pipecat.processors.frameworks.rtvi.processor import RTVIProcessor


class CloudTelemetryObserver(RTVIObserver):
    """Forward response lifecycle without changing speech or conversation text."""

    async def _handle_llm_text_frame(self, frame):
        # Completed answers are already published explicitly by the controller.
        # Keep lifecycle events enabled without publishing partial tokens twice.
        return

    async def on_push_frame(self, data):
        frame = data.frame
        fixed_response = (
            isinstance(frame, TTSSpeakFrame)
            and data.direction == FrameDirection.DOWNSTREAM
            and frame.id not in self._frames_seen
            and data.source not in self._ignored_sources
            and self._params.bot_llm_enabled
        )
        await super().on_push_frame(data)
        if fixed_response:
            # A fixed response is already fully generated. These are protocol
            # notifications only: no LLM timing sample and no pipeline frames.
            # Actual playback boundaries still come from the output transport.
            await self.send_rtvi_message(RTVI.BotLLMStartedMessage())
            await self.send_rtvi_message(RTVI.BotLLMStoppedMessage())


class CloudTelemetryProcessor(RTVIProcessor):
    def create_rtvi_observer(self, *, params=None, **kwargs):
        return CloudTelemetryObserver(self, params=params, **kwargs)


class TelemetrySpeechStartedFrame(VADUserStartedSpeakingFrame):
    """Flux-derived speech boundary for observers only, not turn strategies."""


class TelemetrySpeechStoppedFrame(VADUserStoppedSpeakingFrame):
    """Estimated PCM speech end, expressed in Pipecat's native timing format."""


class TelemetryBoundaryFilter(FrameProcessor):
    """Observers see telemetry boundaries; conversation processors never do.

    Must sit immediately after OrderedFluxSTTService. Only our tagged frames
    are consumed. Actual VAD, transcription, interruption, audio and lifecycle
    frames retain their identity, order and direction.
    """

    def __init__(self):
        # No additional audio/transcript queue or batching stage.
        super().__init__(enable_direct_mode=True)

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, (TelemetrySpeechStartedFrame, TelemetrySpeechStoppedFrame)):
            return
        await self.push_frame(frame, direction)


def cloud_rtvi_params():
    # Preserve explicit completed-answer publishing and suppress partial text.
    # Metrics and speech boundaries are independent of those text switches.
    # bot_llm_enabled controls response lifecycle, not MetricsFrame forwarding.
    # CloudTelemetryObserver suppresses partial LLM text independently.
    return RTVIObserverParams(
        bot_output_enabled=False,
        bot_llm_enabled=True,
        bot_tts_enabled=False,
        metrics_enabled=True,
        user_speaking_enabled=True,
        vad_user_speaking_enabled=True,
        bot_speaking_enabled=True,
    )


def percentiles(values):
    ordered = sorted(v for v in values if v is not None and math.isfinite(v) and v >= 0)
    if not ordered:
        return {"n": 0}
    return {
        "n": len(ordered),
        **{f"p{p}": ordered[math.ceil(len(ordered) * p / 100) - 1] for p in (50, 90, 95, 99)},
        "max": ordered[-1],
    }


class SessionTelemetryReport:
    """One record per audible response; disk I/O only after the call ends.

    Native Cloud metrics remain the dashboard path. This artifact preserves
    exact custom latency definitions without scraping logs or storing speech.
    Local Cloud container files are ephemeral unless exported/mounted.
    """

    FIELDS = (
        "raw_speech_end_to_first_audible_ms", "raw_speech_end_to_hard_eot_ms",
        "hard_eot_to_first_audible_ms", "hosted_llm_ttft_ms", "tts_first_pcm_ms",
        "tts_first_audible_ms",
    )

    def __init__(self, *, session_id=None, call_id=None, directory="logs/telemetry"):
        self.session_id = session_id
        self.call_id = call_id
        self.directory = directory
        self._records = {}
        # Never use externally supplied IDs as filesystem paths.
        self._filename = f"{uuid4().hex}.json"

    def add(self, record):
        if record.get("first_audible_source") != "output_non_silent_pcm":
            return
        self._records.setdefault(record["turn_id"], {
            key: record.get(key) for key in (
                "turn_id", "route", "hosted_llm_used", "first_audible_source", *self.FIELDS
            )
        })

    def payload(self):
        turns = list(self._records.values())
        return {
            "schema_version": 1, "session_id": self.session_id, "call_id": self.call_id,
            "unit": "ms", "percentile_method": "nearest_rank",
            "measurement": "server PCM timestamps; not handset playback",
            "summary": {key: percentiles(t[key] for t in turns) for key in self.FIELDS},
            "by_route": {
                route: percentiles(t["raw_speech_end_to_first_audible_ms"] for t in turns if t["route"] == route)
                for route in sorted({t["route"] for t in turns})
            },
            "turns": turns,
        }

    async def finish(self):
        payload = self.payload()
        logger.info("TELEMETRY SESSION SUMMARY | {}", json.dumps({k: v for k, v in payload.items() if k != "turns"}))
        if not self.directory:
            return
        try:
            path = await asyncio.to_thread(self._write, payload)
            logger.info("TELEMETRY REPORT | session_id={} call_id={} path={}", self.session_id, self.call_id, path)
        except Exception as exc:
            logger.warning("TELEMETRY REPORT WRITE FAILED | {}", type(exc).__name__)

    def _write(self, payload):
        directory = Path(self.directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / self._filename
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
        return str(path)
