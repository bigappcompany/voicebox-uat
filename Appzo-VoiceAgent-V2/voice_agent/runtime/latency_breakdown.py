"""Additive, layer-owned user-to-bot latency timelines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class LatencyContribution:
    """One non-overlapping interval in a latency timeline."""

    key: str
    label: str
    owner_kind: str
    owner: str
    start_at: float
    end_at: float

    @property
    def duration_secs(self) -> float:
        return max(0.0, self.end_at - self.start_at)

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["duration_secs"] = self.duration_secs
        return value

    def line(self) -> str:
        return f"{self.duration_secs:7.3f}s  {self.label:<28} [{self.owner_kind}: {self.owner}]"


@dataclass(frozen=True)
class LatencyBreakdown:
    """A user-speech-stop (or hard-EOT) to first-audible timeline.

    Contributions are deliberately sequential and non-overlapping. Their
    durations therefore sum to ``total_secs`` instead of presenting several
    useful but overlapping provider metrics as if they were additive.
    """

    turn_id: int
    measured_from: str
    started_at: float
    ended_at: float
    contributions: tuple[LatencyContribution, ...]

    @property
    def total_secs(self) -> float:
        return max(0.0, self.ended_at - self.started_at)

    @classmethod
    def from_turn(
        cls,
        metrics: Any,
        *,
        model: str,
        tts_transport: str,
        stt_owner: str = "Deepgram Flux",
        require_audible_pcm: bool = False,
    ) -> "LatencyBreakdown | None":
        hard_eot = metrics.turn_committed_at
        first_audible = metrics.output_first_non_silent_at
        if first_audible is None and not require_audible_pcm:
            first_audible = metrics.tts_first_non_silent_at or metrics.bot_started_at
        if hard_eot is None or first_audible is None or first_audible < hard_eot:
            return None

        last_voiced = metrics.last_voiced_at
        if last_voiced is not None and last_voiced <= hard_eot:
            started_at = last_voiced
            measured_from = "last_voiced_audio"
        else:
            started_at = hard_eot
            measured_from = "hard_eot"

        contributions: list[LatencyContribution] = []
        cursor = started_at

        def add_until(
            end_at: float | None,
            key: str,
            label: str,
            owner_kind: str,
            owner: str,
        ) -> None:
            nonlocal cursor
            if end_at is None or end_at <= cursor or end_at > first_audible:
                return
            contributions.append(
                LatencyContribution(key, label, owner_kind, owner, cursor, end_at)
            )
            cursor = end_at

        if measured_from == "last_voiced_audio":
            profile = metrics.endpoint_profile or "configured profile"
            add_until(
                hard_eot,
                "endpointing.final_transcript",
                "endpointing + final transcript",
                "setting",
                f"{stt_owner}/{profile}",
            )

        add_until(
            metrics.aggregator_stop_at,
            "turn.provider_to_aggregator",
            "turn marker delivery",
            "pipeline",
            "user-turn aggregator",
        )
        add_until(
            metrics.commit_at,
            "turn.commit",
            "turn completion",
            "bot",
            "V2 turn controller",
        )

        route = str(metrics.route or "unknown")
        hosted = "hosted" in route or "speculation" in route or route in {"llm", "v2-llm"}
        response_owner_kind = "service" if hosted else "bot"
        response_owner = model if hosted else route
        safe_text = metrics.first_safe_text_at
        if hosted:
            add_until(
                metrics.llm_first_token_at,
                "llm.first_token",
                "LLM request to first token",
                "service",
                model,
            )
            add_until(
                getattr(metrics, "first_speech_filter_text_at", None),
                "response.speech_filter",
                "first token to speech filter",
                "pipeline",
                "speech stream filter",
            )
            add_until(
                metrics.first_filtered_text_at,
                "response.booking_guard",
                "booking-claim guard",
                "pipeline",
                "booking guard",
            )
        if safe_text is not None and safe_text > cursor:
            add_until(
                safe_text,
                "response.first_safe_text",
                "safe speech chunk readiness" if hosted else "response readiness",
                "pipeline" if hosted else response_owner_kind,
                "safe speech chunker" if hosted else response_owner,
            )

        release_at = getattr(metrics, "response_release_at", None)
        if release_at is not None and metrics.tts_requested_at is not None and release_at < metrics.tts_requested_at:
            add_until(
                release_at,
                "response.release",
                "response routing/release",
                response_owner_kind,
                response_owner,
            )
            add_until(
                metrics.tts_requested_at,
                "response.release_to_tts",
                "TTS dispatch",
                "pipeline",
                "speech chunk dispatch",
            )
        else:
            add_until(
                release_at or metrics.tts_requested_at,
                "response.release_to_tts",
                "response routing/release",
                response_owner_kind,
                response_owner,
            )
        add_until(
            metrics.tts_first_audio_at,
            "tts.first_audio",
            "speech synthesis",
            "service",
            f"Cartesia/{tts_transport}",
        )
        add_until(
            metrics.output_first_packet_at,
            "output.first_packet",
            "audio output queue",
            "pipeline",
            "Pipecat output transport",
        )
        add_until(
            first_audible,
            "output.first_audible",
            "audible audio release",
            "pipeline",
            "Plivo media stream",
        )
        if cursor < first_audible:
            contributions.append(
                LatencyContribution(
                    "pipeline.unattributed",
                    "unreported pipeline time",
                    "pipeline",
                    "timestamp gap",
                    cursor,
                    first_audible,
                )
            )

        return cls(
            turn_id=metrics.turn_id,
            measured_from=measured_from,
            started_at=started_at,
            ended_at=first_audible,
            contributions=tuple(contributions),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "turn_id": self.turn_id,
            "measured_from": self.measured_from,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "total_secs": self.total_secs,
            "contributions": [item.as_dict() for item in self.contributions],
        }

    def turn_contribution_lines(self, min_contribution_secs: float = 0.0) -> list[str]:
        visible: list[LatencyContribution] = []
        rolled_up = 0.0
        for item in self.contributions:
            if item.duration_secs < max(0.0, min_contribution_secs):
                rolled_up += item.duration_secs
            else:
                visible.append(item)
        lines = [item.line() for item in visible]
        if rolled_up:
            lines.append(
                f"{rolled_up:7.3f}s  {'brief pipeline intervals':<28} [pipeline: rolled up]"
            )
        lines.append(
            f"{self.total_secs:7.3f}s  {'TOTAL':<28} [measured_from: {self.measured_from}]"
        )
        return lines
