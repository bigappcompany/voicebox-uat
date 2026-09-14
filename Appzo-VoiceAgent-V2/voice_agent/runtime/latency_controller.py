"""Transport-neutral soft/hard-EOT orchestration. Pipecat adapters call this."""
from dataclasses import dataclass
from typing import Callable

from ..agents.bundle import AgentBundle
from ..routing.deterministic import DeterministicRouter
from ..speech.audio_commit import AudioCommitter, SpeculativeAudioCandidate
from ..turns.transcript_stability import TranscriptStabilityAnalyzer
from .fingerprints import ResponseFingerprint
from .metrics import TurnMetrics
from .response_plan import ResponsePlan
from .session import CallSession


@dataclass
class TurnCandidate:
    turn_id: int; latest_interim: str = ""; final_transcript: str = ""; stable_prefix: str = ""
    soft_eot: bool = False; hard_eot: bool = False; response_plan: ResponsePlan | None = None
    response_fingerprint: str | None = None; audio_candidate: SpeculativeAudioCandidate | None = None; generation: int = 0


class LatencyController:
    def __init__(self, session: CallSession, intent: Callable[[str], str] | None = None) -> None:
        self.session, self.intent, self.router = session, intent or (lambda _: "general"), DeterministicRouter()
        self.stability, self.committer = TranscriptStabilityAnalyzer(), AudioCommitter(); self.turn: TurnCandidate | None = None; self.metrics: dict[int, TurnMetrics] = {}

    def start_turn(self) -> TurnCandidate:
        self.abort_turn(); self.session.turn_id += 1
        self.turn = TurnCandidate(self.session.turn_id); self.metrics[self.session.turn_id] = TurnMetrics(self.session.turn_id)
        return self.turn

    def interim(self, text: str) -> TurnCandidate:
        turn = self.turn or self.start_turn(); turn.latest_interim = text
        hypothesis = self.stability.update(text); turn.stable_prefix = hypothesis.stable_prefix
        metric = self.metrics[turn.turn_id]; metric.mark("first_interim_at")
        if hypothesis.stable_prefix: metric.mark("stable_semantic_prefix_at")
        return turn

    def soft_eot(self, bundle: AgentBundle) -> ResponsePlan | None:
        turn = self.turn
        if not turn: return None
        turn.soft_eot = True; self.metrics[turn.turn_id].mark("soft_eot_at")
        routed = self.router.route(turn.latest_interim, bundle)
        if routed: plan, _ = routed
        else: plan = ResponsePlan("llm", risk_class=str(bundle.risk_policy.get("class", "LOW_PUBLIC")),
                                  slots_read=tuple(sorted(self.session.slots)), allow_speculative_audio=True)
        turn.response_plan = plan
        fp = self._fingerprint(bundle, turn.latest_interim, plan); turn.response_fingerprint = fp.digest()
        return plan

    def prepare_audio(self, text: str, pcm_chunks: list[bytes], sample_rate: int = 24000) -> SpeculativeAudioCandidate | None:
        turn = self.turn
        if not turn or not turn.response_plan or not turn.response_plan.may_prepare_audio(): return None
        candidate = SpeculativeAudioCandidate(turn.response_fingerprint or "", turn.latest_interim, text, list(pcm_chunks), sample_rate)
        turn.audio_candidate = candidate; self.metrics[turn.turn_id].mark("spec_tts_started_at"); return candidate

    def hard_eot(self, final_text: str, bundle: AgentBundle) -> list[bytes] | None:
        turn = self.turn or self.start_turn(); turn.hard_eot = True; turn.final_transcript = final_text; metric = self.metrics[turn.turn_id]
        metric.mark("final_stt_at"); metric.mark("hard_eot_at")
        routed = self.router.route(final_text, bundle)
        plan = routed[0] if routed else ResponsePlan("llm", risk_class=str(bundle.risk_policy.get("class", "LOW_PUBLIC")),
                                                     slots_read=tuple(sorted(self.session.slots)), allow_speculative_audio=True)
        final_fp = self._fingerprint(bundle, final_text, plan).digest(); audio = self.committer.commit(turn.audio_candidate, final_fingerprint=final_fp, plan=plan)
        metric.route, metric.spec_tts = plan.route, "hit" if audio is not None else ("miss" if turn.audio_candidate else "none")
        metric.mark("commit_at"); turn.response_plan = plan
        return audio

    def abort_turn(self) -> None:
        if self.turn: self.committer.abort(self.turn.audio_candidate); self.turn.generation += 1

    def _fingerprint(self, bundle: AgentBundle, text: str, plan: ResponsePlan) -> ResponseFingerprint:
        return ResponseFingerprint.from_plan(tenant_id=bundle.tenant_id, agent_version=bundle.version, state=str(self.session.state.get("name", "OPEN")), intent=self.intent(text), knowledge_version=bundle.knowledge_version, plan=plan, slots=self.session.slots)
