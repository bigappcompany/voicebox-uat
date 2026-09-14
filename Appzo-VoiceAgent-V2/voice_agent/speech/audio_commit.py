from dataclasses import dataclass, field

from ..runtime.response_plan import ResponsePlan


@dataclass
class SpeculativeAudioCandidate:
    fingerprint: str
    transcript_basis: str
    text: str
    pcm_chunks: list[bytes] = field(default_factory=list)
    sample_rate: int = 24000
    invalidated: bool = False
    committed: bool = False

    def append(self, pcm: bytes, max_audio_ms: int = 600) -> bool:
        if self.invalidated or self.committed: return False
        max_bytes = self.sample_rate * 2 * max_audio_ms // 1000
        if sum(map(len, self.pcm_chunks)) + len(pcm) > max_bytes: return False
        self.pcm_chunks.append(pcm); return True


class AudioCommitter:
    def commit(self, candidate: SpeculativeAudioCandidate | None, *, final_fingerprint: str, plan: ResponsePlan) -> list[bytes] | None:
        if candidate is None or candidate.invalidated or not plan.may_prepare_audio() or candidate.fingerprint != final_fingerprint:
            if candidate: candidate.invalidated = True
            return None
        candidate.committed = True
        return list(candidate.pcm_chunks)

    def abort(self, candidate: SpeculativeAudioCandidate | None) -> None:
        if candidate: candidate.invalidated = True; candidate.pcm_chunks.clear()
