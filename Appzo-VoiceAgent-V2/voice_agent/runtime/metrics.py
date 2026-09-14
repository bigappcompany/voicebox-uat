from dataclasses import asdict, dataclass
import time


@dataclass
class TurnMetrics:
    turn_id: int
    first_user_audio_at: float | None = None; last_voiced_at: float | None = None; vad_stop_at: float | None = None
    first_interim_at: float | None = None; stable_semantic_prefix_at: float | None = None; soft_eot_at: float | None = None
    final_stt_at: float | None = None; hard_eot_at: float | None = None; route_started_at: float | None = None; route_completed_at: float | None = None
    retrieval_started_at: float | None = None; retrieval_completed_at: float | None = None; llm_requested_at: float | None = None
    llm_first_token_at: float | None = None; first_safe_text_at: float | None = None; spec_tts_started_at: float | None = None
    spec_tts_first_audio_at: float | None = None; commit_at: float | None = None; first_output_frame_at: float | None = None; bot_started_at: float | None = None
    route: str = ""; model: str = ""; spec_llm: str = "none"; spec_tts: str = "none"; endpoint_mode: str = ""

    def mark(self, name: str) -> None:
        if getattr(self, name) is None: setattr(self, name, time.perf_counter())

    def as_record(self, **labels: object) -> dict[str, object]:
        record = asdict(self); record.update(labels); return record
