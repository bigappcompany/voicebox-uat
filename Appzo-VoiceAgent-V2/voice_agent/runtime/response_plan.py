from dataclasses import dataclass, field
from typing import Any

from .session import PendingQuestion


HIGH_RISK = {"HIGH_PERSONAL", "HIGH_TRANSACTIONAL", "HIGH_REGULATED"}


@dataclass(frozen=True)
class ResponsePlan:
    route: str
    action: str = "continue"
    intent_id: str | None = None
    next_state: str | None = None
    risk_class: str = "LOW_PUBLIC"
    slots_read: tuple[str, ...] = ()
    slots_written: dict[str, Any] = field(default_factory=dict)
    knowledge_ids: tuple[str, ...] = ()
    tool_name: str | None = None
    cache_key: str | None = None
    allow_speculative_audio: bool = False
    requires_booking_guard: bool = False
    booking_authority: str = "none"
    material_slots: tuple[str, ...] = ()
    pending_question: PendingQuestion | None = None
    clear_pending_question: bool = False
    decision_reason: str = ""
    decision_confidence: float = 0.0

    def may_prepare_audio(self) -> bool:
        return self.allow_speculative_audio and self.tool_name is None and self.risk_class not in HIGH_RISK
