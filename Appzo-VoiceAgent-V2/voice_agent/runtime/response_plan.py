from dataclasses import dataclass, field


HIGH_RISK = {"HIGH_PERSONAL", "HIGH_TRANSACTIONAL", "HIGH_REGULATED"}


@dataclass(frozen=True)
class ResponsePlan:
    route: str
    action: str = "continue"
    next_state: str | None = None
    risk_class: str = "LOW_PUBLIC"
    slots_read: tuple[str, ...] = ()
    slots_written: dict[str, str] = field(default_factory=dict)
    knowledge_ids: tuple[str, ...] = ()
    tool_name: str | None = None
    cache_key: str | None = None
    allow_speculative_audio: bool = False

    def may_prepare_audio(self) -> bool:
        return self.allow_speculative_audio and self.tool_name is None and self.risk_class not in HIGH_RISK
