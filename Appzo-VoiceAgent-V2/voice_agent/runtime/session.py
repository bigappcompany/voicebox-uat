from dataclasses import dataclass, field

from ..agents.bundle import AgentBundle


@dataclass
class CallSession:
    call_id: str
    tenant_id: str
    agent: AgentBundle
    state: dict[str, object] = field(default_factory=dict)
    slots: dict[str, object] = field(default_factory=dict)
    history: list[dict[str, str]] = field(default_factory=list)
    turn_id: int = 0

    def __post_init__(self) -> None:
        if self.tenant_id != self.agent.tenant_id:
            raise ValueError("Call session tenant must match its AgentBundle")
        self.state.setdefault("name", self.agent.flow_graph.get("initial_state", "OPEN"))
