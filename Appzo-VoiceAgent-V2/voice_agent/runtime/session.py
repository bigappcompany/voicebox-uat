from dataclasses import dataclass, field
from typing import Any

from ..agents.bundle import AgentBundle


@dataclass(frozen=True)
class PendingQuestion:
    intent: str
    slot: str
    expected_type: str
    asked_turn: int = 0


@dataclass
class AssistantDeliveryState:
    response_id: str
    plan_intent: str | None
    generated_text: str = ""
    spoken_text: str = ""
    interrupted: bool = False
    question_id: str | None = None


@dataclass
class CallSession:
    call_id: str
    tenant_id: str
    agent: AgentBundle
    state: dict[str, object] = field(default_factory=dict)
    slots: dict[str, object] = field(default_factory=dict)
    facts: dict[str, Any] = field(default_factory=dict)
    pending_question: PendingQuestion | None = None
    assistant_delivery: AssistantDeliveryState | None = None
    history: list[dict[str, str]] = field(default_factory=list)
    # Built only at call setup. The V2 media path never fetches Goodbox or a
    # remote vector store after caller EOT.
    knowledge_index: Any | None = None
    turn_id: int = 0

    def __post_init__(self) -> None:
        if self.tenant_id != self.agent.tenant_id:
            raise ValueError("Call session tenant must match its AgentBundle")
        self.state.setdefault("name", self.agent.flow_graph.get("initial_state", "OPEN"))
        if self.pending_question is None:
            state = str(self.state.get("name", "OPEN"))
            states = self.agent.flow_graph.get("states") or {}
            state_config = states.get(state, {}) if isinstance(states, dict) else {}
            required = (state_config.get("required_slots") or []) if isinstance(state_config, dict) else []
            if required:
                slot = str(required[0])
                self.pending_question = PendingQuestion(
                    f"ask_{slot}", slot,
                    {"headcount": "integer_or_range", "hiring_timeline": "duration_or_range"}.get(slot, "string"),
                    0,
                )

    def visible_facts(self) -> dict[str, object]:
        """Return simple prompt/router values while retaining typed provenance."""
        result = dict(self.slots)
        for name, fact in self.facts.items():
            result[name] = getattr(fact, "value", fact)
        return result
