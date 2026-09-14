import hashlib
import json
from dataclasses import dataclass

from .response_plan import ResponsePlan


@dataclass(frozen=True)
class ResponseFingerprint:
    tenant_id: str; agent_version: str; state: str; intent: str; risk_class: str
    knowledge_version: str; material_slots: tuple[tuple[str, str], ...]; tool_dependency: str | None

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.__dict__, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @classmethod
    def from_plan(cls, *, tenant_id: str, agent_version: str, state: str, intent: str, knowledge_version: str, plan: ResponsePlan, slots: dict[str, object]) -> "ResponseFingerprint":
        material = set(plan.slots_read) | set(plan.slots_written)
        return cls(tenant_id, agent_version, state, intent, plan.risk_class, knowledge_version,
                   tuple(sorted((name, str(slots.get(name, plan.slots_written.get(name, "")))) for name in material)), plan.tool_name)
