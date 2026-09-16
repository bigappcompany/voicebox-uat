import hashlib
import json
from dataclasses import dataclass

from .response_plan import ResponsePlan


@dataclass(frozen=True)
class ResponseFingerprint:
    tenant_id: str; agent_version: str; state: str; intent: str; risk_class: str
    knowledge_version: str; knowledge_ids: tuple[str, ...]; material_slots: tuple[tuple[str, str], ...]; tool_dependency: str | None

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.__dict__, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @classmethod
    def from_plan(cls, *, tenant_id: str, agent_version: str, state: str, intent: str, knowledge_version: str, plan: ResponsePlan, slots: dict[str, object]) -> "ResponseFingerprint":
        # Prompt-visible facts can change the answer even when a flow omitted
        # an explicit slots_read declaration. Default to all current slots;
        # a compiled plan may narrow this with material_slots.
        material = set(plan.material_slots or plan.slots_read or slots.keys()) | set(plan.slots_written)
        return cls(
            tenant_id,
            agent_version,
            state,
            intent,
            plan.risk_class,
            knowledge_version,
            tuple(sorted(plan.knowledge_ids)),
            tuple(sorted((name, str(slots.get(name, plan.slots_written.get(name, "")))) for name in material)),
            plan.tool_name,
        )
