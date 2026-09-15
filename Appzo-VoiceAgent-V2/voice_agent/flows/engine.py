from dataclasses import dataclass

from ..runtime.response_plan import ResponsePlan


@dataclass(frozen=True)
class FlowResult:
    next_state: str
    slots_written: dict[str, str]
    action: str


class FlowEngine:
    def transition(self, flow: dict, current_state: str, intent: str, slots: dict[str, str]) -> FlowResult:
        state = (flow.get("states") or {}).get(current_state, {})
        transitions = state.get("transitions") or state.get("allowed_transitions") or {}
        target = transitions.get(intent, current_state)
        actions = state.get("actions") or {}
        updates = (state.get("slot_updates") or {}).get(intent, {})
        return FlowResult(str(target), dict(updates), str(actions.get(intent, "continue")))

    def plan(self, flow: dict, current_state: str, intent: str, slots: dict[str, str], *, risk_class: str = "LOW_PUBLIC") -> ResponsePlan:
        result = self.transition(flow, current_state, intent, slots)
        return ResponsePlan(route="flow", action=result.action, next_state=result.next_state, risk_class=risk_class,
                            slots_written=result.slots_written, allow_speculative_audio=risk_class.startswith("LOW_"))
