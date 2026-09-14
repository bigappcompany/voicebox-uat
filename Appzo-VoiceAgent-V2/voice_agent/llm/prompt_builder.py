from ..agents.bundle import AgentBundle
from ..runtime.response_plan import ResponsePlan


class PromptBuilder:
    def __init__(self, max_history_turns: int = 2, max_knowledge_chars: int = 2400) -> None:
        self.max_history_turns = max_history_turns; self.max_knowledge_chars = max_knowledge_chars

    def build(self, *, agent: AgentBundle, state: dict, slots: dict, route: ResponsePlan, knowledge: list, history: list, user_text: str) -> list[dict[str, str]]:
        action_names = ", ".join(agent.actions.keys()) or "continue"
        system = "\n".join((agent.invariant_prompt, f"RISK: {route.risk_class}", f"STATE: {state.get('name', 'OPEN')}", f"ALLOWED ACTIONS: {action_names}", f"SLOTS: {slots}")).strip()
        docs = "\n".join(getattr(doc, "text", str(doc)) for doc in knowledge)[:self.max_knowledge_chars]
        messages = [{"role": "system", "content": system}]
        if docs: messages.append({"role": "system", "content": "RELEVANT KNOWLEDGE:\n" + docs})
        messages.extend(history[-self.max_history_turns * 2:]); messages.append({"role": "user", "content": user_text})
        return messages
