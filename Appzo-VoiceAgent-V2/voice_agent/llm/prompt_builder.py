import json
import os

from loguru import logger

from ..agents.bundle import AgentBundle
from ..runtime.response_plan import ResponsePlan


class PromptBuilder:
    """Build the bounded runtime request from the immutable Goodbox bundle."""

    def __init__(self, max_history_turns: int = 1, max_knowledge_chars: int = 1200, *, compiled: bool = True) -> None:
        self.max_history_turns = max_history_turns
        self.max_knowledge_chars = max_knowledge_chars
        self.max_invariant_chars = int(os.getenv("V2_MAX_INVARIANT_CHARS", "6000"))
        self.compiled = compiled
        self.section_token_estimates: dict[str, int] = {}

    def build(self, *, agent: AgentBundle, state: dict, slots: dict, route: ResponsePlan, knowledge: list, history: list, user_text: str) -> list[dict[str, str]]:
        state_name = str(state.get("name", "OPEN"))
        state_config = (agent.flow_graph.get("states") or {}).get(state_name, {})
        prompt_states = (agent.compiled_prompt.get("states") or {}) if self.compiled else {}
        prompt_state = prompt_states.get(state_name, {}) if isinstance(prompt_states, dict) else {}
        invariant = str(
            (agent.compiled_prompt.get("invariant") if self.compiled else "")
            or agent.invariant_prompt
        ).strip()
        invariant = self._bounded_invariant(invariant)
        state_actions = state_config.get("allowed_actions") or state_config.get("actions") or agent.actions.keys()
        if isinstance(state_actions, dict):
            state_actions = state_actions.keys()
        action_names = ", ".join(str(action) for action in state_actions) or "continue"
        objective = str(prompt_state.get("objective") or state_config.get("objective") or "Answer the caller's immediate request.")
        known = json.dumps(slots, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        system = "\n".join((
            invariant,
            "RUNTIME CONTEXT (do not mention this context):",
            f"RISK: {route.risk_class}",
            f"STATE: {state_name}",
            f"OBJECTIVE: {objective}",
            f"ALLOWED ACTIONS: {action_names}",
            f"KNOWN FACTS: {known}",
            "Do not ask for a fact that is already present in KNOWN FACTS unless the caller is correcting it.",
        )).strip()
        docs = "\n".join(getattr(doc, "text", str(doc)) for doc in knowledge)[:self.max_knowledge_chars]
        history_slice = history[-self.max_history_turns * 2:]
        self.section_token_estimates = {
            "invariant": self._estimate(invariant),
            "state": self._estimate(system) - self._estimate(invariant),
            "history": sum(self._estimate(str(item.get("content", ""))) for item in history_slice),
            "knowledge": self._estimate(docs),
        }
        self.section_token_estimates["total"] = sum(self.section_token_estimates.values()) + self._estimate(user_text)
        messages = [{"role": "system", "content": system}]
        if docs: messages.append({"role": "system", "content": "RELEVANT KNOWLEDGE:\n" + docs})
        messages.extend(history_slice); messages.append({"role": "user", "content": user_text})
        return messages

    def _bounded_invariant(self, text: str) -> str:
        """Bound hot-path prompt prefill while preserving both policy edges.

        Authoring systems commonly place identity at the beginning and output
        contracts at the end, so a head/tail budget is safer than blind tail
        truncation. A dedicated ``runtime_prompt`` remains preferable.
        """
        if self.max_invariant_chars <= 0 or len(text) <= self.max_invariant_chars:
            return text
        head = int(self.max_invariant_chars * 0.7)
        tail = self.max_invariant_chars - head
        logger.warning(
            "V2 invariant prompt bounded original_chars={} runtime_chars={}; supply runtime_prompt for exact control",
            len(text),
            self.max_invariant_chars,
        )
        return text[:head].rstrip() + "\n[AUTHORING DETAIL OMITTED FROM HOT PATH]\n" + text[-tail:].lstrip()

    @staticmethod
    def _estimate(text: str) -> int:
        # Provider-independent, deterministic telemetry. Provider-reported
        # usage remains authoritative when present in the stream.
        return (len(text) + 3) // 4
