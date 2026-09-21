import json
import os

from typing import Any

from loguru import logger

from ..agents.bundle import AgentBundle
from ..runtime.response_plan import ResponsePlan


class PromptBuilder:
    """Build the bounded runtime request from the immutable Goodbox bundle."""

    def __init__(self, max_history_turns: int = 3, max_knowledge_chars: int = 1200, *, compiled: bool = True) -> None:
        self.max_history_turns = max_history_turns
        self.max_knowledge_chars = max_knowledge_chars
        self.max_invariant_chars = int(os.getenv("V2_MAX_INVARIANT_CHARS", "6000"))
        self.compiled = compiled
        self.section_token_estimates: dict[str, int] = {}

    @classmethod
    def _normalize_for_json(cls, val: Any) -> Any:
        if hasattr(val, "to_primitive") and callable(getattr(val, "to_primitive")):
            val = val.to_primitive()
        if isinstance(val, dict):
            return {str(k): cls._normalize_for_json(v) for k, v in sorted(val.items(), key=lambda item: str(item[0]))}
        if isinstance(val, (list, tuple)):
            return [cls._normalize_for_json(item) for item in val]
        if isinstance(val, (set, frozenset)):
            return [cls._normalize_for_json(item) for item in sorted(list(val), key=str)]
        if isinstance(val, (int, float, bool, str)) or val is None:
            return val
        return str(val)

    def _safe_json_dumps(self, obj: Any) -> str:
        normalized = self._normalize_for_json(obj)
        return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def _json_default(cls, obj: object) -> object:
        if hasattr(obj, "to_primitive") and callable(getattr(obj, "to_primitive")):
            return obj.to_primitive()
        if isinstance(obj, (set, frozenset)):
            return sorted(list(obj))
        return str(obj)

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
        actions_source = state_config.get("allowed_actions") or state_config.get("actions") or getattr(agent, "actions", {})
        if isinstance(actions_source, dict):
            state_actions = list(actions_source.keys())
        elif isinstance(actions_source, (list, tuple, set)):
            state_actions = list(actions_source)
        elif hasattr(actions_source, "keys") and callable(getattr(actions_source, "keys")):
            state_actions = list(actions_source.keys())
        else:
            state_actions = [actions_source] if actions_source else []
        action_names = ", ".join(str(action) for action in state_actions) or "continue"
        objective = str(prompt_state.get("objective") or state_config.get("objective") or "Answer the caller's immediate request.")
        known = self._safe_json_dumps(slots if isinstance(slots, dict) else {})
        system = "\n".join((
            invariant,
            "RUNTIME CONTEXT (do not mention this context):",
            f"RISK: {route.risk_class}",
            f"STATE: {state_name}",
            f"OBJECTIVE: {objective}",
            f"ALLOWED ACTIONS: {action_names}",
            f"KNOWN FACTS: {known}",
            "MEMORY CONTRACT: KNOWN FACTS are authoritative business truth. Use recent dialogue for tone, references, and ambiguity.",
            "Do not contradict a known fact unless the current caller explicitly corrects it.",
            "Do not ask for a fact that is already present in KNOWN FACTS unless the caller is correcting it.",
        )).strip()
        docs = "\n".join(getattr(doc, "text", str(doc)) for doc in knowledge)[:self.max_knowledge_chars]
        history_slice = list(history[-self.max_history_turns * 2:])
        if (
            history_slice
            and history_slice[-1].get("role") == "user"
            and " ".join(str(history_slice[-1].get("content", "")).casefold().split())
            == " ".join(user_text.casefold().split())
        ):
            history_slice.pop()
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
