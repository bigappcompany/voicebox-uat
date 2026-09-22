"""Read-only bridge between Pipecat dialogue and V2 canonical state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return " ".join(parts).strip()
    return ""


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


@dataclass(frozen=True)
class HostedContext:
    workflow_state: str
    canonical_facts: dict[str, object]
    active_question: dict[str, object] | None
    recent_dialogue: tuple[dict[str, str], ...]
    current_user_text: str
    dialogue_version: int
    estimated_tokens: int
    pipecat_read_ms: float = 0.0
    selection_ms: float = 0.0
    token_estimation_ms: float = 0.0


class ConversationContextAdapter:
    """Build bounded prompt context without mutating ``LLMContext``.

    Pipecat owns committed dialogue. V2 canonical state remains separate and
    authoritative. The current user message is removed from the history view
    because PromptBuilder appends it explicitly.
    """

    def __init__(self, pipecat_context: Any, session: Any):
        self.pipecat_context = pipecat_context
        self.session = session

    @staticmethod
    def _estimate(text: str) -> int:
        return (len(text) + 3) // 4

    def _dialogue(self) -> list[dict[str, str]]:
        getter = getattr(self.pipecat_context, "get_messages", None)
        raw = getter() if callable(getter) else getattr(self.pipecat_context, "messages", [])
        messages: list[dict[str, str]] = []
        for item in raw or []:
            if not isinstance(item, dict) or item.get("role") not in {"user", "assistant"}:
                continue
            content = _text_content(item.get("content"))
            if content:
                messages.append({"role": str(item["role"]), "content": content})
        return messages

    def build_hosted_context(
        self,
        *,
        current_user_text: str,
        max_turns: int = 3,
        max_tokens: int = 500,
    ) -> HostedContext:
        import time
        t0 = time.perf_counter()
        dialogue = self._dialogue()
        t1 = time.perf_counter()
        dialogue_version = len(dialogue)

        # The user aggregator has normally committed the hard-EOT message by
        # the time V2 builds the final request. Speculative requests do not
        # have it yet. Handle both paths without duplicating the current turn.
        if (
            dialogue
            and dialogue[-1]["role"] == "user"
            and _normalized(dialogue[-1]["content"]) == _normalized(current_user_text)
        ):
            dialogue = dialogue[:-1]

        t2 = time.perf_counter()
        selected_rev: list[dict[str, str]] = []
        tokens = 0
        token_estimation_time = 0.0
        for message in reversed(dialogue[-max(0, max_turns * 2):]):
            est_t0 = time.perf_counter()
            cost = self._estimate(message["content"])
            est_t1 = time.perf_counter()
            token_estimation_time += (est_t1 - est_t0)

            if selected_rev and tokens + cost > max_tokens:
                break
            if not selected_rev and cost > max_tokens:
                # Keep the newest semantic message, bounded to the configured
                # approximate budget, instead of returning no context.
                chars = max_tokens * 4
                message = {**message, "content": message["content"][-chars:]}
                est_t0 = time.perf_counter()
                cost = self._estimate(message["content"])
                token_estimation_time += (time.perf_counter() - est_t0)
            selected_rev.append(message)
            tokens += cost
        selected = tuple(reversed(selected_rev))
        t3 = time.perf_counter()

        pending = self.session.pending_question
        active_question = None
        if pending is not None:
            active_question = {
                "intent": pending.intent,
                "slot": pending.slot,
                "expected_type": pending.expected_type,
                "asked_turn": pending.asked_turn,
            }
        return HostedContext(
            workflow_state=str(self.session.state.get("name", "OPEN")),
            canonical_facts=dict(self.session.visible_facts()),
            active_question=active_question,
            recent_dialogue=selected,
            current_user_text=current_user_text,
            dialogue_version=dialogue_version,
            estimated_tokens=tokens,
            pipecat_read_ms=round((t1 - t0) * 1000, 3),
            selection_ms=round((t3 - t2) * 1000, 3),
            token_estimation_ms=round(token_estimation_time * 1000, 3),
        )
