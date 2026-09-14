import re

from ..agents.bundle import AgentBundle
from ..runtime.response_plan import ResponsePlan


class DeterministicRouter:
    def route(self, text: str, agent: AgentBundle) -> tuple[ResponsePlan, str] | None:
        normalized = text.lower().strip()
        if re.search(r"\b(bye|goodbye|end call|no thanks)\b", normalized):
            return ResponsePlan("fixed", "end_call", cache_key="goodbye", allow_speculative_audio=True), agent.cached_utterances.get("goodbye", "Thank you for calling. Goodbye.")
        if re.search(r"\b(hello|hi|hey)\b", normalized) and len(normalized.split()) <= 4:
            return ResponsePlan("fixed", cache_key="greeting", allow_speculative_audio=True), agent.cached_utterances.get("greeting", "Hello, how can I help?")
        for key, text_response in agent.cached_utterances.items():
            if key.startswith("faq:") and key[4:].lower() in normalized:
                return ResponsePlan("cache", cache_key=key, allow_speculative_audio=True), text_response
        return None
