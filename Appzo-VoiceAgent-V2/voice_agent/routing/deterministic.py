import re

from ..agents.bundle import AgentBundle
from ..runtime.response_plan import ResponsePlan


class DeterministicRouter:
    def route(self, text: str, agent: AgentBundle) -> tuple[ResponsePlan, str] | None:
        normalized = re.sub(r"[^\w\s]", "", text.lower()).strip()
        if normalized in {"bye", "goodbye", "end call", "end the call", "please end the call", "hang up", "stop calling", "अलविदा"}:
            return ResponsePlan("fixed", "end_call", cache_key="goodbye", allow_speculative_audio=True), agent.cached_utterances.get("goodbye", "Thank you for calling. Goodbye.")
        if normalized in {"hello", "hi", "hey", "नमस्ते"}:
            return ResponsePlan("fixed", cache_key="greeting", allow_speculative_audio=True), agent.cached_utterances.get("greeting", "Hello, how can I help?")
        for key, text_response in agent.cached_utterances.items():
            if key.startswith("faq:") and re.sub(r"[^\w\s]", "", key[4:].lower()).strip() == normalized:
                return ResponsePlan("cache", cache_key=key, allow_speculative_audio=True), text_response
        return None
