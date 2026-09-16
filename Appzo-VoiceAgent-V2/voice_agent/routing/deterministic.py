import re

from ..agents.bundle import AgentBundle
from ..runtime.response_plan import ResponsePlan


class DeterministicRouter:
    def __init__(self, *, extended: bool = True) -> None:
        self.extended = extended

    def route(self, text: str, agent: AgentBundle) -> tuple[ResponsePlan, str] | None:
        normalized = re.sub(r"[^\w\s]", "", text.lower()).strip()
        if normalized in {"bye", "goodbye", "end call", "end the call", "please end the call", "hang up", "stop calling", "अलविदा"}:
            return ResponsePlan("fixed", "end_call", intent_id="goodbye", cache_key="goodbye", allow_speculative_audio=True), agent.cached_utterances.get("goodbye", "Thank you for calling. Goodbye.")
        if normalized in {"hello", "hi", "hey", "नमस्ते"}:
            return ResponsePlan("fixed", intent_id="greeting", cache_key="greeting", allow_speculative_audio=True), agent.cached_utterances.get("greeting", "Hello, how can I help?")
        if not self.extended:
            for key, text_response in agent.cached_utterances.items():
                if key.startswith("faq:") and re.sub(r"[^\w\s]", "", key[4:].lower()).strip() == normalized:
                    return ResponsePlan("cache", intent_id=key, cache_key=key, allow_speculative_audio=True), text_response
            return None
        if normalized in {"please repeat", "repeat that", "say that again", "come again", "क्या कहा", "फिर से बोलिए"}:
            return ResponsePlan("fixed", intent_id="repeat", cache_key="repeat", allow_speculative_audio=True), agent.cached_utterances.get("repeat", "Certainly. Could you please tell me what you would like me to repeat?")
        if normalized in {"im busy", "i am busy", "busy right now", "call later", "अभी व्यस्त हूँ"}:
            return ResponsePlan("fixed", intent_id="busy", cache_key="busy", allow_speculative_audio=True), agent.cached_utterances.get("busy", "Understood. Would you like to share a preferred day and time for a follow-up?")
        if normalized in {"wrong person", "wrong number", "you have the wrong person", "गलत नंबर"}:
            return ResponsePlan("fixed", "end_call", intent_id="wrong-person", cache_key="wrong-person", allow_speculative_audio=True), agent.cached_utterances.get("wrong-person", "I apologize for the inconvenience. I will end the call now.")
        if normalized in {"not interested", "no thanks", "dont call me", "do not call me", "interested नहीं हूँ"}:
            return ResponsePlan("fixed", "end_call", intent_id="not-interested", cache_key="not-interested", allow_speculative_audio=True), agent.cached_utterances.get("not-interested", "Understood. Thank you for your time. Goodbye.")
        if normalized in {"human", "speak to a human", "connect me to a person", "talk to a person", "agent please"}:
            return ResponsePlan("fixed", intent_id="human-transfer", cache_key="human-transfer", allow_speculative_audio=True), agent.cached_utterances.get("human-transfer", "I can record your request for a human follow-up. What day and time would be convenient?")
        for key, text_response in agent.cached_utterances.items():
            if key.startswith("faq:") and re.sub(r"[^\w\s]", "", key[4:].lower()).strip() == normalized:
                return ResponsePlan("cache", intent_id=key, cache_key=key, allow_speculative_audio=True), text_response
        return None
