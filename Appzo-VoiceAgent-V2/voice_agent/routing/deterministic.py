"""State-aware deterministic routing for common voice-agent turns."""

from __future__ import annotations

import re
from typing import Any

from ..agents.bundle import AgentBundle
from ..runtime.intents import CanonicalIntentModel, IntentMatch, normalize_intent_text
from ..runtime.response_plan import ResponsePlan
from ..runtime.session import PendingQuestion


class DeterministicRouter:
    def __init__(self, *, extended: bool = True) -> None:
        self.extended = extended
        self.intents = CanonicalIntentModel()

    def route(
        self,
        text: str,
        agent: AgentBundle,
        *,
        intent: IntentMatch | None = None,
        slots: dict[str, Any] | None = None,
        pending_question: PendingQuestion | None = None,
        turn_id: int = 0,
    ) -> tuple[ResponsePlan, str] | None:
        slots = slots or {}
        match = intent or self.intents.classify(
            text,
            pending_question=pending_question,
            fact_values=slots,
            configured_patterns=agent.routing_policy.get("intent_patterns") or {},
        )
        intent_id = match.intent_id

        if intent_id == "goodbye":
            return self._cached(agent, "goodbye", "Thank you for your time. Goodbye.", intent_id, action="end_call", match=match)
        if intent_id == "greeting":
            return self._cached(agent, "greeting", "Hello. How may I help?", intent_id, match=match)

        exact_faq = self._exact_faq(text, agent)
        if exact_faq is not None:
            return exact_faq
        if not self.extended:
            return None

        fixed = {
            "repeat": ("repeat", "Could you tell me which part you would like repeated?", "continue"),
            "wrong_person": ("wrong-person", "I apologize for the inconvenience. Goodbye.", "end_call"),
            "not_interested": ("not-interested", "Understood. Thank you for your time. Goodbye.", "end_call"),
            "model_identity": ("model-identity", "I'm an AI voice assistant for this hiring call.", "continue"),
        }
        if intent_id in fixed:
            key, fallback, action = fixed[intent_id]
            return self._cached(agent, key, fallback, intent_id, action=action, match=match)

        if intent_id == "out_of_scope":
            oos_count = int(slots.get("oos_count", 0)) + 1
            if oos_count >= 2:
                speech = agent.cached_utterances.get(
                    "out-of-scope:repeat",
                    "As mentioned, I can only assist with hiring and staffing inquiries. If you don't have hiring needs right now, I can follow up later or end the call.",
                )
            else:
                speech = agent.cached_utterances.get(
                    "out-of-scope", "I can only help with hiring and staffing on this call."
                )
            return self._plan("fixed", intent_id, speech, writes={"oos_count": oos_count}, match=match)

        if intent_id == "company_identity":
            company = str(agent.identity.get("company_name") or "the hiring company")
            speech = agent.cached_utterances.get(
                "identity", f"I'm calling from {company} about your hiring plans."
            )
            return self._plan("identity", intent_id, speech, match=match)

        if intent_id == "incomplete_response" and pending_question is not None:
            key = f"ask:{pending_question.slot}"
            fallback = self._question_for(pending_question.slot)
            return self._plan(
                "fixed", intent_id, agent.cached_utterances.get(key, fallback),
                next_state=self._state_for_slot(pending_question.slot),
                pending=pending_question, match=match,
            )

        if str(slots.get("callback_state") or "IDLE") not in (None, "", "IDLE") or (pending_question and pending_question.slot.startswith("callback_")):
            return None

        if intent_id in {"provide_hiring_status", "provide_role", "provide_headcount", "provide_timeline", "correction"}:
            return self._next_requirement(agent, slots, match, turn_id)

        return None

    def _next_requirement(
        self, agent: AgentBundle, slots: dict[str, Any], match: IntentMatch, turn_id: int,
    ) -> tuple[ResponsePlan, str]:
        if slots.get("hiring_timeline") not in (None, "") and slots.get("hiring_status") in (None, ""):
            slots["hiring_status"] = "future"
        elif (slots.get("roles") or slots.get("headcount")) and slots.get("hiring_status") in (None, ""):
            slots["hiring_status"] = "yes"

        missing = next(
            (name for name in ("hiring_status", "roles", "headcount", "hiring_timeline") if slots.get(name) in (None, "", [])),
            None,
        )
        if missing is None:
            speech = agent.cached_utterances.get(
                "requirements:complete",
                "Got it. Would you like a hiring specialist to follow up?",
            )
            return self._plan(
                "fixed", match.intent_id, speech, next_state="FOLLOWUP",
                pending=PendingQuestion("ask_callback_consent", "followup_consent", "boolean", turn_id),
                writes={"callback_state": "FOLLOWUP_OFFERED"}, match=match,
            )
        speech = agent.cached_utterances.get(f"ask:{missing}", self._question_for(missing))
        return self._plan(
            "fixed", match.intent_id, speech, next_state=self._state_for_slot(missing),
            pending=PendingQuestion(f"ask_{missing}", missing, self._type_for_slot(missing), turn_id),
            match=match,
        )

    @staticmethod
    def _question_for(slot: str) -> str:
        return {
            "hiring_status": "Are you hiring now or in the next few months?",
            "roles": "What roles are you planning to hire for?",
            "departments": "Which departments are hiring?",
            "headcount": "Roughly how many people would you need?",
            "hiring_timeline": "What hiring timeline are you targeting?",
            "callback_day": "What day would work best?",
            "callback_time": "What time would be convenient?",
            "callback_preference": "What day and time would be convenient?",
        }.get(slot, "Could you share that detail?")

    @staticmethod
    def _type_for_slot(slot: str) -> str:
        return {"headcount": "integer_or_range", "hiring_timeline": "duration_or_range", "roles": "role_list"}.get(slot, "string")

    @staticmethod
    def _state_for_slot(slot: str) -> str:
        if slot.startswith("callback_"):
            return "CALLBACK"
        return "HIRING_STATUS" if slot == "hiring_status" else "REQUIREMENTS"

    @staticmethod
    def _plan(
        route: str, intent_id: str, speech: str, *, next_state: str | None = None,
        pending: PendingQuestion | None = None, writes: dict[str, Any] | None = None,
        action: str = "continue", match: IntentMatch | None = None,
    ) -> tuple[ResponsePlan, str]:
        return (
            ResponsePlan(
                route, action=action, intent_id=intent_id, next_state=next_state,
                slots_written=writes or {}, pending_question=pending,
                allow_speculative_audio=True, material_slots=tuple(sorted((writes or {}).keys())),
                decision_reason=match.reason if match else "deterministic",
                decision_confidence=match.confidence if match else .99,
            ),
            speech,
        )

    def _cached(
        self, agent: AgentBundle, key: str, fallback: str, intent_id: str, *,
        action: str = "continue", match: IntentMatch | None = None,
    ) -> tuple[ResponsePlan, str]:
        speech = agent.cached_utterances.get(key, fallback)
        plan, speech = self._plan("fixed", intent_id, speech, action=action, match=match)
        return ResponsePlan(**{**plan.__dict__, "cache_key": key}), speech

    @staticmethod
    def _exact_faq(text: str, agent: AgentBundle) -> tuple[ResponsePlan, str] | None:
        normalized = normalize_intent_text(text)
        for key, response in agent.cached_utterances.items():
            if not key.startswith("faq:"):
                continue
            alias = normalize_intent_text(key[4:])
            if alias == normalized:
                return (
                    ResponsePlan(
                        "cache", intent_id="faq_services", cache_key=key,
                        allow_speculative_audio=True, decision_reason="exact_faq", decision_confidence=1.0,
                    ),
                    response,
                )
        return None
