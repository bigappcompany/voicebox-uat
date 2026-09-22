"""Canonical, deterministic intent classification for latency-critical turns."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable


CANONICAL_INTENTS = frozenset({
    "greeting", "provide_hiring_status", "provide_role", "provide_headcount",
    "provide_timeline", "faq_services", "faq_pricing", "existing_agency",
    "request_human", "callback_consent_yes", "callback_consent_no",
    "conversation_continue_yes", "conversation_continue_no",
    "callback_day", "callback_time", "callback_day_time", "repeat",
    "correction", "out_of_scope", "model_identity", "wrong_person", "busy",
    "not_interested", "goodbye", "thank_you", "company_identity", "incomplete_response", "unknown",
    "recall_callback_preference", "callback_preference_acknowledged",
})


def normalize_intent_text(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s:.-]", " ", text.casefold()).split())


@dataclass(frozen=True)
class IntentMatch:
    intent_id: str
    confidence: float
    reason: str
    slots: dict[str, Any] = field(default_factory=dict)


class CanonicalIntentModel:
    """Small context-aware classifier used before any hosted-model request.

    It deliberately handles only high-confidence conversational shapes. An
    ambiguous utterance remains ``unknown`` and is eligible for hosted fallback.
    """

    _yes = frozenset({
        "yes", "yep", "yup", "yeah", "sure", "okay", "ok", "please do",
        "sounds good", "that works", "go ahead", "fine", "correct", "yes please",
        "that's correct", "thats correct", "that s correct",
        "confirm it",
        "that's right", "thats right", "that s right",
        "okay that's fine", "okay thats fine", "okay that s fine",
        "that is correct", "that is right",
    })
    _no = frozenset({
        "no", "nope", "no thanks", "not now", "dont", "do not", "please dont",
        "dont call", "do not call", "not interested",
    })
    _days = re.compile(
        r"\b(today|tomorrow|(?:the\s+)?day after tomorrow|later today|next week|"
        r"(?:this|next)?\s*(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b"
    )
    _times = re.compile(
        r"\b((?:around\s+)?(?:at\s+)?(?:[01]?\d|2[0-3])(?::[0-5]\d)?\s*(?:a\.?\s*m\.?|p\.?\s*m\.?)|"
        r"(?:around\s+)?(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
        r"(?:\s+(?:[0-5][0-9]))?\s*(?:a\.?\s*m\.?|p\.?\s*m\.?)|half past\s+"
        r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve))\b"
    )

    def _parse_boolean(self, value: str) -> str:
        # A more robust boolean parser for phrases like "that is true yeah"
        # or negations like "yeah no dont do that"
        val = value.casefold()
        if re.search(r"\b(?:not sure|not certain|dont know|do not know|unsure|maybe|perhaps)\b", val):
            return "ambiguous"
        if re.search(r"\b(?:no|nope|not|dont|do not)\b", val):
            return "no"
        if re.search(r"\b(?:yes|yep|yup|yeah|sure|okay|ok|correct|true|go ahead|fine|sounds good|confirm it|that\s*'?\s*s\s+right|that\s*'?\s*s\s+correct|okay\s+that\s*'?\s*s\s+fine|that\s+is\s+right|that\s+is\s+correct)\b", val):
            return "yes"
        return "ambiguous"

    def classify(
        self,
        text: str,
        *,
        pending_question: Any | None = None,
        fact_values: dict[str, Any] | None = None,
        configured_patterns: dict[str, Any] | None = None,
        current_facts: dict[str, Any] | None = None,
    ) -> IntentMatch:
        value = normalize_intent_text(text)
        facts = fact_values or {}
        if current_facts is None:
            from ..flows.facts import FactExtractor
            current_facts = FactExtractor().extract(text, {}).values
        pending_intent = str(getattr(pending_question, "intent", "") or "")
        pending_slot = str(getattr(pending_question, "slot", "") or "")
        pending_type = str(getattr(pending_question, "expected_type", "") or "")

        configured = self._configured(value, configured_patterns or {})
        if configured is not None:
            return configured

        if (
            re.search(
                r"\b(?:goodbye|bye|bye\s+bye|have\s+a\s+(?:good|great)\s+day|end\s+call|hang\s+up|stop\s+calling)\b",
                value,
            )
            and not re.search(r"\b(?:dont|do not|please dont)\b", value)
        ):
            return IntentMatch("goodbye", .995, "goodbye_phrase")
        if value in {"hello", "hi", "hey", "namaste"}:
            return IntentMatch("greeting", .99, "exact_phrase")
        if re.search(r"\b(?:wrong person|wrong number|wrong contact)\b", value):
            return IntentMatch("wrong_person", .99, "safety_phrase")
        if re.search(r"\b(?:not interested|dont call me|do not call me|remove my number)\b", value):
            return IntentMatch("not_interested", .99, "safety_phrase")
        if re.search(
            r"\b(?:i\s+already\s+told\s+you|i\s+told\s+you\s+already|same\s+time\s+as\s+(?:earlier|before)|"
            r"what\s+time\s+did\s+i\s+(?:give|tell)\s+you|what\s+time\s+did\s+i\s+say|"
            r"as\s+i\s+mentioned\s+before|as\s+mentioned\s+earlier)\b",
            value,
        ):
            return IntentMatch("recall_callback_preference", .98, "recall_phrase")
        if re.search(r"\b(?:speak|talk|connect).*(?:human|person|manager|someone)|\brequest.*human|talk to human|speak to human\b", value):
            return IntentMatch("request_human", .97, "handoff_phrase")
        if re.search(r"\b(?:busy|im busy|i am busy|call.*later|talk.*later|middle of something|not a good time|busy right now|can we talk later|can you call later)\b", value):
            return IntentMatch("busy", .94, "defer_phrase")
        if re.search(r"\b(?:repeat|say that again|come again|what did you say)\b", value):
            return IntentMatch("repeat", .98, "repeat_phrase")
        if re.search(r"\b(?:what|which).*(?:model|ai model)|\bare you (?:an? )?(?:ai|bot)\b", value):
            return IntentMatch("model_identity", .98, "identity_phrase")
        if re.search(
            r"\b(?:where are you calling from|from where|which company|what company|"
            r"who is this|who are you|who is calling)\b", value
        ):
            return IntentMatch("company_identity", .98, "company_identity_phrase")
        if re.search(r"\b(?:reverse|linked list|technical issue|debug|write code|programming|coding|algorithm)\b", value):
            return IntentMatch("out_of_scope", .96, "domain_boundary")
        if re.search(r"\b(?:price|pricing|cost|charges|fee|fees)\b", value):
            return IntentMatch("faq_pricing", .94, "faq_phrase")
        if re.search(r"\b(?:services|staffing|blue collar|white collar|job types|apprenticeships?)\b", value):
            return IntentMatch("faq_services", .94, "faq_phrase")
        if re.search(r"\b(?:agency|vendor|recruiter).*(?:already|existing|currently)\b", value):
            return IntentMatch("existing_agency", .9, "agency_phrase")
        if re.search(r"\b(?:what (?:kind of|type of)?\s*(?:preference|callback|option)|clarify|what do you mean)\b", value):
            return IntentMatch("clarification", .95, "clarification_phrase")
        if (
            re.search(
                r"\b(?:thank\s+you|thanks|thank\s+you\s+so\s+much|thanks\s+a\s+lot|thank\s+you\s+very\s+much|"
                r"many\s+thanks|appreciate\s+it|thank\s+you\s+for\s+(?:your\s+)?(?:help|time))\b",
                value,
            )
            and not re.search(r"\b(?:no|not)\b", value)
        ):
            return IntentMatch("thank_you", .98, "gratitude_phrase")

        day_matches = list(self._days.finditer(value))
        clock_matches = list(self._times.finditer(value))
        # Corrections commonly contain both the abandoned and replacement
        # value: "tomorrow—actually Wednesday at five pm". The latest entity
        # is authoritative for the current turn.
        day = day_matches[-1] if day_matches else None
        clock = clock_matches[-1] if clock_matches else None
        if pending_slot in {"callback_day", "callback_time", "callback_preference"} or pending_intent.startswith("ask_callback"):
            if day and clock:
                return IntentMatch("callback_day_time", .99, "pending_question", {"callback_day": day.group(1).strip(), "callback_time": clock.group(1).strip()})
            if day:
                return IntentMatch("callback_day", .99, "pending_question", {"callback_day": day.group(1).strip()})
            if clock:
                return IntentMatch("callback_time", .99, "pending_question", {"callback_time": clock.group(1).strip()})
        else:
            if day and clock:
                return IntentMatch("callback_day_time", .95, "day_time_phrase", {"callback_day": day.group(1).strip(), "callback_time": clock.group(1).strip()})
            if day:
                return IntentMatch("callback_day", .95, "day_phrase", {"callback_day": day.group(1).strip()})
            if clock:
                return IntentMatch("callback_time", .95, "time_phrase", {"callback_time": clock.group(1).strip()})

        bool_val = self._parse_boolean(value)
        if facts.get("callback_state") == "PREFERENCE_RECORDED" or (
            facts.get("callback_preference") and not pending_question
        ):
            if bool_val == "yes" or value in self._yes or re.search(
                r"\b(?:confirm(?:\s+it)?|sounds\s+good|that\s*'?\s*s\s+(?:correct|right|fine)|okay\s+that\s*'?\s*s\s+fine|correct|yes|yeah|yep|sure|fine)\b",
                value,
            ):
                return IntentMatch("callback_consent_yes", .995, "callback_confirmation")
        if pending_type == "boolean" and pending_slot == "conversation_continue":
            if bool_val == "yes":
                return IntentMatch("conversation_continue_yes", .995, "pending_question")
            if bool_val == "no":
                return IntentMatch("conversation_continue_no", .995, "pending_question")
        if pending_intent in {"offer_callback", "ask_callback_consent"} or pending_slot in {"followup_consent", "callback_consent"}:
            if bool_val == "yes":
                return IntentMatch("callback_consent_yes", .995, "pending_question")
            if bool_val == "no":
                return IntentMatch("callback_consent_no", .995, "pending_question")

        if pending_slot == "hiring_status":
            if bool_val == "yes":
                return IntentMatch("provide_hiring_status", .995, "pending_question", {"hiring_status": "yes"})
            if bool_val == "no":
                return IntentMatch("provide_hiring_status", .995, "pending_question", {"hiring_status": "no"})
        if pending_slot == "headcount":
            if current_facts.get("headcount") is not None:
                return IntentMatch("provide_headcount", .99, "pending_question")
            return IntentMatch("incomplete_response", .85, "pending_question")
        if pending_slot in {"roles", "departments"}:
            if current_facts.get("roles") or current_facts.get("departments"):
                return IntentMatch("provide_role", .99, "pending_question")
            return IntentMatch("incomplete_response", .8, "pending_question")
        if pending_slot == "hiring_timeline":
            if current_facts.get("hiring_timeline") is not None:
                return IntentMatch("provide_timeline", .99, "pending_question")
            return IntentMatch("incomplete_response", .8, "pending_question")

        if re.search(r"\b(?:actually|correction|make that|rather|i meant)\b", value):
            return IntentMatch("correction", .9, "correction_marker")
        if current_facts.get("headcount") is not None:
            return IntentMatch("provide_headcount", .9, "fact_extractor")
        if current_facts.get("roles") or current_facts.get("departments"):
            return IntentMatch("provide_role", .9, "fact_extractor")
        if current_facts.get("hiring_timeline") is not None:
            return IntentMatch("provide_timeline", .9, "fact_extractor")
        if current_facts.get("hiring_status") is not None:
            return IntentMatch("provide_hiring_status", .9, "fact_extractor")
        return IntentMatch("unknown", 0.0, "no_high_confidence_match")

    @staticmethod
    def _configured(value: str, patterns: dict[str, Any]) -> IntentMatch | None:
        for intent, raw_phrases in patterns.items():
            phrases: Iterable[Any] = [raw_phrases] if isinstance(raw_phrases, str) else raw_phrases or []
            for raw in phrases:
                phrase = normalize_intent_text(str(raw))
                if phrase and (value == phrase or phrase in value):
                    return IntentMatch(str(intent), .95, "configured_pattern")
        return None
