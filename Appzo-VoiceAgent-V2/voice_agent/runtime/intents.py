"""Canonical, deterministic intent classification for latency-critical turns."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable


CANONICAL_INTENTS = frozenset({
    "greeting", "provide_hiring_status", "provide_role", "provide_headcount",
    "provide_timeline", "faq_services", "faq_pricing", "existing_agency",
    "request_human", "callback_consent_yes", "callback_consent_no",
    "callback_day", "callback_time", "callback_day_time", "repeat",
    "correction", "out_of_scope", "model_identity", "wrong_person", "busy",
    "not_interested", "goodbye", "company_identity", "incomplete_response", "unknown",
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

        configured = self._configured(value, configured_patterns or {})
        if configured is not None:
            return configured

        if value in {"bye", "goodbye", "end call", "hang up", "stop calling"}:
            return IntentMatch("goodbye", .995, "exact_phrase")
        if value in {"hello", "hi", "hey", "namaste"}:
            return IntentMatch("greeting", .99, "exact_phrase")
        if re.search(r"\b(?:wrong person|wrong number|wrong contact)\b", value):
            return IntentMatch("wrong_person", .99, "safety_phrase")
        if re.search(r"\b(?:not interested|dont call me|do not call me|remove my number)\b", value):
            return IntentMatch("not_interested", .99, "safety_phrase")
        if re.search(r"\b(?:speak|talk|connect).*(?:human|person|manager|someone)\b", value):
            return IntentMatch("request_human", .97, "handoff_phrase")
        if re.search(r"\b(?:im busy|i am busy|call.*later|talk.*later|middle of something)\b", value):
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

        day = self._days.search(value)
        clock = self._times.search(value)
        if pending_slot in {"callback_day", "callback_time", "callback_preference"} or pending_intent.startswith("ask_callback"):
            if day and clock:
                return IntentMatch("callback_day_time", .99, "pending_question", {"callback_day": day.group(1), "callback_time": clock.group(1)})
            if day:
                return IntentMatch("callback_day", .99, "pending_question", {"callback_day": day.group(1)})
            if clock:
                return IntentMatch("callback_time", .99, "pending_question", {"callback_time": clock.group(1)})
        else:
            if day and clock:
                return IntentMatch("callback_day_time", .95, "day_time_phrase", {"callback_day": day.group(1), "callback_time": clock.group(1)})
            if day:
                return IntentMatch("callback_day", .95, "day_phrase", {"callback_day": day.group(1)})
            if clock:
                return IntentMatch("callback_time", .95, "time_phrase", {"callback_time": clock.group(1)})

        if pending_intent in {"offer_callback", "ask_callback_consent"} or pending_slot in {"followup_consent", "callback_consent"}:
            if value in self._yes:
                return IntentMatch("callback_consent_yes", .995, "pending_question")
            if value in self._no:
                return IntentMatch("callback_consent_no", .995, "pending_question")

        if pending_slot == "hiring_status" and value in self._yes:
            return IntentMatch("provide_hiring_status", .995, "pending_question", {"hiring_status": "yes"})
        if pending_slot == "hiring_status" and value in self._no:
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
