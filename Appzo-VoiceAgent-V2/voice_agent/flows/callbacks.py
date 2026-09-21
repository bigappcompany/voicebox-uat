"""Explicit callback-preference state transitions."""

from __future__ import annotations

from typing import Any

from ..runtime.intents import CanonicalIntentModel, IntentMatch
from ..runtime.response_plan import ResponsePlan
from ..runtime.session import PendingQuestion


class CallbackCoordinator:
    IDLE = "IDLE"
    FOLLOWUP_OFFERED = "FOLLOWUP_OFFERED"
    AWAITING_DAY_TIME = "AWAITING_DAY_TIME"
    AWAITING_DAY = "AWAITING_DAY"
    AWAITING_TIME = "AWAITING_TIME"
    PREFERENCE_RECORDED = "PREFERENCE_RECORDED"

    def route(
        self,
        intent: IntentMatch | str,
        slots: dict[str, object],
        *,
        turn_id: int = 0,
    ) -> tuple[ResponsePlan, str] | None:
        state = str(slots.get("callback_state") or self.IDLE)
        if isinstance(intent, str):
            pending = None
            if state == self.FOLLOWUP_OFFERED:
                pending = PendingQuestion("ask_callback_consent", "followup_consent", "boolean")
            elif state == self.AWAITING_TIME:
                pending = PendingQuestion("ask_callback_time", "callback_time", "time")
            elif state == self.AWAITING_DAY:
                pending = PendingQuestion("ask_callback_day", "callback_day", "date")
            elif state == self.AWAITING_DAY_TIME:
                pending = PendingQuestion("ask_callback_day_time", "callback_preference", "date_and_time")
            intent = CanonicalIntentModel().classify(intent, pending_question=pending)
        intent_id = intent.intent_id

        if intent_id == "callback_consent_no" and state == self.FOLLOWUP_OFFERED:
            return self._fixed(
                "callback_consent_no", "Understood. I won't record a callback preference.",
                next_state="FOLLOWUP", clear_pending=True,
                callback_state=self.IDLE, followup_consent="no",
            )
        if intent_id in {"callback_consent_yes", "request_human", "busy"} and state in {
            self.IDLE, self.FOLLOWUP_OFFERED,
        }:
            return self._fixed(
                "callback_consent_yes" if intent_id == "callback_consent_yes" else intent_id,
                "What day and time would be convenient?", next_state="CALLBACK",
                pending=PendingQuestion("ask_callback_day_time", "callback_preference", "date_and_time", turn_id),
                callback_state=self.AWAITING_DAY_TIME, followup_consent="yes",
            )
        if state == self.PREFERENCE_RECORDED and intent_id == "callback_consent_yes":
            preference = str(slots.get("callback_preference") or "your requested time")
            return self._fixed(
                "callback_preference_acknowledged",
                f"Your requested follow-up time is {preference}. Our team will confirm availability.",
                next_state="FOLLOWUP", clear_pending=True,
                callback_state=self.PREFERENCE_RECORDED,
                route="callback-preference-acknowledged",
            )

        callback_context = state in {
            self.AWAITING_DAY_TIME, self.AWAITING_DAY, self.AWAITING_TIME,
            self.FOLLOWUP_OFFERED, self.PREFERENCE_RECORDED,
        }
        if not callback_context:
            return None

        if intent_id == "clarification" and callback_context:
            return self._fixed(
                "callback_clarification",
                "We can note down your preferred day and time for our team to call you back. What day and time works best for you?",
                next_state="CALLBACK",
                pending=PendingQuestion("ask_callback_day_time", "callback_preference", "date_and_time", turn_id),
                route="callback-preference",
            )

        day = str(intent.slots.get("callback_day") or slots.get("callback_day") or "").strip()
        clock = str(intent.slots.get("callback_time") or slots.get("callback_time") or "").strip()
        if day and clock:
            preference = f"{day} at {clock}"
            return self._fixed(
                "callback_day_time", f"I've recorded {preference} as your preference, not a confirmed booking.",
                next_state="FOLLOWUP", clear_pending=True,
                callback_state=self.PREFERENCE_RECORDED, callback_day=day,
                callback_time=clock, callback_preference=preference,
                route="callback-preference",
            )
        if day:
            return self._fixed(
                "callback_day", f"What time {day} would be convenient?", next_state="CALLBACK",
                pending=PendingQuestion("ask_callback_time", "callback_time", "time", turn_id),
                callback_state=self.AWAITING_TIME, callback_day=day,
            )
        if clock:
            return self._fixed(
                "callback_time", f"What day would work for a follow-up at {clock}?", next_state="CALLBACK",
                pending=PendingQuestion("ask_callback_day", "callback_day", "date", turn_id),
                callback_state=self.AWAITING_DAY, callback_time=clock,
            )
        return None

    def offer_plan(self, *, turn_id: int) -> ResponsePlan:
        return ResponsePlan(
            "fixed", intent_id="offer_callback", next_state="CALLBACK",
            slots_written={"callback_state": self.FOLLOWUP_OFFERED},
            pending_question=PendingQuestion("ask_callback_consent", "followup_consent", "boolean", turn_id),
            allow_speculative_audio=True, booking_authority="preference",
        )

    @staticmethod
    def _fixed(
        intent_id: str, speech: str, *, next_state: str | None = None,
        pending: PendingQuestion | None = None, clear_pending: bool = False,
        route: str = "fixed",
        **writes: Any,
    ) -> tuple[ResponsePlan, str]:
        return (
            ResponsePlan(
                route, intent_id=intent_id, next_state=next_state,
                slots_written=writes, pending_question=pending,
                clear_pending_question=clear_pending, allow_speculative_audio=True,
                booking_authority="preference", material_slots=tuple(sorted(writes)),
                decision_reason="callback_state", decision_confidence=.99,
            ),
            speech,
        )
