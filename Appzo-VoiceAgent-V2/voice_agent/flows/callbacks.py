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
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"

    def route(
        self,
        intent: IntentMatch | str,
        slots: dict[str, object],
        *,
        turn_id: int = 0,
    ) -> tuple[ResponsePlan, str] | None:
        state = str(slots.get("callback_state") or self.IDLE)
        if isinstance(intent, str):
            if intent in {
                "request_human", "busy", "recall_callback_preference",
                "callback_consent_yes", "callback_consent_no", "callback_preference_acknowledged",
                "thank_you",
            }:
                intent = IntentMatch(intent, 0.99, "direct_intent")
            else:
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

        if intent_id == "goodbye":
            return None

        if state == self.CLOSING:
            if intent_id in {
                "callback_consent_yes", "callback_consent_no", "callback_preference_acknowledged",
                "thank_you", "affirmative", "ack", "unknown",
            }:
                return self._fixed(
                    "goodbye",
                    "Thank you for your time. Goodbye.",
                    next_state="FOLLOWUP", clear_pending=True,
                    callback_state=self.CLOSED,
                    action="end_call",
                    route="goodbye",
                )

        if intent_id == "callback_consent_no" and state in {
            self.FOLLOWUP_OFFERED, self.AWAITING_DAY_TIME, self.AWAITING_DAY,
            self.AWAITING_TIME, self.PREFERENCE_RECORDED,
        }:
            return self._fixed(
                "callback_consent_no", "Understood. I won't record a callback preference.",
                next_state="FOLLOWUP", clear_pending=True,
                callback_state=self.IDLE, followup_consent="no",
                slots_cleared=("callback_day", "callback_time", "callback_preference"),
            )

        # 3A: request_human
        if intent_id == "request_human":
            pref = str(slots.get("callback_preference") or "")
            if pref:
                return self._fixed(
                    "request_human",
                    f"Sure. I have {pref} as your preferred follow-up time. Our team will confirm availability.",
                    next_state="FOLLOWUP", clear_pending=True,
                    callback_state=self.PREFERENCE_RECORDED,
                    route="callback-preference-acknowledged",
                )
            return self._fixed(
                "request_human",
                "Sure. What day and time would be convenient?",
                next_state="CALLBACK",
                pending=PendingQuestion("ask_callback_day_time", "callback_preference", "date_and_time", turn_id),
                callback_state=self.AWAITING_DAY_TIME, followup_consent="yes",
                route="callback-preference",
            )

        # 3B: busy / talk later
        if intent_id == "busy":
            pref = str(slots.get("callback_preference") or "")
            if pref:
                return self._fixed(
                    "busy",
                    f"Of course. I have {pref} as your preferred follow-up time. Our team will confirm availability.",
                    next_state="FOLLOWUP", clear_pending=True,
                    callback_state=self.PREFERENCE_RECORDED,
                    route="callback-preference-acknowledged",
                )
            return self._fixed(
                "busy",
                "Of course. What day and time would work better?",
                next_state="CALLBACK",
                pending=PendingQuestion("ask_callback_day_time", "callback_preference", "date_and_time", turn_id),
                callback_state=self.AWAITING_DAY_TIME, followup_consent="yes",
                route="callback-preference",
            )

        # 3D: recall_callback_preference
        if intent_id == "recall_callback_preference":
            pref = str(slots.get("callback_preference") or "")
            if pref:
                return self._fixed(
                    "recall_callback_preference",
                    f"You mentioned {pref} as your preferred time. Our team will confirm availability.",
                    next_state="FOLLOWUP", clear_pending=True,
                    callback_state=self.PREFERENCE_RECORDED,
                    route="callback-preference-acknowledged",
                )
            return self._fixed(
                "recall_callback_preference",
                "What day and time would be convenient for a callback?",
                next_state="CALLBACK",
                pending=PendingQuestion("ask_callback_day_time", "callback_preference", "date_and_time", turn_id),
                callback_state=self.AWAITING_DAY_TIME, followup_consent="yes",
                route="callback-preference",
            )

        # A preference was already acknowledged when it was recorded. One
        # brief acknowledgement may close that topic, but generic "okay" must
        # not keep re-reading the preference forever.
        if state == self.PREFERENCE_RECORDED and intent_id in {
            "callback_consent_yes", "callback_preference_acknowledged", "thank_you",
        }:
            preference = str(slots.get("callback_preference") or "your requested time")
            if intent_id == "thank_you":
                return self._fixed(
                    "thank_you",
                    f"You're welcome! Our team will confirm availability for {preference}. Have a great day!",
                    next_state="FOLLOWUP", clear_pending=True,
                    callback_state=self.CLOSING,
                    route="callback-preference-acknowledged",
                )
            return self._fixed(
                "callback_preference_acknowledged",
                f"Great. We'll use {preference} as your preference. Is there anything else I can help with?",
                next_state="FOLLOWUP",
                pending=PendingQuestion("ask_anything_else", "conversation_continue", "boolean", turn_id),
                callback_state=self.IDLE,
                route="callback-preference-acknowledged",
            )

        if intent_id == "callback_consent_yes" and state in {
            self.IDLE, self.FOLLOWUP_OFFERED,
        }:
            return self._fixed(
                "callback_consent_yes",
                "What day and time would be convenient?", next_state="CALLBACK",
                pending=PendingQuestion("ask_callback_day_time", "callback_preference", "date_and_time", turn_id),
                callback_state=self.AWAITING_DAY_TIME, followup_consent="yes",
            )

        # A high-confidence boolean answer inside an incomplete callback flow
        # remains deterministic. It must not fall through to GPT merely
        # because the state is already awaiting one of the date/time fields.
        if intent_id == "callback_consent_yes" and state in {
            self.AWAITING_DAY_TIME, self.AWAITING_DAY, self.AWAITING_TIME,
        }:
            missing_slot = {
                self.AWAITING_DAY_TIME: "callback_preference",
                self.AWAITING_DAY: "callback_day",
                self.AWAITING_TIME: "callback_time",
            }[state]
            expected = {
                "callback_preference": "date_and_time",
                "callback_day": "date",
                "callback_time": "time",
            }[missing_slot]
            return self._fixed(
                "callback_consent_yes",
                {
                    "callback_preference": "What day and time would be convenient?",
                    "callback_day": "What day would work best?",
                    "callback_time": "What time would be convenient?",
                }[missing_slot],
                next_state="CALLBACK",
                pending=PendingQuestion(f"ask_{missing_slot}", missing_slot, expected, turn_id),
                callback_state=state,
                followup_consent="yes",
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
        if intent_id not in {"callback_day", "callback_time", "callback_day_time", "correction"}:
            # Preserve extracted callback facts, but allow the primary FAQ or
            # other intent to choose the spoken answer.
            return None
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
        action: str = "continue",
        slots_cleared: tuple[str, ...] = (),
        **writes: Any,
    ) -> tuple[ResponsePlan, str]:
        return (
            ResponsePlan(
                route, action=action, intent_id=intent_id, next_state=next_state,
                slots_written=writes, pending_question=pending,
                slots_cleared=slots_cleared,
                clear_pending_question=clear_pending, allow_speculative_audio=True,
                booking_authority="preference", material_slots=tuple(sorted(writes)),
                decision_reason="callback_state", decision_confidence=.99,
            ),
            speech,
        )
