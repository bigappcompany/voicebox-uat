"""Deterministic callback-preference state machine."""

from __future__ import annotations

import re

from ..runtime.response_plan import ResponsePlan


class CallbackCoordinator:
    IDLE = "IDLE"
    FOLLOWUP_OFFERED = "FOLLOWUP_OFFERED"
    AWAITING_DAY_TIME = "AWAITING_DAY_TIME"
    AWAITING_DAY = "AWAITING_DAY"
    AWAITING_TIME = "AWAITING_TIME"
    PREFERENCE_RECORDED = "PREFERENCE_RECORDED"

    _days = r"today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|next week"
    _time = (
        r"(?:at\s+(?:[01]?\d|2[0-3])(?::[0-5]\d)?(?:\s*(?:a\.?\s*m\.?|p\.?\s*m\.?))?"
        r"|(?:[01]?\d|2[0-3])(?::[0-5]\d)?\s*(?:a\.?\s*m\.?|p\.?\s*m\.?)"
        r"|(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
        r"(?:\s+[0-5][0-9])?\s*(?:a\.?\s*m\.?|p\.?\s*m\.?))"
    )

    def route(self, text: str, slots: dict[str, object]) -> tuple[ResponsePlan, str] | None:
        normalized = " ".join(re.sub(r"[^\w\s:.]", " ", text.casefold()).split())
        state = str(slots.get("callback_state") or self.IDLE)
        explicit = bool(re.search(r"\b(?:call me back|callback|follow up|follow-up|arrange a call)\b", normalized))
        yes = normalized in {"yes", "yes please", "sure", "okay", "ok", "please do", "that works"}
        no = normalized in {"no", "no thanks", "not now", "not interested", "don't call", "do not call"}

        if state == self.FOLLOWUP_OFFERED and no:
            return self._fixed(
                "callback-declined",
                "Understood. I won't record a callback preference. Is there anything else I can help with?",
                callback_state=self.IDLE,
                followup_consent="no",
            )
        if state == self.FOLLOWUP_OFFERED and yes:
            return self._fixed(
                "callback-consent",
                "What day and time would be convenient for the follow-up?",
                callback_state=self.AWAITING_DAY_TIME,
                followup_consent="yes",
            )
        if explicit and state == self.IDLE:
            return self._fixed(
                "callback-request",
                "What day and time would be convenient for the follow-up?",
                callback_state=self.AWAITING_DAY_TIME,
                followup_consent="yes",
            )

        # A preference has already been captured locally. A subsequent
        # affirmative response must not fall through to the hosted model,
        # where it previously produced claims such as "I will connect with
        # you tomorrow" despite there being no scheduling integration.
        if state == self.PREFERENCE_RECORDED and yes:
            preference = str(slots.get("callback_preference") or "your requested time")
            return self._fixed(
                "callback-preference-acknowledged",
                f"Thanks. I've recorded {preference} as a requested follow-up time. Our team will confirm availability.",
                callback_state=self.PREFERENCE_RECORDED,
            )

        if state not in {self.AWAITING_DAY_TIME, self.AWAITING_DAY, self.AWAITING_TIME, self.FOLLOWUP_OFFERED}:
            return None
        day_match = re.search(rf"\b({self._days})\b", normalized)
        time_match = re.search(rf"\b({self._time})\b", normalized)
        day = day_match.group(1) if day_match else str(slots.get("callback_day") or "")
        clock = time_match.group(1) if time_match else str(slots.get("callback_time") or "")
        if day and clock:
            preference = f"{day} at {clock}"
            return self._fixed(
                "callback-preference",
                f"I've recorded your preference for {preference}. This is a requested time, not a confirmed booking.",
                callback_state=self.PREFERENCE_RECORDED,
                callback_day=day,
                callback_time=clock,
                callback_preference=preference,
            )
        if day:
            return self._fixed(
                "callback-day",
                f"What time {day} would be convenient?",
                callback_state=self.AWAITING_TIME,
                callback_day=day,
            )
        if clock:
            return self._fixed(
                "callback-time",
                f"What day would you prefer for a follow-up at {clock}?",
                callback_state=self.AWAITING_DAY,
                callback_time=clock,
            )
        return None

    def observe_assistant(self, speech: str, slots: dict[str, object]) -> None:
        if re.search(r"would you like.*(?:follow.?up|callback|call)", speech, re.I):
            slots["callback_state"] = self.FOLLOWUP_OFFERED
        elif re.search(r"what time.*(?:convenient|prefer)", speech, re.I):
            slots["callback_state"] = self.AWAITING_TIME
            day = re.search(rf"\b({self._days})\b", speech.casefold())
            if day:
                slots["callback_day"] = day.group(1)
        elif re.search(r"what day.*(?:follow.?up|prefer)", speech, re.I):
            slots["callback_state"] = self.AWAITING_DAY

    @staticmethod
    def _fixed(route: str, speech: str, **writes: str) -> tuple[ResponsePlan, str]:
        return (
            ResponsePlan(
                route,
                intent_id=route,
                slots_written=writes,
                allow_speculative_audio=True,
                booking_authority="preference",
                material_slots=tuple(sorted(writes)),
            ),
            speech,
        )
