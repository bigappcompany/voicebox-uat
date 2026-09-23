"""Deterministic, tenant-scoped fact extraction for low-latency call flows."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
_NUMBER_WORDS = {**_ONES, **_TENS, "hundred": 100}


def _number(value: str) -> int | None:
    value = value.casefold().strip().replace("-", " ")
    if value.isdigit():
        return int(value)
    if value in _NUMBER_WORDS:
        return _NUMBER_WORDS[value]
    parts = value.split()
    if len(parts) == 2 and parts[0] in _TENS and parts[1] in _ONES:
        return _TENS[parts[0]] + _ONES[parts[1]]
    if len(parts) == 2 and parts[0] in _ONES and parts[1] == "hundred":
        return _ONES[parts[0]] * 100
    return None


@dataclass(frozen=True)
class NumericRange:
    minimum: int
    maximum: int
    unit: str = ""
    approximate: bool = False

    def __str__(self) -> str:
        prefix = "about " if self.approximate else ""
        unit = f" {self.unit}" if self.unit else ""
        return f"{prefix}{self.minimum}-{self.maximum}{unit}"

    def to_primitive(self) -> str:
        return str(self)


@dataclass(frozen=True)
class FactValue:
    value: Any
    confidence: float
    source_turn: int
    raw_text: str
    approximate: bool = False
    corrected_from: Any | None = None

    def to_primitive(self) -> Any:
        val = getattr(self.value, "to_primitive", None)
        return val() if callable(val) else self.value


@dataclass(frozen=True)
class FactUpdate:
    values: dict[str, Any] = field(default_factory=dict)
    corrected: tuple[str, ...] = ()
    records: dict[str, FactValue] = field(default_factory=dict)


class FactExtractor:
    """Extract conservative facts without adding a sequential model request."""

    _number_pattern = (
        r"(?:\d{1,3}|"
        r"(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:[\s-](?:one|two|three|four|five|six|seven|eight|nine))?|"
        r"one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
        r"hundred)"
    )
    _callback_day = re.compile(
        r"\b(today|tomorrow|(?:the\s+)?day after tomorrow|later today|next week|"
        r"(?:this|next)?\s*(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b"
    )
    _callback_time = re.compile(
        r"\b((?:around\s+)?(?:at\s+)?(?:[01]?\d|2[0-3])(?::[0-5]\d)?\s*(?:a\s*m|p\s*m)|"
        r"(?:around\s+)?(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
        r"(?:\s+(?:[0-5][0-9]))?\s*(?:a\s*m|p\s*m)|half past\s+"
        r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve))\b"
    )

    def __init__(self, profile: dict[str, Any] | None = None, *, default_profile: str = "") -> None:
        self.profile = dict(profile or {})
        self.name = str(self.profile.get("name") or default_profile).casefold()

    def extract(
        self,
        text: str,
        current: dict[str, Any],
        *,
        pending_question: Any | None = None,
        source_turn: int = 0,
    ) -> FactUpdate:
        normalized = " ".join(re.sub(r"[^\w\s-]", " ", text.casefold()).split())
        if not normalized:
            return FactUpdate()
        updates: dict[str, Any] = {}
        approximate_fields: set[str] = set()
        pending_slot = str(getattr(pending_question, "slot", "") or "")

        # Callback facts are independent of the primary intent. A caller can
        # provide a preferred time and ask an FAQ in the same utterance.
        callback_context = bool(
            pending_slot.startswith("callback_")
            or pending_slot == "callback_preference"
            or str(current.get("callback_state") or "IDLE") != "IDLE"
            or re.search(r"\b(?:callback|call back|follow up|meeting)\b", normalized)
        )
        if callback_context:
            days = list(self._callback_day.finditer(normalized))
            times = list(self._callback_time.finditer(normalized))
            if days:
                updates["callback_day"] = days[-1].group(1).strip()
            if times:
                updates["callback_time"] = times[-1].group(1).strip()
            if "callback_day" in updates or "callback_time" in updates:
                day_value = updates.get("callback_day") or current.get("callback_day")
                time_value = updates.get("callback_time") or current.get("callback_time")
                if day_value and time_value:
                    updates["callback_preference"] = f"{day_value} at {time_value}"

        if re.search(r"\b(?:not hiring|no hiring|aren't hiring|are not hiring|not looking to hire|not hiring right now|not hiring now)\b", normalized):
            updates["hiring_status"] = "no"
        elif pending_slot == "hiring_status" and re.search(r"\b(?:not right now|not now)\b", normalized):
            updates["hiring_status"] = "no"
        elif re.search(r"\b(?:we are hiring|we're hiring|need to hire|planning to hire|plan to hire|will hire)\b", normalized):
            updates["hiring_status"] = "yes"
        elif pending_slot == "hiring_status" and normalized in {"yes", "yep", "yup", "yeah", "sure", "okay", "ok"}:
            updates["hiring_status"] = "yes"
        elif pending_slot == "hiring_status" and normalized in {"no", "nope", "no thanks"}:
            updates["hiring_status"] = "no"

        timeline = re.search(
            rf"\b(?:in|within|over|about|around|next)?\s*({self._number_pattern})"
            rf"(?:\s*(?:to|-|or)?\s*({self._number_pattern}))?\s*"
            r"(day|days|week|weeks|month|months|year|years)\b",
            normalized,
        )
        if not timeline and current.get("hiring_timeline") not in (None, ""):
            timeline = re.search(
                rf"\b(?:actually|correction|make that|rather|i meant)\s+({self._number_pattern})"
                rf"(?:\s*(?:to|-|or)?\s*({self._number_pattern}))?\s*"
                r"(day|days|week|weeks|month|months|year|years)\b",
                normalized,
            )
        if timeline:
            first, second = _number(timeline.group(1)), _number(timeline.group(2) or "")
            unit = timeline.group(3)
            approximate = bool(re.search(r"\b(?:about|around|roughly|approximately)\b", normalized))
            if first is not None and second is not None and first != second:
                updates["hiring_timeline"] = NumericRange(min(first, second), max(first, second), unit, approximate)
            elif first is not None:
                updates["hiring_timeline"] = f"{timeline.group(1)} {unit}"
            if approximate:
                approximate_fields.add("hiring_timeline")
            if updates.get("hiring_status") == "no":
                updates["hiring_status"] = "future"
            elif current.get("hiring_status") in (None, "") and "hiring_status" not in updates:
                updates["hiring_status"] = "future"

        # Reject ambiguous sequences of digits (e.g. "eight two five", "1 2 3", or "eight two" without a noun or connector)
        digit_word = r"(?:zero|one|two|three|four|five|six|seven|eight|nine|\d{1,3})"
        has_noun = bool(re.search(r"\b(?:people|persons|employees|hires|candidates|positions|members|roles|staff|engineers|developers|department|team)\b", normalized))
        is_ambiguous_digit_seq = bool(
            re.search(rf"\b{digit_word}\s+{digit_word}\s+{digit_word}\b", normalized)
            or (
                not has_noun
                and re.search(rf"\b{digit_word}\s+{digit_word}\b", normalized)
                and not re.search(r"\b(?:to|-|or|and)\b", normalized)
            )
        )

        headcount = None
        if not is_ambiguous_digit_seq:
            headcount = re.search(
                rf"\b(?:need|hire|hiring|require|about|above|around|roughly|approximately|lets say|let s say)?\s*"
                rf"({self._number_pattern})(?:\s*(?:to|-|or|and)?\s*({self._number_pattern}))?\s+"
                r"(?:people|persons|employees|hires|candidates|positions|members|roles|staff|engineers|developers|"
                r"(?:(?:for|per|in|each|across)(?:\s+(?:each|the|our|every))?\s+)?(?:department|departments|dept|depts|team|teams))\b",
                normalized,
            )
            if not headcount and current.get("headcount") not in (None, ""):
                # Corrections commonly omit the noun: "actually six, not five".
                headcount = re.search(
                    rf"\b(?:actually|correction|make that|rather|i meant)\s+({self._number_pattern})"
                    rf"(?:\s+(?:to|-|or|and)\s+({self._number_pattern}))?\b",
                    normalized,
                )
            if not headcount and pending_slot == "headcount":
                headcount = re.search(
                    rf"\b(?:about|above|around|roughly|approximately|lets say|let s say)?\s*"
                    rf"({self._number_pattern})(?:\s+(?:to|-|or|and)\s+({self._number_pattern}))?\b",
                    normalized,
                )
                if headcount:
                    tail = normalized[headcount.end():].strip()
                    if re.match(r"^(?:a\s*m|p\s*m|o'?clock|days?|weeks?|months?|years?)\b", tail):
                        headcount = None
        if headcount:
            first, second = _number(headcount.group(1)), _number(headcount.group(2) or "")
            approximate = bool(re.search(r"\b(?:about|above|around|roughly|approximately|lets say|let s say)\b", normalized))
            if first is not None and second is not None and first != second:
                if first < second:
                    approximate = True
                    updates["headcount"] = NumericRange(first, second, "people", approximate)
            elif first is not None:
                updates["headcount"] = first
            if approximate:
                approximate_fields.add("headcount")
            if current.get("hiring_status") in (None, "") and "hiring_status" not in updates:
                updates["hiring_status"] = "yes"

        aliases = self.profile.get("role_aliases") or {
            "operations": ["operations", "operation", "ops", "operation check", "operations check"],
            "technology": ["technology", "tech", "engineering", "engineers", "developers", "developer"],
            "office": ["office", "administration", "administrative", "admin"],
            "sales": ["sales"],
            "marketing": ["marketing", "growth", "branding", "digital marketing", "marketing department"],
            "human resources": ["human resources", "hr"],
            "finance": ["finance", "accounting"],
        }
        found: list[str] = []
        if isinstance(aliases, dict):
            for canonical, phrases in aliases.items():
                values = [phrases] if isinstance(phrases, str) else list(phrases or [])
                values = [v for v in values if str(v).casefold() not in {"it", "i.t."}]
                if any(re.search(rf"\b{re.escape(str(value).casefold())}\b", normalized) for value in values):
                    found.append(str(canonical))

        it_matched = False
        if re.search(
            r"\b(?:information technology|it\s+(?:department|team|roles?|jobs?|positions?|functions?|staff|support|personnel|professionals?|openings?)|tech\s+roles?)\b",
            normalized,
        ):
            it_matched = True
        elif pending_slot in {"roles", "departments"} and (
            normalized in {"it", "i t"}
            or re.fullmatch(r"(?:for\s+)?(?:it|i t)", normalized)
        ):
            it_matched = True
        elif re.search(r"\bIT\b", text):
            pronoun_usage = bool(
                re.search(
                    r"\b(?:confirm|leave|forget|do|schedule|that['’]?s|thats|drop|skip|cancel|got|take|make|send|hear|see|keep|get)\s+IT\b",
                    text,
                    re.I,
                )
                or re.search(r"\bIT\s+(?:is|was|will|can|could|would|should|has|had)\b", text)
            )
            if not pronoun_usage:
                it_matched = True

        if it_matched and "technology" not in found:
            found.append("technology")

        # Preserve separate quantities when a caller names multiple hiring
        # groups in one turn, e.g. "six to seven in tech and a couple in the
        # office." The aggregate headcount remains useful to the workflow,
        # while headcount_by_role retains the detail for the prompt and CRM.
        per_role_counts: dict[str, Any] = {}
        # Keep quantities attached to the role they describe. The previous
        # matcher covered only technology and office, losing statements such
        # as "three-four for tech and a couple for operations".
        if isinstance(aliases, dict):
            for role, phrases in aliases.items():
                values = [phrases] if isinstance(phrases, str) else list(phrases or [])
                for raw_alias in values:
                    alias = str(raw_alias).casefold()
                    if alias in {"it", "i.t."}:
                        continue
                    role_count_pattern = re.compile(
                        rf"\b(?:(about|around|roughly|approximately)\s+)?"
                        rf"(?:(a\s+couple)(?:\s+of)?|({self._number_pattern})"
                        rf"(?:\s*(?:to|-|or|and)\s*|\s+)({self._number_pattern})?)\s+"
                        r"(?:people|persons|employees|hires|candidates|positions|members|staff)\s+"
                        rf"(?:in|for)\s+(?:my|the|our)?\s*{re.escape(alias)}"
                        r"(?:\s+(?:department|team|roles?))?\b"
                    )
                    role_count = role_count_pattern.search(normalized)
                    if not role_count:
                        continue
                    approximate = bool(role_count.group(1))
                    raw_first, raw_second = role_count.group(3), role_count.group(4)
                    if role_count.group(2):
                        first = second = 2
                        approximate = True
                    else:
                        first, second = _number(raw_first or ""), _number(raw_second or "")
                    if first is None:
                        continue
                    role_name = str(role)
                    if role_name not in found:
                        found.append(role_name)
                    if second is not None and second != first:
                        per_role_counts[role_name] = NumericRange(min(first, second), max(first, second), "people", True)
                    else:
                        per_role_counts[role_name] = first
                    break

        if per_role_counts:
            updates["headcount_by_role"] = per_role_counts
            total_min = sum(getattr(value, "minimum", value) for value in per_role_counts.values())
            total_max = sum(getattr(value, "maximum", value) for value in per_role_counts.values())
            updates["headcount"] = NumericRange(
                total_min, total_max, "people",
                any(getattr(value, "approximate", False) for value in per_role_counts.values()),
            )
            approximate_fields.add("headcount")
        if found:
            existing_roles = list(current.get("roles") or [])
            updates["roles"] = sorted(set(existing_roles + found))
            if current.get("hiring_status") in (None, "") and "hiring_status" not in updates:
                updates["hiring_status"] = "yes"
            # Broad function names in the default recruitment profile are
            # useful as both role families and departments. Tenants can
            # provide department_aliases to separate those concepts.
            department_aliases = self.profile.get("department_aliases")
            existing_depts = list(current.get("departments") or [])
            if department_aliases is None:
                updates["departments"] = sorted(set(existing_depts + found))
            elif isinstance(department_aliases, dict):
                departments = [
                    str(canonical)
                    for canonical, phrases in department_aliases.items()
                    if any(
                        re.search(rf"\b{re.escape(str(value).casefold())}\b", normalized)
                        for value in ([phrases] if isinstance(phrases, str) else list(phrases or []))
                    )
                ]
                if departments:
                    updates["departments"] = sorted(set(existing_depts + departments))
                elif existing_depts:
                    updates["departments"] = sorted(set(existing_depts))
            if headcount and "each" in normalized and "headcount" in updates:
                updates["headcount_by_role"] = {
                    role: updates["headcount"] for role in updates["roles"]
                }

        if "headcount" in updates and "headcount_by_role" not in updates:
            previous_by_role = current.get("headcount_by_role")
            if isinstance(previous_by_role, dict) and previous_by_role:
                updates["headcount_by_role"] = {
                    str(role): updates["headcount"] for role in previous_by_role
                }

        corrected = tuple(
            name for name, value in updates.items()
            if name in current and current.get(name) not in (None, "", []) and current.get(name) != value
        )
        records = {
            name: FactValue(
                value=value,
                confidence=.99 if pending_slot == name else .92,
                source_turn=source_turn,
                raw_text=text,
                approximate=name in approximate_fields,
                corrected_from=current.get(name) if name in corrected else None,
            )
            for name, value in updates.items()
        }
        return FactUpdate(updates, corrected, records)

    @staticmethod
    def known_facts(slots: dict[str, Any]) -> dict[str, Any]:
        return {
            name: value for name, value in slots.items()
            if value not in (None, "", False, []) and name != "callback_state"
        }
