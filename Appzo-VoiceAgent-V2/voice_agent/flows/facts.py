"""Deterministic, tenant-scoped fact extraction for low-latency call flows."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
}


def _number(value: str) -> int | None:
    value = value.casefold().strip()
    if value.isdigit():
        return int(value)
    return _NUMBER_WORDS.get(value)


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

    _number_pattern = r"(?:\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty)"

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

        if re.search(r"\b(?:not hiring|no hiring|aren't hiring|are not hiring|not right now|not now)\b", normalized):
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

        headcount = re.search(
            rf"\b(?:need|hire|hiring|require|about|around|roughly|approximately|lets say|let s say)?\s*"
            rf"({self._number_pattern})(?:\s*(?:to|-|or)?\s*({self._number_pattern}))?\s+"
            r"(?:people|persons|employees|hires|candidates|positions|members|roles)\b",
            normalized,
        )
        if not headcount and current.get("headcount") not in (None, ""):
            # Corrections commonly omit the noun: "actually six, not five".
            headcount = re.search(
                rf"\b(?:actually|correction|make that|rather|i meant)\s+({self._number_pattern})"
                rf"(?:\s*(?:to|-|or)?\s*({self._number_pattern}))?\b",
                normalized,
            )
        if not headcount and pending_slot == "headcount":
            headcount = re.fullmatch(
                rf"(?:about|around|roughly|approximately|lets say|let s say)?\s*({self._number_pattern})"
                rf"(?:\s*(?:to|-|or)?\s*({self._number_pattern}))?",
                normalized,
            )
        if headcount:
            first, second = _number(headcount.group(1)), _number(headcount.group(2) or "")
            approximate = bool(re.search(r"\b(?:about|around|roughly|approximately|lets say|let s say)\b", normalized))
            if first is not None and second is not None and first != second:
                approximate = True
                updates["headcount"] = NumericRange(min(first, second), max(first, second), "people", approximate)
            elif first is not None:
                updates["headcount"] = first
            if approximate:
                approximate_fields.add("headcount")
            if current.get("hiring_status") in (None, "") and "hiring_status" not in updates:
                updates["hiring_status"] = "yes"

        aliases = self.profile.get("role_aliases") or {
            "operations": ["operations", "operation", "ops", "operation check", "operations check"],
            "technology": ["technology", "tech", "engineering", "engineers", "developers", "developer", "it"],
            "sales": ["sales"],
            "human resources": ["human resources", "hr"],
            "finance": ["finance", "accounting"],
        }
        found: list[str] = []
        if isinstance(aliases, dict):
            for canonical, phrases in aliases.items():
                values = [phrases] if isinstance(phrases, str) else list(phrases or [])
                if any(re.search(rf"\b{re.escape(str(value).casefold())}\b", normalized) for value in values):
                    found.append(str(canonical))
        if found:
            updates["roles"] = sorted(set(found))
            if current.get("hiring_status") in (None, "") and "hiring_status" not in updates:
                updates["hiring_status"] = "yes"
            # Broad function names in the default recruitment profile are
            # useful as both role families and departments. Tenants can
            # provide department_aliases to separate those concepts.
            department_aliases = self.profile.get("department_aliases")
            if department_aliases is None:
                updates["departments"] = sorted(set(found))
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
                    updates["departments"] = sorted(set(departments))
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
