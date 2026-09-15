"""Small deterministic slot validators for compiled low-risk workflows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SlotValidation:
    valid: bool
    value: str | int | None
    reason: str | None = None


class SlotValidator:
    """Validate extracted/configured slot values without asking the LLM to judge them.

    Extraction is intentionally separate: callers may use a local extractor,
    an approved pattern, or a verified tool.  This class only enforces the
    bundle's declared type/enum/pattern contract.
    """

    def validate(self, schema: dict[str, Any], value: Any) -> SlotValidation:
        text = str(value or "").strip()
        if not text:
            return SlotValidation(False, None, "empty")
        slot_type = str(schema.get("type") or "string").lower()
        if slot_type in {"integer", "int"}:
            if not re.fullmatch(r"[+-]?\d+", text):
                return SlotValidation(False, None, "not_integer")
            numeric = int(text)
            if "minimum" in schema and numeric < int(schema["minimum"]):
                return SlotValidation(False, None, "below_minimum")
            if "maximum" in schema and numeric > int(schema["maximum"]):
                return SlotValidation(False, None, "above_maximum")
            return SlotValidation(True, numeric)
        choices = schema.get("enum") or schema.get("choices") or []
        if choices and text.casefold() not in {str(choice).casefold() for choice in choices}:
            return SlotValidation(False, None, "not_allowed")
        pattern = schema.get("pattern")
        if pattern and not re.fullmatch(str(pattern), text):
            return SlotValidation(False, None, "pattern_mismatch")
        return SlotValidation(True, text)
