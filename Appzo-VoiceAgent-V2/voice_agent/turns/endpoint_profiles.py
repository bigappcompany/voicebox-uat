from dataclasses import dataclass
import re


@dataclass(frozen=True)
class EndpointProfile:
    vad_stop_secs: float
    expected_answer: str
    speech_timeout_ms: int
    soft_eot_confidence: float
    hard_eot_confidence: float


PROFILES = {
    "YES_NO": EndpointProfile(.2, "yes_no", 2500, .55, .8),
    "SHORT_ENTITY": EndpointProfile(.35, "entity", 4000, .6, .85),
    "FREEFORM": EndpointProfile(.6, "freeform", 7000, .7, .9),
    "KYC_DIGITS": EndpointProfile(.85, "digits", 9000, .8, .95),
    "ADDRESS": EndpointProfile(.9, "address", 12000, .8, .95),
}


@dataclass(frozen=True)
class FluxEndpointProfile:
    eager_eot_threshold: float
    eot_threshold: float
    eot_timeout_ms: int


FLUX_PROFILES = {
    "balanced": FluxEndpointProfile(.55, .70, 3000),
    # The default sales-call profile favours responsiveness while retaining a
    # timeout long enough for short natural pauses.
    "fast": FluxEndpointProfile(.35, .55, 1200),
    "yes_no": FluxEndpointProfile(.30, .50, 800),
    "short_entity": FluxEndpointProfile(.30, .52, 1000),
    "requirements": FluxEndpointProfile(.35, .55, 1400),
    "freeform": FluxEndpointProfile(.40, .62, 2200),
}


def flux_profile(name: str) -> FluxEndpointProfile:
    return FLUX_PROFILES.get(name.casefold(), FLUX_PROFILES["fast"])


def profile_for_prompt(text: str) -> str:
    """Choose the next turn's profile from the question just spoken."""
    value = " ".join(text.casefold().split())
    if re.search(
        r"\b(?:would|are|do|did|can|will|is|have)\s+you\b|"
        r"\b(?:is|does|will)\s+your\b|\bare\s+there\b",
        value,
    ):
        return "yes_no"
    if re.search(r"\b(?:what|which)\s+(?:day|time|date|name)\b|\bwhen\b", value):
        return "short_entity"
    if re.search(r"\b(?:roles?|headcount|how many|timeline|departments?)\b", value):
        return "requirements"
    return "freeform"
