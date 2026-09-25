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


import os

FLUX_PROFILES = {
    "balanced": FluxEndpointProfile(.55, .70, 3000),
    # The default sales-call profile favours responsiveness while retaining a
    # timeout long enough for short natural pauses.
    "fast": FluxEndpointProfile(.30, .60, 1200),
    # yes_no: caller gave a simple yes/no; eager fires at Deepgram minimum (.30),
    # hard EOT secures high-confidence confirmation (.55).
    "yes_no": FluxEndpointProfile(.30, .55, 500),
    # short_entity: name / date / time — 280ms eager lead for TTS pre-warming.
    "short_entity": FluxEndpointProfile(.30, .58, 750),
    # requirements: caller listing roles / headcount — needs a longer window to
    # complete a sentence, but 300ms eager gap allows background candidate generation.
    "requirements": FluxEndpointProfile(.30, .60, 1000),
    # freeform: open-ended answer — match old requirements timeout so we don't
    # cut off mid-sentence while eager at .32 gives early speculation start.
    "freeform": FluxEndpointProfile(.32, .62, 1400),
}

LATENCY_TEST_FLUX_PROFILES = {
    "balanced": FluxEndpointProfile(.55, .70, 3000),
    "fast": FluxEndpointProfile(.30, .60, 1200),
    "yes_no": FluxEndpointProfile(.30, .55, 400),
    "short_entity": FluxEndpointProfile(.30, .58, 550),
    "requirements": FluxEndpointProfile(.30, .60, 750),
    "freeform": FluxEndpointProfile(.32, .62, 1100),
}


def flux_profile(name: str) -> FluxEndpointProfile:
    preset = os.getenv("V2_ENDPOINT_PROFILE_PRESET", "current").lower()
    profiles = LATENCY_TEST_FLUX_PROFILES if preset == "latency_test" else FLUX_PROFILES
    return profiles.get(name.casefold(), profiles["fast"])


def detected_profile_for_prompt(text: str) -> str | None:
    """Detect a targeted endpoint profile from the question just spoken, or None if unspecific."""
    value = " ".join(text.casefold().split())
    # Ask for detailed hiring requirements before looking for a yes/no shape:
    # "Could you share the roles ...?" is grammatically a question but is not
    # a binary answer and must not use the aggressively short timeout.
    if re.search(r"\b(?:roles?|headcount|how many|timeline|departments?|requirements?|skills?)\b", value):
        return "requirements"
    # Collecting one compact entity benefits from the entity profile even if
    # the sentence also contains a polite auxiliary such as "could you".
    if re.search(r"\b(?:what|which)\s+(?:day|time|date|name)\b|\bwhen\b", value):
        return "short_entity"
    # Open-ended questions asking for elaboration or explanation need freeform
    if re.search(
        r"\b(?:tell me (?:more|a little more)|what else|how (?:can|may) i|help (?:you|with)|anything else|"
        r"explain|describe|more about|share more)\b",
        value,
    ):
        return "freeform"
    if re.search(
        r"\b(?:would|are|do|did|can|will|is|have)\s+you\b|"
        r"\b(?:is|does|will)\s+your\b|\bare\s+there\b|"
        r"\bis\s+this\s+(?:a\s+)?(?:good|okay|ok)\s+time\b",
        value,
    ):
        return "yes_no"
    return None


def profile_for_prompt(text: str) -> str:
    """Choose the next turn's profile from the question just spoken."""
    return detected_profile_for_prompt(text) or "freeform"
