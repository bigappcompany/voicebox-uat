from dataclasses import dataclass


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
