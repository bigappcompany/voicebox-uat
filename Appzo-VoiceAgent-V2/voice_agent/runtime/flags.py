from dataclasses import dataclass
import os


def _enabled(name: str, default: bool) -> bool:
    master = os.getenv("ENABLE_V2_IMPROVEMENTS")
    if master is not None and master.strip().lower() not in {"1", "true", "yes", "on"}:
        return False
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RuntimeFlags:
    enable_agent_bundle: bool = True
    enable_local_retrieval: bool = True
    enable_dynamic_endpoints: bool = True
    enable_flux: bool = False
    enable_semantic_spec_reuse: bool = True
    enable_safe_chunker: bool = True
    enable_spec_tts: bool = True
    enable_local_llm: bool = False
    enable_fillers: bool = False
    enable_hedged_models: bool = False
    enable_cached_greeting: bool = True
    enable_callback_state_machine: bool = True
    enable_structured_facts: bool = True
    enable_zero_delay_spec_promotion: bool = True
    enable_stable_interim_speculation: bool = True
    enable_flux_tuning: bool = True
    enable_compiled_prompts: bool = True
    enable_extended_deterministic_routing: bool = True
    enable_audible_pcm_metrics: bool = True

    @classmethod
    def from_env(cls) -> "RuntimeFlags":
        values = {}
        for name, field in cls.__dataclass_fields__.items():
            env_name = name.upper()
            if name == "enable_semantic_spec_reuse":
                # Match the public plan name while retaining compatibility
                # with the earlier development flag.
                env_name = (
                    "ENABLE_SEMANTIC_FINGERPRINT_REUSE"
                    if "ENABLE_SEMANTIC_FINGERPRINT_REUSE" in os.environ
                    else "ENABLE_SEMANTIC_SPEC_REUSE"
                )
            values[name] = _enabled(env_name, bool(field.default))
        return cls(**values)
