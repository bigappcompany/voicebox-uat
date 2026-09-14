from dataclasses import dataclass
import os


@dataclass(frozen=True)
class RuntimeFlags:
    enable_agent_bundle: bool = True
    enable_local_retrieval: bool = False
    enable_dynamic_endpoints: bool = False
    enable_flux: bool = False
    enable_semantic_spec_reuse: bool = False
    enable_safe_chunker: bool = False
    enable_spec_tts: bool = False
    enable_local_llm: bool = False
    enable_fillers: bool = False
    enable_hedged_models: bool = False

    @classmethod
    def from_env(cls) -> "RuntimeFlags":
        values = {}
        for name, field in cls.__dataclass_fields__.items():
            raw = os.getenv(name.upper())
            values[name] = field.default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}
        return cls(**values)
