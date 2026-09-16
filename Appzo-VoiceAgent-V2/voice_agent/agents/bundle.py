from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentBundle:
    agent_id: str
    version: str
    tenant_id: str
    identity: dict[str, Any] = field(default_factory=dict)
    invariant_prompt: str = ""
    language_profile: dict[str, Any] = field(default_factory=dict)
    flow_graph: dict[str, Any] = field(default_factory=dict)
    state_schema: dict[str, Any] = field(default_factory=dict)
    slot_schema: dict[str, Any] = field(default_factory=dict)
    actions: dict[str, Any] = field(default_factory=dict)
    risk_policy: dict[str, Any] = field(default_factory=dict)
    routing_policy: dict[str, Any] = field(default_factory=dict)
    stt_profile: dict[str, Any] = field(default_factory=dict)
    tts_profile: dict[str, Any] = field(default_factory=dict)
    knowledge_profile: dict[str, Any] = field(default_factory=dict)
    cached_utterances: dict[str, str] = field(default_factory=dict)
    compiled_prompt: dict[str, Any] = field(default_factory=dict)
    fact_profile: dict[str, Any] = field(default_factory=dict)
    cache_policy: dict[str, Any] = field(default_factory=dict)

    @property
    def knowledge_version(self) -> str:
        return str(self.knowledge_profile.get("version", self.version))
