"""Call-start bootstrap: the only control-plane-to-runtime crossing per call."""
from dataclasses import dataclass, field
from typing import Any

from ..agents.compiler import AgentCompiler
from ..agents.registry import AgentBundleRegistry
from ..knowledge.index import KnowledgeRecord, TenantKnowledgeIndex
from .latency_controller import LatencyController
from .session import CallSession


@dataclass
class RuntimeBootstrap:
    registry: AgentBundleRegistry
    compiler: AgentCompiler
    knowledge_index: TenantKnowledgeIndex = field(default_factory=TenantKnowledgeIndex)

    async def start_call(self, call_data: dict[str, Any], authoring_payload: dict[str, Any]) -> LatencyController:
        source_digest = self.compiler.source_digest(authoring_payload)
        bundle = await self.registry.get_by_source(source_digest)
        if bundle is None:
            bundle = self.compiler.compile_goodbox(authoring_payload)
            await self.registry.put(bundle, source_digest=source_digest)
        for document in bundle.knowledge_profile.get("documents", []):
            self.knowledge_index.add(
                KnowledgeRecord(
                    tenant_id=bundle.tenant_id,
                    agent_id=bundle.agent_id,
                    knowledge_version=bundle.knowledge_version,
                    document_id=str(document["id"]),
                    text=str(document["text"]),
                    risk_class=str(document.get("risk_class") or "LOW_PUBLIC"),
                    questions=tuple(str(item) for item in document.get("questions") or []),
                )
            )
        call_id = str(call_data.get("call_id") or call_data.get("stream_id") or "unknown-call")
        return LatencyController(
            CallSession(call_id, bundle.tenant_id, bundle, knowledge_index=self.knowledge_index)
        )
