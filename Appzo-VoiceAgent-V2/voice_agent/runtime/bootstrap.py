"""Call-start bootstrap: the only control-plane-to-runtime crossing per call."""
from dataclasses import dataclass
from typing import Any

from ..agents.compiler import AgentCompiler
from ..agents.registry import AgentBundleRegistry
from .latency_controller import LatencyController
from .session import CallSession


@dataclass
class RuntimeBootstrap:
    registry: AgentBundleRegistry
    compiler: AgentCompiler

    async def start_call(self, call_data: dict[str, Any], authoring_payload: dict[str, Any]) -> LatencyController:
        bundle = self.compiler.compile_goodbox(authoring_payload)
        await self.registry.put(bundle)
        call_id = str(call_data.get("call_id") or call_data.get("stream_id") or "unknown-call")
        return LatencyController(CallSession(call_id, bundle.tenant_id, bundle))
