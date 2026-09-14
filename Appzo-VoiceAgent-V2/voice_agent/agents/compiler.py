"""Compilation boundary between authoring payloads and small runtime artifacts."""
from typing import Any

from .bundle import AgentBundle


class AgentCompiler:
    def compile_goodbox(self, payload: dict[str, Any]) -> AgentBundle:
        agent = payload.get("call_agent") or {}
        model = payload.get("model_config") or {}
        transcriber = payload.get("transcriber_config") or {}
        synthesizer = payload.get("synthesizer_config") or {}
        agent_id = str(payload.get("chatbot_id") or payload.get("agent_id") or "goodbox-agent")
        version = str(payload.get("agent_version") or payload.get("version") or "goodbox-current")
        tenant_id = str(payload.get("tenant_id") or payload.get("org_code") or "goodbox")
        system = str(payload.get("system_prompt") or payload.get("prompt") or "You are a concise telephone assistant.").strip()
        return AgentBundle(
            agent_id=agent_id, version=version, tenant_id=tenant_id,
            identity={"name": agent.get("name", "assistant")}, invariant_prompt=system,
            language_profile={"primary": payload.get("primary_language", "multi")},
            flow_graph=agent.get("flow_graph") or {"initial_state": "OPEN", "states": {"OPEN": {}}},
            state_schema=agent.get("state_schema") or {}, slot_schema=agent.get("slot_schema") or {},
            actions=agent.get("actions") or {}, risk_policy=agent.get("risk_policy") or {"class": "LOW_PUBLIC"},
            routing_policy={"model": model.get("model"), "provider": model.get("provider")},
            stt_profile={"model": transcriber.get("model", "nova-3"), "endpointing": transcriber.get("endpointing")},
            tts_profile={"model": synthesizer.get("model", "sonic-3.5"), "voice_id": synthesizer.get("voice_id")},
            knowledge_profile=payload.get("knowledge_profile") or {"version": version},
            cached_utterances=agent.get("cached_utterances") or {},
        )
