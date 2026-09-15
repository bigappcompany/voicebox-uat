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
        prompt_parts = (
            str(payload.get("system_prompt") or "").strip(),
            str(payload.get("prompt") or "").strip(),
        )
        system = "\n\n".join(part for part in prompt_parts if part)
        if not system:
            system = "You are a concise, helpful telephone voice assistant."
        raw_knowledge = (
            payload.get("knowledge_profile")
            or payload.get("knowledge_documents")
            or agent.get("knowledge_profile")
            or agent.get("knowledge_documents")
            or []
        )
        knowledge_profile = self._knowledge_profile(raw_knowledge, version)
        return AgentBundle(
            agent_id=agent_id, version=version, tenant_id=tenant_id,
            identity={
                "name": agent.get("name", "assistant"),
                "company_name": payload.get("company_name") or agent.get("company_name") or "The Hiring Company",
            }, invariant_prompt=system,
            language_profile={"primary": payload.get("primary_language", "multi")},
            flow_graph=agent.get("flow_graph") or payload.get("flow_graph") or {"initial_state": "OPEN", "states": {"OPEN": {}}},
            state_schema=agent.get("state_schema") or payload.get("state_schema") or {},
            slot_schema=agent.get("slot_schema") or payload.get("slot_schema") or {},
            actions=agent.get("actions") or payload.get("actions") or {},
            risk_policy=agent.get("risk_policy") or payload.get("risk_policy") or {"class": "LOW_PUBLIC"},
            routing_policy={
                "model": model.get("model"),
                "provider": model.get("provider"),
                **(payload.get("routing_policy") or agent.get("routing_policy") or {}),
            },
            stt_profile={"model": transcriber.get("model", "nova-3"), "endpointing": transcriber.get("endpointing"),
                         "keyterms": transcriber.get("keyterms") or []},
            tts_profile={"model": synthesizer.get("model", "sonic-3.5"), "voice_id": synthesizer.get("voice_id")},
            knowledge_profile=knowledge_profile,
            cached_utterances=(agent.get("cached_utterances") or payload.get("cached_utterances") or {}),
        )

    @staticmethod
    def _knowledge_profile(raw: Any, version: str) -> dict[str, Any]:
        """Normalize optional Goodbox knowledge without a runtime control-plane call.

        Goodbox deployments currently differ in whether they return a list of
        passages, a ``documents`` list, or a profile object. The compiler makes
        all of those forms a tenant-local, immutable bundle artifact.
        """
        profile = dict(raw) if isinstance(raw, dict) else {"documents": raw}
        documents = profile.get("documents") or profile.get("records") or profile.get("items") or []
        if isinstance(documents, (str, bytes)):
            documents = [documents]
        normalized: list[dict[str, str]] = []
        for index, document in enumerate(documents if isinstance(documents, list) else []):
            if isinstance(document, str):
                text, document_id, risk = document.strip(), f"doc-{index + 1}", "LOW_PUBLIC"
            elif isinstance(document, dict):
                text = str(document.get("text") or document.get("content") or document.get("answer") or "").strip()
                document_id = str(document.get("id") or document.get("document_id") or document.get("name") or f"doc-{index + 1}")
                risk = str(document.get("risk_class") or "LOW_PUBLIC")
            else:
                continue
            if text:
                normalized.append({"id": document_id, "text": text, "risk_class": risk})
        profile["version"] = str(profile.get("version") or version)
        profile["documents"] = normalized
        return profile
