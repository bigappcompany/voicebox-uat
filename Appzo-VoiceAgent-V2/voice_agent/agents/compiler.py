"""Compilation boundary between authoring payloads and small runtime artifacts."""
import hashlib
import json
import os
import re
from typing import Any

from loguru import logger

from .bundle import AgentBundle


class AgentCompiler:
    @staticmethod
    def source_digest(payload: dict[str, Any]) -> str:
        """Stable cache key for an immutable Goodbox authoring snapshot."""
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

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
        runtime_prompt = payload.get("runtime_prompt") or agent.get("runtime_prompt") or {}
        if isinstance(runtime_prompt, str):
            runtime_prompt = {"invariant": runtime_prompt}
        if not isinstance(runtime_prompt, dict):
            runtime_prompt = {}
        # Goodbox authoring prompts can be ~20k characters. Sending a blind
        # head/tail slice on every caller turn is both slow and semantically
        # unstable: a policy in the omitted middle silently disappears. A
        # tenant can provide `runtime_prompt.invariant` for exact control. For
        # existing agents, compile a deterministic compact invariant once at
        # call setup; it preserves high-signal authoring rules without putting
        # truncation logic on the hot path.
        if not str(runtime_prompt.get("invariant") or "").strip():
            runtime_prompt = {
                **runtime_prompt,
                "invariant": self._compile_runtime_invariant(system),
            }
        fact_profile = payload.get("fact_profile") or agent.get("fact_profile") or {}
        if not isinstance(fact_profile, dict):
            fact_profile = {}
        profile_name = str(fact_profile.get("name") or payload.get("runtime_profile") or agent.get("runtime_profile") or "").strip()
        if profile_name:
            fact_profile = {**fact_profile, "name": profile_name}
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
            compiled_prompt=dict(runtime_prompt),
            fact_profile=fact_profile,
            cache_policy=dict(payload.get("cache_policy") or agent.get("cache_policy") or {}),
        )

    @staticmethod
    def _compile_runtime_invariant(source: str) -> str:
        """Derive a bounded fallback invariant once, outside the media path.

        This is deliberately an extraction, not a summarization call: no
        external model, latency, or authoring data leaves the process. It
        keeps source-order so later tenant rules retain their usual override
        semantics. Supplying `runtime_prompt.invariant` remains the exact,
        recommended authoring interface.
        """
        limit = max(800, int(os.getenv("V2_COMPILED_RUNTIME_PROMPT_CHARS", "3600")))
        source = " ".join(str(source).split())
        if len(source) <= limit:
            return source

        # Split after sentence punctuation and retain sentences carrying
        # identity, safety, language, output, campaign, or business-fact
        # instructions. The first sentence is included as it commonly names
        # the caller, company, or campaign.
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", source) if part.strip()]
        high_signal = re.compile(
            r"\b(?:you are|name is|company|calling from|identity|language|english|hindi|hinglish|"
            r"must|must not|never|do not|don't|only|always|avoid|policy|safe|consent|"
            r"schedule|book|callback|appointment|confirm|output|response|sentence|concise|"
            r"hiring|staffing|recruit|role|candidate|service|offer|pricing|payment)\b",
            re.I,
        )
        selected: list[str] = []
        used = 0
        for index, sentence in enumerate(sentences):
            if index and not high_signal.search(sentence):
                continue
            addition = len(sentence) + (1 if selected else 0)
            if used + addition > limit:
                continue
            selected.append(sentence)
            used += addition
        if not selected:
            selected = [source[:limit].rsplit(" ", 1)[0]]
        result = " ".join(selected)
        logger.warning(
            "V2 compiled runtime prompt original_chars={} runtime_chars={}; supply runtime_prompt.invariant for exact authoring",
            len(source),
            len(result),
        )
        return result

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
