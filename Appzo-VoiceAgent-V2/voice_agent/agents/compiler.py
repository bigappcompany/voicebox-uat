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
        is_recruitment = profile_name.casefold() == "recruitment" or bool(
            re.search(r"\b(?:hiring|staffing|recruit(?:ment|ing)?)\b", system, re.I)
        )
        if is_recruitment and not profile_name:
            fact_profile = {**fact_profile, "name": "recruitment"}
        company_name = str(payload.get("company_name") or agent.get("company_name") or "The Hiring Company")
        assistant_name = str(agent.get("name") or "Riya")
        defaults = self._recruitment_defaults() if is_recruitment else {}
        raw_knowledge = (
            payload.get("knowledge_profile")
            or payload.get("knowledge_documents")
            or payload.get("faqs")
            or agent.get("knowledge_profile")
            or agent.get("knowledge_documents")
            or agent.get("faqs")
            or []
        )
        knowledge_profile = self._knowledge_profile(raw_knowledge, version)
        if not knowledge_profile.get("documents"):
            knowledge_profile = self._knowledge_profile(
                self._faq_pairs_from_prompt(system), version
            )
        explicit_routing = payload.get("routing_policy") or agent.get("routing_policy") or {}
        routing_policy = self._deep_merge(defaults.get("routing_policy", {}), explicit_routing)
        routing_policy = {"model": model.get("model"), "provider": model.get("provider"), **routing_policy}
        cached_utterances = {
            **defaults.get("cached_utterances", {}),
            **(agent.get("cached_utterances") or payload.get("cached_utterances") or {}),
        }
        cached_utterances = {
            key: str(value).replace("The Hiring Company", company_name).replace("Riya", assistant_name)
            for key, value in cached_utterances.items()
        }
        explicit_flow = agent.get("flow_graph") or payload.get("flow_graph") or {}
        flow_graph = self._deep_merge(defaults.get("flow_graph", {}), explicit_flow)
        if not flow_graph:
            flow_graph = {"initial_state": "OPEN", "states": {"OPEN": {}}}
        slot_schema = self._deep_merge(
            defaults.get("slot_schema", {}), agent.get("slot_schema") or payload.get("slot_schema") or {},
        )
        base_keyterms = list(defaults.get("keyterms", []))
        base_keyterms.append(company_name)
        aliases = fact_profile.get("role_aliases") or {}
        if isinstance(aliases, dict):
            base_keyterms.extend(str(name) for name in aliases)
        configured_keyterms = list(transcriber.get("keyterms") or [])
        state_keyterms = defaults.get("state_keyterms", {})
        return AgentBundle(
            agent_id=agent_id, version=version, tenant_id=tenant_id,
            identity={
                "name": assistant_name,
                "company_name": company_name,
            }, invariant_prompt=system,
            language_profile={"primary": payload.get("primary_language", "multi")},
            flow_graph=flow_graph,
            state_schema=agent.get("state_schema") or payload.get("state_schema") or {},
            slot_schema=slot_schema,
            actions=agent.get("actions") or payload.get("actions") or {},
            risk_policy=agent.get("risk_policy") or payload.get("risk_policy") or {"class": "LOW_PUBLIC"},
            routing_policy=routing_policy,
            stt_profile={"model": transcriber.get("model", "nova-3"), "endpointing": transcriber.get("endpointing"),
                         "keyterms": list(dict.fromkeys([*configured_keyterms, *base_keyterms])),
                         "state_keyterms": state_keyterms,
                         "language_hints": transcriber.get("language_hints") or ["en", "hi"]},
            tts_profile={"model": synthesizer.get("model", "sonic-3.5"), "voice_id": synthesizer.get("voice_id")},
            knowledge_profile=knowledge_profile,
            cached_utterances=cached_utterances,
            compiled_prompt=dict(runtime_prompt),
            fact_profile=fact_profile,
            cache_policy={
                "namespace": f"{tenant_id}:{version}:{payload.get('primary_language', 'multi')}",
                **dict(payload.get("cache_policy") or agent.get("cache_policy") or {}),
            },
        )

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        result = dict(base or {})
        for key, value in dict(override or {}).items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = AgentCompiler._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    @staticmethod
    def _recruitment_defaults() -> dict[str, Any]:
        states = {
            "OPENING": {"objective": "identify the call purpose", "required_slots": ["hiring_status"], "endpoint_profile": "yes_no"},
            "HIRING_STATUS": {"objective": "learn current or future hiring status", "required_slots": ["hiring_status"], "endpoint_profile": "yes_no"},
            "REQUIREMENTS": {"objective": "collect only missing hiring facts", "required_slots": ["roles", "headcount", "hiring_timeline"], "endpoint_profile": "requirements"},
            "FOLLOWUP": {"objective": "offer a human follow-up without promising a booking", "required_slots": ["followup_consent"], "endpoint_profile": "yes_no"},
            "CALLBACK": {"objective": "record callback day and time preferences", "required_slots": ["callback_day", "callback_time"], "endpoint_profile": "short_entity"},
            "CLOSING": {"objective": "close politely", "required_slots": [], "endpoint_profile": "freeform"},
        }
        return {
            "flow_graph": {"initial_state": "OPENING", "states": states},
            "slot_schema": {
                "hiring_status": {"type": "string", "enum": ["yes", "no", "future", "unknown"]},
                "roles": {"type": "string"}, "departments": {"type": "string"},
                "headcount": {"type": "string"}, "hiring_timeline": {"type": "string"},
                "followup_consent": {"type": "string", "enum": ["yes", "no"]},
                "callback_day": {"type": "string"}, "callback_time": {"type": "string"},
                "callback_preference": {"type": "string"}, "callback_state": {"type": "string"},
            },
            "cached_utterances": {
                "greeting": "Hello, I'm Riya from The Hiring Company.",
                "ask:hiring_status": "Are you hiring now or in the next few months?",
                "ask:roles": "What roles are you planning to hire for?",
                "ask:headcount": "Roughly how many people would you need?",
                "ask:hiring_timeline": "What hiring timeline are you targeting?",
                "ask:callback_day": "What day would work best?",
                "ask:callback_time": "What time would be convenient?",
                "model-identity": "I'm an AI voice assistant for The Hiring Company.",
                "out-of-scope": "I can only help with hiring and staffing on this call.",
                "goodbye": "Thank you for your time. Goodbye.",
            },
            "routing_policy": {
                "intent_patterns": {
                    "request_human": ["speak to someone", "talk to a person", "talent acquisition manager"],
                    "faq_services": ["what services", "blue collar", "white collar", "job types"],
                    "out_of_scope": ["reverse the linked list", "technical issue"],
                }
            },
            "keyterms": [
                "The Hiring Company", "Talent Acquisition", "operations", "technology",
                "staffing", "permanent", "contract", "temporary", "apprenticeship", "NAPS", "NATS",
            ],
            "state_keyterms": {
                "REQUIREMENTS": ["operations", "technology", "engineering", "finance", "sales", "headcount"],
                "CALLBACK": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "AM", "PM"],
            },
        }

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
        if isinstance(raw, dict) and not documents and not ({"version", "documents", "records", "items"} & set(raw)):
            documents = [
                {"id": f"faq-{index + 1}", "question": question, "answer": answer}
                for index, (question, answer) in enumerate(raw.items())
            ]
        if isinstance(documents, (str, bytes)):
            documents = [documents]
        normalized: list[dict[str, Any]] = []
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
                questions = (
                    document.get("questions") or document.get("aliases")
                    or document.get("question") or []
                ) if isinstance(document, dict) else []
                if isinstance(questions, str):
                    questions = [questions]
                normalized.append({
                    "id": document_id, "text": text, "risk_class": risk,
                    "questions": [str(item) for item in questions if str(item).strip()],
                })
        profile["version"] = str(profile.get("version") or version)
        profile["documents"] = normalized
        return profile

    @staticmethod
    def _faq_pairs_from_prompt(source: str) -> list[dict[str, Any]]:
        """Extract only explicit Q/A authoring blocks; never invent knowledge."""
        pairs: list[dict[str, Any]] = []
        pattern = re.compile(
            r"(?:^|\n)\s*(?:Q(?:uestion)?\s*[:.-])\s*(?P<question>[^\n]+)\n"
            r"\s*(?:A(?:nswer)?\s*[:.-])\s*(?P<answer>[^\n]+)",
            re.I,
        )
        for index, match in enumerate(pattern.finditer(source)):
            pairs.append({
                "id": f"faq-{index + 1}",
                "questions": [match.group("question").strip()],
                "text": match.group("answer").strip(),
                "risk_class": "LOW_PUBLIC",
            })
        return pairs
