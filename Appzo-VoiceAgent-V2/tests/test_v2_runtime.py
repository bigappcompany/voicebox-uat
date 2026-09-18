import unittest
from unittest.mock import patch

from voice_agent.agents.bundle import AgentBundle
from voice_agent.agents.compiler import AgentCompiler
from voice_agent.agents.registry import AgentBundleRegistry
from voice_agent.knowledge.index import KnowledgeRecord, TenantKnowledgeIndex
from voice_agent.llm.prompt_builder import PromptBuilder
from voice_agent.runtime.latency_controller import LatencyController
from voice_agent.runtime.response_plan import ResponsePlan
from voice_agent.runtime.session import CallSession
from voice_agent.runtime.bootstrap import RuntimeBootstrap
from voice_agent.runtime.flags import RuntimeFlags
from voice_agent.speech.audio_commit import AudioCommitter, SpeculativeAudioCandidate
from voice_agent.speech.safe_chunker import SafeSpeechChunker
from voice_agent.flows.slots import SlotValidator
from voice_agent.flows.engine import FlowEngine
from voice_agent.flows.facts import NumericRange


def bundle(*, risk="LOW_PUBLIC"):
    return AgentBundle("agent", "v2", "tenant-a", invariant_prompt="Be concise.", risk_policy={"class": risk}, cached_utterances={"goodbye": "Bye."})


class V2RuntimeTests(unittest.TestCase):
    def test_goodbox_compiler_emits_immutable_runtime_bundle(self):
        compiled = AgentCompiler().compile_goodbox({"chatbot_id": "a", "tenant_id": "t", "prompt": "hello", "call_agent": {}})
        self.assertEqual((compiled.agent_id, compiled.tenant_id, compiled.invariant_prompt), ("a", "t", "hello"))
        self.assertEqual(compiled.identity["company_name"], "The Hiring Company")

    def test_call_start_compiles_and_namespaces_goodbox_knowledge(self):
        async def run():
            bootstrap = RuntimeBootstrap(AgentBundleRegistry(), AgentCompiler())
            runtime = await bootstrap.start_call(
                {"call_id": "c"},
                {
                    "chatbot_id": "a",
                    "tenant_id": "t",
                    "knowledge_documents": [{"id": "fees", "content": "Contract staffing fees depend on the role."}],
                },
            )
            return runtime.session
        import asyncio
        session = asyncio.run(run())
        records = session.knowledge_index.search(
            tenant_id="t", agent_id="a", knowledge_version=session.agent.knowledge_version, query="contract fees"
        )
        self.assertEqual([record.document_id for record in records], ["fees"])

    def test_bootstrap_compiles_once_at_call_start(self):
        async def run():
            registry = AgentBundleRegistry(); runtime = await RuntimeBootstrap(registry, AgentCompiler()).start_call({"call_id": "c"}, {"chatbot_id": "a", "tenant_id": "t"})
            return runtime, await registry.get("a")
        import asyncio
        runtime, saved = asyncio.run(run())
        self.assertEqual((runtime.session.call_id, saved.tenant_id), ("c", "t"))

    def test_requested_v2_improvements_default_on_with_provider_swaps_off(self):
        flags = RuntimeFlags()
        self.assertTrue(flags.enable_agent_bundle)
        self.assertTrue(flags.enable_spec_tts)
        self.assertTrue(flags.enable_structured_facts)
        self.assertTrue(flags.enable_semantic_spec_reuse)
        self.assertFalse(flags.enable_flux)
        self.assertFalse(flags.enable_local_llm)

    def test_goodbox_flux_alias_uses_the_v2_multilingual_flux_model(self):
        from goodbox_server import _runtime_from_goodbox

        config = {
            "model_config": {"provider": "azure", "model": "gpt-4.1-mini"},
            "transcriber_config": {"provider": "deepgram", "model": "flux"},
            "synthesizer_config": {"provider": "cartesia", "voice_id": "voice"},
            "call_agent": {},
        }
        with patch("goodbox_server._required", return_value="test-key"), patch(
            "goodbox_server._shared_llm_client", return_value=object()
        ):
            runtime = _runtime_from_goodbox(config)
        self.assertEqual(runtime.stt_model, "flux-general-multi")

    def test_index_cannot_return_another_tenant_document(self):
        index = TenantKnowledgeIndex(); index.add(KnowledgeRecord("tenant-b", "agent", "v2", "secret", "private rate"))
        self.assertEqual(index.search(tenant_id="tenant-a", agent_id="agent", knowledge_version="v2", query="rate"), [])

    def test_chunker_waits_for_safe_boundary_and_flushes(self):
        chunker = SafeSpeechChunker(min_chars=10, min_words=2, max_wait_ms=10000)
        self.assertEqual(chunker.push("Your rate is ", now=1), [])
        self.assertEqual(chunker.push("five percent, subject to policy. Next", now=1.1), ["Your rate is five percent, subject to policy."])
        self.assertEqual(chunker.flush(), ["Next"])

    def test_high_risk_audio_never_commits(self):
        plan = ResponsePlan("llm", risk_class="HIGH_PERSONAL", allow_speculative_audio=True)
        candidate = SpeculativeAudioCandidate("same", "hello", "hello", [b"pcm"])
        self.assertIsNone(AudioCommitter().commit(candidate, final_fingerprint="same", plan=plan)); self.assertTrue(candidate.invalidated)

    def test_semantic_fingerprint_commits_but_material_slot_change_does_not(self):
        agent = bundle(); session = CallSession("call", "tenant-a", agent, slots={"product": "loan"})
        controller = LatencyController(session, intent=lambda _: "loan_faq")
        controller.start_turn(); controller.interim("what is loan rate"); self.assertIsNotNone(controller.soft_eot(agent))
        controller.prepare_audio("The rate depends on policy.", [b"pcm"])
        self.assertEqual(controller.hard_eot("what is the loan interest rate", agent), [b"pcm"])
        controller.start_turn(); controller.interim("what is loan rate"); controller.soft_eot(agent); controller.prepare_audio("text", [b"pcm"])
        session.slots["product"] = "mortgage"
        self.assertIsNone(controller.hard_eot("what is loan rate", agent))

    def test_new_turn_aborts_old_speculative_audio(self):
        agent = bundle(); controller = LatencyController(CallSession("call", "tenant-a", agent))
        controller.start_turn(); controller.interim("hello"); controller.soft_eot(agent); old = controller.prepare_audio("Hi", [b"pcm"])
        controller.start_turn(); self.assertTrue(old.invalidated)

    def test_prompt_is_bounded_and_excludes_irrelevant_bundle_shape(self):
        agent = bundle(); agent = AgentBundle(**{**agent.__dict__, "actions": {"ASK": {}}, "flow_graph": {"huge": "ignored"}})
        messages = PromptBuilder(max_knowledge_chars=10).build(agent=agent, state={"name": "OPEN"}, slots={}, route=ResponsePlan("llm"), knowledge=["x" * 50], history=[], user_text="hello")
        self.assertEqual(messages[-1]["content"], "hello"); self.assertLessEqual(len(messages[1]["content"]), len("RELEVANT KNOWLEDGE:\n") + 10)

    def test_prompt_builder_serializes_structured_slot_values_safely(self):
        agent = bundle()
        slots = {
            "headcount": NumericRange(3, 4, "people", True),
            "hiring_timeline": NumericRange(3, 4, "months"),
            "roles": ["operations"],
            "headcount_by_role": {"operations": NumericRange(3, 4, "people", True)},
            "tags": {"urgent", "remote"},
        }
        messages = PromptBuilder().build(
            agent=agent, state={"name": "OPEN"}, slots=slots,
            route=ResponsePlan("llm"), knowledge=[], history=[], user_text="hello",
        )
        self.assertIn('"headcount":"about 3-4 people"', messages[0]["content"])
        self.assertIn('"hiring_timeline":"3-4 months"', messages[0]["content"])
        self.assertIn('"roles":["operations"]', messages[0]["content"])


    def test_slot_validator_enforces_compiled_type_bounds_and_enums(self):
        validator = SlotValidator()
        self.assertEqual(validator.validate({"type": "integer", "minimum": 2}, "4").value, 4)
        self.assertFalse(validator.validate({"type": "integer", "minimum": 2}, "one").valid)
        self.assertFalse(validator.validate({"enum": ["yes", "no"]}, "maybe").valid)

    def test_flow_plan_keeps_state_transition_and_slot_update_deterministic(self):
        plan = FlowEngine().plan(
            {"states": {"OPEN": {"transitions": {"qualified": "REQUIREMENTS"}, "actions": {"qualified": "continue"}, "slot_updates": {"qualified": {"qualified": "yes"}}}}},
            "OPEN",
            "qualified",
            {},
            risk_class="LOW_WORKFLOW",
        )
        self.assertEqual((plan.next_state, plan.slots_written), ("REQUIREMENTS", {"qualified": "yes"}))


if __name__ == "__main__": unittest.main()
