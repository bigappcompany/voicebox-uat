import unittest

from voice_agent.agents.compiler import AgentCompiler
from voice_agent.flows.callbacks import CallbackCoordinator
from voice_agent.flows.facts import FactExtractor, NumericRange
from voice_agent.knowledge.index import KnowledgeRecord, TenantKnowledgeIndex
from voice_agent.routing.deterministic import DeterministicRouter
from voice_agent.runtime.intents import CanonicalIntentModel
from voice_agent.runtime.session import CallSession


class V2CompletionTests(unittest.TestCase):
    def setUp(self):
        self.bundle = AgentCompiler().compile_goodbox({
            "chatbot_id": "recruiter",
            "tenant_id": "tenant",
            "company_name": "Example Staffing",
            "prompt": "You are Riya, a hiring and staffing voice assistant.",
            "faqs": [{
                "id": "faq-services",
                "question": "do you support blue collar roles",
                "answer": "We support both blue-collar and white-collar hiring.",
            }],
        })
        self.session = CallSession("call", "tenant", self.bundle)
        self.facts = FactExtractor(default_profile="recruitment")
        self.intents = CanonicalIntentModel()
        self.router = DeterministicRouter()
        self.callbacks = CallbackCoordinator()

    def _turn(self, text, turn_id):
        update = self.facts.extract(
            text, self.session.visible_facts(),
            pending_question=self.session.pending_question, source_turn=turn_id,
        )
        self.session.slots.update(update.values)
        self.session.facts.update(update.records)
        match = self.intents.classify(
            text, pending_question=self.session.pending_question,
            fact_values=self.session.visible_facts(),
            configured_patterns=self.bundle.routing_policy.get("intent_patterns"),
        )
        routed = self.callbacks.route(match, self.session.visible_facts(), turn_id=turn_id)
        if routed is None:
            routed = self.router.route(
                text, self.bundle, intent=match, slots=self.session.visible_facts(),
                pending_question=self.session.pending_question, turn_id=turn_id,
            )
        self.assertIsNotNone(routed)
        plan, speech = routed
        self.session.slots.update(plan.slots_written)
        if plan.clear_pending_question:
            self.session.pending_question = None
        if plan.pending_question:
            self.session.pending_question = plan.pending_question
        if plan.next_state:
            self.session.state["name"] = plan.next_state
        return update, match, plan, speech

    def test_replay_preserves_requirements_and_never_reasks_known_fields(self):
        update, _, _, speech = self._turn(
            "not right now, in the next three four months", 1
        )
        self.assertEqual(update.values["hiring_status"], "future")
        self.assertEqual(update.values["hiring_timeline"], NumericRange(3, 4, "months"))
        self.assertIn("roles", speech.casefold())

        _, _, _, speech = self._turn("operation check", 2)
        self.assertIn("how many", speech.casefold())
        _, match, _, speech = self._turn("about", 3)
        self.assertEqual(match.intent_id, "incomplete_response")
        self.assertIn("how many", speech.casefold())

        self._turn("let's say five people", 4)
        facts = self.session.visible_facts()
        self.assertEqual(facts["roles"], ["operations"])
        self.assertEqual(facts["headcount"], 5)
        self.assertEqual(facts["hiring_timeline"], NumericRange(3, 4, "months"))
        self.assertEqual(self.session.pending_question.slot, "followup_consent")

    def test_ranges_approximations_and_corrections_keep_provenance(self):
        first = self.facts.extract(
            "around three to four people", {},
            pending_question=type("Pending", (), {"slot": "headcount"})(), source_turn=3,
        )
        self.assertEqual(first.values["headcount"], NumericRange(3, 4, "people", True))
        second = self.facts.extract("actually six", first.values, source_turn=4)
        self.assertEqual(second.values["headcount"], 6)
        self.assertEqual(second.records["headcount"].corrected_from, first.values["headcount"])
        self.assertEqual(second.records["headcount"].source_turn, 4)

    def test_callback_collects_only_missing_fields_and_records_preference(self):
        self.session.slots["callback_state"] = self.callbacks.FOLLOWUP_OFFERED
        from voice_agent.runtime.session import PendingQuestion
        self.session.pending_question = PendingQuestion(
            "ask_callback_consent", "followup_consent", "boolean", 1
        )
        self._turn("yep", 2)
        self._turn("the day after tomorrow", 3)
        self.assertEqual(self.session.pending_question.slot, "callback_time")
        _, _, plan, speech = self._turn("three pm", 4)
        self.assertEqual(plan.slots_written["callback_state"], "PREFERENCE_RECORDED")
        self.assertIn("not a confirmed booking", speech.casefold())
        self.assertIsNone(self.session.pending_question)

    def test_compiler_populates_flow_cache_knowledge_and_keyterms(self):
        self.assertEqual(self.bundle.flow_graph["initial_state"], "OPENING")
        self.assertIn("ask:headcount", self.bundle.cached_utterances)
        self.assertIn("Example Staffing", self.bundle.stt_profile["keyterms"])
        self.assertIn("REQUIREMENTS", self.bundle.stt_profile["state_keyterms"])
        self.assertEqual(self.bundle.knowledge_profile["documents"][0]["id"], "faq-services")
        self.assertTrue(self.bundle.cache_policy["namespace"].startswith("tenant:"))

    def test_tenant_knowledge_returns_confident_exact_alias(self):
        index = TenantKnowledgeIndex()
        document = self.bundle.knowledge_profile["documents"][0]
        index.add(KnowledgeRecord(
            "tenant", "recruiter", self.bundle.knowledge_version,
            document["id"], document["text"], questions=tuple(document["questions"]),
        ))
        match = index.search_matches(
            tenant_id="tenant", agent_id="recruiter",
            knowledge_version=self.bundle.knowledge_version,
            query="do you support blue collar roles",
        )[0]
        self.assertEqual(match.confidence, 1.0)
        self.assertEqual(match.reason, "exact_alias")


if __name__ == "__main__":
    unittest.main()
