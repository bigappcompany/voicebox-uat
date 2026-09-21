import unittest
from voice_agent.flows.facts import FactExtractor, NumericRange
from voice_agent.runtime.intents import CanonicalIntentModel
from voice_agent.routing.deterministic import DeterministicRouter
from voice_agent.flows.callbacks import CallbackCoordinator
from voice_agent.agents.bundle import AgentBundle
from voice_agent.runtime.session import PendingQuestion


class TestUserRequestedCases(unittest.TestCase):
    def setUp(self):
        self.fe = FactExtractor(default_profile="recruitment")
        self.im = CanonicalIntentModel()
        self.bundle = AgentBundle(
            "test-agent", "1.0", "test-tenant",
            cached_utterances={
                "ask:hiring_status": "Are you hiring now or in the next few months?",
                "ask:roles": "What roles are you planning to hire for?",
                "ask:headcount": "Roughly how many people would you need?",
                "ask:hiring_timeline": "What hiring timeline are you targeting?",
                "model-identity": "I'm an AI voice assistant for this hiring call.",
                "out-of-scope": "I can only help with hiring and staffing on this call.",
                "out-of-scope:repeat": "As mentioned, I can only assist with hiring and staffing inquiries. If you don't have hiring needs right now, I can follow up later or end the call.",
                "goodbye": "Thank you for your time. Goodbye.",
            },
        )
        self.router = DeterministicRouter()
        self.callbacks = CallbackCoordinator()

    def test_case_1_not_right_now_next_three_four_months(self):
        update = self.fe.extract("not right now, next three four months", {})
        self.assertEqual(update.values["hiring_status"], "future")
        self.assertEqual(update.values["hiring_timeline"], NumericRange(3, 4, "months"))

    def test_case_2_operation_check(self):
        update = self.fe.extract("operation check", {})
        self.assertEqual(update.values["roles"], ["operations"])
        match = self.im.classify("operation check", current_facts=update.values)
        self.assertEqual(match.intent_id, "provide_role")

    def test_case_3_about_after_headcount_incomplete(self):
        pq = PendingQuestion("ask_headcount", "headcount", "integer", 1)
        match = self.im.classify("about", pending_question=pq)
        self.assertEqual(match.intent_id, "incomplete_response")
        res = self.router.route("about", self.bundle, intent=match, pending_question=pq, slots={"roles": ["operations"]})
        self.assertIsNotNone(res)
        plan, speech = res
        self.assertEqual(plan.next_state, "REQUIREMENTS")
        self.assertEqual(plan.pending_question.slot, "headcount")
        self.assertIn("how many", speech.casefold())

    def test_case_4_five_people(self):
        update = self.fe.extract("five people", {})
        self.assertEqual(update.values["headcount"], 5)

    def test_case_5_what_kind_of_preference_clarification(self):
        slots = {"callback_state": "AWAITING_DAY_TIME", "roles": ["operations"], "headcount": 5}
        pq = PendingQuestion("ask_callback_day_time", "callback_preference", "date_and_time", 1)
        match = self.im.classify("what kind of preference?", pending_question=pq, fact_values=slots)
        self.assertEqual(match.intent_id, "clarification")
        res = self.callbacks.route(match, slots, turn_id=2)
        self.assertIsNotNone(res)
        plan, speech = res
        self.assertEqual(plan.next_state, "CALLBACK")
        self.assertIn("preferred day and time", speech.casefold())
        self.assertEqual(slots["roles"], ["operations"])
        self.assertEqual(slots["headcount"], 5)

    def test_case_6_yes_after_hiring_question(self):
        pq = PendingQuestion("ask_hiring_status", "hiring_status", "boolean", 1)
        match = self.im.classify("yes", pending_question=pq)
        self.assertEqual(match.intent_id, "provide_hiring_status")
        self.assertEqual(match.slots.get("hiring_status"), "yes")

    def test_case_7_three_four_people_in_operations(self):
        update = self.fe.extract("three four people in operations", {})
        self.assertEqual(update.values["roles"], ["operations"])
        self.assertEqual(update.values["headcount"], NumericRange(3, 4, "people", True))

    def test_case_8_in_three_to_four_months(self):
        current = {"hiring_timeline": NumericRange(3, 4, "months")}
        update = self.fe.extract("in three to four months", current)
        self.assertEqual(update.values["hiring_timeline"], NumericRange(3, 4, "months"))

    def test_case_9_speak_to_someone(self):
        match = self.im.classify("I'd like to speak to someone")
        self.assertEqual(match.intent_id, "request_human")

    def test_case_10_the_day_after_tomorrow(self):
        match = self.im.classify("the day after tomorrow")
        self.assertEqual(match.intent_id, "callback_day")
        self.assertIn("day after tomorrow", match.slots["callback_day"])

    def test_case_11_three_pm(self):
        match = self.im.classify("three pm")
        self.assertEqual(match.intent_id, "callback_time")
        self.assertEqual(match.slots["callback_time"], "three pm")

    def test_case_12_reverse_the_linked_list_oos(self):
        match = self.im.classify("reverse the linked list")
        self.assertEqual(match.intent_id, "out_of_scope")

    def test_case_13_repeated_oos(self):
        match = self.im.classify("reverse the linked list")
        slots = {}
        res1 = self.router.route("reverse the linked list", self.bundle, intent=match, slots=slots)
        self.assertIsNotNone(res1)
        plan1, speech1 = res1
        self.assertNotIn("Are you hiring now", speech1)
        self.assertIn("I can only help with hiring", speech1)
        slots.update(plan1.slots_written)
        self.assertEqual(slots.get("oos_count"), 1)

        res2 = self.router.route("debug my python code", self.bundle, intent=match, slots=slots)
        self.assertIsNotNone(res2)
        plan2, speech2 = res2
        self.assertNotIn("Are you hiring now", speech2)
        self.assertIn("As mentioned", speech2)
        self.assertEqual(plan2.slots_written.get("oos_count"), 2)

    def test_case_14_what_model_do_you_use(self):
        match = self.im.classify("what model do you use?")
        self.assertEqual(match.intent_id, "model_identity")
        res = self.router.route("what model do you use?", self.bundle, intent=match)
        self.assertIsNotNone(res)
        plan, speech = res
        self.assertIn("AI voice assistant", speech)

    def test_case_15_ill_talk_to_you_later(self):
        match = self.im.classify("I'll talk to you later")
        self.assertEqual(match.intent_id, "busy")

    def test_case_16_yep_callback_consent(self):
        pq = PendingQuestion("ask_callback_consent", "followup_consent", "boolean", 1)
        match = self.im.classify("yep", pending_question=pq)
        self.assertEqual(match.intent_id, "callback_consent_yes")

    def test_case_17_seven_pm(self):
        match = self.im.classify("seven pm")
        self.assertEqual(match.intent_id, "callback_time")
        self.assertEqual(match.slots["callback_time"], "seven pm")

    def test_case_18_today(self):
        match = self.im.classify("today")
        self.assertEqual(match.intent_id, "callback_day")
        self.assertEqual(match.slots["callback_day"], "today")

    def test_case_19_goodbye(self):
        match = self.im.classify("goodbye")
        self.assertEqual(match.intent_id, "goodbye")
        res = self.router.route("goodbye", self.bundle, intent=match)
        self.assertIsNotNone(res)
        plan, speech = res
        self.assertEqual(plan.action, "end_call")
        self.assertIn("Goodbye", speech)

    def test_callback_preference_can_be_corrected_after_recording(self):
        slots = {
            "callback_state": self.callbacks.PREFERENCE_RECORDED,
            "callback_day": "tomorrow",
            "callback_time": "five pm",
            "callback_preference": "tomorrow at five pm",
        }
        match = self.im.classify("make it Wednesday at five pm", fact_values=slots)
        plan, speech = self.callbacks.route(match, slots, turn_id=3)
        self.assertEqual(plan.slots_written["callback_day"], "wednesday")
        self.assertEqual(plan.slots_written["callback_preference"], "wednesday at five pm")
        self.assertIn("wednesday", speech.casefold())

    def test_latest_day_wins_inside_correction_utterance(self):
        pending = PendingQuestion("ask_callback_day_time", "callback_preference", "date_and_time", 1)
        match = self.im.classify(
            "tomorrow, actually no, make it Wednesday at five pm",
            pending_question=pending,
        )
        self.assertEqual(match.slots["callback_day"], "wednesday")


if __name__ == "__main__":
    unittest.main()
