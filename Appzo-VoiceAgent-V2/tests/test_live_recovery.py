import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from main import StreamingFAQController


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.controller = StreamingFAQController("test", client=AsyncMock())
        self.controller._speak_fixed = AsyncMock()
        self.controller._run_llm = AsyncMock()
        await self.controller._start_turn()

    async def asyncTearDown(self):
        await self.controller._cancel_turn_work(self.controller._state)
        await asyncio.sleep(0)

    async def recover(self, state):
        with patch("main.asyncio.sleep", new=AsyncMock()):
            await self.controller._wait_for_final_transcript(state)

    async def test_empty_turn_asks_for_repeat(self):
        state = self.controller._state
        await self.recover(state)
        self.controller._speak_fixed.assert_awaited_once()
        self.assertTrue(state.committed)
        self.controller._run_llm.assert_not_awaited()

    async def test_interim_recovery_runs_llm(self):
        state = self.controller._state
        state.latest_interim = "hello"
        await self.recover(state)
        await state.final_request.task
        self.controller._run_llm.assert_awaited_once()
        self.controller._speak_fixed.assert_not_awaited()

    async def test_old_turn_cannot_speak_after_barge_in(self):
        state = self.controller._state
        state.turn_stopped = True
        await self.controller._start_turn()
        await self.recover(state)
        self.controller._speak_fixed.assert_not_awaited()

    async def test_duplicate_stop_keeps_one_recovery_task(self):
        await self.controller._on_turn_stopped()
        task = self.controller._state.final_wait_task
        await self.controller._on_turn_stopped()
        self.assertIs(task, self.controller._state.final_wait_task)

    async def test_empty_final_does_not_erase_text(self):
        await self.controller._on_final_transcript("hello")
        await self.controller._on_final_transcript("")
        self.assertEqual(self.controller._state.final_transcript, "hello")
