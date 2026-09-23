import asyncio
import json
import os
import unittest
from unittest.mock import patch

import httpx

from goodbox_server import GoodboxApi


class GoodboxTranscriptCorrelationTests(unittest.TestCase):
    def test_call_stop_forwards_non_secret_call_correlation(self):
        received = []

        async def exercise():
            async def handler(request):
                received.append(json.loads(request.content))
                return httpx.Response(201, json={"data": {"id": "history-id", "call_id": "call-id"}})

            api = GoodboxApi()
            await api._client.aclose()
            api._client = httpx.AsyncClient(
                transport=httpx.MockTransport(handler), base_url="https://goodbox.test"
            )
            try:
                result = await api.call_stop(
                    "stream-id",
                    "voice-call-id",
                    [{"role": "user", "content": "hello"}],
                    call_data={
                        "call_id": "call-id",
                        "phone_id": "phone-id",
                        "chatbot_id": "chatbot-id",
                        "provider": "plivo",
                        "custom_variables": {"source": "test"},
                    },
                )
            finally:
                await api.close()
            return result

        with patch.dict(os.environ, {"GOODBOX_API_BASE_URL": "https://goodbox.test/v1"}):
            result = asyncio.run(exercise())
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0]["call_id"], "call-id")
        self.assertEqual(received[0]["phone_id"], "phone-id")
        self.assertEqual(received[0]["chatbot_id"], "chatbot-id")
        self.assertEqual(received[0]["provider"], "plivo")
        self.assertEqual(result["id"], "history-id")

    def test_call_stop_retries_legacy_payload_after_schema_rejection(self):
        received = []

        async def exercise():
            async def handler(request):
                payload = json.loads(request.content)
                received.append(payload)
                if "call_id" in payload:
                    return httpx.Response(422, json={"detail": "unexpected field"})
                return httpx.Response(201, json={"data": {"id": "history-id"}})

            api = GoodboxApi()
            await api._client.aclose()
            api._client = httpx.AsyncClient(
                transport=httpx.MockTransport(handler), base_url="https://goodbox.test"
            )
            try:
                return await api.call_stop(
                    "stream-id", "voice-call-id", [], call_data={"call_id": "call-id"}
                )
            finally:
                await api.close()

        with patch.dict(os.environ, {"GOODBOX_API_BASE_URL": "https://goodbox.test/v1"}):
            asyncio.run(exercise())
        self.assertEqual(len(received), 2)
        self.assertIn("call_id", received[0])
        self.assertNotIn("call_id", received[1])
