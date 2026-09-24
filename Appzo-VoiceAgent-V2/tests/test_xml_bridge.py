import base64
import json
import os
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse
from xml.etree import ElementTree

from fastapi.testclient import TestClient
from pipecat.runner.types import CallData, RunnerArguments

from bot import _call_data
from goodbox_server import app


class XmlBridgeTests(unittest.TestCase):
    def test_callback_routes_cloud_and_preserves_goodbox_metadata(self):
        with patch.dict(os.environ, {
            "PIPECAT_CLOUD_PLIVO_WS_URL": "wss://ap-south.api.pipecat.daily.co/ws/plivo?serviceHost=agent.org",
            "GOODBOX_CHATBOT_ID": "configured-agent",
        }):
            response = TestClient(app).post(
                "/v1/plivo/callback/phone", data={"CallUUID": "call", "From": "123", "To": "456"}
            )
        self.assertEqual(response.status_code, 200)
        stream = ElementTree.fromstring(response.text).find("Stream")
        self.assertEqual(stream.attrib["bidirectional"], "true")
        query = parse_qs(urlparse(stream.text).query)
        self.assertEqual(query["serviceHost"], ["agent.org"])
        body = json.loads(base64.b64decode(query["body"][0]))
        args = RunnerArguments(body=body, call_data=CallData(call_id="call", stream_id="stream"))
        data = _call_data(args)
        self.assertEqual(data["from"], "123")
        self.assertEqual(data["to"], "456")
        self.assertEqual(data["phone_id"], "phone")
        self.assertEqual(data["chatbot_id"], "configured-agent")
        self.assertEqual(data["stream_id"], "stream")

    def test_local_route_unchanged(self):
        with patch.dict(os.environ, {"PIPECAT_CLOUD_PLIVO_WS_URL": "", "PUBLIC_BASE_URL": "https://example.ngrok.app"}):
            response = TestClient(app).post("/v1/plivo/callback/phone?chatbot_id=explicit")
        url = urlparse(ElementTree.fromstring(response.text).find("Stream").text)
        self.assertEqual(url.netloc, "example.ngrok.app")
        self.assertEqual(url.path, "/v1/plivo/ws")

    def test_http_start_url_rejected(self):
        with patch.dict(os.environ, {"PIPECAT_CLOUD_PLIVO_WS_URL": "https://api.pipecat.daily.co/v1/public/agent/start"}):
            with self.assertRaises(ValueError):
                TestClient(app).post("/v1/plivo/callback/phone")
