"""Goodbox-controlled Plivo media endpoint for the shared Pipecat agent.

Goodbox remains the source of truth for the phone number, call initiation,
prompt, behavior, and service settings. This service only handles the media
webhook that Goodbox's Plivo number reaches after a call is placed.
"""

import base64
import os
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode, urlparse, urlunparse
from xml.sax.saxutils import escape

import httpx
from fastapi import FastAPI, Form, Query, Response, WebSocket
from loguru import logger
from openai import AsyncAzureOpenAI, AsyncOpenAI

from main import AgentRuntimeConfig, run_bot
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.plivo import PlivoFrameSerializer
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)


CALL_START_PATH = "/voice-calls/call-start"
CALL_STOP_PATH = "/voice-calls/call-stop"


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value.rstrip("/")


def _float(value: Any, default: float, low: float, high: float) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default


def _websocket_url(public_base_url: str, path: str, query: dict[str, str]) -> str:
    parsed = urlparse(public_base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunparse(
        (scheme, parsed.netloc, path, "", urlencode(query), "")
    )


def _goodbox_prompt(data: dict[str, Any]) -> str:
    """Preserve Goodbox instructions while adding the controller's wire format."""
    parts = [
        str(data.get("system_prompt") or "").strip(),
        str(data.get("prompt") or "").strip(),
    ]
    prompt = "\n\n".join(part for part in parts if part)
    if not prompt:
        prompt = "You are a concise, helpful telephone voice assistant."
    return f"""{prompt}

V1 LANGUAGE OVERRIDE
For this V1 telephone runtime, support English, Hindi, and natural Hinglish.
This overrides any earlier instruction that limits the conversation to English
only. Understand questions spoken in Hindi or Hinglish and answer in the same
language style as the caller. Keep business names, product names, and technical
terms in English when that sounds more natural.

VOICE OUTPUT CONTRACT
Follow all instructions and guardrails above. For every response that should be
spoken while the call should continue, output exactly `OK|` followed by the
spoken text. When the caller asks to end the call, says goodbye, or confirms
that no further help is needed, output exactly `END|` followed by one short
spoken closing. `END|` is how V1 performs the earlier `end_call` instruction.
Do not output markdown or explain this contract. If you must refuse, put the
refusal text after `OK|`.
"""


def _runtime_from_goodbox(data: dict[str, Any]) -> AgentRuntimeConfig:
    model_config = data.get("model_config") or {}
    transcriber = data.get("transcriber_config") or {}
    synthesizer = data.get("synthesizer_config") or {}
    agent = data.get("call_agent") or {}

    stt_provider = str(transcriber.get("provider") or "deepgram").lower()
    tts_provider = str(synthesizer.get("provider") or "cartesia").lower()
    llm_provider = str(model_config.get("provider") or "azure").lower()
    if stt_provider != "deepgram":
        raise ValueError(
            f"This test agent currently supports Goodbox Deepgram STT, not {stt_provider!r}"
        )
    if tts_provider != "cartesia":
        raise ValueError(
            f"This test agent currently supports Goodbox Cartesia TTS, not {tts_provider!r}"
        )

    if llm_provider == "azure":
        resource = _required("AZURE_LLM_RESOURCE_NAME")
        llm_key = _required("AZURE_LLM_API_KEY")
        llm_client = AsyncAzureOpenAI(
            api_key=llm_key,
            azure_endpoint=f"https://{resource}.openai.azure.com",
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        )
    elif llm_provider in {"groq", "openai"}:
        # Retain an escape hatch for the existing browser-test providers.
        env_key = "GROQ_API_KEY" if llm_provider == "groq" else "OPENAI_API_KEY"
        llm_key = _required(env_key)
        llm_client = AsyncOpenAI(
            api_key=llm_key,
            base_url=("https://api.groq.com/openai/v1" if llm_provider == "groq" else None),
        )
    else:
        raise ValueError(f"Unsupported Goodbox LLM provider: {llm_provider!r}")

    primary_language = str(data.get("primary_language") or "").upper()
    stt_language = "multi" if primary_language in {"", "ENGLISH", "HINDI"} else "multi"
    requested_stt_model = str(transcriber.get("model") or "nova-3").strip()
    # Goodbox currently returns the generic alias "flux". V1 uses Deepgram's
    # Nova /v1/listen service and local Smart Turn detection; sending a Flux
    # model name to that endpoint makes Deepgram reject the WebSocket with 400.
    # Keep V1 on its original turn architecture and use the supported Nova-3
    # multilingual model for this Goodbox alias.
    stt_model = (
        "nova-3" if requested_stt_model.lower().startswith("flux") else requested_stt_model
    )
    if stt_model != requested_stt_model:
        logger.warning(
            "Goodbox requested STT model {!r}; V1 is using {!r} with its "
            "Nova /v1/listen pipeline",
            requested_stt_model,
            stt_model,
        )
    voice_id = str(synthesizer.get("voice_id") or os.getenv("CARTESIA_VOICE_ID", "")).strip()
    if not voice_id:
        raise ValueError("Goodbox synthesizer_config.voice_id or CARTESIA_VOICE_ID is required")

    refusal = str(agent.get("guardrail_refusal_message") or "Sorry, I can't help with that request.")
    return AgentRuntimeConfig(
        llm_api_key=llm_key,
        llm_model=str(model_config.get("model") or "gpt-4.1-mini"),
        llm_max_tokens=_int(agent.get("max_completion_tokens"), 100, 16, 1024),
        system_prompt=_goodbox_prompt(data),
        deepgram_api_key=_required("DEEPGRAM_API_KEY"),
        stt_model=stt_model,
        stt_language=stt_language,
        deepgram_endpointing_ms=_int(transcriber.get("endpointing"), 300, 50, 2000),
        cartesia_api_key=_required("CARTESIA_API_KEY"),
        cartesia_voice_id=voice_id,
        cartesia_model=str(synthesizer.get("model") or "sonic-3.5"),
        cartesia_speed=_float(synthesizer.get("speed"), 1.0, 0.5, 2.0),
        vad_confidence=_float(agent.get("confidence"), 0.75, 0.0, 1.0),
        vad_start_secs=_float(agent.get("start_secs"), 0.3, 0.05, 2.0),
        vad_stop_secs=_float(agent.get("stop_secs"), 0.7, 0.05, 3.0),
        vad_min_volume=_float(agent.get("min_volume"), 0.6, 0.0, 1.0),
        intro_message=str(agent.get("intro_message") or "").strip() or None,
        enforce_apple_scope=False,
        llm_client=llm_client,
        refusal_message=refusal,
        operational_error_message="Sorry, I couldn't process that request just now. Please try again.",
    )


class GoodboxApi:
    def __init__(self) -> None:
        self._base_url = _required("GOODBOX_API_BASE_URL")
        self._client = httpx.AsyncClient(timeout=30)

    async def close(self) -> None:
        await self._client.aclose()

    async def call_start(self, call_data: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "stream_id": call_data.get("stream_id"),
            "call_id": call_data.get("call_id"),
            "from": call_data.get("from"),
            "to": call_data.get("to"),
            "phone_id": call_data.get("phone_id"),
            "provider": "plivo",
            "voice_call_id": call_data.get("voice_call_id"),
            "chatbot_id": call_data.get("chatbot_id"),
            "custom_variables": call_data.get("custom_variables") or {},
            "inbox_key": os.getenv("GOODBOX_INBOX_KEY", "voice_standard_inbox"),
        }
        response = await self._client.post(f"{self._base_url}{CALL_START_PATH}", json=payload)
        response.raise_for_status()
        body = response.json()
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, dict):
            raise ValueError("Goodbox CALL_START returned no configuration data")
        return data

    async def call_stop(
        self, stream_id: str | None, voice_call_id: str | None, messages: list[dict[str, str]]
    ) -> None:
        payload = {
            "stream_id": stream_id,
            "voice_call_id": voice_call_id,
            "messages": messages,
            "inbox_key": os.getenv("GOODBOX_INBOX_KEY", "voice_standard_inbox"),
        }
        response = await self._client.post(f"{self._base_url}{CALL_STOP_PATH}", json=payload)
        response.raise_for_status()


@dataclass
class CallTranscript:
    stream_id: str | None
    voice_call_id: str | None
    messages: list[dict[str, str]] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        content = (content or "").strip()
        if content:
            self.messages.append({"role": role, "content": content})


app = FastAPI(title="Goodbox Plivo Voice Agent")


@app.on_event("startup")
async def startup() -> None:
    _required("GOODBOX_API_BASE_URL")
    _required("PUBLIC_BASE_URL")
    logger.info("Goodbox Plivo voice server ready")


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/v1/plivo/callback/{phone_id}")
async def plivo_callback(
    phone_id: str,
    CallUUID: str | None = Form(None),
    From: str | None = Form(None),
    To: str | None = Form(None),
) -> Response:
    public_base_url = _required("PUBLIC_BASE_URL")
    body = {
        "phone_id": phone_id,
        "call_id": CallUUID,
        "from": From,
        "to": To,
        "provider": "plivo",
    }
    import json

    encoded = base64.urlsafe_b64encode(json.dumps(body).encode()).decode()
    stream_url = _websocket_url(public_base_url, "/v1/plivo/ws", {"body": encoded})
    xml = f"""<Response><Stream streamTimeout=\"3600\" keepCallAlive=\"true\" bidirectional=\"true\" contentType=\"audio/x-mulaw;rate=8000\">{escape(stream_url)}</Stream></Response>"""
    return Response(content=xml, media_type="application/xml")


@app.websocket("/v1/plivo/ws")
async def plivo_media(websocket: WebSocket, body: str = Query("")) -> None:
    #logger.log("Received plivo ws connection")
    await websocket.accept()
    try:
        import json

        callback_data = json.loads(base64.urlsafe_b64decode(body.encode()).decode()) if body else {}
        _, parsed_call = await parse_telephony_websocket(websocket)
        call_data = parsed_call.model_dump(by_alias=True) if hasattr(parsed_call, "model_dump") else dict(parsed_call)
        call_data.update(callback_data)
        call_data["provider"] = "plivo"

        goodbox = GoodboxApi()
        try:
            config = await goodbox.call_start(call_data)
            phone_provider = config.get("phone_provider") or {}
            transcript = CallTranscript(
                stream_id=call_data.get("stream_id"),
                voice_call_id=config.get("voice_call_id"),
            )
            serializer = PlivoFrameSerializer(
                stream_id=call_data["stream_id"],
                call_id=call_data.get("call_id"),
                auth_id=str(phone_provider.get("api_key") or ""),
                auth_token=str(phone_provider.get("secret_key") or ""),
                params=PlivoFrameSerializer.InputParams(auto_hang_up=True),
            )
            transport = FastAPIWebsocketTransport(
                websocket=websocket,
                params=FastAPIWebsocketParams(
                    audio_in_enabled=True,
                    audio_out_enabled=True,
                    add_wav_header=False,
                    serializer=serializer,
                    allowed_origins=[],
                ),
            )
            try:
                await run_bot(
                    transport,
                    runtime_config=_runtime_from_goodbox(config),
                    transcript_callback=transcript.add,
                )
            finally:
                # Goodbox receives all completed user/assistant turns even when
                # the carrier disconnects or the pipeline exits with an error.
                await goodbox.call_stop(
                    transcript.stream_id, transcript.voice_call_id, transcript.messages
                )
        finally:
            await goodbox.close()
    except (ValueError, httpx.HTTPError) as exc:
        logger.error(f"Goodbox/Plivo call setup failed: {exc}")
        await websocket.close(code=1011)
    except Exception:
        logger.exception("Unhandled Goodbox Plivo media error")
        await websocket.close(code=1011)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "goodbox_server:app",
        host=os.getenv("GOODBOX_SERVER_HOST", "0.0.0.0"),
        port=int(os.getenv("GOODBOX_SERVER_PORT", "8000")),
        reload=False,
    )
