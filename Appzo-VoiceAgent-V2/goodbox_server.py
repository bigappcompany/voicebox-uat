"""Goodbox-controlled Plivo media endpoint for the shared Pipecat agent.

Goodbox remains the source of truth for the phone number, call initiation,
prompt, behavior, and service settings. This service only handles the media
webhook that Goodbox's Plivo number reaches after a call is placed.
"""

import asyncio
import base64
import hashlib
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from xml.sax.saxutils import escape

import httpx
from fastapi import FastAPI, Form, Query, Response, WebSocket
from loguru import logger
from openai import AsyncAzureOpenAI, AsyncOpenAI

from main import AgentRuntimeConfig, run_bot
from voice_agent.agents.compiler import AgentCompiler
from voice_agent.agents.registry import AgentBundleRegistry
from voice_agent.runtime.bootstrap import RuntimeBootstrap
from voice_agent.runtime.flags import RuntimeFlags
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.plivo import PlivoFrameSerializer
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)


CALL_START_PATH = "/voice-calls/call-start"
CALL_STOP_PATH = "/voice-calls/call-stop"
_LLM_CLIENTS: dict[tuple[str, str, str], object] = {}


def setup_logging() -> Path:
    """Configure shared Loguru sinks once for the serving process."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_path = Path(os.getenv("VOICEAGENT_RUNTIME_LOG", "").strip() or (
        Path(__file__).resolve().parent / "logs" / "voiceagent.log"
    )).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        enqueue=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )
    logger.add(
        str(log_path),
        level="DEBUG",
        rotation="50 MB",
        enqueue=True,
        backtrace=False,
        diagnose=False,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
    )
    return log_path

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


def _plivo_stream_url(body: dict[str, Any]) -> str:
    """Send media to Cloud when configured; otherwise retain local execution."""
    import json

    encoded = base64.b64encode(json.dumps(body).encode()).decode()
    cloud_url = os.getenv("PIPECAT_CLOUD_PLIVO_WS_URL", "").strip()
    if not cloud_url:
        return _websocket_url(_required("PUBLIC_BASE_URL"), "/v1/plivo/ws", {"body": encoded})
    parsed = urlparse(cloud_url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    if (
        parsed.scheme != "wss"
        or not parsed.hostname
        or parsed.path != "/ws/plivo"
        or not query.get("serviceHost")
        or parsed.fragment
        or parsed.username
    ):
        raise ValueError("PIPECAT_CLOUD_PLIVO_WS_URL must be a wss:// host /ws/plivo URL with serviceHost")
    query["body"] = encoded
    return urlunparse(parsed._replace(query=urlencode(query)))


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


def _shared_llm_client(provider: str, endpoint: str, api_key: str) -> object:
    """Reuse HTTP/TLS connections without making credentials part of the key."""
    key_id = hashlib.sha256(api_key.encode()).hexdigest()
    key = (provider, endpoint, key_id)
    existing = _LLM_CLIENTS.get(key)
    if existing is not None:
        return existing
    if provider == "azure":
        client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        )
    elif provider == "groq":
        client = AsyncOpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    else:
        client = AsyncOpenAI(api_key=api_key)
    _LLM_CLIENTS[key] = client
    return client


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
        llm_client = _shared_llm_client("azure", f"https://{resource}.openai.azure.com", llm_key)
    elif llm_provider in {"groq", "openai"}:
        # Retain an escape hatch for the existing browser-test providers.
        env_key = "GROQ_API_KEY" if llm_provider == "groq" else "OPENAI_API_KEY"
        llm_key = _required(env_key)
        llm_client = _shared_llm_client(
            llm_provider,
            "https://api.groq.com/openai/v1" if llm_provider == "groq" else "https://api.openai.com/v1",
            llm_key,
        )
    else:
        raise ValueError(f"Unsupported Goodbox LLM provider: {llm_provider!r}")

    primary_language = str(data.get("primary_language") or "").upper()
    stt_language = "multi" if primary_language in {"", "ENGLISH", "HINDI"} else "multi"
    requested_stt_model = str(transcriber.get("model") or "flux").strip()
    # Goodbox uses the short `flux` alias.  The V2 adapter uses the dedicated
    # Deepgram Flux /v2/listen service, where the explicit multilingual model
    # is required; unlike Nova it also supplies the native turn boundaries.
    stt_model = (
        "flux-general-multi"
        if requested_stt_model.lower() in {"flux", "flux-general", "flux-general-multi"}
        else requested_stt_model
    )
    logger.info("Goodbox STT requested={!r}; V2 runtime model={!r}", requested_stt_model, stt_model)
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
        # Explicit V2 experiment override; retain Goodbox settings for rollback.
        vad_stop_secs=_float(os.getenv("V2_VAD_STOP_SECS", "0.2"), 0.2, 0.05, 3.0),
        vad_min_volume=_float(agent.get("min_volume"), 0.6, 0.0, 1.0),
        intro_message=str(agent.get("intro_message") or "").strip() or None,
        llm_client=llm_client,
        owns_llm_client=False,
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
        logger.info(
            "GOODBOX CALL START | status={} call_id={} stream_id={} voice_call_id={} chatbot_id={}",
            response.status_code,
            call_data.get("call_id"),
            call_data.get("stream_id"),
            data.get("voice_call_id"),
            call_data.get("chatbot_id"),
        )
        if not data.get("voice_call_id"):
            logger.warning(
                "GOODBOX CALL START MISSING ID | call_id={} stream_id={} chatbot_id={}",
                call_data.get("call_id"),
                call_data.get("stream_id"),
                call_data.get("chatbot_id"),
            )
        return data

    async def call_stop(
        self,
        stream_id: str | None,
        voice_call_id: str | None,
        messages: list[dict[str, str]],
        *,
        call_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist a transcript against the same provider call Goodbox started.

        Older Goodbox deployments accepted only the three legacy fields below.
        The extra, non-secret correlation fields let newer deployments resolve
        the dashboard's voice-campaign record by Plivo call ID. If an older
        deployment validates its schema strictly, retry its rejected 400/422
        request once with the legacy payload so a compatibility issue cannot
        discard a completed transcript.
        """
        legacy_payload = {
            "stream_id": stream_id,
            "voice_call_id": voice_call_id,
            "messages": messages,
            "inbox_key": os.getenv("GOODBOX_INBOX_KEY", "voice_standard_inbox"),
        }
        payload = dict(legacy_payload)
        correlation = {
            key: (call_data or {}).get(key)
            for key in ("call_id", "phone_id", "chatbot_id", "provider", "custom_variables")
            if (call_data or {}).get(key) is not None
        }
        payload.update(correlation)
        logger.info(
            "GOODBOX CALL STOP BEGIN | stream_id={} voice_call_id={} call_id={} phone_id={} chatbot_id={} messages={}",
            stream_id, voice_call_id, correlation.get("call_id"), correlation.get("phone_id"),
            correlation.get("chatbot_id"), len(messages),
        )
        response = await self._client.post(f"{self._base_url}{CALL_STOP_PATH}", json=payload)
        if correlation and response.status_code in {400, 422}:
            logger.warning(
                "GOODBOX CALL STOP CORRELATION REJECTED | status={} call_id={} retrying legacy payload",
                response.status_code, correlation.get("call_id"),
            )
            response = await self._client.post(f"{self._base_url}{CALL_STOP_PATH}", json=legacy_payload)
        response.raise_for_status()
        try:
            response_body = response.json()
        except ValueError:
            response_body = {}
        result = response_body.get("data") if isinstance(response_body, dict) else None
        result = result if isinstance(result, dict) else {}
        role_counts = {
            role: sum(1 for item in messages if item.get("role") == role)
            for role in ("user", "assistant")
        }
        logger.info(
            "GOODBOX CALL STOP | status={} stream_id={} voice_call_id={} call_id={} messages={} user_messages={} assistant_messages={} response_id={} response_call_id={} response_voice_call_id={}",
            response.status_code,
            stream_id,
            voice_call_id,
            correlation.get("call_id"),
            len(messages),
            role_counts["user"],
            role_counts["assistant"],
            result.get("id"),
            result.get("call_id"),
            result.get("voice_call_id"),
        )
        return result


@dataclass
class CallTranscript:
    stream_id: str | None
    voice_call_id: str | None
    call_data: dict[str, Any] = field(default_factory=dict)
    messages: list[dict[str, str]] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        content = (content or "").strip()
        if content:
            self.messages.append({"role": role, "content": content})


app = FastAPI(title="Goodbox Plivo Voice Agent")
v2_bootstrap = RuntimeBootstrap(AgentBundleRegistry(), AgentCompiler())


@app.on_event("startup")
async def startup() -> None:
    log_path = setup_logging()
    _required("GOODBOX_API_BASE_URL")
    public_base_url = _required("PUBLIC_BASE_URL")
    app.state.instance_id = uuid.uuid4().hex[:12]
    app.state.started_at = time.time()
    app.state.callback_count = 0
    app.state.media_count = 0
    logger.info("Goodbox Plivo voice server ready")
    logger.info("Server instance ID: {}", app.state.instance_id)
    logger.info("Public health URL: {}/health", public_base_url)
    logger.info("Runtime log file: {}", log_path)
    logger.info("Latency observer enabled: true")
    logger.info("Audible PCM metrics enabled: {}", RuntimeFlags.from_env().enable_audible_pcm_metrics)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "instance_id": getattr(app.state, "instance_id", None),
        "started_at": getattr(app.state, "started_at", None),
        "latency_observer_enabled": True,
        "audible_pcm_metrics_enabled": RuntimeFlags.from_env().enable_audible_pcm_metrics,
        "plivo_callback_count": getattr(app.state, "callback_count", 0),
        "plivo_media_count": getattr(app.state, "media_count", 0),
    }


@app.post("/v1/plivo/callback/{phone_id}")
async def plivo_callback(
    phone_id: str,
    CallUUID: str | None = Form(None),
    From: str | None = Form(None),
    To: str | None = Form(None),
    chatbot_id: str | None = Query(None),
) -> Response:
    app.state.callback_count = getattr(app.state, "callback_count", 0) + 1
    logger.info("PLIVO CALLBACK | call_id={} phone_id={}", CallUUID, phone_id)
    body = {
        "phone_id": phone_id,
        "call_id": CallUUID,
        "from": From,
        "to": To,
        "provider": "plivo",
        "chatbot_id": chatbot_id or os.getenv("GOODBOX_CHATBOT_ID"),
    }
    stream_url = _plivo_stream_url(body)
    logger.info("PLIVO XML | media_destination={}", "cloud" if os.getenv("PIPECAT_CLOUD_PLIVO_WS_URL", "").strip() else "local")
    xml = f"""<Response><Stream streamTimeout=\"3600\" keepCallAlive=\"true\" bidirectional=\"true\" contentType=\"audio/x-mulaw;rate=8000\">{escape(stream_url)}</Stream></Response>"""
    return Response(content=xml, media_type="application/xml")


@app.websocket("/v1/plivo/ws")
async def plivo_media(websocket: WebSocket, body: str = Query("")) -> None:
    #logger.log("Received plivo ws connection")
    await websocket.accept()
    app.state.media_count = getattr(app.state, "media_count", 0) + 1
    logger.info("PLIVO MEDIA | websocket accepted")
    telephony_connected_at = time.perf_counter()
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
            # Compile authoring configuration at call setup. The existing V1
            # Pipecat controller remains the feature-flagged rollback path;
            # no Goodbox lookup occurs after this point in the media turn path.
            v2_controller = await v2_bootstrap.start_call(call_data, config)
            logger.info(
                "V2 bundle ready tenant={} agent={} version={} state={}",
                v2_controller.session.tenant_id,
                v2_controller.session.agent.agent_id,
                v2_controller.session.agent.version,
                v2_controller.session.state.get("name"),
            )
            phone_provider = config.get("phone_provider") or {}
            transcript = CallTranscript(
                stream_id=call_data.get("stream_id"),
                voice_call_id=config.get("voice_call_id"),
                call_data=call_data,
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
                    v2_session=v2_controller.session,
                    telephony_stream_id=call_data["stream_id"],
                    telephony_connected_at=telephony_connected_at,
                )
            finally:
                # Goodbox receives all completed user/assistant turns even when
                # the carrier disconnects or the pipeline exits with an error.
                # Carrier disconnect cancellation must not discard the only
                # transcript persistence request. Keep it bounded so shutdown
                # cannot hang indefinitely.
                await asyncio.wait_for(
                    asyncio.shield(
                        goodbox.call_stop(
                            transcript.stream_id,
                            transcript.voice_call_id,
                            transcript.messages,
                            call_data=transcript.call_data,
                        )
                    ),
                    timeout=15,
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
