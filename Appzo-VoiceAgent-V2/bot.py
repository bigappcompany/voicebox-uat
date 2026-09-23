"""Pipecat Cloud entrypoint for the Goodbox/Plivo V2 runtime.

Pipecat Cloud's base image discovers ``bot(args)``.  The existing
``goodbox_server.py`` remains the webhook server used for local Compose and
direct Plivo deployments; this adapter gives Cloud-managed Plivo sessions the
same Goodbox call configuration and V2 pipeline.
"""

from __future__ import annotations

import time
from typing import Any

from loguru import logger
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

from goodbox_server import (
    CallTranscript,
    GoodboxApi,
    _runtime_from_goodbox,
    v2_bootstrap,
)
from main import run_bot


def _as_dict(value: Any) -> dict[str, Any]:
    """Return a best-effort mapping from Cloud's typed runner payloads."""
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        result = value.model_dump(by_alias=True)
        return result if isinstance(result, dict) else {}
    if isinstance(value, dict):
        return dict(value)
    return {
        name: getattr(value, name)
        for name in (
            "stream_id", "call_id", "from", "to", "phone_id",
            "voice_call_id", "chatbot_id", "custom_variables",
        )
        if getattr(value, name, None) is not None
    }


def _call_data(runner_args: RunnerArguments) -> dict[str, Any]:
    """Normalize Cloud call metadata for Goodbox's CALL_START contract."""
    data = _as_dict(getattr(runner_args, "body", None))
    data.update({
        key: value
        for key, value in _as_dict(getattr(runner_args, "call_data", None)).items()
        if value is not None and key != "body"
    })
    data["provider"] = "plivo"
    # Cloud transports do not guarantee a distinct media stream ID. A stable
    # call ID remains suitable for transcript correlation and cache scoping.
    if not data.get("stream_id"):
        data["stream_id"] = data.get("call_id")
    return data


async def bot(runner_args: RunnerArguments) -> None:
    """Run a Pipecat Cloud-managed Plivo call through VoiceAgent V2."""
    transport = await create_transport(
        runner_args,
        {
            "plivo": lambda: FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
            ),
        },
    )
    call_data = _call_data(runner_args)
    if not call_data.get("call_id"):
        raise ValueError("Pipecat Cloud Plivo session is missing call_id")

    goodbox = GoodboxApi()
    transcript: CallTranscript | None = None
    try:
        config = await goodbox.call_start(call_data)
        v2_controller = await v2_bootstrap.start_call(call_data, config)
        transcript = CallTranscript(
            stream_id=call_data.get("stream_id"),
            voice_call_id=config.get("voice_call_id"),
            call_data=call_data,
        )
        logger.info(
            "Pipecat Cloud V2 bundle ready tenant={} agent={} version={}",
            v2_controller.session.tenant_id,
            v2_controller.session.agent.agent_id,
            v2_controller.session.agent.version,
        )
        await run_bot(
            transport,
            runner_args,
            runtime_config=_runtime_from_goodbox(config),
            transcript_callback=transcript.add,
            v2_session=v2_controller.session,
            telephony_stream_id=call_data["stream_id"],
            telephony_connected_at=time.perf_counter(),
        )
    finally:
        try:
            if transcript is not None:
                await goodbox.call_stop(
                    transcript.stream_id,
                    transcript.voice_call_id,
                    transcript.messages,
                    call_data=transcript.call_data,
                )
        finally:
            await goodbox.close()


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
