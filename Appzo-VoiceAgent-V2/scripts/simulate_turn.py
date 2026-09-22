import asyncio
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from goodbox_server import setup_logging
from main import LiveLatencyObserver, TurnMetrics
from voice_agent.runtime.latency_breakdown import LatencyBreakdown


async def main():
    setup_logging()
    logger.info("Running simulated turn to verify latency telemetry...")

    metrics = TurnMetrics(
        turn_id=1,
        last_voiced_at=10.0,
        provider_eot_at=10.20,
        aggregator_stop_at=10.21,
        turn_committed_at=10.20,
        commit_at=10.22,
        llm_request_started_at=10.22,
        llm_stream_opened_at=10.35,
        llm_first_token_at=10.45,
        first_filtered_text_at=10.46,
        first_safe_text_at=10.55,
        tts_requested_at=10.56,
        tts_first_audio_at=10.75,
        output_first_packet_at=10.76,
        output_first_non_silent_at=10.85,
        bot_started_at=10.76,
        route="hosted",
        endpoint_profile="fast",
    )

    state = SimpleNamespace(metrics=metrics)
    controller = SimpleNamespace(_state=state, _model="openai/gpt-4.1-mini")
    session = SimpleNamespace(
        tenant_id="goodbox-test",
        agent=SimpleNamespace(version="v2.1"),
        state={"name": "OPEN"},
        call_id="sim-call-001",
    )

    observer = LiveLatencyObserver(
        controller,
        tts_transport="websocket-stream",
        session=session,
        call_origin_at=time.time(),
    )

    @observer.event_handler("on_latency_breakdown")
    async def on_latency_breakdown(_observer, breakdown: LatencyBreakdown):
        logger.info(
            "── LATENCY BREAKDOWN | turn={} ──\n{}",
            breakdown.turn_id,
            "\n".join(
                f"  {line}"
                for line in breakdown.turn_contribution_lines(
                    observer._breakdown_min_secs
                )
            ),
        )

    await observer._report_latency(state)
    logger.info("Simulated turn completed. Check above output or logs/voiceagent.log!")


if __name__ == "__main__":
    asyncio.run(main())
