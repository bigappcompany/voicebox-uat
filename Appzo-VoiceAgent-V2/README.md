# Appzo VoiceAgent V2

## Live routing

`goodbox_server.py` enables V2 routing by default. Restart the server and look
for `V2 ROUTING enabled`, followed by `V2 ROUTE` for each completed caller turn.
Exact greetings/closings and explicitly configured `cached_utterances` FAQs
bypass the model; other turns use the hosted provider with bounded history.
Set `ENABLE_V2_ROUTING=false` to restore the inherited controller.

V2 is configured with `V2_TTS_TRANSPORT=auto` and
`V2_STREAM_SPEECH=true`: it performs a short call-start health check, then uses
direct Cartesia WebSocket streaming when available. Set
`V2_TTS_TRANSPORT=websocket` to force it during a provider/network test. Its
`SafeSpeechChunker` releases a complete short phrase to Cartesia without
waiting for a full sentence. `V2_TTS_TRANSPORT=http` is an emergency audio
fallback only: it needs full synthesis and cannot meet the interactive latency
target.
The V2 WebSocket adapter uses Cartesia's header-based authentication and an
IPv4 provider connection. This avoids the current host's stalled IPv6
CloudFront route while keeping the API key out of connection URLs and logs.
Finalized STT turns have no artificial settle delay; unfinalized fragments use
`V2_TURN_SETTLE_SECS` (default `0`). Both are included in reported EOT latency.
Only VAD-confirmed starts cancel stale generation; late transcription starts do
not overwrite or interrupt an in-flight turn.
For low-risk, no-tool response plans, `ENABLE_SPEC_TTS=true` warms a second,
private Cartesia WebSocket at call start. Interim LLM audio is retained locally
until hard EOT validates the exact response fingerprint; then a capped PCM
prefix is released before normal Cartesia synthesis continues. A mismatch,
barge-in, high-risk plan, tool dependency, or failed private socket discards it
and falls back to the normal public WebSocket stream.

Set `V2_COMPANY_NAME` to the approved tenant name to enforce a consistent
identity and enable exact identity-question responses. Simple time preferences
following a request for callback time use a local route; they are not bookings.
Provide tenant STT keyterms through Goodbox `transcriber_config.keyterms`.
No product-specific fallback, prompt, or STT keyterms are present in V2.

Goodbox call-start configuration is compiled into an `AgentBundle`, including
optional structured flows, slots, cached utterances, and tenant-scoped local
knowledge passages. V2 performs no Goodbox/database lookup from caller EOT to
first audio. No callback scheduling tool is connected, so callback times remain
preferences rather than bookings.

Latency summaries use nearest-rank percentiles, exclude error/retry routes,
and measure detected EOT to transport audio start. Per-turn records include
tenant, bundle version, state, route, WebSocket transport, speculative LLM/TTS,
safe-text, PCM, and first-audio timestamps. They do not prove caller-perceived
time until tested over representative calls.

V2 is an incremental, multi-tenant latency runtime. The original Plivo/Pipecat
service has been copied here as the rollback-compatible media path; the new
`voice_agent/` package supplies compiled bundles, tenant-local retrieval,
semantic speculation guards, safe speech chunking, audio two-phase commit,
and observability primitives.

Read [the execution plan](EXECUTION_PLAN.md) before enabling runtime flags.

## Verify

```bash
# Use Python 3.10+ (the workspace's ../.venv313/bin/python is suitable).
python -m unittest discover -s tests -v
python -m compileall -q voice_agent scripts
python scripts/verify_cartesia_ws.py
```

## Existing call path

Set the same environment values used by V1, then run `goodbox_server.py` and
`scripts/dial_goodbox_test.py`. Do not turn on speculative audio for regulated
or tool-dependent flows; the V2 controller enforces this again at commit time.
