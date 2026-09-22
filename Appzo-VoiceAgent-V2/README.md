# Appzo VoiceAgent V2

The current rollout is documented in
[IMPROVEMENTS_EXECUTION_PLAN.md](IMPROVEMENTS_EXECUTION_PLAN.md); the source
design is [improvements.md](improvements.md).

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
Goodbox's `flux` alias resolves to Deepgram `flux-general-multi`. Flux owns
start/EOT detection through `ExternalUserTurnStrategies`. A stable interim can
start private LLM work before EagerEOT; EagerEOT raises confidence, while only
final Flux EOT may make a validated result audible. Flux is never combined with local
Silero/SmartTurn ownership. Finalized Flux turns have no artificial settle
delay. Nova fallback turns use `V2_TURN_SETTLE_SECS` (default `0.18`) so a
trailing final transcript replaces a provisional aggregate rather than creating
a duplicate response. A transcription-only start is initially treated as
untrusted, but meaningful subsequent STT text confirms it as a barge-in and
flushes queued bot audio before the next answer starts.
For low-risk, no-tool response plans, `ENABLE_SPEC_TTS=true` warms a second,
private Cartesia WebSocket at call start. Interim LLM audio is retained locally
until hard EOT validates the exact response fingerprint; then a capped PCM
prefix is released before normal Cartesia synthesis continues. A mismatch,
barge-in, high-risk plan, tool dependency, or failed private socket discards it
and falls back to the normal public WebSocket stream.

Known greetings are stored persistently as Plivo-native 8 kHz μ-law under
`.runtime-cache/greetings`. The first call for a new text/voice/version captures
the normal TTS greeting; subsequent calls play the cached asset while the live
pipeline warms. Caller speech cancels it with `clearAudio`.

All improvements default on. Set `ENABLE_V2_IMPROVEMENTS=false` to disable
them as a group. Individual rollback flags are listed in the improvement
execution plan. The default Flux profile is `fast`; use
`V2_FLUX_ENDPOINT_PROFILE=balanced` while evaluating false cuts for slower
speech.

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

### Flux correctness and timing update

Flux preserves the native `UserStoppedSpeakingFrame` so Pipecat's controller
and strategy both leave the speaking state. `FluxStopStrategy` checks for
completion immediately when final text arrives, supporting either arrival
order without the external strategy's 500 ms fallback wait. The aggregator's full text is
authoritative; individual final segments no longer replace it.

Latency starts at receipt of Flux EndOfTurn, and each response logs
`provider-turn` and `provider-EOT->aggregator`. Compare new summaries with
provider EOT timestamps, not the older aggregator-based summaries.

Every completed response also logs an additive `LATENCY BREAKDOWN`. When a
local last-voiced timestamp is available it covers caller speech-stop through
first audible output; otherwise it starts at hard EOT. Each non-overlapping
part has a stable key and an owner (`setting`, `service`, `bot`, or `pipeline`),
and the same structured list is stored under `latency_breakdown` in the JSON
`LATENCY RECORD`. The parts always sum to the reported total. Set
`V2_LATENCY_BREAKDOWN_MIN_MS` (default `1`) to roll briefer parts into one
pipeline line in the human-readable view without changing the JSON detail.

Stable Flux interims and EagerEndOfTurn can prepare private speculative work.
TurnResumed invalidates it. Classified routes may reuse a response only when
the tenant/state/intent/knowledge/material-slot fingerprint matches;
unclassified hosted turns still require exact normalized text.
Set `V2_MIN_USEFUL_SPEC_LEAD_MS` (default `100`) to control when a reused
candidate is counted as a genuine latency speculation hit. Later candidates
remain reusable to avoid duplicate model work, but are reported as
`late-reuse` and cannot inflate speculation-hit metrics.

Callback time variants (including `p.m.`) use a local preference state machine.
All hosted output passes through the booking-claim guard. It releases checked
clause boundaries instead of buffering an entire sentence. No scheduling
service is connected. Live latency and recognition quality still require
representative calls; unit tests do not establish a latency SLO.

```bash
# Use Python 3.10+ (the workspace's ../.venv313/bin/python is suitable).
python -m unittest discover -s tests -v
python -m compileall -q voice_agent scripts
python scripts/verify_cartesia_ws.py
# After representative calls, this accepts either raw JSONL or server logs:
python scripts/record_baseline.py path/to/server.log
```

## Existing call path

Set the same environment values used by V1, then run `goodbox_server.py` and
`scripts/dial_goodbox_test.py`. Do not turn on speculative audio for regulated
or tool-dependent flows; the V2 controller enforces this again at commit time.

The default `goodbox` test route is controlled by the Goodbox backend and can
therefore run on a remote voice worker. To guarantee that a benchmark call
reaches the locally running server, set Plivo API credentials locally and use
the explicit local route:

```bash
PLIVO_AUTH_ID=... \
PLIVO_AUTH_TOKEN=... \
python3 scripts/dial_goodbox_test.py --route local --to +918000000000
```

`PLIVO_SOURCE_NUMBER` is optional; when absent, the helper reads the configured
non-secret source number from Goodbox. Direct mode supplies `PUBLIC_BASE_URL` as
the outbound call's answer URL and does not modify the Plivo application. A
real local call must increment both `plivo_callback_count` and
`plivo_media_count` on `/health`.
# Low-latency call validation

The V2 Goodbox path uses dynamic Deepgram Flux profiles. The default fast
profile is `eager=0.35`, `eot=0.55`, `timeout=1200ms`; the runtime switches to
shorter yes/no and entity profiles based on the question the bot asks.

Capture a call and generate the layered report:

```bash
mkdir -p logs
python3 goodbox_server.py 2>&1 | tee logs/call.log
```

After the call finishes, run in another terminal (or after stopping the
server):

```bash
python3 scripts/record_baseline.py logs/call.log
```

After at least 50 fully measured turns, make the latency targets enforceable:

```bash
python3 scripts/record_baseline.py logs/call.log --enforce-targets --min-samples 50
```

The report separates true `last_voiced_audio -> first_audible` measurements
from `hard_eot -> first_audible` fallbacks. `AUDIO CADENCE` records report
whole-utterance packet gaps to diagnose broken or word-by-word playback.
