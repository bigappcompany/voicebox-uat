# Improvements execution plan

This is the executable companion to [improvements.md](improvements.md). It
records the implementation order, shipped behavior, rollback controls,
verification, and remaining live-call work. Unit tests do not establish a
production latency SLO.

## Outcome

The Goodbox → Plivo → Pipecat path remains intact. V2 now moves safe work
before hard end-of-turn, keeps speculative speech private until validation,
and removes avoidable work from hard-EOT → audio. The active providers remain
Deepgram Flux, GPT-4.1-mini/Azure-compatible chat completions, and Cartesia
WebSocket TTS. No unbenchmarked provider was silently introduced.

All runtime improvements are enabled by default. Set
`ENABLE_V2_IMPROVEMENTS=false` for aggregate rollback, or disable an
individual flag below. `ENABLE_V2_ROUTING=false` remains the controller-level
rollback.

## Execution sequence and delivered behavior

1. **Cached greeting audio**
   - The cache key includes tenant, agent, agent version, voice, TTS model,
     speed, and intro text.
   - A hit plays as paced Plivo-native 8 kHz μ-law while the pipeline warms.
     Caller speech cancels it and sends `clearAudio`.
   - A miss uses normal TTS once, trims/captures its PCM, and atomically creates
     the persistent entry. This lazy first-call exception exists because the
     current Goodbox authoring service has no publish-time synthesis hook.

2. **Deterministic callback state machine**
   - Tracks offered, consented, awaiting day/time, and preference recorded.
   - Local day/time routes never claim a confirmed booking.
   - Booking sentence buffering applies only to booking-sensitive hosted
     plans; ordinary hosted speech streams directly.

3. **Structured facts**
   - Recruitment facts are extracted before planning and stored as session
     slots. Corrections replace prior values.
   - Interim facts remain provisional until final validation.
   - Tenant bundles may supply a fact profile and role aliases.

4. **Immediate speculation promotion**
   - Default commit wait is zero.
   - A matching candidate becomes canonical immediately, without a duplicate
     LLM request, and releases already-safe text.

5. **Stable-interim speculation**
   - Flux may begin private LLM work after a stable prefix survives an 80 ms
     debounce; EagerEOT remains a higher-confidence trigger.
   - Restarts are capped at two. TurnResumed, barge-in, fingerprint changes,
     and new turns cancel stale LLM/private-TTS work.

6. **Flux endpoint profile**
   - `fast` is the V2 default: eager threshold `0.45`, final threshold `0.65`,
     timeout `2500 ms`.
   - `balanced` remains available: `0.55`, `0.70`, `3000 ms`.
   - Flux retains sole turn ownership; Silero/SmartTurn is not layered onto it.

7. **Private speculative TTS**
   - A private Cartesia connection warms at call start.
   - The first safe phrase starts synthesis in an independent task, so LLM
     token consumption continues.
   - At most 600 ms is retained. Only ready, matching, low-risk, no-tool PCM
     can commit; everything else is discarded.

8. **Semantic response fingerprints**
   - Fingerprints include tenant, version, state, classified intent, risk,
     knowledge IDs, material slots, and tool dependency.
   - Classified turns may reuse semantically equivalent text. Generic hosted
     turns retain exact normalized matching.
   - `HIGH_*` plans cannot commit speculative audio.

9. **Compiled AgentBundle prompts**
   - Goodbox authoring fields are normalized into immutable runtime fields;
     identical authoring snapshots reuse a bundle by source digest.
   - Hosted prompts contain stable policy, active-state objective/actions,
     facts, selected tenant knowledge, one prior exchange, and the current
     user. Section token estimates are logged.

10. **Deterministic routing and approved caches**
    - Greeting, identity, repeat, goodbye, busy, wrong-person,
      not-interested, human follow-up, callback steps, and authored `faq:*`
      utterances bypass hosted generation.
    - Text answers remain in the tenant/versioned bundle. Dynamic personal,
      KYC, eligibility, balance, and transaction results are never cached.

11. **First-audible measurement**
    - Live TTS PCM uses an RMS threshold and logs `EOT->first-audible`
      separately from first packet/bot-start.
    - Cached greetings log first packet and first audible packet.
    - Summaries report nearest-rank p50/p90/p95/p99.
      `scripts/record_baseline.py` groups by route, endpoint mode, and model.

12. **Provider replacement gate**
    - Nemotron, Magpie, or a local LLM stays disabled until a representative
      English/Hindi/Hinglish telephony benchmark passes quality review.
    - Provider swaps remain an operational experiment after these runtime
      fixes, as required by the source plan, not an automatic migration.

## Feature flags

| Flag | Default | Scope |
|---|---:|---|
| `ENABLE_V2_IMPROVEMENTS` | `true` | Master switch |
| `ENABLE_CACHED_GREETING` | `true` | Persistent native greeting |
| `ENABLE_CALLBACK_STATE_MACHINE` | `true` | Local callback flow |
| `ENABLE_STRUCTURED_FACTS` | `true` | Fact extraction and retention |
| `ENABLE_ZERO_DELAY_SPEC_PROMOTION` | `true` | Immediate promotion |
| `ENABLE_STABLE_INTERIM_SPECULATION` | `true` | Pre-EagerEOT LLM start |
| `ENABLE_FLUX_TUNING` | `true` | Fast Flux default |
| `ENABLE_SPEC_TTS` | `true` | Private capped PCM |
| `ENABLE_SEMANTIC_FINGERPRINT_REUSE` | `true` | Classified semantic reuse |
| `ENABLE_COMPILED_PROMPTS` | `true` | State-scoped prompt fields |
| `ENABLE_EXTENDED_DETERMINISTIC_ROUTING` | `true` | Expanded local routes |
| `ENABLE_AUDIBLE_PCM_METRICS` | `true` | First-audible telemetry |

Provider-swap flags remain false: `ENABLE_LOCAL_LLM`, `ENABLE_FILLERS`, and
`ENABLE_HEDGED_MODELS`. Goodbox STT configuration, rather than the dormant
`ENABLE_FLUX` migration flag, selects the deployed Flux service.

## Verification and rollout gates

```bash
../.venv313/bin/python -m unittest discover -s tests -v
../.venv313/bin/python -m compileall -q .
```

The regression suite covers tenant isolation, bundle reuse, fact corrections,
callback safety, ordinary hosted streaming, stable-interim start, TurnResumed
cancellation, exact fallback for unclassified turns, semantic fingerprints,
high-risk rejection, private PCM commit, greeting invalidation/barge-in, and
audible detection.

Live promotion requires representative calls measuring:

- p50/p90/p95/p99 native EOT → first audible audio;
- raw speech end → first audible audio;
- false-EOT and TurnResumed rates;
- repeated-question, slot-correction, and booking-violation rates;
- speculation and speculative-TTS hit/miss rates;
- English, Hindi, Hinglish, noisy mobile, long-pause, number/date, and
  self-correction cases.

The server emits one non-sensitive `LATENCY RECORD | {...}` line per response.
Save a call log and summarize it with:

```bash
../.venv313/bin/python scripts/record_baseline.py path/to/server.log
```

Sub-second p50/p95 is a rollout target, not a guarantee. If p95 misses, segment
by route and inspect LLM TTFT, first-safe-text, Flux raw-audio → EOT, and
first-audible TTS before changing providers.
