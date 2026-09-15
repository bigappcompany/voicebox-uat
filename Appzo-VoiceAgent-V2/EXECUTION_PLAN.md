# Optimize V2 execution plan

## Outcome and scope

V2 is a staged migration, not a replacement of the known-working Plivo/Pipecat
call path. This delivery implements the offline/runtime foundation through
phases 1–5, 7–10 and the safety gates needed to run them: compiled tenant
bundles, local tenant-scoped retrieval, flow/slot control, route selection,
semantic speculation validation, safe text chunking, direct Cartesia WebSocket
streaming, two-phase speculative audio, metrics, flags, and provider-client
reuse. The copied V1 service remains the rollback path while V2 is evaluated
in shadow mode.

The runtime uses `V2_TTS_TRANSPORT=auto`: it selects direct Cartesia WebSocket
streaming only after a call-start handshake succeeds, otherwise it uses the
audible HTTP emergency fallback. Private speculative Cartesia PCM is enabled
only on the healthy WebSocket path and only for low-risk, no-tool plans.
The deployed adapter uses header authentication and IPv4 for Cartesia WSS,
matching the verified server-side provider route.
Flux, regional ingress, and self-hosted Gemma remain provider/deployment
experiments; they cannot be safely enabled without their credentials,
recorded-call evaluation corpus, and production infrastructure.

## Execution sequence

1. Freeze a baseline with `scripts/record_baseline.py`; store traces outside
   source control and record the provider/version/endpoint settings.
2. Compile the Goodbox call-start payload into an immutable `AgentBundle` at
   call setup. Load it into `AgentBundleRegistry`; never fetch configuration in
   the EOT-to-audio path.
3. Instantiate a tenant-scoped `CallSession`, flow engine, local knowledge
   index, and runtime flags. Reject cross-tenant lookups at the index boundary.
4. On every interim transcript, update `TranscriptStabilityAnalyzer`. At soft
   EOT, only low-risk, no-tool candidates may prepare response/audio work.
5. At hard EOT, reroute using the final transcript and compare a complete
   `ResponseFingerprint`. Commit prepared audio only on an exact semantic and
   policy match; otherwise abort it and use the normal response path.
6. Use deterministic/cached responses before retrieval or an LLM. Build the
   LLM prompt from the invariant prefix, active state, allowed actions, slots,
   relevant documents, bounded history, and user turn only.
7. Stream generated text through `SafeSpeechChunker`; instrument both first
   audio and first meaningful audio. Keep sentence aggregation as a flaggable
   fallback until prosody tests pass.
8. Run replay, cancellation, tenant-isolation, chunking, fingerprint, and
   audio-commit tests. Then shadow V2 beside V1 on recorded/internal calls.
9. A/B one variable at a time: endpoint profile, STT ownership, chunking,
   semantic reuse, speculative TTS, and only then model routing. Promote by
   TTFMA plus false-cut/correctness gates, not TTFA alone.

## Rollout gates

Feature flags default to safe settings. `ENABLE_SPEC_TTS` is only honored for
`LOW_PUBLIC`/`LOW_WORKFLOW` plans with no tool. `HIGH_*` plans cannot commit
speculative audio. A new speech turn invalidates all work for the prior turn.
Every latency record contains tenant, bundle version, route, endpoint profile,
model, and enabled flags. V1 can be restored by routing traffic back to the
copied `main.py`/`goodbox_server.py` entrypoint.

## Acceptance criteria for this delivery

- No remote config lookup is required after bundle compilation for a call.
- Tenant namespaces, flow transitions, slot validation, and prompt budgets are
  covered by unit tests.
- A speculative artifact is private until a hard-EOT fingerprint match.
- High-risk plans never commit speculative audio.
- Safe chunks never end inside an unsafe number, currency/date expression, or
  incomplete negation.
- Metrics distinguish route, soft/hard EOT, LLM, safe-text, TTS, and commit
  timing.
- `python -m unittest discover -s tests -v` passes from this directory.

## Next operational work

Capture representative English/Hindi/Hinglish telephony calls before setting
numeric SLOs. Use `scripts/record_baseline.py` to write a versioned baseline,
then attach actual Pipecat/Deepgram/Cartesia adapters to `LatencyController`
behind the supplied flags. Do not enable Flux together with local turn-stop
ownership, and do not promote a local model until it passes the same bundle,
retrieval, policy, and slot evaluation corpus as the hosted model.
