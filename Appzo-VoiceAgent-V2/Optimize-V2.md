# Optimize V2 — Multi-Tenant Ultra-Low-Latency Voice Agent Runtime

**Status:** Proposed V2 architecture  
**Primary goal:** Minimize *time from the end of a caller’s meaningful utterance to the first meaningful bot audio* while preserving correctness, interruption quality, tenant isolation, and safety for regulated clients.  
**Primary runtime:** Pipecat + Plivo + Deepgram + LLM + Cartesia  
**Current reference implementation:** `main.py` + `goodbox_server.py`  
**Intended reader:** Codex / implementation engineer  
**Last updated:** 2026-09-14

---

## 0. Executive Summary

The current system is already more advanced than a stock STT → LLM → TTS voice agent: it uses Pipecat for transport/turn orchestration, Deepgram Nova-3 for streaming STT, a custom voice controller for hosted LLM calls, GPT-4.1-mini, Cartesia, and Plivo. Goodbox acts primarily as a control plane that supplies prompt/configuration at call startup and receives the transcript at call end.

Optimize V2 should **not** be built as “a faster prompt” or “a faster model swap.” It should be built as a **multi-tenant real-time agent runtime** with six core principles:

1. **Compile client configurations into runtime artifacts.** Do not execute large authoring prompts directly at runtime.
2. **Move deterministic work out of the LLM.** Conversation flow, state transitions, slot collection, guardrails, and many FAQs can be deterministic or retrieval-driven.
3. **Overlap work with the user’s speech.** Route, retrieve, call the LLM, and even pre-synthesize a small amount of TTS before hard end-of-turn.
4. **Separate “soft EOT” from “hard EOT.”** Start expensive work early; only release audio after final validation.
5. **Optimize the first meaningful audio chunk, not total completion time.**
6. **Measure every stage before swapping models.** Model changes come after turn detection, prompt shape, TTS chunking, connection reuse, and speculation are instrumented.

The target post-EOT path should approach:

```text
hard EOT
  ↓
validate speculative ResponsePlan
  ↓
release already-buffered safe audio
  ↓
Plivo playback
```

rather than:

```text
hard EOT
  ↓
wait for final transcript
  ↓
route
  ↓
retrieve
  ↓
LLM TTFT
  ↓
generate sentence
  ↓
TTS TTFB
  ↓
playback
```

---

# 1. Stop Treating the “Bot Prompt” as One Prompt

The prompt shown for the recruitment client is an **authoring document**, not an optimal runtime representation. It mixes:

- identity/persona,
- business objective,
- allowed scope,
- language rules,
- conversation state machine,
- FAQ content,
- fixed wordings,
- action logic,
- callback/exit behavior,
- commercial restrictions,
- end-call conditions.

For a recruitment firm this may be manageable; for banks handling loans, KYC, account FAQs, authentication, complaints, eligibility explanations, or transaction-related flows, a monolithic system prompt becomes hard to control and expensive to repeatedly prefill.

### V2 rule

A tenant’s agent configuration MUST be represented as structured runtime components:

```text
AgentBundle
├── identity
├── invariant_policy
├── language_profile
├── flow_graph
├── state_schema
├── slot_schema
├── actions
├── knowledge_sources
├── response_policy
├── risk_policy
├── stt_profile
├── tts_profile
├── model_routing_profile
└── cached_utterances
```

The LLM MUST NOT be responsible for inferring the entire flow graph from prose on every turn.

### Multi-tenant examples

Recruitment:

```text
OPENING
  → QUALIFY_HIRING
  → REQUIREMENTS
  → FOLLOW_UP
  → CLOSING
```

Banking:

```text
INTENT
  ├── PUBLIC_FAQ
  ├── LOAN_FAQ
  ├── LOAN_STATUS
  ├── KYC_EXPLANATION
  ├── KYC_COLLECTION
  ├── AUTHENTICATION
  ├── COMPLAINT
  ├── TRANSACTIONAL_ACTION
  └── HUMAN_ESCALATION
```

A public loan FAQ and a KYC identity flow MUST NOT have the same generative freedom.

### Pipecat note

Pipecat’s `FlowManager` formalizes state transitions, functions, message handling, and runtime LLM switching. Optimize V2 does not need to immediately replace the custom controller with Pipecat Flows, but its runtime design should be compatible with this model.

Reference:
- https://docs.pipecat.ai/api-reference/pipecat-flows/flow-manager

---

# 2. Introduce an Agent Compiler

The authoring system (Goodbox today, a future in-house UI tomorrow) should remain human-friendly. Runtime should be machine-optimized.

### Publish path

```text
Goodbox / Internal Agent UI
          │
          │ Publish
          ▼
   ┌───────────────┐
   │ AgentCompiler │
   └───────┬───────┘
           │
           ├── normalized flow graph
           ├── compact invariant prompt
           ├── state/slot schemas
           ├── action definitions
           ├── risk rules
           ├── Moss/search index metadata
           ├── STT keyterms
           ├── response templates
           ├── pre-synthesized audio manifest
           └── routing policy
           │
           ▼
      AgentBundle vN
```

### Runtime path

At worker start or bundle refresh:

```text
AgentBundle → RAM
Knowledge index → RAM/local process
Model clients → warm/shared
Fixed audio → memory/local cache
```

Per call:

```text
agent_id
agent_version
call metadata
custom variables
```

should be sufficient to instantiate the session.

### Requirements

Implement:

```python
@dataclass(frozen=True)
class AgentBundle:
    agent_id: str
    version: str
    tenant_id: str
    identity: dict
    invariant_prompt: str
    language_profile: dict
    flow_graph: dict
    state_schema: dict
    slot_schema: dict
    actions: dict
    risk_policy: dict
    routing_policy: dict
    stt_profile: dict
    tts_profile: dict
    knowledge_profile: dict
    cached_utterances: dict
```

Add:

```python
class AgentBundleRegistry:
    async def get(self, agent_id: str, version: str | None = None) -> AgentBundle: ...
    async def refresh(self, agent_id: str) -> None: ...
```

### Hard requirement

**No external configuration/database lookup may occur between user EOT and first bot audio unless the business action itself requires a remote tool.**

---

# 3. Make Runtime Prompts Tiny and Dynamic

The main response LLM should receive only information relevant to the current turn.

### Bad

```text
persona
+ all objectives
+ all steps
+ every branch
+ every FAQ
+ every action
+ full call history
+ user
```

### Target

```text
SYSTEM:
You are Riya for The Hiring Company.
Only make claims supported by supplied policy/knowledge.
Use concise spoken English.
Do not invent commercials.

STATE:
REQUIREMENTS

OBJECTIVE:
Capture missing timeline; then offer approved follow-up.

KNOWN SLOTS:
role=backend engineers
volume=12
timeline=unknown

ALLOWED NEXT ACTIONS:
ASK_TIMELINE
OFFER_FOLLOWUP
EXIT

RELEVANT KNOWLEDGE:
<only required passages>

USER:
"We need them around December."
```

Banking example:

```text
SYSTEM:
You are the approved banking support assistant.
Never infer loan approval, eligibility, KYC status, balance, or account facts.
Use retrieved policy and verified tool results only.

STATE:
LOAN_FAQ

RISK:
PUBLIC_INFORMATION

KNOWLEDGE:
<approved passages>

USER:
"What are the foreclosure charges?"
```

### Prompt caching invariant

Put stable content first:

```text
tenant invariant prefix
→ flow/state contract
→ dynamic state/slots
→ retrieved knowledge
→ recent conversation
→ current user
```

Keep the invariant prefix byte-for-byte stable across calls for the same agent version wherever possible.

### Prompt budget target

Track:

- invariant tokens,
- state tokens,
- retrieval tokens,
- history tokens,
- user tokens.

Create a per-route budget. Do not allow prompt size to grow unbounded over a call.

---

# 4. Remove `OK|`, `END|`, `NO|` as the Long-Term Control Protocol

The current protocol is useful but couples control semantics to generated speech tokens.

V2 should separate:

```text
CONTROL
- continue
- end call
- invoke tool
- transfer
- refuse
- ask clarification

SPEECH
- actual utterance
```

### Target object

```python
@dataclass
class ResponsePlan:
    route: str
    action: str
    next_state: str | None
    risk_class: str
    slots_read: tuple[str, ...]
    slots_written: dict
    knowledge_ids: tuple[str, ...]
    tool_name: str | None
    cache_key: str | None
    allow_speculative_audio: bool
```

Speech can then be generated independently.

### Why

The first generated token can be speech content rather than protocol overhead. This also permits deterministic hang-up, transfer, or tool behavior without trusting a text prefix.

### Migration

Phase 1:
- Keep existing `OK|/END|/NO|`.
- Add `ResponsePlan` internally.

Phase 2:
- Router/state engine decides control.
- LLM generates speech only.

Phase 3:
- Remove prefix parser.

---

# 5. Create Multiple Response Paths

Do not route every turn through the same full LLM stack.

### Required hierarchy

```text
1. Fixed deterministic response
2. Cached audio/text
3. Retrieval + approved template
4. Retrieval + lightweight rewrite
5. Local fast model
6. Main/heavy model
7. Tool/API + generated explanation
8. Human escalation
```

### Examples

| Turn | Route |
|---|---|
| Greeting | pre-synthesized audio |
| “Bye” | deterministic closing + EndFrame |
| “Are you a robot?” | approved fixed response |
| Known public FAQ | retrieval/template |
| Slot confirmation | deterministic template |
| KYC field collection | deterministic validator/flow |
| Loan status | authenticated tool |
| Ambiguous policy | main model |
| Exception/risk | heavy model/human |

### Goal

The lowest-latency successful path is the one that avoids LLM and/or TTS entirely.

---

# 6. Decompose the Latency Problem into Distinct Speculation Layers

Use explicit terminology in code and metrics:

### A. Turn speculation
Predict whether the speaker is near end-of-turn.

### B. Transcript speculation
Predict that the stable interim transcript will preserve the important semantics.

### C. Response speculation
Start retrieval/model generation before hard EOT.

### D. Token speculative decoding
Draft tokens with a smaller model and verify with a target model.

### E. TTS speculation
Synthesize audio before it is permitted to play.

### F. Perception masking
Play an acknowledgement while a genuinely slow operation continues.

Do not use one metric called `speculation_hit`. Track these separately.

---

# 7. Dynamic and Speculative Endpointing

Turn completion is one of the highest-priority optimization areas.

Pipecat currently defaults to:
- start: VAD and transcription strategies,
- stop: `LocalSmartTurnAnalyzerV3`.

Pipecat notes that built-in STT P99 latency assumptions use `VADParams.stop_secs=0.2`.

Reference:
- https://docs.pipecat.ai/api-reference/server/utilities/turn-management/user-turn-strategies

### Current issue to verify

The generic config uses `vad_stop_secs=0.2`, while the Goodbox phone adapter has historically defaulted to `0.7` if not supplied.

Do not blindly change this. Instrument and A/B it.

### State-aware endpoint profile

Add:

```python
@dataclass(frozen=True)
class EndpointProfile:
    vad_stop_secs: float
    expected_answer: str
    speech_timeout_ms: int
    soft_eot_confidence: float
    hard_eot_confidence: float
```

Example profiles:

```text
YES_NO:
  aggressive

SHORT_ENTITY:
  moderately aggressive

FREEFORM:
  normal

KYC_DIGITS:
  conservative

ADDRESS:
  conservative

USER_WITH_LONG_PAUSES:
  adaptive/conservative
```

### Acceptance metric

Optimize **false cut rate** and **EOT-to-first-meaningful-audio** jointly. Do not optimize EOT latency alone.

---

# 8. Re-Evaluate Nova-3 + Smart Turn vs Deepgram Flux

The current V1 adapter maps Goodbox `flux` back to Nova-3 because the old architecture expects Nova streaming STT plus local turn management.

V2 should benchmark this decision again.

### Candidate A

```text
Deepgram Nova-3
+ Pipecat Silero VAD
+ LocalSmartTurnAnalyzerV3
+ custom speculation
```

### Candidate B

```text
Deepgram Flux multilingual
+ Flux EagerEndOfTurn
+ ExternalUserTurnStrategies
+ custom speculation
```

Pipecat’s `DeepgramFluxSTTService` supports:

- `eager_eot_threshold`,
- `eot_threshold`,
- `eot_timeout_ms`,
- keyterms,
- multilingual hints,
- mid-stream setting updates.

It emits eager EOT transcripts as interim frames and emits a resume event if speech continues.

Reference:
- https://docs.pipecat.ai/api-reference/server/services/stt/deepgram

### Critical implementation rule

If Flux owns start/stop detection, use Pipecat `ExternalUserTurnStrategies` to avoid double turn management.

### Benchmark

Use recorded real calls by:
- English,
- Hindi,
- Hinglish,
- noisy telephony,
- yes/no,
- freeform,
- digit strings,
- long pauses.

Measure:
- final transcript latency,
- eager EOT latency,
- false endpoint rate,
- transcript accuracy,
- interruption quality,
- EOT → audio p50/p95/p99.

---

# 9. Add “Soft EOT” and “Hard EOT”

Do not model EOT as binary.

### State machine

```text
TALKING
  ↓
SOFT_EOT
  ↓
HARD_EOT
```

`SOFT_EOT` permits expensive speculative work.

`HARD_EOT` permits audio playback.

### Soft EOT actions

Allowed:
- retrieval,
- intent classification,
- route selection,
- local model call,
- heavy model call,
- first-safe-chunk parsing,
- speculative TTS into private buffer.

Not allowed:
- audible output,
- irreversible action,
- external side effect.

### Hard EOT actions

- validate final transcript/slots,
- validate ResponsePlan fingerprint,
- commit or discard speculative work,
- release audio.

### Resume behavior

If user speech resumes from soft EOT:

```text
cancel/mark stale speculative work
discard speculative audio
return to TALKING
```

---

# 10. Improve Streaming STT Usage

Streaming STT is already enabled. V2 should exploit it more intelligently.

### Current behavior to replace

```text
interim changed
→ wait debounce
→ if min words/chars
→ speculative LLM
```

### V2 stability analyzer

Track:

```python
@dataclass
class TranscriptHypothesis:
    text: str
    stable_prefix: str
    unstable_suffix: str
    stable_since: float
    intent_guess: str | None
    slot_guess: dict
    semantic_hash: str | None
```

### Example

```text
"I want a personal"
"I want a personal lone"
"I want a personal loan"
"I want a personal loan for"
"I want a personal loan for my business"
```

The system should recognize that:
- `personal loan` has stabilized,
- route may be known,
- retrieval can begin,
- suffix is still unstable.

Do not require entire interim equality before useful speculative work begins.

---

# 11. Replace Exact Transcript Equality with Semantic Response Fingerprints

Current speculation reuse is very conservative: the final transcript must match the speculative transcript after normalization.

V2 should validate based on response semantics.

### ResponsePlan fingerprint

```python
@dataclass(frozen=True)
class ResponseFingerprint:
    tenant_id: str
    agent_version: str
    state: str
    intent: str
    risk_class: str
    knowledge_version: str
    material_slots: tuple[tuple[str, str], ...]
    tool_dependency: str | None
```

Hash this object.

### Example safe reuse

Interim:

```text
"how much is your personal loan interest"
```

Final:

```text
"what is the personal loan interest rate"
```

If they map to the same response fingerprint, speculative answer may be reusable.

### High-risk invalidation

For banking, include every material slot:

```text
amount=250000
term_months=24
product=personal_loan
```

A change invalidates speculative output immediately.

### Rule

Text equivalence is optional. **Semantic and policy equivalence is mandatory.**

---

# 12. Treat Traditional Speculative Decoding as a Separate Optimization

Do not confuse early LLM calls with token-level speculative decoding.

Traditional speculative decoding:

```text
draft model → candidate tokens
target model → verify
accepted tokens → output
```

### With hosted GPT-4.1-mini

You do not control Azure/OpenAI’s inference kernel. You cannot bolt an external draft decoder into the hosted target model.

### With self-hosted Gemma

You can experiment with:
- assisted generation,
- same-family draft models,
- self-speculation,
- model-server-specific speculative decoding.

### Priority

Lower than:
1. endpointing,
2. prompt prefill,
3. first-safe TTS chunk,
4. connection reuse,
5. response speculation,
6. speculative TTS.

Why: voice first-audio often depends more on TTFT/prefill and TTS startup than on total decode throughput.

---

# 13. Replace Sentence-Only TTS Release with Adaptive Boundary Parsing

Current Cartesia integration uses sentence aggregation, which waits for a sentence boundary.

Pipecat’s Cartesia service supports `TextAggregationMode.SENTENCE` and `TextAggregationMode.TOKEN`.

Reference:
- https://docs.pipecat.ai/api-reference/server/services/tts/cartesia

### V2 requirement

Build a custom **SafeSpeechChunker** between model token stream and TTS.

The chunker should release text at safe, natural boundaries without waiting for full sentences.

### It MUST understand

- complete word boundaries,
- punctuation,
- commas/clauses,
- numbers,
- currency,
- dates,
- abbreviations,
- negation,
- URLs/emails if supported,
- Hindi/Hinglish tokenization as needed.

### Metrics

Track:
- first LLM token → first safe text chunk,
- safe chunk character count,
- safe chunk word count,
- prosody quality rating,
- correction/rollback impossibility rate.

---

# 14. Do Not Literally Send “First 4 Tokens” to TTS

A fixed four-token rule is unsafe and linguistically arbitrary.

Examples:

```text
"Your loan app..."
"five lakh at..."
"I'm sorry, but..."
```

All are incomplete.

### SafeSpeechChunker interface

```python
class SafeSpeechChunker:
    def push(self, text_delta: str) -> list[str]:
        """Return zero or more safe chunks."""

    def flush(self) -> list[str]:
        ...
```

### Release heuristics

Require:
- not inside a word,
- not inside a number/date/currency expression,
- not immediately after an incomplete negation,
- minimum semantic payload,
- prefer clause punctuation,
- hard maximum wait timer.

### Starting experiment ranges

These are **benchmark hypotheses, not fixed requirements**:
- minimum ~6–12 words,
- or ~40–80 characters,
- forced release after a short timer if a safe boundary exists.

Tune by voice/model/language.

---

# 15. Control Cartesia Buffering Explicitly

Switching to token streaming alone is not sufficient.

Pipecat documents that Cartesia sentence aggregation and token aggregation interact differently with buffering settings.

Reference:
- https://docs.pipecat.ai/api-reference/server/services/tts/cartesia

### V2 experiment

Test:

```text
custom SafeSpeechChunker
+ Cartesia WebSocket streaming
+ explicit low server buffer
```

against:

```text
Pipecat SENTENCE mode
```

### Required metrics

- `tts_request_at`,
- `cartesia_first_audio_at`,
- `first_audio_chunk_size`,
- audio underruns,
- prosody rating,
- total TTS websocket reconnect count.

### Avoid

Do not unknowingly stack:
- client-side chunk buffer,
- Pipecat sentence buffer,
- Cartesia server buffer.

---

# 16. Add Speculative Speech Synthesis

This is one of the highest-upside V2 features.

### Current concept

```text
interim → speculative LLM
hard EOT → validate
then TTS
```

### V2

```text
interim
  ↓
speculative ResponsePlan
  ↓
speculative LLM
  ↓
first safe phrase
  ↓
speculative Cartesia
  ↓
PCM stored privately

hard EOT
  ↓
validate
  ↓
release already-generated audio
```

### Constraint

Speculative audio MUST NEVER be sent to Plivo before commit.

### Data structure

```python
@dataclass
class SpeculativeAudioCandidate:
    fingerprint: str
    transcript_basis: str
    text: str
    pcm_chunks: list[bytes]
    sample_rate: int
    created_at: float
    committed: bool = False
    invalidated: bool = False
```

---

# 17. Implement Speculative Audio as a Two-Phase Commit

### PREPARE

Perform:
- speculative route,
- retrieval,
- LLM,
- safe chunk extraction,
- TTS synthesis.

Store audio privately.

### COMMIT

On hard EOT:
1. obtain final transcript,
2. recompute/confirm material slots,
3. compute final `ResponseFingerprint`,
4. compare with prepared fingerprint,
5. verify no policy/risk violation,
6. release buffered audio.

### ABORT

Abort on:
- user resumes speech,
- slot changed,
- intent changed,
- knowledge version changed,
- tool result required,
- risk class changed,
- controller moved to another turn.

### Banking rule

No speculative audible output for a fact derived from:
- authentication state,
- account balance,
- KYC verification result,
- loan decision,
- transaction result,
until the authoritative tool response is committed.

---

# 18. Cap Speculative TTS

Do not synthesize a full answer speculatively.

### Goal

Prepare only enough audio to hide TTS startup and begin naturally.

Example target to benchmark:

```text
~300–700 ms of playable audio
```

This is not a fixed production constant.

### Why

- limits wasted synthesis,
- reduces stale audio memory,
- minimizes compute burn on misses,
- first-audio latency is what matters,
- normal TTS can continue while buffered audio is playing.

### Controller setting

```python
SPEC_TTS_MAX_AUDIO_MS = 500  # experiment default, not final
```

Make tenant/route configurable.

---

# 19. Dual-Model Routing: Use It for Routing, Not Just Fillers

A light model that only says “Got it” is usually a waste.

Use a small/local model for:
- intent classification,
- slot extraction,
- state repair,
- FAQ route selection,
- simple grounded responses,
- response rewriting.

Escalate to heavy model for:
- ambiguity,
- multi-step reasoning,
- uncommon policy questions,
- tool synthesis,
- difficult multilingual cases,
- high-complexity generation.

### Router decision

```python
@dataclass
class RouteDecision:
    route: str
    confidence: float
    expected_latency_ms: int | None
    requires_tool: bool
    risk_class: str
```

### Key optimization

Run routing on interim speech whenever possible.

Do not do:

```text
hard EOT → router call → main model
```

Prefer:

```text
interim → router
hard EOT → route already available
```

---

# 20. Use Fillers Only for Genuinely Slow Operations

If the answer is ready in 300–400 ms, a filler can make the interaction feel slower.

### Use acknowledgement audio for

- bank API calls,
- CRM reads,
- document retrieval,
- human transfer setup,
- unusually slow reasoning,
- remote action execution.

### Do not use filler for

- normal FAQ,
- fixed response,
- cached answer,
- fast local generation.

### Filler decision

```text
expected meaningful answer < threshold
→ no filler

expected answer clearly above threshold
→ approved neutral acknowledgement
```

Benchmark thresholds through listening tests.

### Better implementation

Pre-synthesize approved fillers per tenant voice.

Runtime path:

```text
decision → cached PCM → output
```

No LLM. No TTS call.

---

# 21. Restrict Filler Language for Banks and Regulated Workflows

Fillers MUST NOT imply:
- approval,
- eligibility,
- completion,
- successful KYC,
- correctness of a claim,
- transaction success.

Bad:

```text
"That should be fine."
"Absolutely."
"Your KYC looks good."
```

Safe:

```text
"One moment while I check that."
"Let me verify that for you."
```

### Risk policy

```python
@dataclass
class FillerPolicy:
    enabled: bool
    allowed_categories: tuple[str, ...]
    phrase_ids: tuple[str, ...]
```

For high-risk routes, use only neutral phrases.

---

# 22. Hierarchical Routing Is More Valuable Than Filler Generation

Target routing:

```text
interim transcript
    ↓
local route classifier
    ├── fixed response
    ├── retrieval/template
    ├── local LLM
    ├── main LLM
    ├── tool
    └── escalation
```

### Optional hedged execution

For absolute minimum tail latency, allow controlled parallelism:

```text
local path ─┐
            ├─ race/validate/cancel loser
heavy path ─┘
```

Only use hedging when:
- cost budget allows it,
- route uncertainty is high,
- latency SLA requires it.

Track wasted calls.

---

# 23. Preserve Fine-Grained Plivo WebSocket Media Chunking

Do not create artificial audio buffering at the carrier boundary.

The current Plivo stream uses telephony μ-law at 8 kHz. Maintain provider-compatible low-latency framing.

### Rule

Do not aggregate many small inbound frames into large batches unless measurements prove it helps.

Avoid turning approximately real-time frame cadence into 80–100+ ms buffered blocks.

### Measure

- inbound media inter-arrival jitter,
- serializer delay,
- resampling delay,
- STT enqueue delay.

---

# 24. Optimize Outbound Media Separately

Measure:

```text
Cartesia first PCM
→ Pipecat frame
→ serializer/resample
→ WebSocket send
→ Plivo
→ audible playback
```

Do not assume TTS first byte equals user-heard audio.

### Instrument timestamps

```python
tts_first_audio_at
serializer_first_audio_at
ws_first_send_at
plivo_ack_or_send_complete_at  # if observable
bot_started_at
```

### Rule

Only tune packet size when traces show queueing or underflow.

---

# 25. Remove ngrok from the Production Critical Path

ngrok is acceptable for local development.

Production target:

```text
Plivo
  ↓
regional public ingress
  ↓
Pipecat worker/service
```

### Region selection

Benchmark RTT to:
- Plivo edge,
- Deepgram,
- LLM provider/inference cluster,
- Cartesia.

Choose region based on the **critical-path composite**, not just proximity to users.

### Deployment experiments

At minimum benchmark:
- India region,
- Singapore,
- nearest Azure/OpenAI region used by tenant,
- nearest Cartesia/Deepgram effective path.

Use actual p50/p95 network timings.

---

# 26. Reuse Persistent Model Connections

The current call-specific runtime creates an OpenAI/Azure client and closes it at call cleanup. V2 should prefer process-wide shared clients.

### Target

```python
class ProviderClientPool:
    azure_clients: dict[str, AsyncAzureOpenAI]
    openai_clients: dict[str, AsyncOpenAI]
    # optional local inference HTTP clients
```

Calls should borrow/reuse clients.

### Benefits

- connection pooling,
- DNS/TLS reuse,
- lower first-turn setup cost,
- lower socket churn.

### Constraint

Tenant credentials must remain isolated. Pool by provider credential identity/config, not by arbitrary tenant.

### Also warm

- model server,
- Moss/local index,
- TTS service path where supported,
- common prompt prefixes where provider supports caching.

---

# 27. Evaluate Gemma 4 E4B as a Self-Hosted Candidate

Do not replace GPT-4.1-mini only because Gemma is self-hostable.

The primary attraction is operational control:

- same-region inference,
- KV cache,
- prefix cache,
- quantization,
- batch scheduling,
- queue control,
- speculative decoding,
- GPU allocation,
- deterministic serving stack.

### Required benchmark dimensions

- TTFT p50/p95/p99,
- tokens/sec,
- concurrent-call degradation,
- instruction adherence,
- tool correctness,
- slot extraction,
- multilingual/Hinglish quality,
- hallucination,
- policy compliance,
- memory footprint,
- GPU cost per concurrent call.

### Compare against GPT-4.1-mini

GPT-4.1-mini is a low-latency hosted model with strong instruction following/tool use and a large context window. It remains a strong control baseline.

Reference:
- https://developers.openai.com/api/docs/models/gpt-4.1-mini

Gemma 4 E4B official repository/model card:
- https://huggingface.co/google/gemma-4-E4B-it

### Important

Do not use public benchmark scores alone to approve Gemma for banking use. Use product-specific evals.

---

# 28. Do Not Globally Replace GPT-4.1-mini with E4B Until It Passes Tenant Evals

Create an anonymized eval corpus covering:

```text
recruitment
public banking FAQ
loan FAQ
KYC
authentication wording
Hinglish
Hindi
numbers/dates/currency
noisy transcript variants
interruptions
ambiguous requests
prompt attacks
tool calls
unsupported claims
goodbyes
callbacks
complaints
```

### Required eval fields

```python
EvalResult(
    semantic_correctness,
    policy_compliance,
    tool_accuracy,
    slot_accuracy,
    state_transition_accuracy,
    unsupported_claim_rate,
    refusal_accuracy,
    first_token_latency,
    completion_latency,
)
```

### Promotion policy

Define route-specific quality gates.

Example:
- E4B may be allowed for public FAQ earlier.
- E4B may be forbidden from final banking transaction narration until passing stricter evals.

---

# 29. Use a Hybrid Local + Hosted Model Architecture

Recommended initial V2:

```text
                  ┌─ fixed/template
                  │
interim → router ─┼─ Gemma 4 E4B
                  │
                  ├─ GPT-4.1-mini
                  │
                  ├─ backend tool
                  │
                  └─ human escalation
```

### Gemma candidate work

- intent,
- classification,
- slot extraction,
- flow transition suggestions,
- grounded FAQ rewriting,
- short responses.

### Hosted/heavy candidate work

- complex ambiguity,
- policy explanation,
- difficult multilingual turns,
- long tool response synthesis,
- high-stakes edge cases.

### Rule

The router must be deterministic where possible. Do not add a 300 ms LLM router just to save a 100 ms model call.

---

# 30. Benchmark Larger Sparse/MoE Gemma Variants Too

Do not assume the smallest model gives the best overall quality/latency tradeoff.

If a larger sparse/MoE Gemma variant offers substantially stronger quality while activating a small subset of parameters per token, benchmark it on the same hardware and concurrency profile.

### Test

```text
E4B
vs
larger Gemma sparse/MoE candidate
vs
GPT-4.1-mini
```

### Measure

- single-stream TTFT,
- 8/16/32 concurrent calls,
- cache-hit TTFT,
- no-cache TTFT,
- GPU memory,
- output quality.

Model naming/availability can change; pin exact model revisions in the benchmark manifest.

---

# 31. Use Self-Hosting to Enable True Speculative Decoding Experiments

If Gemma is self-hosted, test:

```text
smaller draft model
→ larger target model
```

or model-server-native speculative modes.

### Do not assume improvement

A very small target model may already decode quickly enough that draft verification overhead hurts.

### Benchmark separately

Measure:
- target-only decode,
- draft+verify decode,
- TTFT,
- first 8/16/32 token latency,
- throughput under concurrent calls.

Voice objective is **time to first safe phrase**, not full-response tokens/sec.

---

# 32. Prioritize Prefix/KV Caching Before Speculative Decoding

Voice responses are usually short, which makes prompt prefill disproportionately important.

Per tenant:

```text
stable invariant prefix
+ stable tool schemas
+ stable behavior contract
```

should be cacheable.

Dynamic suffix:

```text
state
slots
retrieval
short history
current user
```

### Self-hosting requirement

Expose metrics for:
- prefix cache hit,
- KV reuse,
- prefill tokens/sec,
- decode tokens/sec,
- TTFT cache hit/miss.

### Hosted providers

Use provider-supported prompt caching where available by keeping the prefix stable.

---

# 33. Share Inference Fleets Across Tenants

Do not deploy one model process per customer unless required by compliance.

Prefer:

```text
shared inference fleet
├── tenant A prefix/cache namespace
├── tenant B prefix/cache namespace
└── tenant C prefix/cache namespace
```

### Tenant isolation

Cache key MUST include:

```text
tenant_id
agent_id
agent_version
model_revision
policy_version
```

Never reuse retrieved content or generated answers across tenant namespaces accidentally.

---

# 34. Do Not Remove Deepgram Just Because Gemma Supports Audio

An audio-capable multimodal model is not automatically a better streaming STT component.

Deepgram provides useful primitives for this architecture:

- streaming interims,
- finalized transcripts,
- language handling,
- keyterms,
- EOT signals,
- timing,
- predictable telephony behavior.

The speculative runtime relies on these signals.

### V2 default

Keep a cascaded architecture:

```text
audio → streaming STT → controller → text LLM → TTS
```

until an end-to-end audio model demonstrably beats it on:
- latency,
- interruption control,
- transcript observability,
- compliance,
- multilingual quality,
- cost.

---

# 35. Target End-State Architecture

```text
                           AUTHORING / CONTROL PLANE
 ┌──────────────────────────────────────────────────────────────────┐
 │ Goodbox or Internal UI                                           │
 │      ↓ publish                                                   │
 │ AgentCompiler                                                    │
 │      ↓                                                           │
 │ versioned AgentBundle + KB/index metadata + audio cache manifest │
 └──────────────────────────────┬───────────────────────────────────┘
                                │ background load/refresh
                                ▼

                           REAL-TIME RUNTIME

 Plivo 20ms-ish media
         │
         ▼
 Pipecat Transport / Serializer
         │
         ▼
 Streaming STT (Nova-3 or Flux)
         │
         ├──── interim ──────────────────────────────────────────────┐
         │                                                          │
         ▼                                                          ▼
 Turn Controller                                          Transcript Stability
         │                                                          │
         │                                                   Router / Flow Engine
         │                                                          │
         │                                          ┌───────────────┴──────────────┐
         │                                          │                              │
         │                                     Local retrieval              State/slots
         │                                          │                              │
         │                                          └──────────────┬───────────────┘
         │                                                         │
         │                                                         ▼
         │                                                   ResponsePlan
         │                                                         │
         │                                     ┌───────────────────┼──────────────────┐
         │                                     │                   │                  │
         │                               fixed/cached           local LLM        hosted LLM
         │                                     │                   │                  │
         │                                     └───────────────────┴──────────┬───────┘
         │                                                                  │
         │                                                           SafeSpeechChunker
         │                                                                  │
 SOFT EOT ──────────────────────────────────────────────────────────────────►│
         │                                                                  ▼
         │                                                        speculative Cartesia
         │                                                                  │
         │                                                         private PCM buffer
         │                                                                  │
 HARD EOT                                                                  │
         │                                                                  │
         └────────────── final ResponseFingerprint validation ──────────────┘
                                      │
                               valid / invalid
                                  │        │
                                  │        └→ discard/recompute
                                  ▼
                          release committed PCM
                                  │
                                  ▼
                         Pipecat output → Plivo
```

### Core invariant

**No speculative result becomes audible until committed.**

---

# 36. Optimize the Overlapped Critical Path

The old sequential critical path:

```text
EOT
+ STT final
+ routing
+ retrieval
+ LLM TTFT
+ safe text accumulation
+ TTS TTFB
+ output transport
```

V2 tries to move these before hard EOT:

```text
while user speaks:
- streaming transcript
- stable prefix detection
- route
- retrieval
- LLM TTFT
- partial generation
- initial TTS
```

Resulting hard-EOT critical path:

```text
hard EOT confirmation
+ final semantic validation
+ buffered-audio commit
+ media output
```

### Optimization objective

Primary SLA:

```text
last meaningful user audio
→ first meaningful bot audio
```

Secondary SLAs:
- interruption responsiveness,
- false endpoint rate,
- semantic correctness,
- speculative waste,
- tool correctness.

---

# 37. Integrate Optimizations as One Latency Controller, Not Independent Hacks

These features can interfere with each other:

- aggressive EOT + fillers can cause interruptions,
- Flux EagerEOT + custom EOT logic can double-trigger,
- token TTS + hidden server buffering can add delay,
- first-token TTS + speculative response can speak invalid text,
- speculative TTS + filler can delay an answer that is already ready,
- sequential routing can add more latency than it saves,
- speculative decoding can hurt a small model,
- hedged model calls can explode cost.

### Required V2 component

```python
class LatencyController:
    """
    Coordinates turn speculation, routing, response speculation,
    TTS speculation, commitment, cancellation, and metrics.
    """
```

It should own:

```text
turn_id
soft_eot_state
hard_eot_state
transcript hypothesis
route candidate
response fingerprint
LLM candidate(s)
TTS candidate
audio buffer
cancellation token
deadline/budget
metrics
```

### Rule

There must be exactly one authority that decides whether speculative work is:
- valid,
- stale,
- committed,
- cancelled,
- audible.

---

# 38. Implementation Plan

Although the design above contains the 37 requested architecture sections, implementation should proceed incrementally.

## Phase 0 — Freeze Baseline

Before edits:
- pin Pipecat version,
- pin Deepgram settings,
- pin GPT deployment/model revision,
- pin Cartesia model/voice,
- pin Plivo configuration,
- capture 100+ representative call traces if possible.

Produce baseline:
- p50/p90/p95/p99 EOT → bot audio,
- LLM TTFT,
- STT finalization latency,
- TTS first audio,
- false endpoint rate,
- interruption failure rate.

## Phase 1 — Instrumentation First

Extend `TurnMetrics`.

Suggested:

```python
@dataclass
class TurnMetrics:
    turn_id: int

    first_user_audio_at: float | None = None
    last_voiced_at: float | None = None
    vad_stop_at: float | None = None

    first_interim_at: float | None = None
    stable_semantic_prefix_at: float | None = None
    soft_eot_at: float | None = None
    final_stt_at: float | None = None
    hard_eot_at: float | None = None

    route_started_at: float | None = None
    route_completed_at: float | None = None

    retrieval_started_at: float | None = None
    retrieval_completed_at: float | None = None

    llm_requested_at: float | None = None
    llm_first_token_at: float | None = None
    first_safe_text_at: float | None = None

    spec_tts_started_at: float | None = None
    spec_tts_first_audio_at: float | None = None

    commit_at: float | None = None
    first_output_frame_at: float | None = None
    bot_started_at: float | None = None

    route: str = ""
    model: str = ""
    spec_llm: str = "none"
    spec_tts: str = "none"
    endpoint_mode: str = ""
```

Add histograms by:
- tenant,
- language,
- flow state,
- route,
- model,
- endpoint profile.

## Phase 2 — Shared Provider Clients

Refactor call-specific LLM client creation into a process-level client pool.

Do not change conversation semantics in this phase.

## Phase 3 — AgentBundle + Compiler

Implement structured config loading while preserving existing Goodbox API.

The Goodbox response becomes input to the compiler/adapter:

```text
Goodbox payload
→ GoodboxAgentAdapter
→ AgentBundle
```

Later, Goodbox can be removed without changing runtime.

## Phase 4 — State/Flow Engine

Extract:
- step logic,
- slot tracking,
- action transitions,
- end-call behavior.

Keep LLM speech generation initially.

## Phase 5 — Local Retrieval

Load Moss/search index at worker startup/background refresh.

Do not perform cloud retrieval in the hot path.

Implement tenant-keyed knowledge lookup.

## Phase 6 — Endpoint Experiments

A/B:
- current,
- VAD 0.2 + Smart Turn,
- state-aware endpoint profiles,
- Flux + EagerEOT.

No TTS changes yet.

## Phase 7 — SafeSpeechChunker

Replace sentence-only release with adaptive chunking.

A/B against current `TextAggregationMode.SENTENCE`.

## Phase 8 — Response Fingerprints

Replace exact transcript reuse with semantic response validation.

Keep speculative LLM but improve hit rate safely.

## Phase 9 — Speculative TTS

Implement private audio buffer and two-phase commit.

Initially enable only:
- low-risk public FAQs,
- non-tool routes.

## Phase 10 — Deterministic/Cached Routes

Add:
- greeting cache,
- goodbye cache,
- approved fixed FAQ responses,
- callback phrases,
- neutral fillers.

## Phase 11 — Model Router

Introduce:
- local model candidate,
- hosted model fallback.

Shadow local model before routing production speech to it.

## Phase 12 — Self-Hosted Model Experiments

Benchmark Gemma variants with:
- prefix caching,
- quantization,
- concurrency,
- speculative decoding.

---

# 39. Proposed Code Layout

```text
voice_agent/
├── app/
│   ├── goodbox_server.py
│   └── health.py
│
├── runtime/
│   ├── pipeline.py
│   ├── session.py
│   ├── latency_controller.py
│   ├── response_plan.py
│   ├── fingerprints.py
│   └── metrics.py
│
├── agents/
│   ├── bundle.py
│   ├── registry.py
│   ├── compiler.py
│   ├── goodbox_adapter.py
│   └── schemas.py
│
├── turns/
│   ├── endpoint_profiles.py
│   ├── transcript_stability.py
│   ├── soft_eot.py
│   └── hard_eot.py
│
├── routing/
│   ├── router.py
│   ├── deterministic.py
│   ├── local_model.py
│   └── policy.py
│
├── knowledge/
│   ├── index.py
│   ├── moss_backend.py
│   ├── retrieval.py
│   └── tenant_namespace.py
│
├── llm/
│   ├── client_pool.py
│   ├── hosted.py
│   ├── gemma.py
│   ├── prompt_builder.py
│   └── speculation.py
│
├── speech/
│   ├── safe_chunker.py
│   ├── speculative_tts.py
│   ├── audio_commit.py
│   ├── fixed_audio_cache.py
│   └── filler_policy.py
│
├── flows/
│   ├── engine.py
│   ├── state.py
│   ├── slots.py
│   └── actions.py
│
├── providers/
│   ├── deepgram.py
│   ├── cartesia.py
│   ├── plivo.py
│   └── model_clients.py
│
└── evals/
    ├── corpus/
    ├── latency/
    ├── semantic/
    └── replay.py
```

Do not require this exact module layout if a smaller refactor is cleaner, but preserve the responsibility boundaries.

---

# 40. Proposed Core Runtime Objects

## CallSession

```python
@dataclass
class CallSession:
    call_id: str
    tenant_id: str
    agent: AgentBundle
    state: dict
    slots: dict
    history: list
    turn_id: int = 0
```

## TurnCandidate

```python
@dataclass
class TurnCandidate:
    turn_id: int
    latest_interim: str = ""
    final_transcript: str = ""
    stable_prefix: str = ""
    soft_eot: bool = False
    hard_eot: bool = False
    response_plan: ResponsePlan | None = None
    response_fingerprint: str | None = None
    llm_task: asyncio.Task | None = None
    tts_task: asyncio.Task | None = None
    audio_candidate: SpeculativeAudioCandidate | None = None
```

## Cancellation

Every speculative task MUST be bound to:
- call ID,
- turn ID,
- candidate generation/version.

Never let late completion from turn N emit into turn N+1.

---

# 41. Prompt Builder Contract

```python
class PromptBuilder:
    def build(
        self,
        *,
        agent: AgentBundle,
        state: dict,
        slots: dict,
        route: RouteDecision,
        knowledge: list,
        history: list,
        user_text: str,
    ) -> list[dict]:
        ...
```

### Prompt ordering

1. invariant tenant prefix,
2. risk/safety constraints,
3. current flow state,
4. objective + allowed actions,
5. known material slots,
6. relevant knowledge,
7. minimum useful recent history,
8. current user turn.

### Do not include

- irrelevant branches,
- all FAQs,
- full authoring prose,
- unused action descriptions,
- entire transcript by default.

---

# 42. Knowledge Architecture

Moss/search should be treated as the **local retrieval/index layer**, not the whole control plane/database.

### Separate concerns

```text
Config DB / AgentBundle store:
- tenant
- versions
- flows
- policies
- model settings
- action settings

Knowledge index:
- FAQ passages
- policy passages
- product knowledge
- metadata
```

### Tenant isolation

Each retrieval record MUST include:

```text
tenant_id
agent_id
knowledge_version
document_id
risk_class
valid_from / valid_to where relevant
```

### Banking

For regulated statements, store source provenance and version so generated speech can be audited.

---

# 43. Safety and Banking Constraints

Optimization MUST NOT allow speculative speedups to bypass correctness.

### Never speculatively commit

- authentication success,
- KYC result,
- loan approval/denial,
- eligibility,
- balance,
- transaction result,
- personal customer data,
- dynamic rates unless retrieved from authoritative source.

### Safe speculative classes

Initially:
- greeting,
- navigation,
- public FAQ,
- generic product explanation,
- neutral acknowledgement.

### Risk classes

```text
LOW_PUBLIC
LOW_WORKFLOW
MEDIUM_POLICY
HIGH_PERSONAL
HIGH_TRANSACTIONAL
HIGH_REGULATED
```

Every `ResponsePlan` must carry one.

---

# 44. Metrics and Observability

Primary metric:

```text
last meaningful user audio
→ first meaningful bot audio
```

Do NOT use only “first bot audio,” because a filler can game the metric.

Track both:

```text
TTFA = time to first audio
TTFMA = time to first meaningful answer audio
```

### Required metrics

#### Turn
- user speech duration,
- VAD stop latency,
- soft EOT latency,
- hard EOT latency,
- false endpoint,
- resumed-after-soft-EOT.

#### STT
- first interim,
- final transcript,
- finalization delay,
- edit distance between speculative and final,
- semantic fingerprint stability.

#### Retrieval
- query time,
- top-k,
- cache hit,
- selected knowledge IDs.

#### LLM
- request start,
- TTFT,
- first-safe-chunk,
- completion,
- route,
- model,
- prompt tokens,
- cached prompt tokens if exposed.

#### Speculation
- response speculation attempts,
- hits,
- semantic reuse hits,
- cancellations,
- wasted tokens.

#### TTS
- request,
- first audio,
- buffered speculative ms,
- commit rate,
- discard rate,
- audio underruns.

#### Transport
- first output frame,
- outbound jitter,
- bot-start event.

### Dashboards

Report p50/p90/p95/p99 by:
- tenant,
- agent version,
- language,
- route,
- model,
- endpoint profile,
- network region.

---

# 45. Latency Budget

Create an explicit budget for each stage.

Example *engineering target structure* (numbers should be set after baseline; do not treat these as promises):

```text
hard-EOT confirmation         X ms
final semantic validation     X ms
commit                        X ms
output transport              X ms
-----------------------------------
post-hard-EOT target          Y ms
```

Separately:

```text
soft EOT → speculative LLM
soft EOT → speculative TTS
```

The point of V2 is to move expensive work out of the post-hard-EOT budget.

---

# 46. Testing Strategy

## Unit tests

- flow transition logic,
- slot validators,
- transcript stability,
- response fingerprint equality,
- safe chunk parser,
- speculative cancellation,
- audio commit rules,
- tenant namespace isolation.

## Replay tests

Feed recorded media/transcripts into the controller and reproduce:
- interim sequences,
- pauses,
- corrections,
- barge-in,
- false EOT.

## Chaos tests

Inject:
- slow Deepgram finals,
- slow LLM,
- cancelled TTS,
- dropped WebSocket,
- late speculative result,
- tool timeout,
- Goodbox/control-plane outage.

## Security tests

- cross-tenant knowledge leak,
- prompt injection,
- “ignore your instructions,”
- unauthorized personal data,
- stale policy document,
- wrong tool result attached to turn.

---

# 47. Model Evaluation Harness

Each model benchmark must use the exact runtime prompt builder.

Do not benchmark models in a generic chatbot prompt and extrapolate.

### Run

```text
GPT-4.1-mini
Gemma 4 E4B-it
candidate larger Gemma
other future fast models
```

### Same tests

- exact same AgentBundle,
- exact same retrieved context,
- exact same state/slots,
- exact same expected response contract.

### Record

```json
{
  "semantic_score": 0,
  "policy_score": 0,
  "tool_score": 0,
  "slot_score": 0,
  "ttft_ms": 0,
  "safe_chunk_ms": 0,
  "tokens_per_second": 0
}
```

---

# 48. Rollout Strategy

Use feature flags.

```text
ENABLE_AGENT_BUNDLE
ENABLE_LOCAL_RETRIEVAL
ENABLE_DYNAMIC_ENDPOINTS
ENABLE_FLUX
ENABLE_SEMANTIC_SPEC_REUSE
ENABLE_SAFE_CHUNKER
ENABLE_SPEC_TTS
ENABLE_LOCAL_LLM
ENABLE_FILLERS
ENABLE_HEDGED_MODELS
```

### Rollout order

1. shadow-only,
2. internal calls,
3. low-risk tenant,
4. low-risk routes in banking,
5. expand based on metrics.

### Instant fallback

Every optimization must have a fallback to the known V1 path.

---

# 49. Codex Implementation Rules

When implementing this document:

1. **Do not perform a wholesale rewrite first.**
2. Preserve the existing working Plivo/Pipecat pipeline until a replacement path is tested.
3. Add instrumentation before optimizing.
4. Keep each optimization behind a feature flag.
5. Use Pipecat’s current turn-management APIs rather than inventing duplicate frame semantics unless necessary.
6. Prefer Pipecat `UserTurnStrategies`, `ExternalUserTurnStrategies`, service settings, frame types, and transport primitives where they already solve the problem.
7. Keep speculative work cancel-safe and turn-scoped.
8. Never let stale async tasks write to a newer turn.
9. Never release speculative audio before fingerprint validation.
10. Preserve transcript collection and call cleanup even when the call errors.
11. Add tests for every new cancellation/commit path.
12. Do not modify model/STT/TTS providers and turn logic in the same benchmark commit unless the experiment explicitly requires it.
13. Record config/version metadata with every latency trace.
14. Treat bank/regulated routes as a stricter policy tier than public FAQ/recruitment.
15. Do not optimize benchmark numbers by adding meaningless filler audio.

---

# 50. First Concrete Refactor From Current Code

Start from the current controller and make these changes first.

### A. Introduce `ResponsePlan`

Do not change audible behavior yet.

### B. Introduce `TurnCandidate`

Move speculative task ownership into one turn object.

### C. Extend `TurnMetrics`

Add timestamps described above.

### D. Add process-wide LLM client pool

Remove per-call close for shared clients.

### E. Add `AgentBundle`

Build it from current Goodbox `call_start` response.

### F. Keep current Goodbox behavior

Goodbox remains source-of-truth temporarily; it should become an adapter, not a runtime dependency.

### G. Verify VAD settings

Log effective:
- `confidence`,
- `start_secs`,
- `stop_secs`,
- `min_volume`,
- Deepgram endpointing,
- selected Smart Turn/turn strategy.

### H. Add experiment ID

Every call log:

```text
tenant
agent_version
runtime_version
experiment_flags
provider models
endpoint profile
```

Only then begin A/B optimization.

---

# 51. Pipecat-Specific Implementation Guidance

Keep referencing current Pipecat docs during implementation.

## Turn management

Use:
- `LLMUserAggregatorParams`,
- `UserTurnStrategies`,
- `VADUserTurnStartStrategy`,
- `TranscriptionUserTurnStartStrategy`,
- `TurnAnalyzerUserTurnStopStrategy`,
- `LocalSmartTurnAnalyzerV3`,
- or `ExternalUserTurnStrategies` for Flux.

Docs:
- https://docs.pipecat.ai/api-reference/server/utilities/turn-management/user-turn-strategies

## Deepgram

Use `DeepgramSTTService` for Nova or `DeepgramFluxSTTService` for Flux.

For Flux:
- do not combine with competing local turn-stop ownership,
- use runtime setting updates if testing dynamic EOT.

Docs:
- https://docs.pipecat.ai/api-reference/server/services/stt/deepgram

## Cartesia

Be explicit about:
- WebSocket service,
- aggregation mode,
- sample rate,
- server buffering,
- streaming chunks.

Docs:
- https://docs.pipecat.ai/api-reference/server/services/tts/cartesia

## Flows

Use Pipecat Flows as a conceptual/reference architecture for state transitions and functions even if the first V2 implementation keeps the custom controller.

Docs:
- https://docs.pipecat.ai/api-reference/pipecat-flows/flow-manager

## Service settings

Keep provider configuration explicit and versioned.

Docs:
- https://docs.pipecat.ai/pipecat/fundamentals/service-settings

---

# 52. Definition of Done for Optimize V2

Optimize V2 is not done when one synthetic latency benchmark is faster.

It is done when:

- tenant behavior is compiled into structured runtime artifacts,
- no authoring-system fetch exists on the conversational hot path,
- flow/state logic is separate from speech generation,
- deterministic/cached routes exist,
- knowledge is retrieved locally or from a low-latency local index,
- endpointing is measured and state-aware,
- soft/hard EOT is implemented or equivalent behavior exists,
- speculative LLM work is semantically validated,
- first-safe-chunk TTS beats sentence-only latency without unacceptable prosody loss,
- speculative TTS is two-phase committed,
- provider clients/connections are reused where appropriate,
- all speculation is cancel-safe,
- regulated routes have stricter commit rules,
- p50/p95/p99 latency is observable end to end,
- TTFMA improves materially rather than only TTFA,
- local model routing passes product-specific quality gates,
- every feature has a rollback flag,
- production no longer depends on ngrok.

---

# 53. Priority Order

If engineering capacity is limited, implement in this order:

```text
1. Instrumentation
2. Effective VAD/turn settings audit
3. State-aware endpoint experiments
4. Shared connections/client pools
5. AgentBundle + compiled prompt
6. Local retrieval
7. SafeSpeechChunker
8. Semantic ResponseFingerprint
9. Speculative TTS two-phase commit
10. Deterministic/cached response routes
11. Model router
12. Gemma shadow evaluation
13. Prefix/KV caching
14. Speculative decoding
15. Advanced hedging
```

This order intentionally postpones model replacement.

---

# 54. Key Hypotheses to Validate

1. The current effective VAD/turn configuration may be contributing more latency than the LLM.
2. Sentence aggregation may be hiding a large chunk of otherwise-available streaming latency gains.
3. Semantic speculation reuse can materially outperform exact-transcript reuse.
4. Speculative TTS can hide most TTS startup time if commit validation is reliable.
5. A compiled prompt + local retrieval will reduce prefill and improve model consistency.
6. A local model may outperform hosted GPT in TTFT only if it is kept warm, colocated, and correctly served.
7. Prefix caching may matter more than speculative decoding for short voice responses.
8. Fixed/cached routes will beat any generative optimization for common FAQs and control utterances.
9. TTFMA is a more honest product metric than TTFA.
10. The best architecture will differ by route risk; there should not be one universal fast path.

---

# 55. References

### Pipecat
- Turn strategies: https://docs.pipecat.ai/api-reference/server/utilities/turn-management/user-turn-strategies
- Deepgram STT / Flux: https://docs.pipecat.ai/api-reference/server/services/stt/deepgram
- Cartesia TTS: https://docs.pipecat.ai/api-reference/server/services/tts/cartesia
- FlowManager: https://docs.pipecat.ai/api-reference/pipecat-flows/flow-manager
- Service settings: https://docs.pipecat.ai/pipecat/fundamentals/service-settings

### Models
- GPT-4.1-mini: https://developers.openai.com/api/docs/models/gpt-4.1-mini
- Gemma 4 E4B instruction model: https://huggingface.co/google/gemma-4-E4B-it

---

# Appendix A — Existing V1 Behaviors That Must Be Preserved During Migration

- Plivo bidirectional media remains functional.
- Pipecat transport remains interruption-safe.
- Deepgram streaming interim/final transcript handling remains functional.
- User-turn start and stop callbacks remain authoritative.
- Speculative LLM work is cancelled on a new real turn.
- Completed user/assistant turns continue to be persisted.
- Call cleanup still sends final transcript metadata to the control plane.
- End-call semantics still terminate Plivo reliably.
- Browser/test path remains separately functional if still needed.
- Existing latency observer remains available until V2 metrics supersede it.

---

# Appendix B — Response Fingerprint Example

```python
from dataclasses import dataclass
import hashlib
import json

@dataclass(frozen=True)
class ResponseFingerprint:
    tenant_id: str
    agent_version: str
    state: str
    intent: str
    risk_class: str
    knowledge_version: str
    material_slots: tuple[tuple[str, str], ...]
    tool_dependency: str | None

    def digest(self) -> str:
        payload = {
            "tenant_id": self.tenant_id,
            "agent_version": self.agent_version,
            "state": self.state,
            "intent": self.intent,
            "risk_class": self.risk_class,
            "knowledge_version": self.knowledge_version,
            "material_slots": self.material_slots,
            "tool_dependency": self.tool_dependency,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()
```

---

# Appendix C — Safe Chunker Pseudocode

```python
class SafeSpeechChunker:
    def __init__(self, min_chars=40, max_wait_ms=120):
        self.buffer = ""
        self.first_token_at = None
        self.min_chars = min_chars
        self.max_wait_ms = max_wait_ms

    def push(self, delta: str, now: float) -> list[str]:
        if self.first_token_at is None:
            self.first_token_at = now

        self.buffer += delta

        if self._inside_number_or_currency(self.buffer):
            return []

        if self._ends_in_unsafe_negation(self.buffer):
            return []

        if self._has_clause_boundary(self.buffer) and len(self.buffer) >= self.min_chars:
            return [self._pop_safe_prefix()]

        if self._wait_exceeded(now) and self._has_safe_word_boundary(self.buffer):
            return [self._pop_safe_prefix()]

        return []
```

The production implementation must be language-aware and covered by tests.

---

# Appendix D — Two-Phase Audio Commit Pseudocode

```python
async def prepare_speculative_audio(turn: TurnCandidate):
    plan = await build_response_plan(turn.latest_interim)
    fp = fingerprint(plan)

    text_stream = stream_llm(plan)
    async for safe_chunk in safe_chunker(text_stream):
        pcm = await synthesize_chunk(safe_chunk)
        turn.audio_candidate = SpeculativeAudioCandidate(
            fingerprint=fp,
            transcript_basis=turn.latest_interim,
            text=safe_chunk,
            pcm_chunks=[pcm],
            sample_rate=24000,
        )
        if audio_ms(pcm) >= SPEC_TTS_MAX_AUDIO_MS:
            break


async def commit_turn(turn: TurnCandidate, final_text: str):
    final_plan = await build_response_plan(final_text)
    final_fp = fingerprint(final_plan)

    candidate = turn.audio_candidate

    if candidate and candidate.fingerprint == final_fp and not candidate.invalidated:
        candidate.committed = True
        await release_pcm(candidate.pcm_chunks)
    else:
        if candidate:
            candidate.invalidated = True
        await generate_normal_response(final_plan)
```

Real implementation must avoid duplicate TTS continuation and must preserve audio context/prosody.

---

# Appendix E — Example AgentBundle Fragments

## Recruitment

```yaml
flow:
  state: QUALIFY_HIRING
  allowed_transitions:
    hiring: REQUIREMENTS
    no_hiring: FUTURE_NEED
    existing_agency: EXISTING_PARTNER
    not_interested: EXIT

slots:
  role:
    type: string
  volume:
    type: integer
  timeline:
    type: string

risk:
  class: LOW_WORKFLOW
```

## Banking

```yaml
flow:
  state: KYC_COLLECTION
  allowed_transitions:
    valid_field: KYC_NEXT_FIELD
    invalid_field: KYC_RETRY
    user_refuses: HUMAN_OR_EXIT

slots:
  pan:
    type: string
    sensitive: true
  dob:
    type: date
    sensitive: true

risk:
  class: HIGH_PERSONAL

speculation:
  llm: restricted
  tts_prepare: false
  audio_commit: false
```

---

# Appendix F — Experiment Matrix

| Experiment | Variant A | Variant B | Primary metric |
|---|---|---|---|
| VAD stop | current | 0.2s | TTFMA + false EOT |
| Turn model | Smart Turn | Flux EagerEOT | TTFMA + false EOT |
| TTS aggregation | sentence | safe chunk | first meaningful audio |
| Spec reuse | exact transcript | semantic fingerprint | hit rate / wrong commit |
| TTS | normal | speculative | post-EOT latency |
| Prompt | monolithic | compiled dynamic | TTFT + quality |
| LLM | GPT-4.1-mini | Gemma E4B | quality + TTFT |
| Cache | off | prefix/KV | TTFT |
| Decode | normal | speculative | first-safe-chunk |
| Network | ngrok | regional ingress | transport latency |

---

# Appendix G — Non-Negotiable Invariants

1. A speculative result must never escape its turn.
2. Speculative audio must never be audible before commit.
3. High-risk banking facts must come from authoritative data.
4. Tenant knowledge and caches must be strictly namespaced.
5. Latency optimization must not reduce interruption quality below an agreed threshold.
6. Every optimization must be observable.
7. Every optimization must be independently disable-able.
8. A faster but less correct route is not an optimization.
9. “First audio” is not sufficient; measure first meaningful audio.
10. Prefer Pipecat’s native primitives where they already provide the needed lifecycle semantics.
