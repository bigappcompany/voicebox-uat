# Improvements.md — Finalized Latency & Runtime Improvements for Appzo VoiceAgent V2

**Status:** Finalized improvement plan  
**Target runtime:** Pipecat + Plivo + Deepgram Flux/Nova + hosted/local LLM + Cartesia/Magpie  
**Current observed baseline:** p50 ≈ 523 ms, p95 ≈ 1,012 ms native-EOT → bot-audio  
**Primary objective:** Reduce caller-perceived latency while preserving interruption safety, correctness, and multi-tenant behavior.

**Implementation record:** [IMPROVEMENTS_EXECUTION_PLAN.md](IMPROVEMENTS_EXECUTION_PLAN.md)

---

# 0. Executive Summary

The current V2 pipeline is already close to the performance target. The remaining latency is concentrated in a few places rather than spread evenly across the stack.

The strongest evidence from the current logs is:

```text
native-EOT → bot-audio:
p50 ≈ 523 ms
p95 ≈ 1,012 ms

provider-EOT → aggregator:
≈ 2–7 ms

deterministic routes:
≈ 92–103 ms EOT → bot-audio

hosted/speculative routes:
≈ 273–1,012 ms EOT → bot-audio
```

This means:

```text
Pipecat frame transport is NOT the primary remaining bottleneck.

The remaining latency is mainly in:
1. delayed first-safe-text generation,
2. speculation starting too late,
3. unnecessary sentence-level/booking-level buffering,
4. delayed provider end-of-turn,
5. long/redundant hosted prompts,
6. runtime TTS use for known fixed utterances,
7. underuse of deterministic routing.
```

The finalized implementation order is:

```text
1. Cached greeting audio
2. Deterministic callback/booking state machine
3. Structured conversation-fact retention
4. Remove 80 ms speculation commit wait
5. Start speculation from stable interims before EagerEOT
6. Tune Flux endpointing
7. Make speculative TTS actually produce PCM before hard EOT
8. Upgrade exact-match speculation to semantic response fingerprints
9. Finish AgentBundle prompt compilation
10. Expand deterministic routing + answer caching
11. Measure first audible phoneme, not only first audio packet
12. Only then benchmark model/provider replacement
```

The central architecture principle is:

> **Move work earlier, not speech earlier.**

The runtime should become increasingly aggressive about:
- routing,
- retrieval,
- prompt construction,
- LLM start,
- speculative TTS preparation,

while remaining conservative about:
- audible output,
- irreversible actions,
- regulated claims,
- callback/booking confirmation,
- banking/KYC statements.

---

# 1. Move Greeting Audio Completely Out of Runtime TTS

## Problem

The greeting is fully known before the call starts, yet it currently waits for:
- Pipecat worker initialization,
- provider startup,
- Cartesia websocket readiness,
- Cartesia TTFB,
- provider-generated leading silence.

This is unnecessary.

For the current run, greeting startup included roughly:

```text
pipeline startup
+
Cartesia TTFB ~117 ms
+
leading silence ~106 ms
```

This delay is avoidable.

## Target architecture

At agent publish time:

```text
intro text
   ↓
selected voice
   ↓
TTS generation
   ↓
trim leading/trailing silence
   ↓
convert to 8 kHz μ-law
   ↓
cache by agent version
```

Then at call time:

```text
Plivo stream established
      │
      ├── immediately send cached greeting audio
      │
      └── concurrently initialize:
            Deepgram/Flux
            Cartesia
            LLM clients
            Pipecat worker
            retrieval/runtime state
```

This turns greeting playback into a parallel warm-up window.

## Cache key

Recommended:

```python
GreetingCacheKey = (
    tenant_id,
    agent_id,
    agent_version,
    voice_id,
    tts_model,
    speed,
    intro_text_hash,
)
```

## Output format

Prefer storing greeting audio already in Plivo-native form:

```text
audio/x-mulaw
8000 Hz
mono
```

This avoids:

```text
24 kHz PCM
→ Pipecat
→ resampling
→ μ-law conversion
```

for a static asset.

## Concurrency concern

Do not allow:
- direct WebSocket greeting writes,
- Pipecat transport writes,

to race on the same socket.

Use either:

```text
A. one shared outbound writer
```

or:

```text
B. send greeting first,
   then transfer outbound ownership to Pipecat
```

## Barge-in

The greeting must remain interruptible.

If caller speech begins:

```text
user-start event
→ clear pending greeting audio
→ resume normal Pipecat interaction
```

## Metrics

Add:

```text
PlivoStart → greeting first packet
PlivoStart → greeting first non-silent sample
PlivoStart → pipeline ready
```

## Acceptance criteria

- No runtime TTS request for greeting.
- No duplicate greeting through Pipecat.
- Caller can barge in.
- First greeting phoneme consistently starts earlier than current path.
- Cached greeting invalidates when text/voice/version changes.

## Pipecat/Plivo reference

Keep Pipecat handling the normal live transport. Use Plivo’s bidirectional media path only for the fixed cached opening.

Pipecat transport docs:
- https://docs.pipecat.ai/api-reference/server/services/transport/fastapi-websocket

---

# 2. Replace Broad Booking Guarding with Deterministic Callback Routing

## Problem

The current booking/callback safety layer buffers too much generated text before releasing speech.

This is visible in slow hosted turns where:

```text
LLM TTFT            ~539 ms
first safe text     ~920 ms
bot audio           ~1,012 ms
```

The system is waiting too long for enough language to determine whether a hosted response makes an unsafe scheduling claim.

This is a policy problem being solved too late in the generation path.

## Correct architecture

Move callback/booking semantics into deterministic state.

Example:

```text
FOLLOWUP_OFFERED
    │
    ├── user declines
    │      ↓
    │    CONTINUE_OR_CLOSE
    │
    └── user accepts
           ↓
       CALLBACK_DAY_TIME
           │
           ├── only day
           │      ↓
           │   ASK_TIME
           │
           ├── only time
           │      ↓
           │   ASK_DAY
           │
           └── day + time
                  ↓
             RECORD_PREFERENCE
```

Important distinction:

```text
"recorded preference"
!=
"scheduled"
```

Unless a real booking/calendar tool confirms the event.

## Runtime rules

Allowed deterministic wording:

```text
"What day would you prefer?"
"What time would be convenient?"
"Thanks, I’ve noted that preference."
```

Forbidden without tool confirmation:

```text
"Your call is scheduled."
"Confirmed for tomorrow."
"Our manager will call you at 4 PM."
```

## BookingClaimGuard changes

Current broad usage:

```text
hosted response
→ BookingClaimGuard
→ sentence buffering
→ SafeSpeechChunker
```

Target:

```text
normal hosted response
→ SafeSpeechChunker directly

booking-sensitive hosted response
→ BookingClaimGuard
→ SafeSpeechChunker
```

## Guard activation

Only enable the guard if:

```python
response_plan.requires_booking_guard is True
```

Possible triggers:
- current state is scheduling-related,
- user requests booking,
- route involves appointment/follow-up,
- hosted response references a booking tool result.

## Benefits

Expected affected-turn improvement:

```text
~200–450 ms
```

depending on how much sentence buffering is removed.

## Acceptance criteria

- Ordinary FAQ/qualification text streams immediately.
- Scheduling claims remain safe.
- No “confirmed/scheduled” wording without authoritative tool result.
- Callback flow works even if hosted LLM is disabled.

---

# 3. Add Structured Conversation-Fact Retention

## Problem

The current conversation sometimes receives information and asks for it again.

Example pattern:

```text
User:
"We plan to hire in five months."

Later bot:
"What is your expected timeline?"
```

This increases:
- user frustration,
- response length,
- LLM context size,
- TTS duration,
- unnecessary turns.

## Correct architecture

Use structured slots as the source of truth for facts.

For recruitment:

```python
RecruitmentSlots = {
    "hiring_status": None,
    "roles": [],
    "departments": [],
    "headcount": None,
    "headcount_by_role": {},
    "hiring_timeline": None,
    "followup_consent": None,
    "callback_day": None,
    "callback_time": None,
}
```

For banks, define a completely separate schema.

Example:

```python
BankingSlots = {
    "intent": None,
    "loan_product": None,
    "requested_amount": None,
    "tenure": None,
    "kyc_topic": None,
    "authenticated": False,
}
```

## Extraction order

Do not add another hosted LLM call sequentially after every user turn.

Use:

```text
1. deterministic parser
2. local/fast extractor
3. hosted extractor only if needed
```

Examples suitable for deterministic extraction:
- yes/no,
- day names,
- common relative dates,
- simple headcounts,
- callback consent,
- phone-style time expressions.

Examples possibly needing a small model:
- multiple roles/departments,
- freeform timeline phrases,
- ambiguous multi-slot utterances.

## Slot update timing

On each completed user turn:

```text
final transcript
   ↓
extract/update slots
   ↓
flow transition
   ↓
build response plan
   ↓
hosted prompt if needed
```

For speculative execution, provisional slots may be extracted from stable interims but must not be committed until final validation.

## Hosted prompt integration

Prompt builder should inject:

```text
KNOWN FACTS:
Hiring status: yes
Roles: operations, technology
Headcount: five each
Timeline: five months
Follow-up consent: unknown

DO NOT ask again for facts already known.
```

## History reduction

Once slots are reliable, reduce conversation history.

Benchmark:

```text
1 previous turn
vs
2 previous turns
vs
3 previous turns
```

The old optimized controller used only the previous user/assistant exchange. That is worth re-testing because structured slots can replace much of raw history.

## Acceptance criteria

- Known facts are not re-requested.
- Slots update correctly on correction:
  - user says "actually six, not five"
  - slot becomes six.
- Slot updates are tenant-specific.
- Sensitive banking slots are never logged insecurely.
- Conversation history can be reduced without losing required context.

---

# 4. Remove the Fixed ~80 ms Speculation Commit Wait

## Problem

The current speculation-hit path appears to intentionally or indirectly wait about 80 ms before promoting the candidate.

Observed pattern:

```text
hard EOT
→ ~80 ms
→ speculation-hit commit
```

This is now significant because the full p95 target miss is only around 12 ms above one second.

## Correct behavior

When hard EOT arrives:

```text
final transcript
   ↓
candidate validation
   ↓
if valid:
    promote immediately
```

Do NOT wait for:
- another token,
- another timer,
- another debounce window.

The already-running request should continue as the canonical request.

## Important distinction

Promote:

```text
request ownership
```

not necessarily:

```text
audio immediately
```

Text still must satisfy:
- safe chunking,
- policy guard,
- TTS commit rules.

## Target flow

```python
if candidate_is_valid:
    state.final_request = candidate
    candidate.speculative = False
    candidate.promoted = True
    await release_available_safe_text(candidate)
```

## Cancellation

On invalid candidate:

```text
cancel immediately
start final request
```

## Metrics

Add:

```text
hard_eot_at
candidate_validated_at
candidate_promoted_at
```

Target:

```text
hard EOT → promoted
< 10 ms
```

excluding any intentional validation work.

## Acceptance criteria

- No fixed 80 ms timer on speculation hit.
- No duplicate LLM request after promotion.
- No stale candidate leakage into next turn.
- Speculation miss behavior remains correct.

---

# 5. Start Speculation Earlier Using Stable Interims, Not Only EagerEOT

## Problem

Current V2 speculation often starts too close to hard EOT.

The speculative TTS channel is warm, but it usually reports:

```text
spec_tts=none
```

because no safe text exists early enough.

Example pattern:

```text
SOFT EOT
→ 230 ms
→ HARD EOT

LLM TTFT
→ 571 ms
```

The LLM cannot possibly generate anything before hard EOT.

## Recommended architecture

Use three confidence levels.

### Level 1 — Normal stable interim

Allowed:
- deterministic routing,
- retrieval,
- prompt construction,
- intent guess,
- low-risk LLM start.

Not allowed:
- audible output,
- irreversible state mutation.

### Level 2 — Flux EagerEOT

Allowed:
- promote speculation confidence,
- start private speculative TTS once safe text exists.

Still not allowed:
- audible output.

### Level 3 — Hard EOT

Allowed:
- final transcript validation,
- commit candidate,
- release buffered audio.

## Transcript stability

Use the existing `TranscriptStabilityAnalyzer` for every interim.

Target structure:

```python
@dataclass
class StableTranscriptState:
    full_interim: str
    stable_prefix: str
    unstable_suffix: str
    stable_words: int
    stable_since: float
    semantic_route: str | None
```

## Early speculation criteria

Possible starting policy:

```text
stable_words >= 4–6
AND
stable_duration >= 60–120 ms
AND
route confidence high enough
AND
no high-risk slot ambiguity
```

These values must be benchmarked.

## Old-main behavior worth preserving

The old optimized controller used:

```text
min words: 3
min chars: 12
debounce: 80 ms
max restarts: 2
```

V2 should borrow the early-start philosophy but keep the newer safety model.

## Restart limit

Keep bounded speculative churn:

```text
early LLM restarts <= 2
speculative TTS starts <= 1
```

## TurnResumed

On Flux `TurnResumed`:

```text
cancel or stale-mark:
- response candidate
- private TTS task
- prepared PCM

retain:
- stable retrieval cache
if still semantically reusable
```

## Acceptance criteria

- Earlier LLM start observed before EagerEOT on stable interims.
- No audible output before hard EOT.
- TurnResumed reliably cancels stale work.
- Speculation hit rate improves.
- False spoken speculative output remains zero.

---

# 6. Tune Flux Endpointing for Caller-Perceived Latency

## Problem

Provider EOT → Pipecat aggregator is already excellent:

```text
~2–7 ms
```

The larger delay is:

```text
raw speech end → provider EOT
~450–900 ms
```

Therefore the main EOT latency is upstream of Pipecat aggregation.

## Metrics to distinguish

Always track both:

```text
A. provider EOT → bot audio
```

This measures response generation.

And:

```text
B. last meaningful caller audio → bot audio
```

This measures caller-perceived latency.

## Flux parameters to A/B

Tune:

```text
eager_eot_threshold
eot_threshold
eot_timeout_ms
```

Start conservatively.

Example experiment grid:

```text
eager_eot_threshold:
0.55
0.50
0.45
0.40

final eot threshold:
current
slightly lower

timeout:
current
-100 ms
-200 ms
```

Do not change all parameters at once.

## Test corpus

Must include:
- short yes/no,
- "uh... yes",
- long qualification responses,
- numbers,
- dates,
- phone numbers,
- rupee amounts,
- Hindi,
- Hinglish,
- noisy mobile calls,
- long natural pauses,
- self-corrections.

## Resume safety

Retain Flux resume handling:

```text
early EOT prediction
   ↓
caller resumes
   ↓
cancel speculative work
```

## Important recommendation

Do **not** use lower EagerEOT confidence as the only mechanism for earlier speculation.

First use stable interims for upstream work.

Then independently tune EagerEOT.

This separates:

```text
more compute overlap
```

from:

```text
more aggressive turn prediction
```

## Parallel benchmark

Keep an alternative branch:

```text
Nova-3
+ Silero VAD
+ Pipecat LocalSmartTurnAnalyzerV3
```

and later:

```text
Nemotron streaming ASR
+ Pipecat turn management
```

Pipecat turn management reference:
- https://docs.pipecat.ai/api-reference/server/utilities/turn-management/user-turn-strategies

Deepgram Pipecat reference:
- https://docs.pipecat.ai/api-reference/server/services/stt/deepgram

## Acceptance criteria

- Reduced raw-speech-end → hard EOT.
- No unacceptable false endpoint increase.
- Resume recovery works.
- Improvement holds at p95, not only p50.

---

# 7. Make Speculative TTS Produce Useful PCM Before Hard EOT

## Problem

The private speculative Cartesia connection is warming correctly, but the pipeline rarely gets safe text early enough.

This means the speculative TTS architecture exists but is underutilized.

## Target flow

```text
stable interim
    ↓
early LLM request
    ↓
safe phrase generated
    ↓
EagerEOT confidence reached
    ↓
private TTS starts
    ↓
PCM buffered privately
    ↓
hard EOT
    ↓
validate
    ↓
commit PCM
```

## Private TTS task

Do not block the LLM receive loop while waiting for TTS.

Bad:

```text
LLM token processing
→ await speculative TTS preparation
→ resume LLM
```

Target:

```python
state.spec_tts_task = asyncio.create_task(
    prepare_speculative_audio(...)
)
```

LLM streaming continues independently.

## Cap speculative audio

Prepare only the first:

```text
~300–600 ms
```

of audio initially.

This is enough to hide:
- TTS TTFB,
- initial synthesis,
- output startup.

Do not speculatively synthesize the full response.

## Candidate state

```python
@dataclass
class SpecTTSState:
    fingerprint: str
    text_basis: str
    pcm_chunks: list[bytes]
    audio_ms: int
    ready_at: float | None
    committed: bool = False
    invalidated: bool = False
```

## Commit

At hard EOT:

```text
candidate valid?
   yes → release prepared PCM
   no  → discard
```

Continue normal TTS after buffered audio begins.

## Metrics

Add:

```text
spec_tts_started_at
spec_tts_first_pcm_at
spec_tts_audio_ms
spec_tts_commit_at
spec_tts_discard_at
```

Report:

```text
spec_tts=hit
spec_tts=miss
spec_tts=cancelled
spec_tts=not_started
```

## Acceptance criteria

- Non-zero `spec_tts=hit` rate on eligible hosted turns.
- No speculative audio leaks before commit.
- No double-speech when committed PCM transitions to normal TTS.
- TurnResumed cancels prepared audio.

---

# 8. Upgrade Exact Transcript Matching to Semantic Response Fingerprints

## Phase 1

Keep exact normalized text matching while making speculation earlier.

This is the safest way to tune:
- thresholds,
- cancellation,
- timing.

## Problem with exact matching

These are semantically equivalent:

```text
"how much is the personal loan interest"

"what is the interest rate for the personal loan"
```

Exact matching causes a miss even if:
- route,
- knowledge,
- state,
- material slots,

are unchanged.

## Response fingerprint

Introduce:

```python
@dataclass(frozen=True)
class ResponseFingerprint:
    tenant_id: str
    agent_version: str
    state: str
    intent_id: str
    risk_class: str
    knowledge_ids: tuple[str, ...]
    material_slots: tuple[tuple[str, str], ...]
    tool_dependency: str | None
```

## Match rule

Speculative candidate may be promoted if:

```text
candidate_fingerprint == final_fingerprint
```

even if transcript wording differs.

## High-risk requirement

For banking/KYC, include every response-relevant slot.

Example:

```text
loan_product=home_loan
amount=500000
tenure_months=24
```

A changed amount invalidates the candidate.

## Knowledge dependency

If retrieval returns:

```text
KB doc 17
KB doc 23
```

those IDs become part of the fingerprint.

If final transcript maps to another KB source, invalidate.

## Migration order

```text
1. exact text only
2. exact intent + slots
3. full response fingerprint
```

## Acceptance criteria

- Semantic hit rate improves.
- No increase in wrong-answer commit.
- Every speculative commit is auditable from fingerprint inputs.

---

# 9. Finish the AgentBundle Prompt Compiler

## Problem

The architecture has an AgentBundle, but the runtime prompt can still include a large amount of authoring prose.

The goal is not merely to store Goodbox config in another object.

The goal is to transform:

```text
human authoring configuration
```

into:

```text
runtime-optimized policy/state/knowledge representation
```

## Compile-time responsibilities

At publish time, produce:

```text
identity
invariant policy
language policy
flow graph
state schema
slot schema
deterministic responses
actions
knowledge index
risk rules
routing policy
STT keyterms
cached audio manifest
```

## Hosted prompt should contain only

```text
1. compact invariant system policy
2. current state
3. current objective
4. known slots
5. allowed actions
6. selected knowledge
7. minimal conversation history
8. current user
```

## Example

Instead of:

```text
full hiring script
all steps
all FAQs
all callback wording
all branch instructions
```

send:

```text
STATE: REQUIREMENTS

KNOWN:
roles = operations, tech
headcount = 5 each
timeline = unknown

OBJECTIVE:
collect timeline
then offer follow-up

ALLOWED:
ASK_TIMELINE
OFFER_FOLLOWUP
EXIT
```

## Token instrumentation

Log:

```text
prompt_total_tokens
prompt_invariant_tokens
prompt_state_tokens
prompt_history_tokens
prompt_knowledge_tokens
```

Graph:

```text
input tokens
vs
LLM TTFT
```

## Prompt cache compatibility

Keep invariant prefix stable byte-for-byte for:
- same tenant,
- same agent version,
- same tool schema.

This improves hosted prompt caching and self-hosted prefix/KV reuse.

## Acceptance criteria

- Large Goodbox authoring text does not directly appear in every hosted request.
- Current flow state determines which instructions are sent.
- Prompt tokens decrease materially.
- Response quality does not regress.

---

# 10. Expand Deterministic Routing and Stable Answer Caching

## Evidence

Current deterministic routes are dramatically faster than hosted turns.

Observed deterministic examples:

```text
~92 ms
~103 ms
```

EOT → bot audio.

This proves that avoiding hosted generation is the strongest optimization for common turns.

## Deterministic candidates

Implement local handling for:

```text
greeting
identity
repeat
goodbye
busy
wrong person
not interested
callback consent
callback day
callback time
yes/no
simple confirmation
call closing
human transfer request
known stable FAQs
```

## Routing hierarchy

```text
1. deterministic
2. exact cache
3. retrieval/template
4. local model
5. hosted model
6. tool/human
```

## Cache key

For stable responses:

```python
(
    tenant_id,
    agent_version,
    normalized_intent,
    state,
    relevant_slot_hash,
    language,
    voice_version,
)
```

## Text cache vs audio cache

### Text cache

Useful when:
- response text stable,
- voice selection dynamic.

### Audio cache

Useful when:
- response text stable,
- voice fixed.

Examples:
- greeting,
- goodbye,
- identity,
- common neutral acknowledgement.

## Banking safety

Never cache:
- account-specific facts,
- KYC status,
- loan decision,
- dynamic eligibility,
- balances,
- transaction results.

Only cache:
- static public FAQ,
- approved policy wording,
- navigation speech.

## Acceptance criteria

- Hosted-route percentage decreases.
- Deterministic-route coverage increases without behavior regressions.
- All caches tenant/version namespaced.
- No cross-tenant reuse.

---

# 11. Measure First Audible Phoneme, Not Only First Audio Frame

## Problem

Provider TTFB can look healthy while the actual waveform still begins with silence.

Observed pattern:

```text
Cartesia TTFB ~82–95 ms

but TTFA:
~200–320 ms

with leading silence:
~120–225 ms
```

The user cares about:

```text
first audible phoneme
```

not:

```text
first PCM packet
```

## New metric

Add a non-silence detector to outbound PCM telemetry.

Example:

```python
def is_audible_pcm(audio: bytes, threshold: float) -> bool:
    ...
```

Record:

```text
tts_first_packet_at
tts_first_non_silent_pcm_at
output_first_packet_at
output_first_non_silent_pcm_at
```

## Main user metric

Prefer:

```text
last meaningful caller audio
→ first meaningful bot phoneme
```

over merely:

```text
native EOT
→ first output frame
```

## Use cases

This is essential when comparing:

```text
Cartesia
vs
Magpie
vs
cached audio
```

A provider with 50 ms TTFB and 200 ms leading silence is not really 50 ms from the user's perspective.

## Fixed audio

Trim:
- greeting,
- fillers,
- closing,
- cached FAQ audio,

offline.

## Acceptance criteria

- All latency dashboards include first non-silent audio.
- Provider benchmarks use the same audible threshold.
- No optimization is accepted purely because first-packet timing improves.

---

# 12. Only After the Runtime Fixes, Benchmark Model/Provider Replacement

## Reason

The current pipeline already demonstrates:

```text
~100 ms deterministic response path
```

This means the transport/orchestration path is capable of low latency.

The slow hosted path is still dominated by:
- safe-text delay,
- LLM TTFT,
- EOT timing,
- prompt size,
- speculation timing.

Model replacement should therefore come after runtime fixes.

---

## 12.1 ASR Benchmark

Compare:

```text
A. Deepgram Flux + ExternalTurn
B. Deepgram Nova-3 + Pipecat SmartTurn
C. NVIDIA Nemotron 3.5 Streaming + Pipecat SmartTurn/custom EOT
```

Hold constant:
- controller,
- LLM,
- TTS,
- prompt,
- routing.

Measure:

```text
raw audio end → hard EOT
final transcript latency
WER / semantic accuracy
false endpoint rate
resume rate
Hindi/Hinglish quality
numbers/dates quality
```

Pipecat NVIDIA STT reference:
- https://docs.pipecat.ai/api-reference/server/services/stt/nvidia

---

## 12.2 TTS Benchmark

Compare:

```text
Cartesia
vs
local NVIDIA Magpie
```

Measure:

```text
request → first packet
request → first audible phoneme
quality
prosody
Hindi/Hinglish pronunciation
voice suitability
concurrency jitter
p95/p99
```

Do not compare provider marketing TTFB against caller-perceived TTFA.

Pipecat NVIDIA TTS reference:
- https://docs.pipecat.ai/api-reference/server/services/tts/nvidia

Pipecat Cartesia reference:
- https://docs.pipecat.ai/api-reference/server/services/tts/cartesia

---

## 12.3 LLM Benchmark

Compare:

```text
GPT-4.1-mini
vs
Nemotron 3 Nano
vs
other local fast models
```

Measure:

```text
TTFT
first-safe-text
tokens/sec
prompt cache hit behavior
concurrency
Hinglish/Hindi quality
flow compliance
tool correctness
hallucination
response brevity
```

Do not globally switch until tenant-specific evals pass.

## Important locality benefit

The strongest reason to self-host is:

```text
Pipecat
→ local/VPC model
```

instead of:

```text
Pipecat
→ WAN
→ provider
```

You gain:
- lower network variance,
- prefix/KV caching,
- queue control,
- batching control,
- GPU scheduling control,
- speculative decoding options.

---

# 13. Final Recommended Runtime Flow

```text
                 CALL START
                     │
                     ├── cached greeting μ-law → Plivo immediately
                     │
                     └── warm/init runtime concurrently
                              │
                              ▼

                    STREAMING CALL AUDIO
                              │
                              ▼
                     Deepgram Flux / ASR
                              │
                ┌─────────────┴──────────────┐
                │                            │
           normal interims               EagerEOT
                │                            │
                ▼                            ▼
       TranscriptStability             confidence boost
                │                            │
                ▼                            │
      route / retrieval / LLM                │
                │                            │
                └──────────────┬─────────────┘
                               ▼
                       SafeSpeechChunker
                               │
                               ▼
                   private speculative TTS
                               │
                         buffered PCM
                               │
                               ▼
                           HARD EOT
                               │
                      validate candidate
                               │
               ┌───────────────┴────────────────┐
               │                                │
             valid                            invalid
               │                                │
        commit buffered audio            discard/recompute
               │
               ▼
           Pipecat → Plivo
```

---

# 14. Revised Latency Budget

## Fixed/deterministic route

Target:

```text
hard EOT
→ route
→ cached/fixed response
→ TTS/audio
```

Expected:

```text
~50–150 ms
```

depending on TTS/cache path.

## Speculative hosted route

Ideal:

```text
while user speaks:
route
retrieval
LLM TTFT
safe text
TTS prepare

hard EOT:
validate
release
```

Target:

```text
hard EOT → audio
~50–250 ms
```

for successful speculative hits.

## Non-speculative hosted route

Target:

```text
hard EOT
→ hosted LLM
→ safe chunk
→ TTS
```

Expected:

```text
~350–700 ms
```

with p95 under 1 second.

## Caller-perceived target

Track separately:

```text
last meaningful user speech
→ first meaningful bot phoneme
```

Consistently sub-second caller-perceived latency requires:
- aggressive but safe endpointing,
- early speculation,
- short first phrase,
- low TTS startup.

---

# 15. Implementation Order

Do not implement all 12 simultaneously.

Recommended sequence:

## Phase 1 — Cheap deterministic wins

1. cached greeting audio,
2. deterministic booking/callback flow,
3. slot/fact retention,
4. remove fixed speculation commit delay.

## Phase 2 — Speculation improvements

5. stable-interim speculation,
6. Flux tuning,
7. speculative TTS hit path.

## Phase 3 — Semantic/prompt improvements

8. response fingerprints,
9. prompt compiler,
10. deterministic routing expansion.

## Phase 4 — Measurement correctness

11. first-audible-phoneme metrics.

## Phase 5 — Provider/model experiments

12. ASR/TTS/LLM replacement tests.

---

# 16. Feature Flags

Every improvement should be independently reversible.

Recommended:

```text
ENABLE_CACHED_GREETING
ENABLE_CALLBACK_STATE_MACHINE
ENABLE_STRUCTURED_FACTS
ENABLE_ZERO_DELAY_SPEC_PROMOTION
ENABLE_STABLE_INTERIM_SPECULATION
ENABLE_FLUX_TUNING
ENABLE_SPEC_TTS
ENABLE_SEMANTIC_FINGERPRINT_REUSE
ENABLE_COMPILED_PROMPTS
ENABLE_EXTENDED_DETERMINISTIC_ROUTING
ENABLE_AUDIBLE_PCM_METRICS

ASR_PROVIDER=flux|nova|nemotron
TTS_PROVIDER=cartesia|magpie
LLM_PROVIDER=azure|local
```

---

# 17. Required New Metrics

Add to turn metrics:

```python
@dataclass
class TurnMetrics:
    turn_id: int

    # input / turn
    first_user_audio_at: float | None = None
    last_meaningful_user_audio_at: float | None = None
    first_interim_at: float | None = None
    stable_interim_at: float | None = None
    eager_eot_at: float | None = None
    hard_eot_at: float | None = None

    # route
    route_started_at: float | None = None
    route_ready_at: float | None = None

    # llm
    speculative_llm_started_at: float | None = None
    final_llm_started_at: float | None = None
    llm_first_token_at: float | None = None
    first_safe_text_at: float | None = None

    # speculation
    candidate_validated_at: float | None = None
    candidate_promoted_at: float | None = None

    # tts
    spec_tts_started_at: float | None = None
    spec_tts_first_pcm_at: float | None = None
    tts_first_packet_at: float | None = None
    tts_first_non_silent_pcm_at: float | None = None

    # output
    first_output_packet_at: float | None = None
    first_non_silent_output_at: float | None = None

    # labels
    route: str = ""
    speculation: str = "none"
    spec_tts: str = "none"
    endpoint_profile: str = ""
```

---

# 18. Dashboard Metrics

Primary:

```text
RawSpeechEnd → FirstMeaningfulAudio
```

Secondary:

```text
ProviderEOT → FirstMeaningfulAudio
HardEOT → CandidatePromoted
HardEOT → FirstSafeText
FirstSafeText → FirstAudibleTTS
```

Distribution:

```text
p50
p90
p95
p99
```

Break down by:

```text
tenant
route
language
state
model
ASR mode
TTS provider
speculation hit/miss
spec_tts hit/miss
```

---

# 19. Quality Guardrails

Latency improvements are invalid if they create:

```text
more interruptions
wrong slot values
duplicate questions
wrong callback claims
cross-tenant cache leaks
wrong banking facts
stale speculative speech
```

Track alongside latency:

```text
false EOT rate
TurnResumed rate
wrong speculation commit rate
slot correction rate
repeated-question rate
booking-claim violation rate
barge-in recovery failures
```

---

# 20. Banking and Regulated Tenant Rules

For high-risk tenants:

## Never speculate audibly on

```text
KYC result
authentication success
balance
loan approval
loan eligibility
transaction success
personal account data
dynamic rates
```

## Safe early work

Allowed:

```text
intent classification
retrieval
prompt building
public FAQ generation
generic navigation
```

## Audio commit

High-risk speech requires:
- hard EOT,
- authoritative tool result if required,
- validated response fingerprint.

---

# 21. Regression Test Matrix

Every optimization must pass:

| Case | Expected |
|---|---|
| User says yes quickly | Fast deterministic transition |
| User pauses mid-sentence | No premature audible response |
| User resumes after EagerEOT | Spec work cancelled |
| User corrects number | Old speculation invalidated |
| User gives only callback day | Ask only for time |
| User gives day + time | Record preference |
| User asks known FAQ | Local/cache route |
| User asks unexpected FAQ | Hosted route |
| User says goodbye | deterministic close |
| User barges into greeting | greeting cleared |
| Hindi/Hinglish turn | correct STT/route/TTS |
| Slow speaker | no aggressive false cut |
| Booking wording | no false confirmation |
| Banking user requests status | tool-required path |

---

# 22. Codex Implementation Guidance

When using Codex to implement this plan:

1. Do not rewrite the entire controller at once.
2. Preserve the working V2 turn lifecycle.
3. Add metrics before changing behavior.
4. Implement one feature flag per change.
5. Keep Pipecat’s turn/event semantics authoritative.
6. Do not duplicate Pipecat turn handling unnecessarily.
7. Keep every speculative async task scoped to:
   - call ID,
   - turn ID,
   - candidate generation.
8. Cancel stale work immediately.
9. Do not let private speculative audio reach the transport.
10. Keep deterministic and hosted paths separate in metrics.
11. Benchmark one layer at a time.
12. Do not change ASR + LLM + TTS simultaneously in the same benchmark.
13. Log effective runtime thresholds/config on every call.
14. Keep tenant and agent version in every cache key.
15. Preserve fallback to the current working path.

---

# 23. Pipecat References

Keep implementation aligned with current Pipecat primitives.

## Turn management
- https://docs.pipecat.ai/api-reference/server/utilities/turn-management/user-turn-strategies

## Deepgram STT
- https://docs.pipecat.ai/api-reference/server/services/stt/deepgram

## Cartesia TTS
- https://docs.pipecat.ai/api-reference/server/services/tts/cartesia

## NVIDIA STT
- https://docs.pipecat.ai/api-reference/server/services/stt/nvidia

## NVIDIA TTS
- https://docs.pipecat.ai/api-reference/server/services/tts/nvidia

## FastAPI WebSocket transport
- https://docs.pipecat.ai/api-reference/server/services/transport/fastapi-websocket

## Pipecat Flows
- https://docs.pipecat.ai/api-reference/pipecat-flows/flow-manager

---

# 24. Final Priority Summary

## Highest-confidence wins

```text
1. cached greeting
2. booking/callback deterministic state
3. structured facts
4. remove 80 ms commit wait
5. stable-interim speculation
```

## Next biggest latency work

```text
6. Flux tuning
7. speculative TTS hit path
8. semantic fingerprint reuse
```

## Structural improvements

```text
9. prompt compilation
10. deterministic routing expansion
11. first-audible-phoneme metrics
```

## Provider/model work

```text
12. ASR/TTS/LLM replacement benchmarking
```

---

# 25. Definition of Done

This improvements plan is complete when:

- greeting no longer waits for runtime TTS,
- callback/booking is deterministic,
- known facts are not re-asked,
- speculation promotion has no artificial delay,
- hosted speculation starts before EagerEOT when stable,
- Flux thresholds have been benchmarked,
- speculative TTS produces measurable hit cases,
- semantic reuse is available behind a flag,
- hosted prompts are compiled/minimal,
- deterministic routes cover common turns,
- first audible phoneme is measured,
- model/provider tests are performed only after runtime optimization,
- p95 remains below one second over a representative workload,
- caller-perceived raw-speech-end latency improves,
- no regression occurs in interruption safety or correctness.

The main goal is not merely:

```text
p95 < 1000 ms
```

It is:

```text
fast
+
natural
+
safe
+
consistent
+
multi-tenant
+
observable
```
