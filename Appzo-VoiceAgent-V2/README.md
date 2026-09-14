# Appzo VoiceAgent V2

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
```

## Existing call path

Set the same environment values used by V1, then run `goodbox_server.py` and
`scripts/dial_goodbox_test.py`. Do not turn on speculative audio for regulated
or tool-dependent flows; the V2 controller enforces this again at commit time.
