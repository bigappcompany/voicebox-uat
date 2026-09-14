#!/usr/bin/env python3
"""Validate and persist versioned, non-sensitive latency baseline records."""
import argparse, json
from pathlib import Path

REQUIRED = {"tenant_id", "agent_version", "route", "endpoint_mode", "model", "hard_eot_at", "bot_started_at"}

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("input", type=Path); parser.add_argument("--output", type=Path, default=Path("logs/v2_baseline.jsonl")); args = parser.parse_args()
    records = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    bad = [index + 1 for index, item in enumerate(records) if REQUIRED - item.keys()]
    if bad: raise SystemExit(f"Missing required baseline fields on lines: {bad}")
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text("\n".join(json.dumps(item, sort_keys=True) for item in records) + "\n")
    print(f"Wrote {len(records)} baseline records to {args.output}")

if __name__ == "__main__": main()
