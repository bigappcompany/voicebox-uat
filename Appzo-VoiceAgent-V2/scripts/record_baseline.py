#!/usr/bin/env python3
"""Validate, summarize and persist versioned, non-sensitive latency records."""
import argparse, json, math
from collections import defaultdict
from pathlib import Path

REQUIRED = {"tenant_id", "agent_version", "route", "endpoint_mode", "model", "hard_eot_at", "bot_started_at"}
MARKER = "LATENCY RECORD | "
CADENCE_MARKER = "AUDIO CADENCE RECORD | "


def parse_records(text: str) -> list[dict]:
    records = []
    cadence = {}
    server_log = any(MARKER in line for line in text.splitlines())
    for line_number, line in enumerate(text.splitlines(), 1):
        clean = line.strip()
        if not clean:
            continue
        if MARKER in clean:
            clean = clean.split(MARKER, 1)[1]
            kind = "latency"
        elif CADENCE_MARKER in clean:
            clean = clean.split(CADENCE_MARKER, 1)[1]
            kind = "cadence"
        elif server_log:
            continue
        else:
            kind = "latency"
        try:
            item = json.loads(clean)
            key = (item.get("call_id"), item.get("turn_id"))
            if kind == "cadence":
                cadence[key] = item
            else:
                records.append(item)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON latency record on line {line_number}: {exc.msg}") from exc
    for item in records:
        item.update(cadence.get((item.get("call_id"), item.get("turn_id")), {}))
    return records


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * percentile) - 1)]


def summarize(records: list[dict]) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for item in records:
        response_at = item.get("first_audible_at")
        if response_at is None and "first_audible_source" in item:
            continue
        response_at = response_at or item["bot_started_at"]
        latency_ms = (float(response_at) - float(item["hard_eot_at"])) * 1000
        if latency_ms < 0 or item.get("route") in {"v2-error", "stt-empty-retry"}:
            continue
        key = "|".join(str(item[name]) for name in ("route", "endpoint_mode", "model"))
        groups[key].append(latency_ms)
    return {
        key: {
            "n": len(values),
            "p50_ms": round(_percentile(values, .50), 1),
            "p90_ms": round(_percentile(values, .90), 1),
            "p95_ms": round(_percentile(values, .95), 1),
            "p99_ms": round(_percentile(values, .99), 1),
        }
        for key, values in sorted(groups.items()) if values
    }


def summarize_contributions(records: list[dict], measured_from: str | None = None) -> dict[str, dict[str, float | int | str]]:
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    total_secs = 0.0
    for record in records:
        breakdown = record.get("latency_breakdown") or {}
        if measured_from is not None and breakdown.get("measured_from") != measured_from:
            continue
        total_secs += float(breakdown.get("total_secs") or 0.0)
        for item in breakdown.get("contributions") or []:
            key = (
                str(item.get("key", "unknown")),
                str(item.get("owner_kind", "pipeline")),
                str(item.get("owner", "unknown")),
            )
            groups[key].append(float(item.get("duration_secs") or 0.0))
    result = {}
    for (key, owner_kind, owner), values in sorted(groups.items()):
        contribution_total = sum(values)
        result[key] = {
            "owner_kind": owner_kind,
            "owner": owner,
            "n": len(values),
            "p50_ms": round(_percentile(values, .50) * 1000, 1),
            "p95_ms": round(_percentile(values, .95) * 1000, 1),
            "total_share_pct": round(100 * contribution_total / total_secs, 1) if total_secs else 0.0,
        }
    return result


def summarize_user_perceived(records: list[dict]) -> dict[str, float | int]:
    values = [
        float(item["latency_breakdown"]["total_secs"]) * 1000
        for item in records
        if (item.get("latency_breakdown") or {}).get("measured_from") == "last_voiced_audio"
    ]
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "p50_ms": round(_percentile(values, .50), 1),
        "p95_ms": round(_percentile(values, .95), 1),
        "p99_ms": round(_percentile(values, .99), 1),
    }


def evaluate_targets(records: list[dict], *, min_samples: int = 50) -> tuple[bool, list[str]]:
    true_totals, post_eot, endpointing = [], [], []
    resumed = 0
    cadence_gaps = []
    for item in records:
        breakdown = item.get("latency_breakdown") or {}
        if breakdown.get("measured_from") == "last_voiced_audio":
            true_totals.append(float(breakdown["total_secs"]) * 1000)
        first_audible = item.get("first_audible_at")
        if first_audible is not None and item.get("hard_eot_at") is not None:
            post_eot.append((float(first_audible) - float(item["hard_eot_at"])) * 1000)
        for contribution in breakdown.get("contributions") or []:
            if contribution.get("key") == "endpointing.final_transcript":
                endpointing.append(float(contribution.get("duration_secs") or 0) * 1000)
        resumed += int(item.get("turn_resumed_count") or 0)
        cadence_gaps.append(float(item.get("output_max_packet_gap_ms") or 0))

    lines = []
    enough = len(true_totals) >= min_samples
    lines.append(f"sample_count: {'PASS' if enough else 'INSUFFICIENT'} n={len(true_totals)} required={min_samples}")

    checks = []
    if endpointing:
        checks += [
            ("endpoint_p50", _percentile(endpointing, .50), 200),
            ("endpoint_p95", _percentile(endpointing, .95), 400),
        ]
    if post_eot:
        checks += [
            ("post_eot_p50", _percentile(post_eot, .50), 400),
            ("post_eot_p95", _percentile(post_eot, .95), 600),
        ]
    if true_totals:
        checks += [
            ("true_total_p50", _percentile(true_totals, .50), 600),
            ("true_total_p95", _percentile(true_totals, .95), 800),
        ]
    for name, value, target in checks:
        lines.append(f"{name}: {'PASS' if value <= target else 'FAIL'} value_ms={value:.1f} target_ms<={target}")
    if records:
        resume_rate = resumed / len(records) * 100
        long_gap_rate = sum(gap > 300 for gap in cadence_gaps) / len(cadence_gaps) * 100
        lines.append(f"turn_resumed_rate: {'PASS' if resume_rate < 2 else 'FAIL'} value_pct={resume_rate:.1f} target_pct<2")
        lines.append(f"audio_gap_over_300ms_rate: {'PASS' if long_gap_rate < 1 else 'FAIL'} value_pct={long_gap_rate:.1f} target_pct<1")
    passed = enough and all("FAIL" not in line for line in lines)
    return passed, lines

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("logs/v2_baseline.jsonl"))
    parser.add_argument("--min-samples", type=int, default=50)
    parser.add_argument("--enforce-targets", action="store_true")
    args = parser.parse_args()
    records = parse_records(args.input.read_text())
    bad = [index + 1 for index, item in enumerate(records) if REQUIRED - item.keys()]
    if bad: raise SystemExit(f"Missing required baseline fields on lines: {bad}")
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text("\n".join(json.dumps(item, sort_keys=True) for item in records) + "\n")
    print(f"Wrote {len(records)} baseline records to {args.output}")
    for group, metrics in summarize(records).items():
        print(f"{group}: " + " ".join(f"{name}={value}" for name, value in metrics.items()))
    user_perceived = summarize_user_perceived(records)
    print("User-perceived (last_voiced_audio->first_audible): " + " ".join(f"{name}={value}" for name, value in user_perceived.items()))
    for cohort in ("last_voiced_audio", "hard_eot"):
        contributions = summarize_contributions(records, measured_from=cohort)
        if contributions:
            print(f"Layer contributions ({cohort}):")
            for key, metrics in contributions.items():
                print(f"{key}: " + " ".join(f"{name}={value}" for name, value in metrics.items()))
    passed, gate_lines = evaluate_targets(records, min_samples=args.min_samples)
    print("Acceptance gates:")
    for line in gate_lines:
        print(line)
    if args.enforce_targets and not passed:
        raise SystemExit(2)

if __name__ == "__main__": main()
