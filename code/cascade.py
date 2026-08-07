#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def load_rows(input_glob: str):
    rows_by_cfg = {}
    for path_str in sorted(glob.glob(input_glob)):
        path = Path(path_str)
        rows = {}
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                cfg = rec.get("config_key")
                if cfg is None:
                    continue
                rows_by_cfg.setdefault(cfg, {})
                rows_by_cfg[cfg][rec["sample_id"]] = rec
                rows[rec["sample_id"]] = rec
    return rows_by_cfg


def parse_stage(stage: str) -> str:
    stage = stage.strip()
    return "textonly" if stage in {"0", "text", "textonly"} else f"{int(stage)}f"


def incremental_shared_cost(visited: list[dict]):
    total = 0.0
    prev = None
    for row in visited:
        if prev is None:
            total += float(row["wall_clock_s"])
        else:
            total += max(0.0, float(row["wall_clock_s"]) - float(prev["wall_clock_s"]))
        prev = row
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--cascade", required=True, help="Comma-separated stages, e.g. 16,32,128")
    parser.add_argument("--theta", type=float, required=True)
    parser.add_argument("--mode", choices=["shared", "no_share"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows_by_cfg = load_rows(args.input_glob)
    stages = [parse_stage(part) for part in args.cascade.split(",") if part.strip()]
    if not stages:
        raise ValueError("No valid stages parsed from --cascade")
    for stage in stages:
        if stage not in rows_by_cfg:
            raise FileNotFoundError(f"Missing timed JSONL rows for stage {stage}")

    shared_ids = sorted(set.intersection(*(set(rows_by_cfg[stage].keys()) for stage in stages)))
    if not shared_ids:
        raise ValueError("No shared sample ids across requested stages")

    stop_counts = {stage: 0 for stage in stages}
    latencies = []
    memories = []
    baseline_stage = stages[-1]
    baseline_latencies = []
    baseline_memories = []
    per_item = []
    for sid in shared_ids:
        visited = []
        stop_stage = stages[-1]
        for idx, stage in enumerate(stages):
            row = rows_by_cfg[stage][sid]
            visited.append(row)
            if idx < len(stages) - 1 and float(row.get("base_margin", 0.0)) >= args.theta:
                stop_stage = stage
                break
            stop_stage = stage
        stop_counts[stop_stage] += 1
        if args.mode == "shared":
            latency = incremental_shared_cost(visited)
        else:
            latency = sum(float(row["wall_clock_s"]) for row in visited)
        peak_mem = max(float(row.get("gpu_mem_gb", 0.0)) for row in visited)
        baseline_row = rows_by_cfg[baseline_stage][sid]
        baseline_latency = float(baseline_row["wall_clock_s"])
        baseline_mem = float(baseline_row.get("gpu_mem_gb", 0.0))
        latencies.append(latency)
        memories.append(peak_mem)
        baseline_latencies.append(baseline_latency)
        baseline_memories.append(baseline_mem)
        per_item.append(
            {
                "sample_id": sid,
                "stop_stage": stop_stage,
                "latency_s": round(latency, 6),
                "baseline_latency_s": round(baseline_latency, 6),
                "latency_delta_s": round(latency - baseline_latency, 6),
                "peak_mem_gb": round(peak_mem, 6),
                "baseline_peak_mem_gb": round(baseline_mem, 6),
            }
        )

    report = {
        "cascade": stages,
        "theta": args.theta,
        "mode": args.mode,
        "n_items": len(shared_ids),
        "avg_latency_s": round(float(np.mean(latencies)), 6),
        "avg_peak_mem_gb": round(float(np.mean(memories)), 6),
        "baseline_stage": baseline_stage,
        "baseline_avg_latency_s": round(float(np.mean(baseline_latencies)), 6),
        "baseline_avg_peak_mem_gb": round(float(np.mean(baseline_memories)), 6),
        "wall_clock_delta_s": round(float(np.mean(latencies) - np.mean(baseline_latencies)), 6),
        "wall_clock_reduction_pct": round(
            float((1.0 - (np.mean(latencies) / max(1e-8, np.mean(baseline_latencies)))) * 100.0), 3
        ),
        "stop_counts": stop_counts,
        "per_item": per_item,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
