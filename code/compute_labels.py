#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from phase11_metrics_utils import compute_grid_metrics, read_jsonl_by_id


ROOT = Path(
    os.environ.get(
        "ROOT",
        "${EVIDENCE_ROOT}",
    )
)
BRIDGE = ROOT / "refine-logs/drr_qt_bridge"
PHASE10 = Path(os.environ.get("PHASE10", BRIDGE / "phase10_029j_2026-04-22"))
PHASE11 = Path(os.environ.get("PHASE11", BRIDGE / "phase11_029p_2026-04-27"))
PHASE12 = Path(os.environ.get("PHASE12", BRIDGE / "phase12_029r_2026-04-28"))
PHASE13 = Path(os.environ.get("PHASE13", BRIDGE / "phase13_029s_2026-04-28"))
QWEN3_DEBUG = BRIDGE / "qwen3_debug_2026-04-22"
PHASE6 = BRIDGE / "phase6_extended_2026-04-17"
PHASE8 = BRIDGE / "phase8_029h_2026-04-20"


def first_existing(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def load_clean_summary(path: Path) -> dict[str, Any] | None:
    candidates = [
        path.with_suffix(".merge_summary.json"),
        path.with_name(path.name.replace(".jsonl", ".merge_summary.json")),
        path.with_name(path.name.replace(".jsonl", "_summary.json")),
    ]
    for cand in candidates:
        if cand.exists():
            try:
                return json.loads(cand.read_text(encoding="utf-8"))
            except Exception:
                return None
    return None


def is_official_clean(path: Path) -> tuple[bool, str]:
    summary = load_clean_summary(path)
    if not summary:
        # Older canonical fixed files do not always carry merge summaries.
        return True, "no_summary_assumed_legacy"
    checks = {
        "merged_count_eq_dataset_count": summary.get("merged_count") == summary.get("dataset_count"),
        "missing_count_zero": int(summary.get("missing_count", 0) or 0) == 0,
        "duplicate_count_zero": int(summary.get("duplicate_count", 0) or 0) == 0,
        "error_count_zero": int(summary.get("error_count", 0) or 0) == 0,
    }
    return all(checks.values()), json.dumps(checks, sort_keys=True)


def select_official_candidate(candidates: list[Path]) -> tuple[Path | None, list[dict[str, str]], Path | None]:
    """Return first clean candidate, unclean attempts, and first existing fallback.

    Fallback grids intentionally try high-fidelity protocols first. Failed attempts
    should be recorded as protocol evidence, but they should not block aggregation
    when a later fallback is clean.
    """
    failed_attempts: list[dict[str, str]] = []
    first_seen: Path | None = None
    for path in candidates:
        if not path.exists() or path.stat().st_size == 0:
            continue
        first_seen = first_seen or path
        clean, reason = is_official_clean(path)
        if clean:
            return path, failed_attempts, first_seen
        failed_attempts.append({"path": str(path), "reason": reason})
    return None, failed_attempts, first_seen


def path_registry() -> dict[str, dict[str, dict[str, list[Path]]]]:
    return {
        "V1_short": {
            "Qwen3-VL-8B": {
                "textonly": [QWEN3_DEBUG / "other_configs_8gpu/qwen3vl_textonly_short_fixed.jsonl"],
                "16f": [QWEN3_DEBUG / "qwen3vl_16f_short_fixed.jsonl"],
                "32f": [QWEN3_DEBUG / "other_configs_8gpu/qwen3vl_32f_short_fixed.jsonl"],
                "64f": [QWEN3_DEBUG / "other_configs_8gpu/qwen3vl_64f_short_fixed.jsonl"],
                "128f": [QWEN3_DEBUG / "other_configs_8gpu/qwen3vl_128f_short_fixed.jsonl"],
                "256f": [QWEN3_DEBUG / "short_256f_8gpu/qwen3vl_256f_short_fixed.jsonl"],
            },
            "InternVL3-8B": {
                "textonly": [
                    PHASE13 / "gpu_runs/internvl3_v1_short_all/internvl3_textonly_v2_v1_short.jsonl",
                    PHASE8 / "short_internvl3/internvl3_textonly_short.jsonl",
                ],
                "16f": [
                    PHASE13 / "gpu_runs/internvl3_v1_short_all/internvl3_16f_v2_v1_short.jsonl",
                    PHASE8 / "short_internvl3/internvl3_16f_short.jsonl",
                ],
                "32f": [
                    PHASE13 / "gpu_runs/internvl3_v1_short_all/internvl3_32f_v2_v1_short.jsonl",
                    PHASE8 / "short_internvl3/internvl3_32f_short.jsonl",
                ],
                "64f": [
                    PHASE13 / "gpu_runs/internvl3_v1_short_64f_retry448_gpu4_7/internvl3_64f_v2_v1_short.jsonl",
                    PHASE13 / "gpu_runs/internvl3_v1_short_64f_input224/internvl3_64f_v2_v1_short.jsonl",
                    PHASE13 / "gpu_runs/internvl3_v1_short_64f_retry448/internvl3_64f_v2_v1_short.jsonl",
                    PHASE13 / "gpu_runs/internvl3_v1_short_all/internvl3_64f_v2_v1_short.jsonl",
                    PHASE8 / "short_internvl3/internvl3_64f_short.jsonl",
                ],
            },
            "InternVL3.5-8B": {
                "textonly": [PHASE8 / "short_internvl35/internvl35_textonly_short.jsonl"],
                "16f": [PHASE8 / "short_internvl35/internvl35_16f_short.jsonl"],
                "32f": [PHASE8 / "short_internvl35/internvl35_32f_short.jsonl"],
                "64f": [PHASE8 / "short_internvl35/internvl35_64f_short.jsonl"],
            },
        },
        "V1_medium": {
            "Qwen2.5-VL-7B": {
                "32f": [PHASE13 / "gpu_runs/qwen25_v1_medium_32f/qwen25_32f_v2_v1_medium.jsonl"],
                "256f": [PHASE11 / "gpu_runs/qwen25_v1_medium_256f/qwen25_256f_v1_medium.jsonl"],
            },
            "Qwen3-VL-8B": {
                "32f": [PHASE13 / "gpu_runs/qwen3_v1_medium_32f/qwen3_32f_v2_v1_medium.jsonl"],
                "256f": [PHASE11 / "gpu_runs/qwen3_v1_medium_256f/qwen3_256f_v1_medium.jsonl"],
            },
            "InternVL3-8B": {
                "32f": [PHASE13 / "gpu_runs/internvl3_v1_medium_32_128/internvl3_32f_v2_v1_medium.jsonl"],
                "128f": [
                    PHASE13 / "gpu_runs/internvl3_v1_medium_128f_input224/internvl3_128f_v2_v1_medium.jsonl",
                    PHASE13 / "gpu_runs/internvl3_v1_medium_32_128/internvl3_128f_v2_v1_medium.jsonl",
                ],
            },
            "InternVL3.5-8B": {
                "16f": [PHASE13 / "gpu_runs/internvl35_v1_medium_16_32_128/internvl35_16f_v2_v1_medium.jsonl"],
                "32f": [PHASE13 / "gpu_runs/internvl35_v1_medium_16_32_128/internvl35_32f_v2_v1_medium.jsonl"],
                "128f": [
                    PHASE13 / "gpu_runs/internvl35_v1_medium_16_32_128/internvl35_128f_v2_v1_medium.jsonl",
                    PHASE13 / "gpu_runs/internvl35_v1_medium_128f_input336/internvl35_128f_v2_v1_medium.jsonl",
                    PHASE13 / "gpu_runs/internvl35_v1_medium_128f_input224/internvl35_128f_v2_v1_medium.jsonl",
                ],
            },
        },
        "V2_medium": {
            "Qwen2.5-VL-7B": {
                "textonly": [PHASE10 / "v2_medium_full/qwen25_textonly_v2_medium.jsonl"],
                "16f": [PHASE10 / "v2_medium_full/qwen25_16f_v2_medium.jsonl"],
                "32f": [PHASE13 / "gpu_runs/qwen25_v2_medium_32f/qwen25_32f_v2_medium.jsonl"],
                "64f": [PHASE10 / "v2_medium_full/qwen25_64f_v2_medium.jsonl"],
                "128f": [PHASE10 / "v2_medium_full/qwen25_128f_v2_medium.jsonl"],
                "256f": [PHASE12 / "gpu_runs/qwen25_v2_medium_256f_full/qwen25_256f_v2_medium.jsonl"],
            },
            "Qwen3-VL-8B": {
                "textonly": [PHASE10 / "v2_medium_full/qwen3_textonly_v2_medium.jsonl"],
                "16f": [PHASE10 / "v2_medium_full/qwen3_16f_v2_medium.jsonl"],
                "32f": [PHASE13 / "gpu_runs/qwen3_v2_medium_32f/qwen3_32f_v2_medium.jsonl"],
                "64f": [PHASE10 / "v2_medium_full/qwen3_64f_v2_medium.jsonl"],
                "128f": [PHASE10 / "v2_medium_full/qwen3_128f_v2_medium.jsonl"],
            },
            "InternVL3-8B": {
                "128f_input224": [PHASE13 / "gpu_runs/internvl3_v2_medium_128f_input224/internvl3_128f_v2_medium.jsonl"],
            },
            "InternVL3.5-8B": {
                "128f_input224": [PHASE13 / "gpu_runs/internvl35_v2_medium_128f_input224/internvl35_128f_v2_medium.jsonl"],
            },
        },
        "MLVU": {
            "Qwen2.5-VL-7B": {
                "textonly": [PHASE10 / "mlvu_runs/qwen25/textonly/qwen25_textonly_mlvu.jsonl"],
                "16f": [PHASE10 / "mlvu_runs/qwen25/16f/qwen25_16f_mlvu.jsonl"],
                "32f": [PHASE13 / "gpu_runs/qwen25_mlvu_32f/qwen25_32f_v2_mlvu.jsonl"],
                "64f": [PHASE10 / "mlvu_runs/qwen25/64f/qwen25_64f_mlvu.jsonl"],
                "128f": [PHASE10 / "mlvu_runs/qwen25/128f/qwen25_128f_mlvu.jsonl"],
            },
            "Qwen3-VL-8B": {
                "textonly": [PHASE10 / "mlvu_runs/qwen3/textonly/qwen3_textonly_mlvu.jsonl"],
                "16f": [PHASE10 / "mlvu_runs/qwen3/16f/qwen3_16f_mlvu.jsonl"],
                "32f": [PHASE13 / "gpu_runs/qwen3_mlvu_32f/qwen3_32f_v2_mlvu.jsonl"],
                "64f": [PHASE10 / "mlvu_runs/qwen3/64f/qwen3_64f_mlvu.jsonl"],
                "128f": [PHASE10 / "mlvu_runs/qwen3/128f/qwen3_128f_mlvu.jsonl"],
            },
            "InternVL3.5-8B": {
                "textonly": [PHASE10 / "mlvu_runs/internvl35/textonly/internvl35_textonly_mlvu.jsonl"],
                "16f": [PHASE10 / "mlvu_runs/internvl35/16f/internvl35_16f_mlvu.jsonl"],
                "32f": [PHASE13 / "gpu_runs/internvl35_mlvu_32f/internvl35_32f_v2_mlvu.jsonl"],
                "64f": [PHASE10 / "mlvu_runs/internvl35/64f/internvl35_64f_mlvu.jsonl"],
                "128f_fallback": [
                    PHASE13 / "gpu_runs/internvl35_mlvu_128f_input448/internvl35_128f_v2_mlvu.jsonl",
                    PHASE13 / "gpu_runs/internvl35_mlvu_128f_input336/internvl35_128f_v2_mlvu.jsonl",
                    PHASE13 / "gpu_runs/internvl35_mlvu_128f_input224/internvl35_128f_v2_mlvu.jsonl",
                ],
            },
        },
    }


def summarize_config(path: Path) -> dict[str, Any]:
    rows = read_jsonl_by_id(path)
    total = len(rows)
    correct = sum(int(row.get("final_correct", row.get("base_correct", 0)) or 0) for row in rows.values())
    clean, reason = is_official_clean(path)
    return {
        "path": str(path),
        "total": total,
        "correct": correct,
        "accuracy": round(100.0 * correct / total, 4) if total else 0.0,
        "official_clean": clean,
        "clean_reason": reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate official JSONL files into 029S paper figure data.")
    parser.add_argument("--phase13", type=Path, default=PHASE13)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    registry = path_registry()
    curves: dict[str, Any] = {}
    headroom: dict[str, Any] = {}
    missing: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    failed_attempts: list[dict[str, str]] = []

    for split, models in registry.items():
        curves[split] = {}
        headroom[split] = {}
        for model, configs in models.items():
            curves[split][model] = {}
            row_maps: dict[str, dict[str, dict[str, Any]]] = {}
            for cfg, candidates in configs.items():
                path, attempts, first_seen = select_official_candidate(candidates)
                for attempt in attempts:
                    failed_attempts.append({"split": split, "model": model, "config": cfg, **attempt})
                if not path:
                    curves[split][model][cfg] = None
                    if first_seen:
                        rejected.append(
                            {
                                "split": split,
                                "model": model,
                                "config": cfg,
                                "path": str(first_seen),
                                "reason": "no_clean_candidate",
                            }
                        )
                    else:
                        missing.append({"split": split, "model": model, "config": cfg})
                    continue
                item = summarize_config(path)
                curves[split][model][cfg] = item
                row_maps[cfg] = read_jsonl_by_id(path)
            official_cfgs = [cfg for cfg in configs if cfg in row_maps]
            if len(official_cfgs) >= 2:
                try:
                    metrics = compute_grid_metrics(row_maps, official_cfgs)
                    headroom[split][model] = {k: v for k, v in metrics.items() if k != "per_item"}
                except Exception as exc:
                    headroom[split][model] = {"error": str(exc), "configs": official_cfgs}

    payload = {
        "phase": "phase13_029s_2026-04-28",
        "dry_run": args.dry_run,
        "curves": curves,
        "headroom": headroom,
        "missing": missing,
        "rejected_unclean": rejected,
        "failed_attempts": failed_attempts,
    }

    figure_dir = args.phase13 / "figure_data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        (figure_dir / "cross_model_scaling_curves.json").write_text(
            json.dumps(curves, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (figure_dir / "cross_model_headroom_bars.json").write_text(
            json.dumps(headroom, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    out = args.phase13 / ("aggregate_to_paper_dry_run.json" if args.dry_run else "aggregate_to_paper_summary.json")
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(out),
                "missing": len(missing),
                "rejected_unclean": len(rejected),
                "failed_attempts": len(failed_attempts),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
