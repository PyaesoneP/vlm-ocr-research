#!/usr/bin/env python3
"""Summarize Phase 4 Stage 1 runs for missing real-world candidates."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "benchmark" / "results"
DEFAULT_OUTPUT = RESULTS_DIR / "phase4_realworld_stage1_missing_candidates_summary.json"
DEFAULT_BASELINE = RESULTS_DIR / "phase4_realworld_stage1_qwen_crop_verified_20image.json"
DEFAULT_PREFLIGHT_STATUS = RESULTS_DIR / "phase4_realworld_stage1_missing_candidate_preflight.json"

MISSING_SOURCE_INFO = {
    "florence2_live_region_ocr": {
        "candidate": "Florence-2-large",
        "environment": "conda:florencetf",
        "box_granularity": "line/region",
    },
    "got_ocr2_live_ocr": {
        "candidate": "GOT-OCR2.0",
        "environment": ".venv",
        "box_granularity": "none",
    },
    "smoldocling_live_ocr": {
        "candidate": "SmolDocling-256M",
        "environment": ".venv",
        "box_granularity": "coarse",
    },
    "trocr_base_live_line_ocr": {
        "candidate": "TrOCR-base",
        "environment": ".venv",
        "box_granularity": "heuristic_line",
    },
    "trocr_large_live_line_ocr": {
        "candidate": "TrOCR-large",
        "environment": ".venv",
        "box_granularity": "heuristic_line",
    },
    "nemotron_ocr_v2_live_ocr": {
        "candidate": "Nemotron OCR v2",
        "environment": "conda:aiml",
        "box_granularity": "partial",
    },
    "monkeyocr_live_ocr": {
        "candidate": "MonkeyOCR",
        "environment": "local_server:.venv",
        "box_granularity": "none",
    },
    "paddleocr_vl_live_ocr": {
        "candidate": "PaddleOCR-VL",
        "environment": "docker:paddleocr-vl-sm120",
        "box_granularity": "block",
    },
}


def numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def int_metric(aggregate: dict[str, Any], key: str) -> int:
    value = aggregate.get(key)
    return int(value) if isinstance(value, int) else 0


def extract_source_name(strategy_name: str) -> str:
    match = re.match(r"stage1__(.+)__[^_].*$", strategy_name)
    if match:
        return match.group(1)
    parts = strategy_name.split("__")
    if len(parts) >= 3 and parts[0] == "two_stage":
        return parts[1]
    return strategy_name


def first_failure(strategy: dict[str, Any]) -> str:
    for image in strategy.get("images", []):
        metrics = image.get("metrics", {})
        if metrics.get("failure"):
            return str(metrics["failure"])
        metadata = image.get("output", {}).get("metadata", {})
        if metadata.get("failure"):
            return str(metadata["failure"])
    return ""


def run_scope(path: Path, aggregate: dict[str, Any]) -> str:
    num_images = aggregate.get("num_images")
    if num_images == 1 or "rw11" in path.stem or "rw_11" in path.stem:
        return "smoke_rw11"
    if num_images == 20 or "20image" in path.stem:
        return "full_20"
    return f"{num_images}_images" if isinstance(num_images, int) else "unknown"


def strategy_status(aggregate: dict[str, Any]) -> str:
    num_images = int_metric(aggregate, "num_images")
    failed = int_metric(aggregate, "stage_failed_images")
    if num_images and failed == num_images:
        return "failed"
    if failed:
        return "partial"
    return "ran"


def promotion_reason(aggregate: dict[str, Any]) -> tuple[bool, str]:
    evidence = int_metric(aggregate, "evidence_preserved_total")
    leaks = int_metric(aggregate, "correction_leak_total")
    cer = numeric(aggregate.get("verbatim_cer"))
    word_iou = numeric(aggregate.get("word_iou"))

    if evidence > 13:
        return True, f"evidence preserved {evidence}/21 beats the current 13/21 gate"
    if leaks < 6 and cer is not None and cer <= 0.10:
        return True, f"correction leaks {leaks} with CER {cer:.3f} clears the leak/CER gate"
    if word_iou is not None and word_iou >= 0.80 and evidence >= 12:
        return True, f"word IoU {word_iou:.3f} with evidence {evidence}/21 is geometry-useful"
    return False, "does not clear the Stage 1 promotion gates"


def load_phase4_results(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for strategy in data.get("strategies", []):
            name = str(strategy.get("name", ""))
            source_name = extract_source_name(name)
            if source_name not in MISSING_SOURCE_INFO:
                continue
            aggregate = strategy.get("aggregate", {})
            promote, reason = promotion_reason(aggregate)
            info = MISSING_SOURCE_INFO[source_name]
            row = {
                "candidate": info["candidate"],
                "source_name": source_name,
                "strategy": name,
                "result_path": str(path.relative_to(PROJECT_ROOT)),
                "scope": run_scope(path, aggregate),
                "status": strategy_status(aggregate),
                "environment": info["environment"],
                "box_granularity": info["box_granularity"],
                "num_images": aggregate.get("num_images"),
                "stage_failed_images": aggregate.get("stage_failed_images", 0),
                "verbatim_cer": aggregate.get("verbatim_cer"),
                "verbatim_wer": aggregate.get("verbatim_wer"),
                "evidence_preserved_total": aggregate.get("evidence_preserved_total", 0),
                "evidence_total": aggregate.get("evidence_total", 0),
                "correction_leak_total": aggregate.get("correction_leak_total", 0),
                "word_iou": aggregate.get("word_iou"),
                "word_iou_recall": aggregate.get("word_iou_recall"),
                "word_iou_precision": aggregate.get("word_iou_precision"),
                "latency_stage1_avg": aggregate.get("latency_stage1_avg"),
                "first_failure": first_failure(strategy),
                "promote_to_full_pipeline": promote and strategy_status(aggregate) != "failed",
                "promotion_reason": reason,
            }
            rows.append(row)
    return rows


def load_preflight_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    rows: list[dict[str, Any]] = []
    for source_name, status in sorted(data.get("sources", data).items()):
        if source_name not in MISSING_SOURCE_INFO:
            continue
        info = MISSING_SOURCE_INFO[source_name]
        rows.append({
            "candidate": info["candidate"],
            "source_name": source_name,
            "strategy": f"stage1__{source_name}__same_stage1_boxes",
            "result_path": str(path.relative_to(PROJECT_ROOT)),
            "scope": status.get("scope", "smoke_rw11"),
            "status": status.get("status", "preflight_failed"),
            "environment": info["environment"],
            "box_granularity": info["box_granularity"],
            "num_images": status.get("num_images", 0),
            "stage_failed_images": status.get("stage_failed_images", 0),
            "verbatim_cer": "not_applicable",
            "verbatim_wer": "not_applicable",
            "evidence_preserved_total": 0,
            "evidence_total": status.get("evidence_total", 21),
            "correction_leak_total": 0,
            "word_iou": 0.0,
            "word_iou_recall": 0.0,
            "word_iou_precision": 0.0,
            "latency_stage1_avg": 0.0,
            "first_failure": status.get("reason", ""),
            "promote_to_full_pipeline": False,
            "promotion_reason": "preflight did not clear; no Stage 1 output to promote",
        })
    return rows


def discover_default_inputs(results_dir: Path) -> list[Path]:
    paths: set[Path] = set()
    for pattern in (
        "phase4_realworld_stage1_*rw11*.json",
        "phase4_realworld_stage1_*20image*.json",
        "phase4_realworld_stage1_missing_*.json",
    ):
        paths.update(results_dir.glob(pattern))
    return sorted(paths)


def baseline_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    for strategy in data.get("strategies", []):
        aggregate = strategy.get("aggregate", {})
        return {
            "result_path": str(path.relative_to(PROJECT_ROOT)),
            "strategy": strategy.get("name", ""),
            "evidence_preserved_total": aggregate.get("evidence_preserved_total"),
            "evidence_total": aggregate.get("evidence_total"),
            "correction_leak_total": aggregate.get("correction_leak_total"),
            "verbatim_cer": aggregate.get("verbatim_cer"),
            "word_iou": aggregate.get("word_iou"),
        }
    return {}


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "candidate",
        "scope",
        "status",
        "CER",
        "evidence",
        "leaks",
        "word_iou",
        "promote",
    ]
    print("\t".join(headers))
    for row in rows:
        evidence = f"{row['evidence_preserved_total']}/{row['evidence_total']}"
        values = [
            row["candidate"],
            row["scope"],
            row["status"],
            str(row["verbatim_cer"]),
            evidence,
            str(row["correction_leak_total"]),
            str(row["word_iou"]),
            "yes" if row["promote_to_full_pipeline"] else "no",
        ]
        print("\t".join(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="*", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--preflight-status", type=Path, default=DEFAULT_PREFLIGHT_STATUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = args.inputs if args.inputs is not None else discover_default_inputs(args.results_dir)
    inputs = [path if path.is_absolute() else PROJECT_ROOT / path for path in inputs]
    rows = load_phase4_results(inputs)
    preflight_path = args.preflight_status if args.preflight_status.is_absolute() else PROJECT_ROOT / args.preflight_status
    rows.extend(load_preflight_rows(preflight_path))
    rows.sort(key=lambda row: (row["candidate"], row["scope"], row["result_path"]))

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "purpose": "Phase 4 missing Stage 1 candidate pass summary",
        "baseline": baseline_summary(args.baseline if args.baseline.is_absolute() else PROJECT_ROOT / args.baseline),
        "promotion_gates": {
            "evidence_preserved_total_gt": 13,
            "correction_leak_total_lt_with_cer_lte": {"leaks": 6, "verbatim_cer": 0.10},
            "geometry_useful": {"word_iou_gte": 0.80, "evidence_preserved_total_gte": 12},
        },
        "inputs": [str(path.relative_to(PROJECT_ROOT)) for path in inputs if path.exists()],
        "preflight_status": str(preflight_path.relative_to(PROJECT_ROOT)) if preflight_path.exists() else "",
        "candidates": rows,
        "promoted_candidates": [
            row for row in rows
            if row["scope"] == "full_20" and row["promote_to_full_pipeline"]
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print_table(rows)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
