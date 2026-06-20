#!/usr/bin/env python3
"""Audit Stage 2 misses and false positives for a Phase 4 result file."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = PROJECT_ROOT / "benchmark" / "results" / "phase4_realworld_mixed_best_20image.json"
DEFAULT_DATASET = PROJECT_ROOT / "benchmark" / "test_dataset" / "realworld_writing_errors.json"
DEFAULT_STRATEGY = "two_stage__realworld_source_text__realworld_aligned_words__qwen3vl_4b_grader"
DEFAULT_OUTPUT = PROJECT_ROOT / "benchmark" / "results" / "phase4_stage2_source_text_audit.json"


def normalize(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def contains_or_equals(left: Any, right: Any) -> bool:
    a = normalize(left)
    b = normalize(right)
    return bool(a and b and (a == b or a in b or b in a))


def bbox_iou(a: list[Any], b: list[Any]) -> float:
    if len(a) != 4 or len(b) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def load_strategy(result_path: Path, strategy_name: str) -> dict[str, Any]:
    data = json.loads(result_path.read_text())
    for strategy in data.get("strategies", []):
        if strategy.get("name") == strategy_name:
            return strategy
    names = [strategy.get("name") for strategy in data.get("strategies", [])]
    raise SystemExit(f"Strategy {strategy_name!r} not found. Available: {names}")


def classify_expected(gt: dict[str, Any], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    best = None
    best_score = -1.0
    for index, pred in enumerate(predictions):
        iou = bbox_iou(pred.get("bbox", []), gt.get("bbox", []))
        same_type = pred.get("type") == gt.get("type")
        evidence_match = contains_or_equals(pred.get("evidence_text", ""), gt.get("evidence_text", ""))
        correction_match = contains_or_equals(pred.get("correction", ""), gt.get("correction", ""))
        score = (
            (3 if same_type else 0)
            + (3 if evidence_match else 0)
            + (2 if correction_match else 0)
            + min(iou, 1.0)
        )
        if score > best_score:
            best_score = score
            best = {
                "prediction_index": index,
                "prediction": pred,
                "same_type": same_type,
                "evidence_match": evidence_match,
                "correction_match": correction_match,
                "bbox_iou": iou,
            }

    if best is None:
        return {"status": "missed", "reason": "missed"}

    if best["same_type"] and best["evidence_match"] and best["bbox_iou"] >= 0.5:
        return {"status": "matched", "reason": "matched", **best}
    if not best["same_type"] and (best["evidence_match"] or best["bbox_iou"] >= 0.5):
        return {"status": "unmatched", "reason": "wrong_type", **best}
    if best["same_type"] and not best["evidence_match"]:
        return {"status": "unmatched", "reason": "wrong_evidence_span", **best}
    if best["same_type"] and best["evidence_match"] and not best["correction_match"]:
        return {"status": "unmatched", "reason": "wrong_correction", **best}
    if best["same_type"] and best["evidence_match"] and best["bbox_iou"] < 0.5:
        return {"status": "unmatched", "reason": "bbox_mismatch", **best}
    return {"status": "unmatched", "reason": "missed", **best}


def classify_false_positive(pred: dict[str, Any], gt_errors: list[dict[str, Any]]) -> str:
    for gt in gt_errors:
        if contains_or_equals(pred.get("evidence_text", ""), gt.get("evidence_text", "")):
            return "duplicate_or_wrongly_anchored_true_error"
        if bbox_iou(pred.get("bbox", []), gt.get("bbox", [])) >= 0.5:
            return "bbox_only_mismatch"

    error_type = pred.get("type")
    if error_type == "spelling":
        return "invented_spelling_error"
    if error_type in {"capitalization", "punctuation"}:
        return "invented_punctuation_capitalization_issue"
    if error_type == "grammar":
        return "grammar_overreach"
    return "invented_structural_or_other_error"


def audit(strategy: dict[str, Any], dataset_by_image: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pages = []
    expected_counter: Counter[str] = Counter()
    false_positive_counter: Counter[str] = Counter()

    for record in strategy.get("images", []):
        image = record["image"]
        gt = dataset_by_image.get(image, {})
        gt_errors = gt.get("errors", [])
        pred_errors = record.get("output", {}).get("errors", [])

        expected = []
        matched_prediction_indices: set[int] = set()
        for gt_error in gt_errors:
            result = classify_expected(gt_error, pred_errors)
            expected_counter[result["reason"]] += 1
            if result.get("status") == "matched" and "prediction_index" in result:
                matched_prediction_indices.add(int(result["prediction_index"]))
            expected.append({
                "ground_truth": gt_error,
                **result,
            })

        false_positives = []
        for index, pred in enumerate(pred_errors):
            if index in matched_prediction_indices:
                continue
            reason = classify_false_positive(pred, gt_errors)
            false_positive_counter[reason] += 1
            false_positives.append({
                "prediction_index": index,
                "reason": reason,
                "prediction": pred,
            })

        pages.append({
            "image": image,
            "expected_error_count": len(gt_errors),
            "predicted_error_count": len(pred_errors),
            "expected": expected,
            "false_positives": false_positives,
        })

    return {
        "strategy": strategy.get("name"),
        "summary": {
            "expected_reasons": dict(sorted(expected_counter.items())),
            "false_positive_reasons": dict(sorted(false_positive_counter.items())),
        },
        "pages": pages,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = json.loads(args.dataset.read_text())
    dataset_by_image = {entry["image"]: entry for entry in dataset}
    strategy = load_strategy(args.result, args.strategy)
    report = audit(strategy, dataset_by_image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
