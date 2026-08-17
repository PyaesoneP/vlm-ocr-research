"""Analyze selective thresholds for Phase 4 visual-gain candidate scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_POSITIVE = Path("benchmark/results/phase4_qwen_visual_gain_scores_positive21.json")
DEFAULT_CLEAN = Path("benchmark/results/phase4_qwen_visual_gain_scores_clean44.json")
DEFAULT_OUTPUT = Path("benchmark/results/phase4_qwen_visual_gain_calibration.json")


def is_single_token_pair(row: dict[str, Any]) -> bool:
    visible = str(row.get("visible_form", "")).strip().split()
    normalized = str(row.get("normalized_form", "")).strip().split()
    return len(visible) == 1 and len(normalized) == 1


def load_scores(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    return data.get("scores", [])


def f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else 0.0


def evaluate_threshold(
    positives: list[dict[str, Any]],
    clean: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    tp = sum(1 for row in positives if row.get("top_is_visible") and (row.get("margin") or 0.0) >= threshold)
    fn = len(positives) - tp
    fp = sum(1 for row in clean if (not row.get("top_is_visible")) and (row.get("margin") or 0.0) >= threshold)
    supported_correct = sum(
        1 for row in clean if row.get("top_is_visible") and (row.get("margin") or 0.0) >= threshold
    )
    uncertain_positive = len(positives) - sum(1 for row in positives if (row.get("margin") or 0.0) >= threshold)
    uncertain_clean = len(clean) - sum(1 for row in clean if (row.get("margin") or 0.0) >= threshold)
    return {
        "threshold": threshold,
        "true_positive_errors": tp,
        "false_negative_errors": fn,
        "false_positive_clean": fp,
        "supported_correct_clean": supported_correct,
        "uncertain_positive": uncertain_positive,
        "uncertain_clean": uncertain_clean,
        "error_recall": tp / len(positives) if positives else 0.0,
        "clean_false_positive_rate": fp / len(clean) if clean else 0.0,
        "selector_f1": f1(tp, fp, fn),
        "auto_coverage": (tp + supported_correct + fp) / (len(positives) + len(clean)) if positives or clean else 0.0,
    }


def candidate_thresholds(rows: list[dict[str, Any]]) -> list[float]:
    margins = sorted({float(row.get("margin") or 0.0) for row in rows})
    thresholds = {0.0}
    thresholds.update(margins)
    for a, b in zip(margins, margins[1:]):
        thresholds.add((a + b) / 2)
    return sorted(thresholds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-scores", type=Path, default=DEFAULT_POSITIVE)
    parser.add_argument("--clean-scores", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    positives = load_scores(args.positive_scores)
    clean = load_scores(args.clean_scores)
    rows = positives + clean
    single_positives = [row for row in positives if is_single_token_pair(row)]
    single_clean = [row for row in clean if is_single_token_pair(row)]
    threshold_rows = [evaluate_threshold(positives, clean, t) for t in candidate_thresholds(rows)]
    single_threshold_rows = [
        evaluate_threshold(single_positives, single_clean, t)
        for t in candidate_thresholds(single_positives + single_clean)
    ]
    best_f1 = max(threshold_rows, key=lambda row: (row["selector_f1"], -row["false_positive_clean"]))
    best_single_f1 = max(
        single_threshold_rows,
        key=lambda row: (row["selector_f1"], -row["false_positive_clean"]),
    )
    zero_fp = [row for row in threshold_rows if row["false_positive_clean"] == 0]
    best_zero_fp = max(zero_fp, key=lambda row: (row["true_positive_errors"], row["selector_f1"])) if zero_fp else None
    single_zero_fp = [row for row in single_threshold_rows if row["false_positive_clean"] == 0]
    best_single_zero_fp = (
        max(single_zero_fp, key=lambda row: (row["true_positive_errors"], row["selector_f1"]))
        if single_zero_fp else None
    )
    target_rows = [
        row for row in threshold_rows
        if row["selector_f1"] >= 0.75 and row["false_positive_clean"] <= 2
    ]
    best_target = max(target_rows, key=lambda row: (row["selector_f1"], row["true_positive_errors"])) if target_rows else None
    single_target_rows = [
        row for row in single_threshold_rows
        if row["selector_f1"] >= 0.75 and row["false_positive_clean"] <= 2
    ]
    best_single_target = (
        max(single_target_rows, key=lambda row: (row["selector_f1"], row["true_positive_errors"]))
        if single_target_rows else None
    )

    result = {
        "positive_scores": str(args.positive_scores),
        "clean_scores": str(args.clean_scores),
        "positive_pairs": len(positives),
        "clean_pairs": len(clean),
        "single_token_positive_pairs": len(single_positives),
        "single_token_clean_pairs": len(single_clean),
        "best_f1": best_f1,
        "single_token_best_f1": best_single_f1,
        "best_zero_false_positive": best_zero_fp,
        "single_token_best_zero_false_positive": best_single_zero_fp,
        "best_meeting_phase4_selector_target": best_target,
        "single_token_best_meeting_phase4_selector_target": best_single_target,
        "thresholds": threshold_rows,
        "single_token_thresholds": single_threshold_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "best_f1": best_f1,
        "single_token_best_f1": best_single_f1,
        "best_zero_false_positive": best_zero_fp,
        "single_token_best_zero_false_positive": best_single_zero_fp,
        "best_meeting_phase4_selector_target": best_target,
        "single_token_best_meeting_phase4_selector_target": best_single_target,
    }, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
