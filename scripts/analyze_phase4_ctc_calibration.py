"""Analyze Phase 4 CTC candidate-score calibration."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_SCORES = Path("benchmark/results/phase4_ctc_candidate_scores_minimal_pairs.json")
DEFAULT_OUTPUT = Path("benchmark/results/phase4_ctc_calibration.json")


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def bucket_rows(rows: list[dict[str, Any]], *, split: str | None = None, phrase: bool | None = None) -> list[dict[str, Any]]:
    items = rows
    if split is not None:
        items = [row for row in items if row.get("split") == split]
    if phrase is not None:
        items = [row for row in items if bool(row.get("phrase_shaped")) is phrase]
    return items


def threshold_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    accepted = [
        row for row in rows
        if row.get("margin") is not None
        and finite(row.get("margin"))
        and float(row["margin"]) >= threshold
        and row.get("top_supported")
    ]
    positives = bucket_rows(rows, split="positive")
    controls = bucket_rows(rows, split="clean_control")
    accepted_positive = bucket_rows(accepted, split="positive")
    accepted_controls = bucket_rows(accepted, split="clean_control")
    recovered_positive = [row for row in accepted_positive if row.get("top_is_visible")]
    clean_corruptions = [row for row in accepted_controls if not row.get("top_is_visible")]
    clean_correct = [row for row in accepted_controls if row.get("top_is_visible")]
    return {
        "threshold": threshold,
        "accepted_total": len(accepted),
        "accepted_rate": len(accepted) / len(rows) if rows else 0.0,
        "positive_recovered": len(recovered_positive),
        "positive_total": len(positives),
        "positive_recall_at_threshold": len(recovered_positive) / len(positives) if positives else 0.0,
        "positive_abstained": len(positives) - len(accepted_positive),
        "clean_correct": len(clean_correct),
        "clean_corruptions": len(clean_corruptions),
        "clean_total": len(controls),
        "clean_corruption_rate": len(clean_corruptions) / len(controls) if controls else 0.0,
        "clean_abstained": len(controls) - len(accepted_controls),
        "recovered_pair_ids": [row.get("pair_id") for row in recovered_positive],
        "clean_corruption_pair_ids": [row.get("pair_id") for row in clean_corruptions],
    }


def preference_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = bucket_rows(rows, split="positive")
    controls = bucket_rows(rows, split="clean_control")
    single = bucket_rows(rows, phrase=False)
    phrase = bucket_rows(rows, phrase=True)

    def rate(items: list[dict[str, Any]], key: str) -> float:
        return sum(1 for item in items if item.get(key)) / len(items) if items else 0.0

    margins = [float(row["margin"]) for row in rows if row.get("margin") is not None and finite(row["margin"])]
    return {
        "pairs": len(rows),
        "positive_pairs": len(positives),
        "clean_control_pairs": len(controls),
        "single_token_pairs": len(single),
        "phrase_shaped_pairs": len(phrase),
        "positive_visible_preference": rate(positives, "top_is_visible"),
        "clean_visible_preference": rate(controls, "top_is_visible"),
        "single_token_visible_preference": rate(single, "top_is_visible"),
        "phrase_shaped_visible_preference": rate(phrase, "top_is_visible"),
        "top_supported_rate": rate(rows, "top_supported"),
        "top_unsupported_count": sum(1 for row in rows if row.get("top_unsupported_chars")),
        "margin_min": min(margins) if margins else None,
        "margin_median": sorted(margins)[len(margins) // 2] if margins else None,
        "margin_max": max(margins) if margins else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--threshold",
        type=float,
        action="append",
        dest="thresholds",
        help="Accepted margin threshold. Can be repeated.",
    )
    args = parser.parse_args()

    data = json.loads(args.scores.read_text())
    rows = data.get("scores", [])
    thresholds = args.thresholds or [0.0, 0.25, 0.5, 1.0, 2.0]
    result = {
        "scores": str(args.scores),
        "scorer": data.get("scorer", {}),
        "candidate_mode": data.get("candidate_mode"),
        "preference_summary": preference_summary(rows),
        "threshold_metrics": [threshold_metrics(rows, threshold) for threshold in thresholds],
        "notes": [
            "Accepted predictions require the top candidate to have no unsupported characters.",
            "Labels are used only for post-hoc calibration on the development set.",
            "Phrase-shaped cases are reported separately because the current EasyOCR English alphabet has no space.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["preference_summary"], indent=2))
    print(json.dumps(result["threshold_metrics"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
