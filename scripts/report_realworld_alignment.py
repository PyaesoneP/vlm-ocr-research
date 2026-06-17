#!/usr/bin/env python3
"""Summarize real-world word alignment quality and review targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "benchmark" / "test_dataset" / "realworld_writing_errors.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--low-threshold", type=float, default=0.45)
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()

    entries = json.loads(args.dataset.read_text())
    print("Page summary:")
    for entry in sorted(
        entries,
        key=lambda item: item.get("alignment_summary", {}).get("mean_alignment_score", 0.0),
    ):
        summary = entry.get("alignment_summary", {})
        print(
            "{image}: status={status} mean={mean:.3f} low={low} "
            "expected={expected} draft={draft} delta={delta}".format(
                image=entry["image"],
                status=entry.get("annotation_status"),
                mean=summary.get("mean_alignment_score", 0.0),
                low=summary.get("low_alignment_score_count", 0),
                expected=entry.get("expected_word_count"),
                draft=entry.get("draft_word_count"),
                delta=entry.get("word_count_delta"),
            )
        )

    low_words = []
    for entry in entries:
        for word in entry.get("words", []):
            score = float(word.get("alignment_score", 0.0) or 0.0)
            if score < args.low_threshold:
                low_words.append((score, entry["image"], word))

    print(f"\nLowest word alignments (< {args.low_threshold}):")
    for score, image, word in sorted(
        low_words,
        key=lambda item: (item[0], item[1], int(item[2].get("index", 0))),
    )[:args.limit]:
        print(
            "{image} #{idx} {text!r} <= {draft!r} score={score:.3f} bbox={bbox}".format(
                image=image,
                idx=word.get("index"),
                text=word.get("text"),
                draft=word.get("draft_text"),
                score=score,
                bbox=word.get("bbox"),
            )
        )

    print("\nError anchors:")
    for entry in entries:
        for error in entry.get("errors", []):
            print(
                "{image}: {etype} {evidence!r}->{correction!r} "
                "indices={indices} bbox={bbox} status={status}".format(
                    image=entry["image"],
                    etype=error.get("type"),
                    evidence=error.get("evidence_text"),
                    correction=error.get("correction"),
                    indices=error.get("word_indices"),
                    bbox=error.get("bbox"),
                    status=error.get("annotation_status"),
                )
            )


if __name__ == "__main__":
    main()
