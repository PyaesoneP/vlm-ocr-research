"""Filter CTC agreement review records with label-free independent triggers.

The CTC agreement policy is intentionally conservative: it never automatically
promotes clean-page alternatives, but it can produce too many review records.
By default, this script keeps review alternatives only when a deterministic
source-text candidate points at the same span. Optional switches can also keep
reviews that overlap existing Stage 2/adjudicator predictions or OCR-supported
alternatives; these variants are diagnostic because they may increase clean-page
review noise.

- the word indices overlap a deterministic source-text adjudication candidate.

Development labels are used only for the audit summary written after filtering.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.adjudication import adjudicate_stage2_errors
from pipeline.contracts import TextBox


DEFAULT_AGREEMENT = Path("benchmark/results/phase4_inference_selector_policy_ctc_agreement_20image.json")
DEFAULT_SOURCE_RESULT = Path("benchmark/results/phase4_realworld_mixed_best_20image_adjudicated.json")
DEFAULT_STRATEGY = "two_stage__qwen3vl_4b_verbatim_word_ocr__same_stage1_boxes__qwen3vl_4b_grader"
DEFAULT_OUTPUT = Path("benchmark/results/phase4_inference_selector_policy_ctc_filtered_20image.json")

SUPPORTED_CANONICAL_READING = "SUPPORTED_CANONICAL_READING"
REVIEW_ALTERNATIVE_READING = "REVIEW_ALTERNATIVE_READING"
REJECT_DISAGREEMENT = "REJECT_DISAGREEMENT"
UNCERTAIN_REVIEW = "UNCERTAIN_REVIEW"


def output_boxes_by_image(result_path: Path, strategy_name: str) -> dict[str, list[TextBox]]:
    data = json.loads(result_path.read_text())
    for strategy in data.get("strategies", []):
        if strategy.get("name") != strategy_name:
            continue
        outputs: dict[str, list[TextBox]] = {}
        for row in strategy.get("images", []):
            output = row.get("output", {})
            image = str(output.get("image") or row.get("image") or "")
            outputs[image] = [
                TextBox.from_dict(item, index=i)
                for i, item in enumerate(output.get("boxes", []))
            ]
        return outputs
    names = [strategy.get("name") for strategy in data.get("strategies", [])]
    raise ValueError(f"Strategy not found: {strategy_name}. Available: {names}")


def output_error_spans_by_image(result_path: Path, strategy_name: str) -> dict[str, list[dict[str, Any]]]:
    data = json.loads(result_path.read_text())
    for strategy in data.get("strategies", []):
        if strategy.get("name") != strategy_name:
            continue
        outputs: dict[str, list[dict[str, Any]]] = {}
        for row in strategy.get("images", []):
            output = row.get("output", {})
            image = str(output.get("image") or row.get("image") or "")
            spans = []
            for error in output.get("errors", []) or []:
                indices = [int(i) for i in error.get("word_indices", []) if isinstance(i, int)]
                if not indices:
                    continue
                spans.append({
                    "word_indices": indices,
                    "evidence_text": str(error.get("evidence_text", "")),
                    "correction": str(error.get("correction", "")),
                    "type": str(error.get("type", "")),
                    "trigger": "stage2_prediction",
                })
            outputs[image] = spans
        return outputs
    names = [strategy.get("name") for strategy in data.get("strategies", [])]
    raise ValueError(f"Strategy not found: {strategy_name}. Available: {names}")


def source_candidate_spans(boxes_by_image: dict[str, list[TextBox]]) -> dict[str, list[dict[str, Any]]]:
    spans: dict[str, list[dict[str, Any]]] = {}
    for image, boxes in boxes_by_image.items():
        candidates = []
        changes = adjudicate_stage2_errors(candidates, boxes)
        rows = []
        for candidate, change in zip(candidates, changes):
            rows.append({
                "word_indices": list(candidate.word_indices),
                "evidence_text": candidate.evidence_text,
                "correction": candidate.correction,
                "type": candidate.type,
                "trigger": change,
            })
        spans[image] = rows
    return spans


def overlaps(left: list[int], right: list[int]) -> bool:
    return bool(set(left) & set(right))


def matching_source_triggers(decision: dict[str, Any], spans_by_image: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    indices = [int(i) for i in decision.get("word_indices", []) if isinstance(i, int)]
    matches = []
    for span in spans_by_image.get(str(decision.get("image", "")), []):
        span_indices = [int(i) for i in span.get("word_indices", []) if isinstance(i, int)]
        if overlaps(indices, span_indices):
            matches.append(span)
    return matches


def merge_span_maps(*maps: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    merged: dict[str, list[dict[str, Any]]] = {}
    for span_map in maps:
        for image, rows in span_map.items():
            merged.setdefault(image, []).extend(rows)
    return merged


def filter_decision(
    decision: dict[str, Any],
    spans_by_image: dict[str, list[dict[str, Any]]],
    *,
    keep_ocr_supported: bool,
) -> dict[str, Any]:
    source_triggers = matching_source_triggers(decision, spans_by_image)
    keep_review = False
    reasons = []

    if decision.get("decision") == REVIEW_ALTERNATIVE_READING:
        if keep_ocr_supported and decision.get("review_supported_by_ocr"):
            keep_review = True
            reasons.append("ocr_supported_review")
        if source_triggers:
            keep_review = True
            trigger_types = sorted({str(row.get("trigger", "")) for row in source_triggers})
            reasons.extend(f"overlaps_{trigger}" for trigger in trigger_types if trigger)

    filtered = {
        **decision,
        "source_text_triggers": source_triggers,
        "filtered_review_kept": keep_review,
        "filtered_review_reasons": reasons,
    }
    if decision.get("decision") == REVIEW_ALTERNATIVE_READING and not keep_review:
        filtered["filtered_decision"] = UNCERTAIN_REVIEW
    else:
        filtered["filtered_decision"] = decision.get("decision")
    return filtered


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matched = [row for row in rows if row.get("matched_error")]
    clean = [row for row in rows if not row.get("matched_error")]
    unique_errors: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in matched:
        key = (str(row.get("image", "")), int(row.get("error_index", -1)), str(row.get("evidence_text", "")))
        unique_errors.setdefault(key, []).append(row)

    unique_statuses = Counter()
    for group in unique_errors.values():
        if any(
            row.get("filtered_decision") == SUPPORTED_CANONICAL_READING
            and row.get("selected_contains_visible")
            for row in group
        ):
            unique_statuses["supported_visible"] += 1
        elif any(
            row.get("filtered_review_kept") and row.get("review_contains_visible")
            for row in group
        ):
            unique_statuses["filtered_review_visible"] += 1
        elif any(
            row.get("decision") == REVIEW_ALTERNATIVE_READING
            and row.get("review_contains_visible")
            and not row.get("filtered_review_kept")
            for row in group
        ):
            unique_statuses["review_visible_filtered_out"] += 1
        elif any(row.get("canonical_contains_visible") for row in group):
            unique_statuses["canonical_present_not_selected_visible"] += 1
        elif any(row.get("word_top_contains_visible") or row.get("line_top_contains_visible") for row in group):
            unique_statuses["top_visible_not_usable"] += 1
        else:
            unique_statuses["not_recovered"] += 1

    return {
        "records_total": len(rows),
        "raw_decisions": dict(sorted(Counter(row.get("decision", "") for row in rows).items())),
        "filtered_decisions": dict(sorted(Counter(row.get("filtered_decision", "") for row in rows).items())),
        "matched_error_records": len(matched),
        "unique_matched_errors": len(unique_errors),
        "unique_error_statuses": dict(sorted(unique_statuses.items())),
        "matched_filtered_review_records": sum(1 for row in matched if row.get("filtered_review_kept")),
        "matched_filtered_review_contains_visible": sum(
            1 for row in matched
            if row.get("filtered_review_kept") and row.get("review_contains_visible")
        ),
        "clean_records": len(clean),
        "clean_supported_alternative_records": 0,
        "clean_filtered_review_records": sum(1 for row in clean if row.get("filtered_review_kept")),
        "clean_rejected_records": sum(1 for row in clean if row.get("filtered_decision") == REJECT_DISAGREEMENT),
        "clean_review_filtered_out": sum(
            1 for row in clean
            if row.get("decision") == REVIEW_ALTERNATIVE_READING and not row.get("filtered_review_kept")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agreement", type=Path, default=DEFAULT_AGREEMENT)
    parser.add_argument("--source-result", type=Path, default=DEFAULT_SOURCE_RESULT)
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--keep-stage2-spans",
        action="store_true",
        help="Keep CTC review alternatives that overlap existing Stage 2/adjudicator predicted errors.",
    )
    parser.add_argument(
        "--keep-ocr-supported",
        action="store_true",
        help=(
            "Also keep OCR-supported CTC review alternatives. Diagnostic only: "
            "the current dev run shows OCR support alone is too noisy."
        ),
    )
    args = parser.parse_args()

    agreement = json.loads(args.agreement.read_text())
    boxes_by_image = output_boxes_by_image(args.source_result, args.strategy)
    source_spans = source_candidate_spans(boxes_by_image)
    stage2_spans = (
        output_error_spans_by_image(args.source_result, args.strategy)
        if args.keep_stage2_spans else {}
    )
    spans = merge_span_maps(source_spans, stage2_spans)
    rows = [
        filter_decision(row, spans, keep_ocr_supported=args.keep_ocr_supported)
        for row in agreement.get("decisions", [])
    ]
    result = {
        "agreement": str(args.agreement),
        "source_result": str(args.source_result),
        "strategy": args.strategy,
        "label_free_filter": True,
        "filter_rules": [
            *(["keep_review_if_ocr_supported"] if args.keep_ocr_supported else []),
            *(["keep_review_if_overlaps_stage2_prediction"] if args.keep_stage2_spans else []),
            "keep_review_if_overlaps_source_text_candidate",
        ],
        "source_text_candidate_spans": source_spans,
        "stage2_prediction_spans": stage2_spans,
        "summary": summarize(rows),
        "decisions": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
