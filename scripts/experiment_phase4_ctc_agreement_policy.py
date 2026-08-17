"""Compare word-crop and line-context CTC scores with a conservative policy."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_WORD_GRAPH = Path("benchmark/results/phase4_inference_evidence_graph_ctc_rw1_rw2_rw11.json")
DEFAULT_LINE_GRAPH = Path("benchmark/results/phase4_inference_evidence_graph_ctc_line_rw1_rw2_rw11.json")
DEFAULT_DATASET = Path("benchmark/test_dataset/realworld_writing_errors.json")
DEFAULT_OUTPUT = Path("benchmark/results/phase4_inference_selector_policy_ctc_agreement_rw1_rw2_rw11.json")

SUPPORTED_CANONICAL_READING = "SUPPORTED_CANONICAL_READING"
REVIEW_ALTERNATIVE_READING = "REVIEW_ALTERNATIVE_READING"
REJECT_DISAGREEMENT = "REJECT_DISAGREEMENT"
UNCERTAIN_REVIEW = "UNCERTAIN_REVIEW"


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def norm_words(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def candidate_contains(candidate_text: str, evidence_text: str) -> bool:
    candidate_norm = norm(candidate_text)
    evidence_norm = norm(evidence_text)
    if candidate_norm == evidence_norm:
        return True
    candidate_words = f" {norm_words(candidate_text)} "
    evidence_words = norm_words(evidence_text)
    return bool(evidence_words and f" {evidence_words} " in candidate_words)


def bbox_iou(left: list[float], right: list[float]) -> float:
    if len(left) != 4 or len(right) != 4:
        return 0.0
    xa = max(left[0], right[0])
    ya = max(left[1], right[1])
    xb = min(left[2], right[2])
    yb = min(left[3], right[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_l = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    area_r = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    union = area_l + area_r - inter
    return inter / union if union else 0.0


def error_bbox(sample: dict[str, Any], indices: list[int]) -> list[int]:
    words = sample.get("words", [])
    boxes = [words[index].get("bbox", [0, 0, 0, 0]) for index in indices if 0 <= index < len(words)]
    if not boxes:
        return [0, 0, 0, 0]
    return [
        min(int(box[0]) for box in boxes),
        min(int(box[1]) for box in boxes),
        max(int(box[2]) for box in boxes),
        max(int(box[3]) for box in boxes),
    ]


def load_dataset(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("samples", "pages", "items"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"Unsupported dataset schema: {path}")


def graph_records(path: Path, image_filter: set[str]) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text())
    records = data.get("records", [])
    if image_filter:
        records = [record for record in records if record.get("image") in image_filter]
    return {str(record.get("pair_id", "")): record for record in records}


def best_matching_error(
    record: dict[str, Any],
    dataset_by_image: dict[str, dict[str, Any]],
    *,
    min_iou: float,
) -> dict[str, Any] | None:
    sample = dataset_by_image.get(str(record.get("image", "")))
    if not sample:
        return None
    best: tuple[float, dict[str, Any]] | None = None
    for error_idx, error in enumerate(sample.get("errors", []) or []):
        indices = [int(i) for i in error.get("word_indices", []) if isinstance(i, int)]
        iou = bbox_iou(record.get("bbox", [0, 0, 0, 0]), error_bbox(sample, indices))
        if iou < min_iou:
            continue
        score = iou
        for graph_record in (record.get("_word_record", {}), record.get("_line_record", {})):
            for candidate in graph_record.get("hypotheses", []):
                if candidate_contains(candidate.get("text", ""), str(error.get("evidence_text", ""))):
                    score += 0.5
                    break
        if best is None or score > best[0]:
            best = (score, {**error, "_error_index": error_idx, "_iou": iou})
    return best[1] if best else None


def top(record: dict[str, Any]) -> dict[str, Any]:
    hypotheses = record.get("hypotheses", [])
    return hypotheses[0] if hypotheses else {}


def candidate_rank(record: dict[str, Any], evidence_text: str) -> int | None:
    for rank, candidate in enumerate(record.get("hypotheses", []), start=1):
        if candidate_contains(candidate.get("text", ""), evidence_text):
            return rank
    return None


def best_review_candidate(word_record: dict[str, Any], line_record: dict[str, Any]) -> dict[str, Any]:
    word_top = top(word_record)
    line_top = top(line_record)
    if word_top and not word_top.get("is_canonical"):
        return word_top
    if line_top and not line_top.get("is_canonical"):
        return line_top
    return {}


def agreement_decision(word_record: dict[str, Any], line_record: dict[str, Any]) -> dict[str, Any]:
    word_top = top(word_record)
    line_top = top(line_record)
    word_top_norm = norm(word_top.get("text", ""))
    line_top_norm = norm(line_top.get("text", ""))
    selected = {}
    review = {}
    decision = UNCERTAIN_REVIEW
    reason = "no_context_agreement"

    if word_top.get("is_canonical") or line_top.get("is_canonical"):
        selected = next((row for row in word_record.get("hypotheses", []) if row.get("is_canonical")), {})
        decision = SUPPORTED_CANONICAL_READING
        reason = "canonical_wins_in_at_least_one_context"
    elif word_top_norm and word_top_norm == line_top_norm:
        review = word_top
        decision = REVIEW_ALTERNATIVE_READING
        reason = "same_alternative_wins_both_contexts_review_only"
    elif word_top or line_top:
        review = best_review_candidate(word_record, line_record)
        decision = REJECT_DISAGREEMENT
        reason = "word_line_top_disagree"

    return {
        "record_id": word_record.get("pair_id", line_record.get("pair_id", "")),
        "image": word_record.get("image", line_record.get("image", "")),
        "word_indices": word_record.get("word_indices", line_record.get("word_indices", [])),
        "bbox": word_record.get("bbox", line_record.get("bbox", [])),
        "canonical_text": word_record.get("canonical_text", line_record.get("canonical_text", "")),
        "word_top_text": word_top.get("text", ""),
        "line_top_text": line_top.get("text", ""),
        "word_top_is_canonical": bool(word_top.get("is_canonical")),
        "line_top_is_canonical": bool(line_top.get("is_canonical")),
        "selected_text": selected.get("text", ""),
        "selected_source": selected.get("source", ""),
        "review_text": review.get("text", ""),
        "review_source": review.get("source", ""),
        "review_supported_by_ocr": bool(review.get("supported_by_ocr", False)),
        "decision": decision,
        "reason": reason,
        "word_margin_over_canonical": word_record.get("alternative_margin_over_canonical"),
        "line_margin_over_canonical": line_record.get("alternative_margin_over_canonical"),
    }


def audit_decision(
    decision: dict[str, Any],
    error: dict[str, Any] | None,
    *,
    word_record: dict[str, Any],
    line_record: dict[str, Any],
) -> dict[str, Any]:
    if error is None:
        return {
            "matched_error": False,
            "evidence_text": "",
            "correction": "",
            "word_visible_rank": None,
            "line_visible_rank": None,
            "selected_contains_visible": False,
            "review_contains_visible": False,
            "canonical_contains_visible": False,
            "word_top_contains_visible": False,
            "line_top_contains_visible": False,
        }
    evidence_text = str(error.get("evidence_text", ""))
    return {
        "matched_error": True,
        "error_index": error.get("_error_index"),
        "error_iou": error.get("_iou"),
        "evidence_text": evidence_text,
        "correction": error.get("correction", ""),
        "word_visible_rank": candidate_rank(word_record, evidence_text),
        "line_visible_rank": candidate_rank(line_record, evidence_text),
        "selected_contains_visible": candidate_contains(decision.get("selected_text", ""), evidence_text),
        "review_contains_visible": candidate_contains(decision.get("review_text", ""), evidence_text),
        "canonical_contains_visible": candidate_contains(decision.get("canonical_text", ""), evidence_text),
        "word_top_contains_visible": candidate_contains(decision.get("word_top_text", ""), evidence_text),
        "line_top_contains_visible": candidate_contains(decision.get("line_top_text", ""), evidence_text),
    }


def summarize(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["decision"] for row in decisions)
    matched = [row for row in decisions if row.get("matched_error")]
    clean = [row for row in decisions if not row.get("matched_error")]
    supported = [row for row in matched if row["decision"] == SUPPORTED_CANONICAL_READING]
    review = [row for row in matched if row["decision"] in {REVIEW_ALTERNATIVE_READING, UNCERTAIN_REVIEW}]
    rejected = [row for row in matched if row["decision"] == REJECT_DISAGREEMENT]
    unique_errors: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in matched:
        key = (str(row.get("image", "")), int(row.get("error_index", -1)), str(row.get("evidence_text", "")))
        unique_errors.setdefault(key, []).append(row)

    unique_statuses = Counter()
    for rows in unique_errors.values():
        if any(
            row["decision"] == SUPPORTED_CANONICAL_READING and row.get("selected_contains_visible")
            for row in rows
        ):
            unique_statuses["supported_visible"] += 1
        elif any(
            row["decision"] in {REVIEW_ALTERNATIVE_READING, UNCERTAIN_REVIEW}
            and row.get("review_contains_visible")
            for row in rows
        ):
            unique_statuses["review_visible"] += 1
        elif any(row.get("canonical_contains_visible") for row in rows):
            unique_statuses["canonical_present_not_selected_visible"] += 1
        elif any(row.get("word_top_contains_visible") or row.get("line_top_contains_visible") for row in rows):
            unique_statuses["top_visible_not_usable"] += 1
        else:
            unique_statuses["not_recovered"] += 1

    return {
        "records_total": len(decisions),
        "decisions": dict(sorted(counts.items())),
        "matched_error_records": len(matched),
        "unique_matched_errors": len(unique_errors),
        "unique_error_statuses": dict(sorted(unique_statuses.items())),
        "matched_supported_canonical": len(supported),
        "matched_supported_canonical_contains_visible": sum(
            1 for row in supported if row.get("selected_contains_visible")
        ),
        "matched_review_records": len(review),
        "matched_review_contains_visible": sum(1 for row in review if row.get("review_contains_visible")),
        "matched_rejected_records": len(rejected),
        "matched_canonical_contains_visible": sum(1 for row in matched if row.get("canonical_contains_visible")),
        "matched_word_top_contains_visible": sum(1 for row in matched if row.get("word_top_contains_visible")),
        "matched_line_top_contains_visible": sum(1 for row in matched if row.get("line_top_contains_visible")),
        "clean_records": len(clean),
        "clean_supported_alternative_records": 0,
        "clean_supported_canonical_records": sum(
            1 for row in clean if row["decision"] == SUPPORTED_CANONICAL_READING
        ),
        "clean_review_records": sum(
            1 for row in clean if row["decision"] in {REVIEW_ALTERNATIVE_READING, UNCERTAIN_REVIEW}
        ),
        "clean_rejected_records": sum(1 for row in clean if row["decision"] == REJECT_DISAGREEMENT),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--word-graph", type=Path, default=DEFAULT_WORD_GRAPH)
    parser.add_argument("--line-graph", type=Path, default=DEFAULT_LINE_GRAPH)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image", action="append", dest="images", default=[])
    parser.add_argument("--min-iou", type=float, default=0.05)
    args = parser.parse_args()

    image_filter = set(args.images)
    word_records = graph_records(args.word_graph, image_filter)
    line_records = graph_records(args.line_graph, image_filter)
    dataset = {
        str(sample.get("image", "")): sample
        for sample in load_dataset(args.dataset)
        if not image_filter or sample.get("image") in image_filter
    }

    rows = []
    for record_id in sorted(set(word_records) & set(line_records)):
        word_record = word_records[record_id]
        line_record = line_records[record_id]
        merged_for_match = {
            **word_record,
            "_word_record": word_record,
            "_line_record": line_record,
        }
        decision = agreement_decision(word_record, line_record)
        error = best_matching_error(merged_for_match, dataset, min_iou=args.min_iou)
        rows.append({
            **decision,
            **audit_decision(decision, error, word_record=word_record, line_record=line_record),
        })

    result = {
        "word_graph": str(args.word_graph),
        "line_graph": str(args.line_graph),
        "dataset": str(args.dataset),
        "images": sorted(image_filter),
        "records_total": len(rows),
        "summary": summarize(rows),
        "decisions": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
