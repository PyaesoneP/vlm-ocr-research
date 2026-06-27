"""Run selector-policy experiments on a scored Phase 4 inference graph.

This script does not build candidates or run a model. It takes an already
scored label-free evidence graph and applies several conservative policies for
deciding which reading, if any, should be passed forward as supported visual
evidence. When a development dataset is supplied, it audits those decisions
against reviewed labels after the fact.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_GRAPH = Path("benchmark/results/phase4_inference_evidence_graph_qwen_visual_gain_rw11.json")
DEFAULT_DATASET = Path("benchmark/test_dataset/realworld_writing_errors.json")
DEFAULT_OUTPUT = Path("benchmark/results/phase4_inference_selector_policy_rw11.json")

SUPPORTED_CANONICAL_READING = "SUPPORTED_CANONICAL_READING"
SUPPORTED_ALTERNATIVE_READING = "SUPPORTED_ALTERNATIVE_READING"
UNSUPPORTED_ALTERNATIVE_REVIEW = "UNSUPPORTED_ALTERNATIVE_REVIEW"
REJECT_NOISY_ALTERNATIVE = "REJECT_NOISY_ALTERNATIVE"
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


def first_alpha(text: str) -> str:
    for char in str(text):
        if char.isalpha():
            return char
    return ""


def guarded_alternative_reasons(record: dict[str, Any], top: dict[str, Any]) -> list[str]:
    """Return label-free reasons an alternative is too risky to auto-promote."""

    canonical_text = str(record.get("canonical_text", ""))
    top_text = str(top.get("text", ""))
    canonical_norm = norm(canonical_text)
    top_norm = norm(top_text)
    reasons = []

    if len(canonical_norm) <= 3:
        reasons.append("short_canonical")
    if first_alpha(canonical_text).isupper():
        reasons.append("capitalized_canonical")
    if " " in norm_words(canonical_text):
        reasons.append("phrase_canonical")
    if top.get("source") == "generic_one_edit_variant":
        reasons.append("generic_one_edit_variant")
    if isinstance(top.get("edit_distance"), int) and int(top["edit_distance"]) > 1:
        reasons.append("large_edit_distance")
    if top_text and first_alpha(top_text) == "" and not top_text[0].isalnum():
        reasons.append("non_alnum_alternative_start")
    if canonical_norm and top_norm == canonical_norm[1:]:
        reasons.append("first_character_deletion")
    return reasons


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


def graph_records(path: Path, image_filter: set[str]) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    records = data.get("records", [])
    if image_filter:
        records = [record for record in records if record.get("image") in image_filter]
    return records


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
        if candidate_contains(record.get("canonical_text", ""), str(error.get("evidence_text", ""))):
            score += 0.25
        for candidate in record.get("hypotheses", []):
            if candidate_contains(candidate.get("text", ""), str(error.get("evidence_text", ""))):
                score += 0.5
                break
        if best is None or score > best[0]:
            best = (score, {**error, "_error_index": error_idx, "_iou": iou})
    return best[1] if best else None


def candidate_rank(record: dict[str, Any], evidence_text: str) -> int | None:
    for rank, candidate in enumerate(record.get("hypotheses", []), start=1):
        if candidate_contains(candidate.get("text", ""), evidence_text):
            return rank
    return None


def policy_decision(record: dict[str, Any], *, policy: str, margin_threshold: float) -> dict[str, Any]:
    hypotheses = record.get("hypotheses", [])
    top = hypotheses[0] if hypotheses else {}
    canonical = next((row for row in hypotheses if row.get("is_canonical")), None)
    margin = record.get("alternative_margin_over_canonical")
    if margin is None:
        margin_value = None
    else:
        margin_value = float(margin)

    selected = top if top else {}
    review = {}
    decision = UNCERTAIN_REVIEW
    reason = "no_supported_reading"

    if top.get("is_canonical"):
        decision = SUPPORTED_CANONICAL_READING
        reason = "top_is_canonical"
    elif margin_value is None or margin_value < margin_threshold:
        decision = UNCERTAIN_REVIEW
        selected = {}
        review = top
        reason = "alternative_margin_below_threshold"
    elif policy == "allow_unsupported_guarded" and (guard_reasons := guarded_alternative_reasons(record, top)):
        decision = UNSUPPORTED_ALTERNATIVE_REVIEW
        selected = {}
        review = top
        reason = "guarded_alternative_review:" + ",".join(guard_reasons)
    elif top.get("supported_by_ocr"):
        decision = SUPPORTED_ALTERNATIVE_READING
        reason = "top_alternative_ocr_supported"
    elif policy == "strict_ocr":
        decision = UNSUPPORTED_ALTERNATIVE_REVIEW
        selected = {}
        review = top
        reason = "top_alternative_not_ocr_supported"
    elif policy == "allow_unsupported":
        decision = SUPPORTED_ALTERNATIVE_READING
        reason = "unsupported_alternative_allowed"
    elif policy == "allow_unsupported_guarded":
        decision = SUPPORTED_ALTERNATIVE_READING
        reason = "unsupported_alternative_allowed_after_guards"
    elif policy == "reject_unsupported_when_canonical_supported" and canonical:
        decision = REJECT_NOISY_ALTERNATIVE
        selected = canonical
        reason = "unsupported_top_rejected_to_canonical"
    else:
        decision = UNSUPPORTED_ALTERNATIVE_REVIEW
        selected = {}
        review = top
        reason = "top_alternative_requires_review"

    return {
        "record_id": record.get("pair_id", ""),
        "image": record.get("image", ""),
        "word_indices": record.get("word_indices", []),
        "bbox": record.get("bbox", []),
        "canonical_text": record.get("canonical_text", ""),
        "top_text": top.get("text", ""),
        "top_source": top.get("source", ""),
        "top_supported_by_ocr": bool(top.get("supported_by_ocr", False)),
        "selected_text": selected.get("text", ""),
        "selected_source": selected.get("source", ""),
        "selected_supported_by_ocr": bool(selected.get("supported_by_ocr", False)),
        "review_text": review.get("text", ""),
        "review_source": review.get("source", ""),
        "review_supported_by_ocr": bool(review.get("supported_by_ocr", False)),
        "decision": decision,
        "reason": reason,
        "guarded_alternative_reasons": guarded_alternative_reasons(record, top) if top else [],
        "alternative_margin_over_canonical": margin_value,
    }


def audit_decision(decision: dict[str, Any], error: dict[str, Any] | None, record: dict[str, Any]) -> dict[str, Any]:
    if error is None:
        return {
            "matched_error": False,
            "evidence_text": "",
            "correction": "",
            "visible_rank": None,
            "selected_contains_visible": False,
            "review_contains_visible": False,
            "canonical_contains_visible": False,
            "top_contains_visible": False,
        }
    evidence_text = str(error.get("evidence_text", ""))
    return {
        "matched_error": True,
        "error_index": error.get("_error_index"),
        "error_iou": error.get("_iou"),
        "evidence_text": evidence_text,
        "correction": error.get("correction", ""),
        "visible_rank": candidate_rank(record, evidence_text),
        "selected_contains_visible": candidate_contains(decision.get("selected_text", ""), evidence_text),
        "review_contains_visible": candidate_contains(decision.get("review_text", ""), evidence_text),
        "canonical_contains_visible": candidate_contains(decision.get("canonical_text", ""), evidence_text),
        "top_contains_visible": candidate_contains(decision.get("top_text", ""), evidence_text),
    }


def summarize(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["decision"] for row in decisions)
    matched = [row for row in decisions if row.get("matched_error")]
    clean = [row for row in decisions if not row.get("matched_error")]
    supported = [
        row for row in matched
        if row.get("decision") in {SUPPORTED_CANONICAL_READING, SUPPORTED_ALTERNATIVE_READING}
    ]
    review = [
        row for row in matched
        if row.get("decision") in {UNSUPPORTED_ALTERNATIVE_REVIEW, UNCERTAIN_REVIEW}
    ]
    rejected = [row for row in matched if row.get("decision") == REJECT_NOISY_ALTERNATIVE]
    return {
        "records_total": len(decisions),
        "decisions": dict(sorted(counts.items())),
        "matched_error_records": len(matched),
        "matched_supported_records": len(supported),
        "matched_supported_contains_visible": sum(1 for row in supported if row.get("selected_contains_visible")),
        "matched_review_records": len(review),
        "matched_review_contains_visible": sum(1 for row in review if row.get("review_contains_visible")),
        "matched_rejected_records": len(rejected),
        "matched_rejected_canonical_contains_visible": sum(
            1 for row in rejected if row.get("selected_contains_visible")
        ),
        "matched_canonical_contains_visible": sum(1 for row in matched if row.get("canonical_contains_visible")),
        "matched_top_contains_visible": sum(1 for row in matched if row.get("top_contains_visible")),
        "unsupported_alternative_reviews": sum(
            1 for row in decisions if row["decision"] == UNSUPPORTED_ALTERNATIVE_REVIEW
        ),
        "rejected_noisy_alternatives": sum(
            1 for row in decisions if row["decision"] == REJECT_NOISY_ALTERNATIVE
        ),
        "clean_records": len(clean),
        "clean_supported_alternative_records": sum(
            1 for row in clean if row["decision"] == SUPPORTED_ALTERNATIVE_READING
        ),
        "clean_review_records": sum(
            1 for row in clean if row["decision"] in {UNSUPPORTED_ALTERNATIVE_REVIEW, UNCERTAIN_REVIEW}
        ),
        "clean_rejected_noisy_records": sum(
            1 for row in clean if row["decision"] == REJECT_NOISY_ALTERNATIVE
        ),
        "clean_guarded_review_records": sum(
            1 for row in clean
            if row["decision"] == UNSUPPORTED_ALTERNATIVE_REVIEW
            and str(row.get("reason", "")).startswith("guarded_alternative_review")
        ),
    }


def run_policy(
    records: list[dict[str, Any]],
    *,
    dataset_by_image: dict[str, dict[str, Any]],
    policy: str,
    margin_threshold: float,
    min_iou: float,
) -> dict[str, Any]:
    decisions = []
    for record in records:
        decision = policy_decision(record, policy=policy, margin_threshold=margin_threshold)
        error = best_matching_error(record, dataset_by_image, min_iou=min_iou)
        decisions.append({**decision, **audit_decision(decision, error, record)})
    return {
        "policy": policy,
        "margin_threshold": margin_threshold,
        "summary": summarize(decisions),
        "decisions": decisions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image", action="append", dest="images", default=[])
    parser.add_argument("--policy", action="append", default=[])
    parser.add_argument("--margin-threshold", action="append", type=float, default=[])
    parser.add_argument("--min-iou", type=float, default=0.05)
    args = parser.parse_args()

    policies = args.policy or [
        "strict_ocr",
        "allow_unsupported",
        "allow_unsupported_guarded",
        "reject_unsupported_when_canonical_supported",
    ]
    thresholds = args.margin_threshold or [0.0, 0.5, 1.0]
    image_filter = set(args.images)
    records = graph_records(args.graph, image_filter)
    dataset = {
        str(sample.get("image", "")): sample
        for sample in load_dataset(args.dataset)
        if not image_filter or sample.get("image") in image_filter
    }

    experiments = [
        run_policy(
            records,
            dataset_by_image=dataset,
            policy=policy,
            margin_threshold=threshold,
            min_iou=args.min_iou,
        )
        for policy in policies
        for threshold in thresholds
    ]
    result = {
        "graph": str(args.graph),
        "dataset": str(args.dataset),
        "images": sorted(image_filter),
        "records_total": len(records),
        "experiments": experiments,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for experiment in experiments:
        print(json.dumps({
            "policy": experiment["policy"],
            "margin_threshold": experiment["margin_threshold"],
            **experiment["summary"],
        }, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
