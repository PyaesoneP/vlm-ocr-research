"""Build a scored Phase 4 evidence graph from minimal-pair visual scores.

This script is the bridge between candidate generation and a lattice-aware
grader. It keeps the canonical word evidence immutable, attaches scored
hypotheses, and emits conservative selector decisions:

    SUPPORTED_ERROR
    SUPPORTED_CORRECT
    UNCERTAIN_REVIEW

The visual scores are diagnostic Qwen visual-gain scores, not the final optical
model. The output is meant to make selector behavior auditable before wiring the
evidence into Stage 2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_PAIRS = Path("benchmark/results/phase4_minimal_pair_dataset.json")
DEFAULT_POSITIVE_SCORES = Path("benchmark/results/phase4_qwen_visual_gain_scores_positive21.json")
DEFAULT_CLEAN_SCORES = Path("benchmark/results/phase4_qwen_visual_gain_scores_clean44.json")
DEFAULT_CALIBRATION = Path("benchmark/results/phase4_qwen_visual_gain_calibration.json")
DEFAULT_OUTPUT = Path("benchmark/results/phase4_evidence_graph_qwen_visual_gain.json")

SUPPORTED_ERROR = "SUPPORTED_ERROR"
SUPPORTED_CORRECT = "SUPPORTED_CORRECT"
UNCERTAIN_REVIEW = "UNCERTAIN_REVIEW"


def compact(text: str) -> str:
    return "".join(ch.lower() for ch in str(text) if ch.isalnum())


def token_count(text: str) -> int:
    return len(str(text).strip().split())


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_pairs(path: Path) -> dict[str, dict[str, Any]]:
    data = load_json(path)
    return {str(pair.get("pair_id")): pair for pair in data.get("pairs", [])}


def load_scores(paths: list[Path]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        data = load_json(path)
        for row in data.get("scores", []):
            rows[str(row.get("pair_id"))] = row
    return rows


def threshold_from_calibration(path: Path, *, mode: str) -> float:
    data = load_json(path)
    key = "single_token_best_f1" if mode == "single_token" else "best_f1"
    row = data.get(key) or {}
    return float(row.get("threshold", 0.0) or 0.0)


def candidate_metadata_by_norm(pair: dict[str, Any]) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for candidate in pair.get("candidates", []) or []:
        norm = str(candidate.get("norm") or compact(candidate.get("text", "")))
        if norm and norm not in metadata:
            metadata[norm] = candidate
    return metadata


def pair_scope(pair: dict[str, Any]) -> str:
    visible_tokens = token_count(pair.get("visible_form", ""))
    normalized_tokens = token_count(pair.get("normalized_form", ""))
    if visible_tokens == 1 and normalized_tokens == 1:
        return "single_token"
    return "phrase"


def invalid_word_crop_control(pair: dict[str, Any]) -> bool:
    """Clean controls are word crops; phrase alternatives are not valid there."""

    if pair.get("split") != "clean_control":
        return False
    return token_count(pair.get("normalized_form", "")) > token_count(pair.get("visible_form", ""))


def scored_hypotheses(pair: dict[str, Any], score: dict[str, Any]) -> list[dict[str, Any]]:
    meta_by_norm = candidate_metadata_by_norm(pair)
    hypotheses = []
    for rank, item in enumerate(score.get("scores", []) or [], start=1):
        norm = str(item.get("norm") or compact(item.get("text", "")))
        meta = meta_by_norm.get(norm, {})
        hypotheses.append({
            "rank": rank,
            "text": item.get("text", ""),
            "norm": norm,
            "visual_gain": item.get("visual_gain"),
            "image_logprob": item.get("image_logprob"),
            "blank_logprob": item.get("blank_logprob"),
            "is_visible_label": norm == compact(pair.get("visible_form", "")),
            "is_normalized_label": norm == compact(pair.get("normalized_form", "")),
            "source": meta.get("source", "scored_label"),
            "sources": meta.get("sources", []),
            "supported_by_ocr": bool(meta.get("supported_by_ocr", False)),
            "edit_distance": meta.get("edit_distance"),
            "ious": meta.get("ious", []),
        })
    return hypotheses


def decide(
    pair: dict[str, Any],
    score: dict[str, Any] | None,
    *,
    threshold: float,
    phrase_clean_policy: str,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if score is None:
        return UNCERTAIN_REVIEW, ["missing_score"]
    margin = score.get("margin")
    if margin is None:
        return UNCERTAIN_REVIEW, ["missing_margin"]
    if invalid_word_crop_control(pair):
        reasons.append("invalid_word_crop_phrase_control")
        if phrase_clean_policy == "uncertain":
            return UNCERTAIN_REVIEW, reasons
    if float(margin) < threshold:
        return UNCERTAIN_REVIEW, [*reasons, "below_margin_threshold"]
    if score.get("top_is_visible"):
        if pair.get("split") == "positive":
            return SUPPORTED_ERROR, [*reasons, "visible_form_supported"]
        return SUPPORTED_CORRECT, [*reasons, "clean_visible_form_supported"]
    if score.get("top_is_normalized"):
        if pair.get("split") == "positive":
            return SUPPORTED_CORRECT, [*reasons, "normalized_form_supported"]
        return SUPPORTED_ERROR, [*reasons, "plausible_error_supported_on_clean_control"]
    return UNCERTAIN_REVIEW, [*reasons, "top_candidate_is_neither_label"]


def evidence_record(
    pair: dict[str, Any],
    score: dict[str, Any] | None,
    *,
    threshold: float,
    phrase_clean_policy: str,
) -> dict[str, Any]:
    decision, reasons = decide(
        pair,
        score,
        threshold=threshold,
        phrase_clean_policy=phrase_clean_policy,
    )
    return {
        "pair_id": pair.get("pair_id"),
        "split": pair.get("split"),
        "scope": pair_scope(pair),
        "image": pair.get("image"),
        "word_indices": pair.get("word_indices", []),
        "bbox": pair.get("bbox", [0, 0, 0, 0]),
        "crop_path": pair.get("crop_path", ""),
        "canonical_text": pair.get("canonical_text", ""),
        "visible_form": pair.get("visible_form", ""),
        "normalized_form": pair.get("normalized_form", ""),
        "error_type": pair.get("error_type", ""),
        "decision": decision,
        "decision_reasons": reasons,
        "margin": score.get("margin") if score else None,
        "top_text": score.get("top_text", "") if score else "",
        "top_is_visible": bool(score and score.get("top_is_visible")),
        "top_is_normalized": bool(score and score.get("top_is_normalized")),
        "hypotheses": scored_hypotheses(pair, score or {}),
    }


def f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else 0.0


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [record for record in records if record.get("split") == "positive"]
    clean = [record for record in records if record.get("split") == "clean_control"]
    positive_supported = [record for record in positives if record.get("decision") == SUPPORTED_ERROR]
    clean_false_positive = [record for record in clean if record.get("decision") == SUPPORTED_ERROR]
    false_negative = [record for record in positives if record.get("decision") != SUPPORTED_ERROR]
    return {
        "records_total": len(records),
        "positive_records": len(positives),
        "clean_control_records": len(clean),
        "supported_error_positive": len(positive_supported),
        "supported_correct_positive": sum(1 for record in positives if record.get("decision") == SUPPORTED_CORRECT),
        "uncertain_positive": sum(1 for record in positives if record.get("decision") == UNCERTAIN_REVIEW),
        "supported_error_clean_false_positive": len(clean_false_positive),
        "supported_correct_clean": sum(1 for record in clean if record.get("decision") == SUPPORTED_CORRECT),
        "uncertain_clean": sum(1 for record in clean if record.get("decision") == UNCERTAIN_REVIEW),
        "error_recall": len(positive_supported) / len(positives) if positives else 0.0,
        "clean_false_positive_rate": len(clean_false_positive) / len(clean) if clean else 0.0,
        "selector_f1": f1(len(positive_supported), len(clean_false_positive), len(false_negative)),
    }


def summarize_by_scope(records: list[dict[str, Any]]) -> dict[str, Any]:
    scopes = sorted({str(record.get("scope", "")) for record in records})
    return {scope: summarize([record for record in records if record.get("scope") == scope]) for scope in scopes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--positive-scores", type=Path, default=DEFAULT_POSITIVE_SCORES)
    parser.add_argument("--clean-scores", type=Path, default=DEFAULT_CLEAN_SCORES)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--threshold-mode", choices=["all", "single_token"], default="all")
    parser.add_argument("--phrase-clean-policy", choices=["uncertain", "score"], default="uncertain")
    args = parser.parse_args()

    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else threshold_from_calibration(args.calibration, mode=args.threshold_mode)
    )
    pairs = load_pairs(args.pairs)
    scores = load_scores([args.positive_scores, args.clean_scores])
    records = [
        evidence_record(
            pair,
            scores.get(pair_id),
            threshold=threshold,
            phrase_clean_policy=args.phrase_clean_policy,
        )
        for pair_id, pair in sorted(pairs.items())
    ]
    result = {
        "pairs": str(args.pairs),
        "positive_scores": str(args.positive_scores),
        "clean_scores": str(args.clean_scores),
        "calibration": str(args.calibration),
        "threshold": threshold,
        "threshold_mode": args.threshold_mode,
        "phrase_clean_policy": args.phrase_clean_policy,
        "summary": summarize(records),
        "summary_by_scope": summarize_by_scope(records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "threshold": threshold,
        "summary": result["summary"],
        "summary_by_scope": result["summary_by_scope"],
    }, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
