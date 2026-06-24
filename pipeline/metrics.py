"""Phase 4 metric helpers."""

from __future__ import annotations

import re
from typing import Any

from benchmark.metrics import (
    compute_cer_normalized,
    compute_error_detection_f1,
    compute_iou,
    compute_wer_normalized,
)
from pipeline.contracts import PipelineOutput


NOT_APPLICABLE = "not_applicable"


def _normalize_error_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _contains_normalized_span(text: str, span: str) -> bool:
    normalized_text = f" {_normalize_error_text(text)} "
    normalized_span = _normalize_error_text(span)
    return bool(normalized_span and f" {normalized_span} " in normalized_text)


def _error_text_matches(pred: dict[str, Any], gt: dict[str, Any]) -> bool:
    if pred.get("type") != gt.get("type"):
        return False

    pred_evidence = _normalize_error_text(pred.get("evidence_text", ""))
    gt_evidence = _normalize_error_text(gt.get("evidence_text", ""))
    if pred_evidence and gt_evidence:
        if pred_evidence == gt_evidence:
            return True
        if pred_evidence in gt_evidence or gt_evidence in pred_evidence:
            return True

    pred_correction = _normalize_error_text(pred.get("correction", ""))
    gt_correction = _normalize_error_text(gt.get("correction", ""))
    return bool(pred_correction and gt_correction and pred_correction == gt_correction)


def compute_error_text_f1(
    predicted_errors: list[dict[str, Any]],
    ground_truth_errors: list[dict[str, Any]],
) -> dict[str, float]:
    """Error detection F1 by type + evidence/correction, independent of bbox."""
    if not ground_truth_errors:
        return {"precision": 0.0 if predicted_errors else 1.0, "recall": 1.0, "f1": 1.0 if not predicted_errors else 0.0}

    matched_gt: set[int] = set()
    true_positive = 0
    for pred in predicted_errors:
        for index, gt in enumerate(ground_truth_errors):
            if index in matched_gt:
                continue
            if _error_text_matches(pred, gt):
                matched_gt.add(index)
                true_positive += 1
                break

    false_positive = len(predicted_errors) - true_positive
    false_negative = len(ground_truth_errors) - true_positive
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_truthfulness_metrics(
    observed_text: str,
    ground_truth: dict[str, Any],
) -> dict[str, Any]:
    """Score whether Stage 1 preserved the student's intended error evidence."""
    gt_text = str(ground_truth.get("text", ""))
    gt_errors = ground_truth.get("errors", [])
    metrics: dict[str, Any] = {
        "verbatim_cer": NOT_APPLICABLE,
        "verbatim_wer": NOT_APPLICABLE,
        "evidence_preserved_rate": NOT_APPLICABLE,
        "evidence_preserved_count": 0,
        "evidence_total": 0,
        "correction_leak_count": 0,
        "truthfulness_details": [],
    }
    if gt_text and observed_text.strip():
        metrics["verbatim_cer"] = compute_cer_normalized(observed_text, gt_text)
        metrics["verbatim_wer"] = compute_wer_normalized(observed_text, gt_text)

    if not gt_errors:
        return metrics

    details = []
    preserved_count = 0
    leak_count = 0
    for error in gt_errors:
        evidence = str(error.get("evidence_text", ""))
        correction = str(error.get("correction", ""))
        evidence_preserved = _contains_normalized_span(observed_text, evidence)
        correction_present = _contains_normalized_span(observed_text, correction)
        correction_leaked = bool(correction and correction_present and not evidence_preserved)
        if evidence_preserved:
            preserved_count += 1
        if correction_leaked:
            leak_count += 1
        details.append({
            "type": error.get("type", ""),
            "evidence_text": evidence,
            "correction": correction,
            "evidence_preserved": evidence_preserved,
            "correction_present": correction_present,
            "correction_leaked": correction_leaked,
        })

    metrics["evidence_preserved_count"] = preserved_count
    metrics["evidence_total"] = len(gt_errors)
    metrics["evidence_preserved_rate"] = preserved_count / len(gt_errors)
    metrics["correction_leak_count"] = leak_count
    metrics["truthfulness_details"] = details
    return metrics


def _bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    if len(box_a) != 4 or len(box_b) != 4:
        return 0.0
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_word_iou(
    predicted_boxes: list[dict[str, Any]],
    ground_truth_words: list[dict[str, Any]],
    iou_threshold: float = 0.05,
) -> dict[str, Any]:
    """Greedy word-box IoU against IAM word-level ground truth."""
    valid = [
        box for box in predicted_boxes
        if box.get("bbox") and box.get("bbox") != [0, 0, 0, 0]
    ]
    if not ground_truth_words:
        return {
            "mean_iou": NOT_APPLICABLE,
            "matched": 0,
            "total_gt": 0,
            "total_pred": len(valid),
            "recall": NOT_APPLICABLE,
            "precision": NOT_APPLICABLE,
        }
    if not valid:
        return {
            "mean_iou": 0.0,
            "matched": 0,
            "total_gt": len(ground_truth_words),
            "total_pred": 0,
            "recall": 0.0,
            "precision": 0.0,
        }

    matched_gt: set[int] = set()
    ious: list[float] = []
    for pred in valid:
        p_bbox = pred.get("bbox", [])
        best_iou = 0.0
        best_j = -1
        for j, gt_word in enumerate(ground_truth_words):
            if j in matched_gt:
                continue
            iou = _bbox_iou(p_bbox, gt_word.get("bbox", []))
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_threshold:
            ious.append(best_iou)
            matched_gt.add(best_j)

    return {
        "mean_iou": sum(ious) / len(ious) if ious else 0.0,
        "matched": len(ious),
        "total_gt": len(ground_truth_words),
        "total_pred": len(valid),
        "recall": len(ious) / len(ground_truth_words),
        "precision": len(ious) / len(valid),
    }


def evaluate_phase4_output(
    output: PipelineOutput,
    ground_truth: dict[str, Any] | None = None,
    word_ground_truth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one Phase 4 output without overclaiming on no-error data."""
    metrics: dict[str, Any] = {
        "parse_valid": output.parse_valid,
        "repair_attempted": output.repair_attempted,
        "repair_succeeded": output.repair_succeeded,
        "error_count": len(output.errors),
        "false_positive_count": 0,
        "cer": NOT_APPLICABLE,
        "wer": NOT_APPLICABLE,
        "verbatim_cer": NOT_APPLICABLE,
        "verbatim_wer": NOT_APPLICABLE,
        "evidence_preserved_rate": NOT_APPLICABLE,
        "evidence_preserved_count": 0,
        "evidence_total": 0,
        "correction_leak_count": 0,
        "word_iou": NOT_APPLICABLE,
        "error_text_f1": NOT_APPLICABLE,
        "error_text_precision": NOT_APPLICABLE,
        "error_text_recall": NOT_APPLICABLE,
        "error_detection_f1": NOT_APPLICABLE,
        "error_box_iou": NOT_APPLICABLE,
    }
    if ground_truth is None and word_ground_truth is None:
        return metrics

    text_gt = word_ground_truth or ground_truth or {}
    gt_text = text_gt.get("text", "")
    if gt_text and output.text.strip():
        metrics["cer"] = compute_cer_normalized(output.text, gt_text)
        metrics["wer"] = compute_wer_normalized(output.text, gt_text)

    if ground_truth is not None:
        metrics.update(compute_truthfulness_metrics(output.text, ground_truth))

    if word_ground_truth and word_ground_truth.get("words"):
        word_result = compute_word_iou(
            [box.to_dict() for box in output.boxes],
            word_ground_truth.get("words", []),
        )
        metrics["word_iou"] = word_result["mean_iou"]
        metrics["word_iou_matched"] = word_result["matched"]
        metrics["word_iou_gt"] = word_result["total_gt"]
        metrics["word_iou_pred"] = word_result["total_pred"]
        metrics["word_iou_recall"] = word_result["recall"]
        metrics["word_iou_precision"] = word_result["precision"]

    gt_errors = (ground_truth or {}).get("errors", [])
    if gt_errors:
        pred_errors = [error.to_dict() for error in output.errors]
        text_result = compute_error_text_f1(pred_errors, gt_errors)
        metrics["error_text_f1"] = text_result["f1"]
        metrics["error_text_precision"] = text_result["precision"]
        metrics["error_text_recall"] = text_result["recall"]
        metrics["error_detection_f1"] = compute_error_detection_f1(pred_errors, gt_errors)
        iou_result = compute_iou(pred_errors, gt_errors)
        metrics["error_box_iou"] = iou_result["mean_iou"] if iou_result["matched"] else 0.0
        metrics["error_iou_recall"] = iou_result["recall"]
        metrics["error_iou_precision"] = iou_result["precision"]
    else:
        metrics["false_positive_count"] = len(output.errors)

    return metrics


def _mean_numeric(rows: list[dict[str, Any]], key: str) -> float | str:
    vals = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    return sum(vals) / len(vals) if vals else NOT_APPLICABLE


def aggregate_phase4_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-image Phase 4 metrics."""
    if not rows:
        return {}

    total = len(rows)
    false_positive_total = sum(int(row.get("false_positive_count", 0)) for row in rows)
    evidence_total = sum(int(row.get("evidence_total", 0)) for row in rows)
    evidence_preserved_total = sum(int(row.get("evidence_preserved_count", 0)) for row in rows)
    correction_leak_total = sum(int(row.get("correction_leak_count", 0)) for row in rows)
    return {
        "num_images": total,
        "cer": _mean_numeric(rows, "cer"),
        "wer": _mean_numeric(rows, "wer"),
        "verbatim_cer": _mean_numeric(rows, "verbatim_cer"),
        "verbatim_wer": _mean_numeric(rows, "verbatim_wer"),
        "evidence_preserved_rate": (
            evidence_preserved_total / evidence_total
            if evidence_total else NOT_APPLICABLE
        ),
        "evidence_preserved_total": evidence_preserved_total,
        "evidence_total": evidence_total,
        "correction_leak_total": correction_leak_total,
        "word_iou": _mean_numeric(rows, "word_iou"),
        "word_iou_recall": _mean_numeric(rows, "word_iou_recall"),
        "word_iou_precision": _mean_numeric(rows, "word_iou_precision"),
        "error_text_f1": _mean_numeric(rows, "error_text_f1"),
        "error_text_precision": _mean_numeric(rows, "error_text_precision"),
        "error_text_recall": _mean_numeric(rows, "error_text_recall"),
        "error_detection_f1": _mean_numeric(rows, "error_detection_f1"),
        "error_box_iou": _mean_numeric(rows, "error_box_iou"),
        "valid_json_rate": sum(1 for row in rows if row.get("parse_valid")) / total,
        "repair_attempt_rate": sum(1 for row in rows if row.get("repair_attempted")) / total,
        "repair_success_rate": sum(1 for row in rows if row.get("repair_succeeded")) / total,
        "false_positive_images": sum(1 for row in rows if row.get("false_positive_count", 0) > 0),
        "false_positive_total": false_positive_total,
        "false_positive_avg": false_positive_total / total,
        "stage_failed_images": sum(1 for row in rows if row.get("stage_failed")),
    }
