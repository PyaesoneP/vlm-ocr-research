"""
Metrics computation for VLM-OCR benchmarking.

Covers transcription accuracy (CER/WER), reading-order scoring (Kendall's tau),
error-detection F1, and bounding-box IoU.
"""

from __future__ import annotations

import math
from typing import Any


# ---------------------------------------------------------------------------
# Character / Word Error Rate
# ---------------------------------------------------------------------------

def normalize_ocr_text(text: str) -> str:
    """Normalize OCR text for fair comparison: collapse whitespace, strip."""
    import re
    # Replace newlines with spaces, collapse multiple spaces, strip
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _edit_distance(s1: str, s2: str) -> int:
    """Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)

    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            # Insert, delete, substitute
            curr.append(min(
                curr[-1] + 1,
                prev[j + 1] + 1,
                prev[j] + (0 if c1 == c2 else 1),
            ))
        prev = curr
    return prev[-1]


def compute_cer(prediction: str, ground_truth: str) -> float:
    """Character Error Rate (0.0 = perfect, 1.0 = completely wrong)."""
    if not ground_truth:
        return 1.0 if prediction else 0.0
    return _edit_distance(prediction, ground_truth) / len(ground_truth)


def compute_wer(prediction: str, ground_truth: str) -> float:
    """Word Error Rate computed on whitespace-tokenized words."""
    pred_words = prediction.split()
    gt_words = ground_truth.split()
    if not gt_words:
        return 1.0 if pred_words else 0.0
    return _edit_distance(pred_words, gt_words) / len(gt_words)


def compute_cer_normalized(prediction: str, ground_truth: str) -> float:
    """CER with whitespace normalization — line breaks treated as spaces."""
    return compute_cer(normalize_ocr_text(prediction), normalize_ocr_text(ground_truth))


def compute_wer_normalized(prediction: str, ground_truth: str) -> float:
    """WER with whitespace normalization — line breaks treated as spaces."""
    return compute_wer(normalize_ocr_text(prediction), normalize_ocr_text(ground_truth))


# ---------------------------------------------------------------------------
# Reading Order — Kendall's tau
# ---------------------------------------------------------------------------

def compute_reading_order_tau(
    predicted_order: list[int],
    ground_truth_order: list[int],
) -> float:
    """
    Kendall's tau-b correlation between predicted and ground-truth reading order.

    Both lists should be the same length, where position *i* contains the
    rank of element *i*.  Higher rank = later in reading order.
    """
    n = len(predicted_order)
    if n < 2 or len(ground_truth_order) != n:
        return 0.0

    concordant = 0
    discordant = 0
    ties_pred = 0
    ties_gt = 0

    for i in range(n - 1):
        for j in range(i + 1, n):
            a = predicted_order[i] - predicted_order[j]
            b = ground_truth_order[i] - ground_truth_order[j]

            if a == 0:
                ties_pred += 1
            if b == 0:
                ties_gt += 1

            if a * b > 0:
                concordant += 1
            elif a * b < 0:
                discordant += 1

    total_pairs = n * (n - 1) // 2
    denom = math.sqrt((total_pairs - ties_pred) * (total_pairs - ties_gt))
    if denom == 0:
        return 1.0
    return (concordant - discordant) / denom


# ---------------------------------------------------------------------------
# Error Detection F1
# ---------------------------------------------------------------------------

def compute_error_detection_f1(
    predicted_errors: list[dict[str, Any]],
    ground_truth_errors: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> float:
    """
    Macro-averaged F1 across error types.

    Each error dict should have:
        {"type": "capitalization"|"spelling"|..., "bbox": [x1,y1,x2,y2]}
    """
    error_types = {"capitalization", "spelling", "grammar", "punctuation", "structural"}

    f1_scores: list[float] = []

    for etype in error_types:
        pred_of_type = [e for e in predicted_errors if e.get("type") == etype]
        gt_of_type = [e for e in ground_truth_errors if e.get("type") == etype]

        tp = 0
        matched_gt: set[int] = set()

        for p in pred_of_type:
            for j, g in enumerate(gt_of_type):
                if j in matched_gt:
                    continue
                if _bbox_iou(p.get("bbox", []), g.get("bbox", [])) >= iou_threshold:
                    tp += 1
                    matched_gt.add(j)
                    break

        fp = len(pred_of_type) - tp
        fn = len(gt_of_type) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


# ---------------------------------------------------------------------------
# Bounding Box IoU
# ---------------------------------------------------------------------------

def _bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    """Intersection-over-Union for [x1, y1, x2, y2] boxes."""
    if len(box_a) != 4 or len(box_b) != 4:
        return 0.0

    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def compute_iou(
    predicted_errors: list[dict[str, Any]],
    ground_truth_errors: list[dict[str, Any]],
) -> float:
    """Mean IoU across all matched error bounding-box pairs."""
    if not ground_truth_errors:
        return 1.0 if not predicted_errors else 0.0

    ious: list[float] = []
    matched_gt: set[int] = set()

    for p in predicted_errors:
        best_iou = 0.0
        best_j = -1
        for j, g in enumerate(ground_truth_errors):
            if j in matched_gt:
                continue
            iou = _bbox_iou(p.get("bbox", []), g.get("bbox", []))
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0:
            ious.append(best_iou)
            matched_gt.add(best_j)

    return sum(ious) / len(ious) if ious else 0.0


# ---------------------------------------------------------------------------
# Ground Truth Schema (for reference)
# ---------------------------------------------------------------------------

GROUND_TRUTH_SCHEMA = """
Each entry in the ground-truth JSON array:

{
    "image": "sample_01.jpg",
    "text": "The full transcribed text of the essay page...",
    "reading_order": [0, 1, 2, ...],       // rank of each text block
    "blocks": [
        {
            "bbox": [x1, y1, x2, y2],
            "text": "text in this block",
            "reading_order_rank": 0
        }
    ],
    "errors": [
        {
            "type": "capitalization|spelling|grammar|punctuation|structural",
            "bbox": [x1, y1, x2, y2],
            "description": "Missing capital at sentence start"
        }
    ]
}
"""
