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
        return 0.0
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

        # Skip error types absent from BOTH prediction and ground truth.
        # Including them would punish perfect agreement (both empty → F1=0).
        if not pred_of_type and not gt_of_type:
            continue

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
# Block-Level Alignment & IoU
# ---------------------------------------------------------------------------

def _normalize_bbox_to_xyxy(bbox: list[float], fmt: str = "auto") -> list[float]:
    """Convert [x1,y1,x2,y2] or [x,y,w,h] to [x1,y1,x2,y2].

    Args:
        bbox: 4-element list.
        fmt: "xyxy", "xywh", or "auto" (heuristic: if w < x1 or h < y1, treat as xywh).
    """
    if len(bbox) != 4:
        return [0, 0, 0, 0]
    if fmt == "xywh":
        return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
    if fmt == "auto":
        # Heuristic: xywh has small width/height relative to position,
        # or width < x1 / height < y1 (zero/negative dimensions).
        if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
            return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
    return list(bbox)


def compute_block_iou(
    predicted_blocks: list[dict],
    ground_truth_blocks: list[dict],
    iou_threshold: float = 0.1,
) -> dict:
    """
    Align predicted text blocks to ground-truth blocks by spatial overlap
    and compute mean IoU. Returns dict with mean_iou, matched_pairs, etc.
    """
    if not ground_truth_blocks:
        return {"mean_iou": 1.0 if not predicted_blocks else 0.0, "matched": 0, "total_gt": 0, "total_pred": 0}

    # Greedy matching by best IoU
    matched_gt: set[int] = set()
    ious: list[float] = []

    for p in predicted_blocks:
        p_bbox = _normalize_bbox_to_xyxy(p.get("bbox", []))
        best_iou = 0.0
        best_j = -1
        for j, g in enumerate(ground_truth_blocks):
            if j in matched_gt:
                continue
            g_bbox = _normalize_bbox_to_xyxy(g.get("bbox", []))
            iou = _bbox_iou(p_bbox, g_bbox)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_threshold:
            ious.append(best_iou)
            matched_gt.add(best_j)

    return {
        "mean_iou": sum(ious) / len(ious) if ious else 0.0,
        "matched": len(ious),
        "total_gt": len(ground_truth_blocks),
        "total_pred": len(predicted_blocks),
        "recall": len(ious) / len(ground_truth_blocks) if ground_truth_blocks else 0.0,
        "precision": len(ious) / len(predicted_blocks) if predicted_blocks else 0.0,
    }


def extract_reading_order(blocks: list[dict]) -> list[int]:
    """
    Extract reading order from blocks sorted by y (top-to-bottom),
    then x (left-to-right) within each line. Returns list of indices
    in the original block order, ranked by reading position.
    """
    if not blocks:
        return []
    # Sort by y1 then x1
    indexed = [(i, _normalize_bbox_to_xyxy(b.get("bbox", [0,0,0,0]))[1],
                _normalize_bbox_to_xyxy(b.get("bbox", [0,0,0,0]))[0], b) for i, b in enumerate(blocks)]
    # Group into lines (blocks within 20px y-difference)
    indexed.sort(key=lambda x: (x[1], x[2]))  # y, then x
    return [i for i, _, _, _ in indexed]


def compute_reading_order_from_blocks(
    predicted_blocks: list[dict],
    ground_truth_blocks: list[dict],
    ground_truth_order: list[int],
) -> dict:
    """
    Compute reading order accuracy by comparing predicted block ordering
    (top-to-bottom, left-to-right heuristic) against ground truth order.

    Returns dict with kendall_tau, predicted_order, gt_order.
    """
    pred_order = extract_reading_order(predicted_blocks)

    # Build rank mapping: for each GT block, what's its rank in predicted order?
    # We need to align blocks. Use greedy spatial matching.
    gt_ranks = list(range(len(ground_truth_blocks)))
    pred_ranks = []

    matched_gt: set[int] = set()
    for pi in pred_order:
        p_bbox = _normalize_bbox_to_xyxy(predicted_blocks[pi].get("bbox", []))
        best_iou = 0.0
        best_gt_idx = -1
        for gj in range(len(ground_truth_blocks)):
            if gj in matched_gt:
                continue
            g_bbox = _normalize_bbox_to_xyxy(ground_truth_blocks[gj].get("bbox", []))
            iou = _bbox_iou(p_bbox, g_bbox)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gj
        if best_gt_idx >= 0 and best_iou > 0.05:
            pred_ranks.append(best_gt_idx)
            matched_gt.add(best_gt_idx)

    # Use ground_truth_order to get canonical ordering
    if ground_truth_order and len(ground_truth_order) == len(ground_truth_blocks):
        # ground_truth_order[i] = rank of block i (0 = first in reading order)
        gt_ordered = sorted(range(len(ground_truth_blocks)), key=lambda i: ground_truth_order[i])

        # Map pred to the same ordering.
        # pred_ranks = list of GT indices in predicted visit order.
        # Build parallel rank lists: for each matched GT block, its
        # predicted rank (position in pred_ranks) vs GT rank (from ground_truth_order).
        common = sorted(matched_gt)  # sort by GT index for consistent alignment
        if len(common) >= 2:
            pred_ranks_mapped = []
            gt_ranks_mapped = []
            for gt_idx in common:
                try:
                    pred_rank = pred_ranks.index(gt_idx)
                except ValueError:
                    continue
                pred_ranks_mapped.append(pred_rank)
                gt_ranks_mapped.append(ground_truth_order[gt_idx])
            tau = compute_reading_order_tau(pred_ranks_mapped, gt_ranks_mapped)
        else:
            tau = 0.0
        return {"kendall_tau": tau, "matched_blocks": len(common), "total_gt_blocks": len(ground_truth_blocks)}

    # Fallback: compare pred ranks to GT ranks directly
    if len(pred_ranks) >= 2:
        tau = compute_reading_order_tau(pred_ranks, gt_ranks[:len(pred_ranks)])
    else:
        tau = 0.0
    return {"kendall_tau": tau, "matched_blocks": len(pred_ranks), "total_gt_blocks": len(ground_truth_blocks)}


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
    iou_threshold: float = 0.1,
) -> dict:
    """Mean IoU across matched error bounding-box pairs with precision/recall.

    Returns dict with mean_iou, matched, total_gt, total_pred, precision, recall.
    """
    if not ground_truth_errors:
        return {"mean_iou": 1.0 if not predicted_errors else 0.0,
                "matched": 0, "total_gt": 0, "total_pred": len(predicted_errors),
                "precision": 0.0, "recall": 1.0}

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
        if best_j >= 0 and best_iou >= iou_threshold:
            ious.append(best_iou)
            matched_gt.add(best_j)

    matched = len(ious)
    return {
        "mean_iou": sum(ious) / matched if ious else 0.0,
        "matched": matched,
        "total_gt": len(ground_truth_errors),
        "total_pred": len(predicted_errors),
        "precision": matched / len(predicted_errors) if predicted_errors else 0.0,
        "recall": matched / len(ground_truth_errors) if ground_truth_errors else 0.0,
    }


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
