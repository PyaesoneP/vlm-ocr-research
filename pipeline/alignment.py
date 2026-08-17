"""Word-list alignment helpers for Phase 4 text/box composition."""

from __future__ import annotations

import math
import re
from difflib import SequenceMatcher
from typing import Any

from pipeline.contracts import TextBox
from pipeline.localization import union_bboxes


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def text_similarity(left: str, right: str) -> float:
    """Return a forgiving similarity score for OCR word labels."""
    left_norm = _normalize(left)
    right_norm = _normalize(right)
    left_compact = _compact(left)
    right_compact = _compact(right)
    if not left_compact or not right_compact:
        return 0.0
    if left_norm == right_norm or left_compact == right_compact:
        return 1.0
    return max(
        SequenceMatcher(None, left_norm, right_norm).ratio(),
        SequenceMatcher(None, left_compact, right_compact).ratio(),
    )


def _bbox_iou(left: list[int], right: list[int]) -> float:
    if len(left) != 4 or len(right) != 4:
        return 0.0
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1 = max(lx1, rx1)
    iy1 = max(ly1, ry1)
    ix2 = min(lx2, rx2)
    iy2 = min(ly2, ry2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    left_area = max(0, lx2 - lx1) * max(0, ly2 - ly1)
    right_area = max(0, rx2 - rx1) * max(0, ry2 - ry1)
    union = left_area + right_area - inter
    return inter / union if union else 0.0


def _center_score(left: list[int], right: list[int]) -> float:
    if len(left) != 4 or len(right) != 4:
        return 0.0
    left_cx = (left[0] + left[2]) / 2
    left_cy = (left[1] + left[3]) / 2
    right_cx = (right[0] + right[2]) / 2
    right_cy = (right[1] + right[3]) / 2
    distance = math.hypot(left_cx - right_cx, left_cy - right_cy)
    left_diag = math.hypot(max(1, left[2] - left[0]), max(1, left[3] - left[1]))
    right_diag = math.hypot(max(1, right[2] - right[0]), max(1, right[3] - right[1]))
    scale = max(left_diag, right_diag, 1.0)
    return max(0.0, 1.0 - distance / (scale * 4.0))


def _spatial_score(left: list[int], right: list[int]) -> float:
    return max(_bbox_iou(left, right), _center_score(left, right))


def _span_text(boxes: list[TextBox]) -> str:
    return " ".join(box.text for box in boxes if box.text)


def _copy_with_geometry(
    source: TextBox,
    geometry: list[TextBox],
    *,
    index: int,
    score: float,
) -> TextBox:
    bbox = union_bboxes([box.bbox for box in geometry])
    confidence = sum(box.confidence for box in geometry) / len(geometry) if geometry else source.confidence
    return TextBox(
        bbox=bbox,
        text=source.text,
        confidence=confidence,
        reading_order=source.reading_order,
        index=index,
        source="aligned_tesseract_word_boxes",
    )


def align_text_boxes_to_geometry(
    text_boxes: list[TextBox],
    geometry_boxes: list[TextBox],
    *,
    max_span: int = 3,
    lookahead: int = 8,
    min_score: float = 0.32,
) -> tuple[list[TextBox], dict[str, Any]]:
    """Align canonical OCR text boxes to an alternate geometry source.

    The returned boxes preserve `text_boxes` length, order, and text labels. A box
    uses alternate geometry only when a nearby one-to-many span aligns by text or
    position; otherwise it falls back to the original OCR geometry.
    """
    if not text_boxes:
        return [], {
            "alignment_source": "aligned_tesseract_word_boxes",
            "alignment_text_boxes": 0,
            "alignment_geometry_boxes": len(geometry_boxes),
            "alignment_matched": 0,
            "alignment_fallback": 0,
            "alignment_multi_geometry_matches": 0,
            "alignment_mean_score": 0.0,
        }

    aligned: list[TextBox] = []
    used: set[int] = set()
    pointer = 0
    scores: list[float] = []
    fallback_count = 0
    multi_count = 0

    for index, source in enumerate(text_boxes):
        best: tuple[float, int, int, list[TextBox]] | None = None
        start_min = max(0, pointer - 2)
        start_max = min(len(geometry_boxes), pointer + lookahead)
        for start in range(start_min, start_max):
            if start in used:
                continue
            span: list[TextBox] = []
            for end in range(start, min(len(geometry_boxes), start + max_span)):
                if end in used:
                    break
                span.append(geometry_boxes[end])
                span_bbox = union_bboxes([box.bbox for box in span])
                label_score = text_similarity(source.text, _span_text(span))
                position_score = _spatial_score(source.bbox, span_bbox)
                score = 0.6 * label_score + 0.4 * position_score
                source_token = _compact(source.text)
                spatial_only_one_char = bool(
                    source_token and len(source_token) <= 1 and position_score >= 0.7
                )
                if label_score < 0.5 and not spatial_only_one_char:
                    continue
                if best is None or score > best[0]:
                    best = (score, start, end, list(span))

        if best is not None and best[0] >= min_score:
            score, start, end, span = best
            for used_index in range(start, end + 1):
                used.add(used_index)
            pointer = max(pointer, end + 1)
            scores.append(score)
            if len(span) > 1:
                multi_count += 1
            aligned.append(_copy_with_geometry(source, span, index=index, score=score))
        else:
            fallback_count += 1
            aligned.append(TextBox(
                bbox=list(source.bbox),
                text=source.text,
                confidence=source.confidence,
                reading_order=source.reading_order,
                index=index,
                source=source.source or "text_source_fallback_box",
            ))

    matched = len(text_boxes) - fallback_count
    stats = {
        "alignment_source": "aligned_tesseract_word_boxes",
        "alignment_text_boxes": len(text_boxes),
        "alignment_geometry_boxes": len(geometry_boxes),
        "alignment_matched": matched,
        "alignment_fallback": fallback_count,
        "alignment_multi_geometry_matches": multi_count,
        "alignment_unused_geometry": len(geometry_boxes) - len(used),
        "alignment_mean_score": sum(scores) / len(scores) if scores else 0.0,
    }
    return aligned, stats
