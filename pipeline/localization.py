"""Localization helpers used by Phase 4 prompts, parsing, and metrics."""

from __future__ import annotations

from typing import Iterable

from pipeline.contracts import TextBox


def union_bboxes(bboxes: Iterable[list[int]]) -> list[int]:
    """Return the tight union of non-empty `[x1, y1, x2, y2]` bboxes."""
    valid = [b for b in bboxes if len(b) == 4 and b != [0, 0, 0, 0]]
    if not valid:
        return [0, 0, 0, 0]
    return [
        min(b[0] for b in valid),
        min(b[1] for b in valid),
        max(b[2] for b in valid),
        max(b[3] for b in valid),
    ]


def bbox_from_word_indices(boxes: list[TextBox], word_indices: list[int]) -> list[int]:
    """Union the boxes referenced by 0-based word indices."""
    selected = [boxes[i].bbox for i in word_indices if 0 <= i < len(boxes)]
    return union_bboxes(selected)


def blocks_as_prompt_payload(boxes: list[TextBox]) -> list[dict]:
    """Compact JSON-serializable box payload for LLM prompts."""
    payload = []
    for i, box in enumerate(boxes):
        payload.append({
            "index": box.index if box.index is not None else i,
            "bbox": box.bbox,
            "text": box.text,
        })
    return payload
