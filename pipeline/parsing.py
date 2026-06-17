"""Strict JSON parsing and schema normalization for Phase 4 model outputs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from pipeline.contracts import ALLOWED_ERROR_TYPES, ErrorFinding, Feedback, TextBox


@dataclass
class ParseResult:
    """Normalized model response plus parsing/schema diagnostics."""

    text: str = ""
    boxes: list[TextBox] = field(default_factory=list)
    errors: list[ErrorFinding] = field(default_factory=list)
    feedback: Feedback = field(default_factory=Feedback)
    valid: bool = False
    problems: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)


def _json_candidates(raw: str) -> list[str]:
    candidates = [raw.strip()]
    candidates.extend(m.group(1).strip() for m in re.finditer(r"```(?:json)?\s*(.*?)```", raw, re.S))
    if "{" in raw and "}" in raw:
        candidates.append(raw[raw.find("{"):raw.rfind("}") + 1])
    return [c for c in candidates if c]


def _load_json_object(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    decoder = json.JSONDecoder()

    for candidate in _json_candidates(raw):
        try:
            loaded = json.loads(candidate)
            if isinstance(loaded, dict):
                return loaded, None
            return None, "top-level JSON value must be an object"
        except json.JSONDecodeError:
            pass

    # Last-chance scan for the first valid object embedded in text.
    for match in re.finditer(r"{", raw):
        try:
            loaded, _ = decoder.raw_decode(raw[match.start():])
            if isinstance(loaded, dict):
                return loaded, None
        except json.JSONDecodeError:
            continue

    return None, "could not parse a JSON object"


def _iter_raw_bboxes(items: Any) -> list[list[float]]:
    if not isinstance(items, list):
        return []
    bboxes = []
    for item in items:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox", item.get("box"))
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        try:
            bboxes.append([float(v) for v in bbox[:4]])
        except (TypeError, ValueError):
            continue
    return bboxes


def _looks_normalized_1000(data: dict[str, Any], image_size: tuple[int, int] | None) -> bool:
    if image_size is None:
        return False
    width, height = image_size
    if width <= 1000 and height <= 1000:
        return False

    boxes = _iter_raw_bboxes(data.get("blocks", data.get("boxes", data.get("text_boxes", []))))
    boxes.extend(_iter_raw_bboxes(data.get("errors", [])))
    if not boxes:
        return False

    coords = [coord for bbox in boxes for coord in bbox]
    return min(coords) >= 0 and max(coords) <= 1000


def parse_model_json(
    raw: str,
    *,
    image_size: tuple[int, int] | None = None,
    require_text: bool = False,
    require_boxes: bool = False,
) -> ParseResult:
    """Parse model JSON and normalize it into contracts."""
    data, problem = _load_json_object(raw)
    if data is None:
        return ParseResult(valid=False, problems=[problem or "invalid JSON"])

    problems: list[str] = []
    normalized_1000 = _looks_normalized_1000(data, image_size)

    text = str(data.get("text", data.get("transcription", "")))
    if require_text and not text.strip():
        problems.append("missing non-empty text")

    raw_boxes = data.get("blocks", data.get("boxes", data.get("text_boxes", [])))
    if raw_boxes is None:
        raw_boxes = []
    if not isinstance(raw_boxes, list):
        problems.append("blocks must be a list")
        raw_boxes = []

    boxes = []
    for i, item in enumerate(raw_boxes):
        if not isinstance(item, dict):
            problems.append(f"block {i} must be an object")
            continue
        box = TextBox.from_dict(
            item,
            index=i,
            image_size=image_size,
            normalized_1000=normalized_1000,
        )
        if box.bbox == [0, 0, 0, 0]:
            problems.append(f"block {i} has an empty bbox")
        boxes.append(box)
    if require_boxes and not boxes:
        problems.append("missing localized text boxes")

    raw_errors = data.get("errors", [])
    if raw_errors is None:
        raw_errors = []
    if not isinstance(raw_errors, list):
        problems.append("errors must be a list")
        raw_errors = []

    errors: list[ErrorFinding] = []
    for i, item in enumerate(raw_errors):
        if not isinstance(item, dict):
            problems.append(f"error {i} must be an object")
            continue
        finding = ErrorFinding.from_dict(
            item,
            image_size=image_size,
            normalized_1000=normalized_1000,
        )
        if finding.type not in ALLOWED_ERROR_TYPES:
            problems.append(f"error {i} has invalid type: {finding.type!r}")
            continue
        if finding.bbox == [0, 0, 0, 0]:
            problems.append(f"error {i} has an empty bbox")
        if not finding.description:
            problems.append(f"error {i} is missing description")
        errors.append(finding)

    feedback = Feedback.from_any(data.get("feedback", {}))
    valid = not problems
    return ParseResult(
        text=text,
        boxes=boxes,
        errors=errors,
        feedback=feedback,
        valid=valid,
        problems=problems,
        data=data,
    )
