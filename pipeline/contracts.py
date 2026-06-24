"""Shared contracts for Phase 4 handwritten-feedback pipelines."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ALLOWED_ERROR_TYPES = {
    "capitalization",
    "spelling",
    "grammar",
    "punctuation",
    "structural",
}


def clamp_bbox(
    bbox: Any,
    image_size: tuple[int, int] | None = None,
    normalized_1000: bool = False,
) -> list[int]:
    """Normalize a bbox-like value to ordered integer `[x1, y1, x2, y2]`."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return [0, 0, 0, 0]

    try:
        x1, y1, x2, y2 = [float(v) for v in list(bbox)[:4]]
    except (TypeError, ValueError):
        return [0, 0, 0, 0]

    if normalized_1000 and image_size is not None:
        width, height = image_size
        x1 = x1 / 999 * width
        x2 = x2 / 999 * width
        y1 = y1 / 999 * height
        y2 = y2 / 999 * height

    x1, y1, x2, y2 = [int(round(v)) for v in [x1, y1, x2, y2]]

    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = max(0, x2)
    y2 = max(0, y2)

    if image_size is not None:
        width, height = image_size
        x1 = min(x1, width)
        x2 = min(x2, width)
        y1 = min(y1, height)
        y2 = min(y2, height)

    return [x1, y1, x2, y2]


@dataclass
class TextBox:
    """A localized text span, usually a line or word."""

    bbox: list[int]
    text: str
    confidence: float = 1.0
    reading_order: int | None = None
    index: int | None = None
    source: str = ""

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        index: int | None = None,
        image_size: tuple[int, int] | None = None,
        source: str = "",
        normalized_1000: bool = False,
    ) -> "TextBox":
        bbox = data.get("bbox", data.get("box", [0, 0, 0, 0]))
        return cls(
            bbox=clamp_bbox(bbox, image_size=image_size, normalized_1000=normalized_1000),
            text=str(data.get("text", "")),
            confidence=float(data.get("confidence", 1.0) or 0.0),
            reading_order=data.get("reading_order", data.get("reading_order_rank")),
            index=data.get("index", index),
            source=str(data.get("source", source)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ErrorFinding:
    """A writing error localized on the source page."""

    type: str
    bbox: list[int]
    description: str
    correction: str = ""
    evidence_text: str = ""
    block_index: int | None = None
    word_indices: list[int] = field(default_factory=list)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        image_size: tuple[int, int] | None = None,
        normalized_1000: bool = False,
    ) -> "ErrorFinding":
        word_indices = data.get("word_indices", [])
        if not isinstance(word_indices, list):
            word_indices = []

        return cls(
            type=str(data.get("type", "")),
            bbox=clamp_bbox(
                data.get("bbox", [0, 0, 0, 0]),
                image_size=image_size,
                normalized_1000=normalized_1000,
            ),
            description=str(data.get("description", "")),
            correction=str(data.get("correction", "")),
            evidence_text=str(data.get("evidence_text", data.get("text", ""))),
            block_index=data.get("block_index"),
            word_indices=[int(i) for i in word_indices if isinstance(i, int)],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Feedback:
    """Natural-language feedback for the writer."""

    summary: str = ""
    strengths: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    raw_text: str = ""

    @classmethod
    def from_any(cls, value: Any) -> "Feedback":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(summary=value, raw_text=value)
        if not isinstance(value, dict):
            return cls()
        strengths = value.get("strengths", [])
        improvements = value.get("improvements", [])
        return cls(
            summary=str(value.get("summary", "")),
            strengths=[str(v) for v in strengths] if isinstance(strengths, list) else [],
            improvements=[str(v) for v in improvements] if isinstance(improvements, list) else [],
            raw_text=str(value.get("raw_text", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineOutput:
    """Full output of one Phase 4 strategy on one input image."""

    strategy_name: str
    image: str
    text: str = ""
    boxes: list[TextBox] = field(default_factory=list)
    errors: list[ErrorFinding] = field(default_factory=list)
    feedback: Feedback = field(default_factory=Feedback)
    stage1_latency: float = 0.0
    stage2_latency: float = 0.0
    total_latency: float = 0.0
    parse_valid: bool = True
    repair_attempted: bool = False
    repair_succeeded: bool = False
    raw_response: str = ""
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_ocr_dict(
        cls,
        data: dict[str, Any],
        *,
        strategy_name: str,
        image: str | Path,
        image_size: tuple[int, int] | None = None,
    ) -> "PipelineOutput":
        boxes = [
            TextBox.from_dict(item, index=i, image_size=image_size)
            for i, item in enumerate(data.get("blocks", data.get("boxes", [])))
        ]
        return cls(
            strategy_name=strategy_name,
            image=Path(image).name,
            text=str(data.get("text", "")),
            boxes=boxes,
            stage1_latency=float(data.get("stage1_latency", data.get("latency_s", 0.0)) or 0.0),
            total_latency=float(data.get("stage1_latency", data.get("latency_s", 0.0)) or 0.0),
            raw_response=str(data.get("raw_response", "")),
            metadata={k: v for k, v in data.items() if k.startswith("_")},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "image": self.image,
            "text": self.text,
            "boxes": [box.to_dict() for box in self.boxes],
            "errors": [error.to_dict() for error in self.errors],
            "feedback": self.feedback.to_dict(),
            "stage1_latency": self.stage1_latency,
            "stage2_latency": self.stage2_latency,
            "total_latency": self.total_latency,
            "parse_valid": self.parse_valid,
            "repair_attempted": self.repair_attempted,
            "repair_succeeded": self.repair_succeeded,
            "raw_response": self.raw_response,
            "notes": list(self.notes),
            "metadata": dict(self.metadata),
        }
