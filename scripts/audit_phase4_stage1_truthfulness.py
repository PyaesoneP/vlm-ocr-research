#!/usr/bin/env python3
"""Audit Phase 4 real-world Stage 1 truthfulness from cached OCR outputs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.metrics import compute_iou
from pipeline.localization import union_bboxes


DEFAULT_DATASET = PROJECT_ROOT / "benchmark" / "test_dataset" / "realworld_writing_errors.json"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "pipeline_output" / "phase4_cache"
DEFAULT_OUTPUT = PROJECT_ROOT / "benchmark" / "results" / "phase4_stage1_truthfulness_audit.json"

DEFAULT_SOURCES = [
    "qwen3vl_4b_verbatim_word",
    "qwen3vl_4b_word",
    "qwen3vl_4b_crop_verified_word_v2",
    "tesseract_word_boxes",
    "doctr_live_word_ocr",
    "easyocr_live_word_ocr",
]


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _contains_span(text: str, span: str) -> bool:
    normalized_text = f" {_normalize(text)} "
    normalized_span = _normalize(span)
    return bool(normalized_span and f" {normalized_span} " in normalized_text)


def _bbox_iou(left: list[int], right: list[int]) -> float:
    result = compute_iou([{"bbox": left}], [{"bbox": right}])
    return float(result["mean_iou"] if result.get("matched") else 0.0)


def load_dataset(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text())


def load_cache_entry(cache_dir: Path, source: str, image_name: str) -> dict[str, Any] | None:
    path = cache_dir / source / f"{Path(image_name).stem}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def source_text(entry: dict[str, Any] | None) -> str:
    if not entry:
        return ""
    return str(entry.get("text", ""))


def source_boxes(entry: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not entry:
        return []
    boxes = entry.get("blocks", entry.get("boxes", []))
    return boxes if isinstance(boxes, list) else []


def matching_box_iou(boxes: list[dict[str, Any]], evidence: str, gt_bbox: list[int]) -> float:
    best = 0.0
    evidence_norm = _normalize(evidence)
    for box in boxes:
        if evidence_norm and evidence_norm not in _normalize(str(box.get("text", ""))):
            continue
        bbox = box.get("bbox", [])
        if not isinstance(bbox, list) or len(bbox) < 4:
            continue
        best = max(best, _bbox_iou([int(v) for v in bbox[:4]], gt_bbox))
    return best


def error_bbox_from_words(page: dict[str, Any], error: dict[str, Any]) -> list[int]:
    words = page.get("words", [])
    bboxes = []
    for index in error.get("word_indices", []):
        if isinstance(index, int) and 0 <= index < len(words):
            bbox = words[index].get("bbox", [])
            if isinstance(bbox, list) and len(bbox) >= 4:
                bboxes.append([int(v) for v in bbox[:4]])
    return union_bboxes(bboxes) if bboxes else [0, 0, 0, 0]


def classify_error(
    page: dict[str, Any],
    error: dict[str, Any],
    entry: dict[str, Any] | None,
) -> dict[str, Any]:
    text = source_text(entry)
    boxes = source_boxes(entry)
    evidence = str(error.get("evidence_text", ""))
    correction = str(error.get("correction", ""))
    evidence_present = _contains_span(text, evidence)
    correction_present = _contains_span(text, correction)
    gt_bbox = error.get("bbox") or error_bbox_from_words(page, error)
    box_iou = matching_box_iou(boxes, evidence, gt_bbox) if evidence_present else 0.0

    if entry is None:
        classification = "missing"
    elif evidence_present and correction_present:
        classification = "ambiguous"
    elif evidence_present and boxes and box_iou < 0.05:
        classification = "wrong_line_or_box"
    elif evidence_present:
        classification = "preserved"
    elif correction_present:
        classification = "corrected_leaked"
    elif text.strip():
        classification = "garbled"
    else:
        classification = "missing"

    return {
        "type": error.get("type", ""),
        "evidence_text": evidence,
        "correction": correction,
        "word_indices": error.get("word_indices", []),
        "gt_bbox": gt_bbox,
        "classification": classification,
        "evidence_present": evidence_present,
        "correction_present": correction_present,
        "matching_box_iou": box_iou,
    }


def audit_source(dataset: list[dict[str, Any]], cache_dir: Path, source: str) -> dict[str, Any]:
    pages = []
    counts = {
        "preserved": 0,
        "corrected_leaked": 0,
        "garbled": 0,
        "missing": 0,
        "wrong_line_or_box": 0,
        "ambiguous": 0,
    }
    total_errors = 0
    cached_pages = 0
    for page in dataset:
        entry = load_cache_entry(cache_dir, source, page["image"])
        if entry is not None:
            cached_pages += 1
        errors = []
        for error in page.get("errors", []):
            total_errors += 1
            detail = classify_error(page, error, entry)
            counts[detail["classification"]] += 1
            errors.append(detail)
        pages.append({
            "image": page["image"],
            "has_cache": entry is not None,
            "text": source_text(entry),
            "errors": errors,
        })

    return {
        "source": source,
        "cached_pages": cached_pages,
        "total_pages": len(dataset),
        "total_errors": total_errors,
        "counts": counts,
        "evidence_preserved_rate": counts["preserved"] / total_errors if total_errors else 0.0,
        "correction_leak_rate": counts["corrected_leaked"] / total_errors if total_errors else 0.0,
        "pages": pages,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset)
    audits = [audit_source(dataset, args.cache_dir, source) for source in args.sources]
    result = {
        "dataset": str(args.dataset.relative_to(PROJECT_ROOT)),
        "cache_dir": str(args.cache_dir.relative_to(PROJECT_ROOT)),
        "sources": args.sources,
        "audits": audits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    for audit in audits:
        counts = audit["counts"]
        print(
            f"{audit['source']}: preserved={counts['preserved']}/{audit['total_errors']} "
            f"leaked={counts['corrected_leaked']} wrong_box={counts['wrong_line_or_box']} "
            f"garbled={counts['garbled']} missing={counts['missing']} ambiguous={counts['ambiguous']}"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
