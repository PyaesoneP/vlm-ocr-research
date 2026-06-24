#!/usr/bin/env python3
"""Validate the Phase 4 real-world handwritten error dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "benchmark" / "test_dataset" / "realworld_writing_errors.json"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "benchmark" / "test_dataset" / "realworld_raw"

EXPECTED_PAGE_COUNT = 20
EXPECTED_CLEAN_COUNT = 10
EXPECTED_POSITIVE_COUNT = 10
EXPECTED_ERROR_COUNT = 21

ALLOWED_PAGE_STATUSES = {"auto_aligned_needs_review", "manual_reviewed"}
ALLOWED_REVIEW_STATUSES = {"needs_review", "manual_reviewed"}


def load_json(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise SystemExit(f"{path}: expected a list of page entries")
    return data


def bbox_union(boxes: list[list[int]]) -> list[int]:
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def valid_bbox(value: Any, image_size: tuple[int, int]) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    if not all(isinstance(v, int) for v in value):
        return False
    x1, y1, x2, y2 = value
    width, height = image_size
    return 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height


def image_size(path: Path) -> tuple[int, int] | None:
    if not path.exists():
        return None
    with Image.open(path) as image:
        return image.size


def page_word_tokens(entry: dict[str, Any]) -> list[str]:
    return [str(word.get("text", "")) for word in entry.get("words", [])]


def validate_page(entry: dict[str, Any], image_dir: Path) -> list[str]:
    errors: list[str] = []
    image_name = str(entry.get("image", ""))
    prefix = image_name or "<missing image>"

    image_path = image_dir / image_name
    actual_size = image_size(image_path)
    if actual_size is None:
        errors.append(f"{prefix}: image file is missing")
        actual_size = tuple(entry.get("image_size", [0, 0]))  # type: ignore[assignment]
    elif list(actual_size) != entry.get("image_size"):
        errors.append(f"{prefix}: image_size {entry.get('image_size')} != actual {list(actual_size)}")

    status = entry.get("annotation_status")
    if status not in ALLOWED_PAGE_STATUSES:
        errors.append(f"{prefix}.annotation_status: expected one of {sorted(ALLOWED_PAGE_STATUSES)}, got {status!r}")

    words = entry.get("words", [])
    if not isinstance(words, list) or not words:
        errors.append(f"{prefix}.words: expected non-empty list")
        words = []

    indices = [word.get("index") for word in words]
    expected_indices = list(range(len(words)))
    if indices != expected_indices:
        errors.append(f"{prefix}.words[].index: expected {expected_indices}, got {indices}")

    text_tokens = str(entry.get("text", "")).split()
    word_tokens = page_word_tokens(entry)
    if text_tokens != word_tokens:
        errors.append(f"{prefix}: text tokenization does not match words[].text")

    word_by_index: dict[int, dict[str, Any]] = {}
    for offset, word in enumerate(words):
        location = f"{prefix}.words[{offset}]"
        index = word.get("index")
        if isinstance(index, int):
            word_by_index[index] = word
        if not str(word.get("text", "")).strip():
            errors.append(f"{location}.text: empty word text")
        if actual_size and not valid_bbox(word.get("bbox"), actual_size):
            errors.append(f"{location}.bbox: invalid or out of bounds: {word.get('bbox')}")
        review_status = word.get("review_status")
        if review_status not in ALLOWED_REVIEW_STATUSES:
            errors.append(f"{location}.review_status: expected one of {sorted(ALLOWED_REVIEW_STATUSES)}, got {review_status!r}")
        if status == "manual_reviewed" and review_status != "manual_reviewed":
            errors.append(f"{location}.review_status: reviewed page requires 'manual_reviewed'")

    page_errors = entry.get("errors", [])
    if not isinstance(page_errors, list):
        errors.append(f"{prefix}.errors: expected list")
        page_errors = []

    for offset, error in enumerate(page_errors):
        location = f"{prefix}.errors[{offset}]"
        word_indices = error.get("word_indices")
        if not isinstance(word_indices, list) or not word_indices:
            errors.append(f"{location}.word_indices: expected non-empty list")
            continue
        missing = [idx for idx in word_indices if not isinstance(idx, int) or idx not in word_by_index]
        if missing:
            errors.append(f"{location}.word_indices: unknown indices {missing}")
            continue

        referenced_words = [word_by_index[idx] for idx in word_indices]
        referenced_boxes = [word["bbox"] for word in referenced_words]
        expected_bbox = bbox_union(referenced_boxes)
        if error.get("bbox") != expected_bbox:
            errors.append(f"{location}.bbox: expected union {expected_bbox}, got {error.get('bbox')}")
        if actual_size and not valid_bbox(error.get("bbox"), actual_size):
            errors.append(f"{location}.bbox: invalid or out of bounds: {error.get('bbox')}")

        evidence = str(error.get("evidence_text", ""))
        referenced_text = " ".join(str(word.get("text", "")) for word in referenced_words)
        if evidence and evidence.strip(" .,!?:;\"'") not in referenced_text:
            errors.append(f"{location}.evidence_text: {evidence!r} not found in referenced words {referenced_text!r}")

        error_status = error.get("annotation_status")
        if error_status not in ALLOWED_PAGE_STATUSES:
            errors.append(f"{location}.annotation_status: expected one of {sorted(ALLOWED_PAGE_STATUSES)}, got {error_status!r}")
        if status == "manual_reviewed" and error_status != "manual_reviewed":
            errors.append(f"{location}.annotation_status: reviewed page requires 'manual_reviewed'")

    if status == "manual_reviewed" and any(error.get("annotation_status") != "manual_reviewed" for error in page_errors):
        errors.append(f"{prefix}: reviewed page contains non-reviewed errors")

    return errors


def validate_dataset(entries: list[dict[str, Any]], image_dir: Path) -> list[str]:
    errors: list[str] = []

    if len(entries) != EXPECTED_PAGE_COUNT:
        errors.append(f"dataset: expected {EXPECTED_PAGE_COUNT} pages, got {len(entries)}")

    image_names = [entry.get("image") for entry in entries]
    if len(set(image_names)) != len(image_names):
        errors.append("dataset: duplicate image names found")

    clean_count = sum(1 for entry in entries if not entry.get("errors"))
    positive_count = sum(1 for entry in entries if entry.get("errors"))
    error_count = sum(len(entry.get("errors", [])) for entry in entries)
    if clean_count != EXPECTED_CLEAN_COUNT:
        errors.append(f"dataset: expected {EXPECTED_CLEAN_COUNT} clean pages, got {clean_count}")
    if positive_count != EXPECTED_POSITIVE_COUNT:
        errors.append(f"dataset: expected {EXPECTED_POSITIVE_COUNT} positive pages, got {positive_count}")
    if error_count != EXPECTED_ERROR_COUNT:
        errors.append(f"dataset: expected {EXPECTED_ERROR_COUNT} errors, got {error_count}")

    for entry in entries:
        errors.extend(validate_page(entry, image_dir))

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    args = parser.parse_args()

    entries = load_json(args.dataset)
    errors = validate_dataset(entries, args.image_dir)
    if errors:
        print(f"Validation failed with {len(errors)} issue(s):")
        for error in errors:
            print(f"- {error}")
        sys.exit(1)

    reviewed = sum(entry.get("annotation_status") == "manual_reviewed" for entry in entries)
    total_errors = sum(len(entry.get("errors", [])) for entry in entries)
    print(
        "Validation passed: "
        f"{len(entries)} pages, {reviewed} manually reviewed, {total_errors} errors."
    )


if __name__ == "__main__":
    main()
