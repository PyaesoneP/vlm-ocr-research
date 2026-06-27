#!/usr/bin/env python3
"""Bootstrap the real-world handwritten-error dataset from source_texts.md.

This script creates the annotation skeleton only. Word boxes are intentionally
left empty until an OCR/localizer-assisted annotation pass fills them in.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "benchmark" / "test_dataset" / "realworld_raw"
SOURCE_TEXTS = RAW_DIR / "source_texts.md"
DEFAULT_OUTPUT = PROJECT_ROOT / "benchmark" / "test_dataset" / "realworld_writing_errors.json"


LABEL_TO_ERROR_TYPE = {
    "AGREE": "grammar",
    "CAPITAL": "capitalization",
    "DOUBLE": "grammar",
    "GRAMMAR": "grammar",
    "HOMOPHONE": "spelling",
    "MERGE": "grammar",
    "OMIT": "grammar",
    "SPELL": "spelling",
}


def _paragraph_after(marker: str, section: str) -> str:
    idx = section.find(marker)
    if idx < 0:
        return ""
    tail = section[idx + len(marker):]
    lines: list[str] = []
    for line in tail.splitlines():
        stripped = line.strip()
        if not stripped and lines:
            break
        if stripped.startswith(">"):
            lines.append(stripped[1:].strip())
    return " ".join(lines).strip()


def _parse_intended_error(line: str) -> dict[str, Any] | None:
    match = re.match(r"- \*\*\[([A-Z]+)\]\*\*\s+(.+)", line.strip())
    if not match:
        return None

    source_label, detail = match.groups()
    error_type = LABEL_TO_ERROR_TYPE.get(source_label, "grammar")
    note = ""
    if " — " in detail:
        detail, note = detail.split(" — ", 1)
    near = ""
    near_match = re.search(r"\(near\s+\"(.+?)\"\)", detail)
    if near_match:
        near = near_match.group(1)
        detail = detail[:near_match.start()].strip()

    evidence_text = ""
    correction = ""
    arrow = "→" if "→" in detail else "->" if "->" in detail else ""
    if arrow:
        left, right = detail.split(arrow, 1)
        evidence_text = left.strip().strip('"')
        correction = right.strip().strip('"')
    elif detail.lower().startswith("insert "):
        insert_match = re.search(
            r'insert\s+"(.+?)"\s+between\s+"(.+?)"\s+and\s+"(.+?)"',
            detail,
            re.I,
        )
        if insert_match:
            inserted, before, after = insert_match.groups()
            evidence_text = f"{before} {after}"
            correction = f"{before} {inserted} {after}"
        else:
            evidence_text = detail
            correction = detail
    else:
        evidence_text = detail
        correction = detail

    return {
        "type": error_type,
        "source_label": source_label,
        "word_indices": [],
        "bbox": [0, 0, 0, 0],
        "evidence_text": evidence_text,
        "correction": correction,
        "description": note.strip(" _") if note else "",
        "near": near,
        "annotation_status": "needs_word_indices",
    }


def parse_source_texts(source_path: Path, image_dir: Path | None = None) -> list[dict[str, Any]]:
    raw = source_path.read_text()
    chunks = re.split(r"\n###\s+rw_", raw)
    image_dir = image_dir or source_path.parent
    entries: list[dict[str, Any]] = []
    for chunk in chunks[1:]:
        first_line, _, section_tail = chunk.partition("\n")
        page_number = int(first_line.strip())
        image_name = f"rw_{page_number}.jpg"
        section = section_tail

        text = _paragraph_after("**Write exactly", section)
        corrected_text = _paragraph_after("**Correct version", section)
        errors = [
            parsed
            for line in section.splitlines()
            if (parsed := _parse_intended_error(line)) is not None
        ]
        split = "positive" if errors else "clean"

        image_size = None
        image_path = image_dir / image_name
        if image_path.exists():
            try:
                from PIL import Image

                with Image.open(image_path) as image:
                    image_size = list(image.size)
            except ImportError:
                image_size = None

        entries.append({
            "image": image_name,
            "split": split,
            "text": text,
            "corrected_text": corrected_text or text,
            "image_size": image_size,
            "words": [],
            "errors": errors,
            "annotation_status": "needs_word_boxes" if image_size else "missing_image_or_size",
        })
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_TEXTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Directory containing page images. Defaults to the source_texts.md parent.",
    )
    args = parser.parse_args()

    entries = parse_source_texts(args.source, image_dir=args.image_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n")

    positives = sum(1 for entry in entries if entry["split"] == "positive")
    clean = sum(1 for entry in entries if entry["split"] == "clean")
    errors = sum(len(entry["errors"]) for entry in entries)
    print(f"Wrote {args.output}")
    print(f"pages={len(entries)} clean={clean} positive={positives} intended_errors={errors}")


if __name__ == "__main__":
    main()
