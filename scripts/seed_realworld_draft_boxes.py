#!/usr/bin/env python3
"""Seed draft word boxes for the real-world handwritten dataset.

The generated boxes are a starting point for annotation, not ground truth.
They are stored under `draft_words`; final reviewed boxes should be moved to
`words` after manual alignment with the source text.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "benchmark" / "test_dataset" / "realworld_raw"
DEFAULT_INPUT = PROJECT_ROOT / "benchmark" / "test_dataset" / "realworld_writing_errors.json"


def expected_token_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def tesseract_words(image_path: Path, psm: int) -> list[dict[str, Any]]:
    from PIL import Image
    import pytesseract
    from pytesseract import Output

    with Image.open(image_path) as image:
        data = pytesseract.image_to_data(
            image,
            output_type=Output.DICT,
            config=f"--psm {psm}",
        )

    words: list[dict[str, Any]] = []
    for i, text in enumerate(data.get("text", [])):
        text = str(text).strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf < 0:
            continue
        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])
        words.append({
            "index": len(words),
            "text": text,
            "bbox": [x, y, x + w, y + h],
            "confidence": conf / 100.0,
            "source": f"tesseract_psm{psm}",
        })
    return words


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--image-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--psm", type=int, default=6)
    args = parser.parse_args()

    entries = json.loads(args.input.read_text())
    total_draft_words = 0
    exact_count_matches = 0
    for entry in entries:
        image_path = args.image_dir / entry["image"]
        draft_words = tesseract_words(image_path, args.psm) if image_path.exists() else []
        expected = expected_token_count(entry.get("text", ""))
        total_draft_words += len(draft_words)
        if expected == len(draft_words):
            exact_count_matches += 1
        entry["draft_words"] = draft_words
        entry["draft_word_source"] = f"tesseract_psm{args.psm}"
        entry["expected_word_count"] = expected
        entry["draft_word_count"] = len(draft_words)
        entry["word_count_delta"] = len(draft_words) - expected
        entry["annotation_status"] = (
            "needs_manual_word_review"
            if draft_words
            else "needs_word_boxes"
        )

    args.output.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {args.output}")
    print(
        "pages={pages} draft_words={words} exact_count_matches={matches}".format(
            pages=len(entries),
            words=total_draft_words,
            matches=exact_count_matches,
        )
    )


if __name__ == "__main__":
    main()
