#!/usr/bin/env python3
"""Render word-box overlays for the real-world handwritten dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "benchmark" / "test_dataset" / "realworld_raw"
DEFAULT_DATASET = PROJECT_ROOT / "benchmark" / "test_dataset" / "realworld_writing_errors.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmark" / "visualizations" / "realworld_draft_boxes"
ALIGNED_OUTPUT_DIR = PROJECT_ROOT / "benchmark" / "visualizations" / "realworld_aligned_words"


def draw_entry(entry: dict, image_dir: Path, output_dir: Path, field: str) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    image_path = image_dir / entry["image"]
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    words = entry.get(field, [])
    for word in words:
        bbox = word.get("bbox", [])
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 64, 64), width=2)
        label = f"{word.get('index', '')}:{str(word.get('text', ''))[:14]}"
        draw.text((x1 + 2, max(0, y1 - 16)), label, fill=(255, 64, 64), font=small_font)

    header = (
        f"{entry['image']}  field={field}  split={entry.get('split')}  "
        f"expected={entry.get('expected_word_count')} draft={entry.get('draft_word_count')} "
        f"delta={entry.get('word_count_delta')}"
    )
    draw.rectangle([0, 0, image.width, 34], fill=(255, 255, 255))
    draw.text((10, 8), header, fill=(0, 0, 0), font=font)

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "aligned_words" if field == "words" else "draft_boxes"
    out_path = output_dir / f"{Path(entry['image']).stem}_{suffix}.png"
    image.save(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--image-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--field", choices=["draft_words", "words"], default="draft_words")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0, help="Limit images rendered; 0 means all.")
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = ALIGNED_OUTPUT_DIR if args.field == "words" else DEFAULT_OUTPUT_DIR

    entries = json.loads(args.dataset.read_text())
    if args.limit > 0:
        entries = entries[:args.limit]
    for entry in entries:
        out_path = draw_entry(entry, args.image_dir, output_dir, args.field)
        print(out_path)


if __name__ == "__main__":
    main()
