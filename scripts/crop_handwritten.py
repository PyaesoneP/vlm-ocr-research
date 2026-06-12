#!/usr/bin/env python3
"""
Crop IAM form images to the handwritten region using XML annotations.

The IAM XML `<handwritten-part>` contains `<line>` elements with
character-level `<cmp>` (component) bounding boxes. This script:
  1. Computes the union bounding box of all handwritten `<cmp>` elements
  2. Excludes the last 1-2 lines if they appear to be signatures
  3. Adds padding and crops the image
  4. Saves cropped images and a manifest JSON

Usage:
    python scripts/crop_handwritten.py
    python scripts/crop_handwritten.py --no-signature-filter  # keep all lines
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURATED_MANIFEST = PROJECT_ROOT / "benchmark" / "test_dataset" / "curated_manifest.json"
IAM_XML_DIR = PROJECT_ROOT / "benchmark" / "test_dataset" / "iam_xml" / "archive" / "xml"
CURATED_DIR = PROJECT_ROOT / "benchmark" / "test_dataset" / "curated"
OUTPUT_DIR = PROJECT_ROOT / "benchmark" / "test_dataset" / "handwritten"
MANIFEST_OUT = PROJECT_ROOT / "benchmark" / "test_dataset" / "handwritten_manifest.json"

PADDING = 20  # pixels on each side


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def load_curated_manifest() -> list[dict]:
    with open(CURATED_MANIFEST) as f:
        return json.load(f)


def find_xml(image_name: str) -> Optional[Path]:
    """Find IAM XML for an image by stem match."""
    stem = Path(image_name).stem
    xml_path = IAM_XML_DIR / f"{stem}.xml"
    if xml_path.exists():
        return xml_path
    return None


def extract_handwritten_boxes(xml_path: Path, filter_signatures: bool = True) -> tuple[list[dict], Optional[list[int]]]:
    """
    Extract bounding boxes for all handwritten lines from IAM XML.

    Returns (line_boxes, union_bbox) where:
      - line_boxes: list of {text, bbox: [x1,y1,x2,y2]}
      - union_bbox: [x1,y1,x2,y2] of all lines combined (or None if no boxes)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    form_height = int(root.get("height", 3542))
    handwritten = root.find("handwritten-part")
    if handwritten is None:
        return [], None

    lines = handwritten.findall("line")
    if not lines:
        return [], None

    line_boxes: list[dict] = []
    all_xs, all_ys, all_x2s, all_y2s = [], [], [], []

    for i, line in enumerate(lines):
        text = line.get("text", "").strip()
        if not text:
            continue

        xs, ys, x2s, y2s = [], [], [], []
        for cmp_elem in line.iter("cmp"):
            x = int(cmp_elem.get("x", 0))
            y = int(cmp_elem.get("y", 0))
            w = int(cmp_elem.get("width", 0))
            h = int(cmp_elem.get("height", 0))
            xs.append(x)
            ys.append(y)
            x2s.append(x + w)
            y2s.append(y + h)

        if not xs:
            continue

        bbox = [min(xs), min(ys), max(x2s), max(y2s)]

        # Signature filtering: skip lines that are (a) the last line with
        # "Name:" text, OR (b) very short (1-2 words) AND in the bottom 15%
        is_last = (i == len(lines) - 1)
        is_short = len(text.split()) <= 2
        in_bottom = bbox[1] > form_height * 0.85

        if filter_signatures and ((is_last and "Name:" in text) or (is_short and in_bottom)):
            continue

        line_boxes.append({"text": text, "bbox": bbox})
        all_xs.extend(xs)
        all_ys.extend(ys)
        all_x2s.extend(x2s)
        all_y2s.extend(y2s)

    if not all_xs:
        return line_boxes, None

    union_bbox = [min(all_xs), min(all_ys), max(all_x2s), max(all_y2s)]
    return line_boxes, union_bbox


def crop_image(image_path: Path, bbox: list[int], padding: int = PADDING) -> Image.Image:
    """
    Crop an image to the given bounding box with padding.

    bbox is [x1, y1, x2, y2] in image coordinates.
    Returns the cropped PIL Image.
    """
    img = Image.open(image_path)

    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(img.width, x2 + padding)
    y2 = min(img.height, y2 + padding)

    return img.crop((x1, y1, x2, y2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Crop IAM forms to handwritten region")
    parser.add_argument("--no-signature-filter", action="store_true",
                        help="Keep all handwritten lines including signatures")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute crop boxes without saving files")
    args = parser.parse_args()

    manifest = load_curated_manifest()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    skipped = 0

    for entry in manifest:
        image_name = entry["image"]  # e.g., "a04-039.png"
        image_path = CURATED_DIR / image_name
        xml_path = find_xml(image_name)

        if xml_path is None:
            print(f"SKIP {image_name}: no XML found")
            skipped += 1
            continue

        line_boxes, union_bbox = extract_handwritten_boxes(
            xml_path, filter_signatures=not args.no_signature_filter
        )

        if union_bbox is None:
            print(f"SKIP {image_name}: no handwritten boxes in XML")
            skipped += 1
            continue

        if args.dry_run:
            print(f"DRY-RUN {image_name}: crop={union_bbox}, lines={len(line_boxes)}")
            results.append({
                "image": image_name,
                "crop_bbox": union_bbox,
                "num_lines": len(line_boxes),
                "filtered_signatures": not args.no_signature_filter,
            })
            continue

        # Crop and save
        cropped = crop_image(image_path, union_bbox)
        out_path = OUTPUT_DIR / image_name
        cropped.save(out_path)

        results.append({
            "image": image_name,
            "cropped": str(out_path.relative_to(PROJECT_ROOT)),
            "crop_bbox": union_bbox,
            "original_size": [Image.open(image_path).width, Image.open(image_path).height],
            "cropped_size": [cropped.width, cropped.height],
            "num_lines": len(line_boxes),
            "filtered_signatures": not args.no_signature_filter,
        })
        print(f"OK {image_name}: {union_bbox} → {cropped.width}x{cropped.height} ({len(line_boxes)} lines)")

    # Write manifest
    with open(MANIFEST_OUT, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. {len(results)} cropped, {skipped} skipped.")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Manifest: {MANIFEST_OUT}")


if __name__ == "__main__":
    main()
