#!/usr/bin/env python3
"""
Generate ground_truth.json for the test dataset.

Parses IAM XML annotations (if available) to extract:
  - Full page transcription
  - Per-block bounding boxes and text
  - Reading order (top-to-bottom, left-to-right within same line)

If IAM XML is not available, generates a skeleton JSON with empty text fields
that can be manually populated.

Output: benchmark/test_dataset/ground_truth.json

Usage:
    python scripts/generate_ground_truth.py
    python scripts/generate_ground_truth.py --subset curated   # curated subset only
    python scripts/generate_ground_truth.py --all               # all images
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DATASET = PROJECT_ROOT / "benchmark" / "test_dataset"
IAM_XML_DIR = TEST_DATASET / "iam_xml" / "archive" / "xml"
CURATED_MANIFEST = TEST_DATASET / "curated_manifest.json"
OUTPUT_PATH = TEST_DATASET / "ground_truth.json"


# ---------------------------------------------------------------------------
# IAM XML parsing
# ---------------------------------------------------------------------------

def parse_iam_xml(xml_path: Path) -> dict[str, Any]:
    """
    Parse an IAM form XML file.

    Extracts line-level text and bounding boxes.
    Returns a dict with 'text', 'blocks', and 'reading_order'.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    blocks: list[dict[str, Any]] = []
    all_text: list[str] = []

    for line in root.iter("line"):
        text_attr = line.get("text", "")
        if not text_attr:
            continue

        # Compute bounding box from <cmp> children within <word> elements.
        # IAM XML stores character-level coordinates: <cmp x="..." y="..." width="..." height="..."/>
        xs, ys, x2s, y2s = [], [], [], []
        for cmp_elem in line.iter("cmp"):
            cx = cmp_elem.get("x")
            cy = cmp_elem.get("y")
            cw = cmp_elem.get("width")
            ch = cmp_elem.get("height")
            if cx is not None and cy is not None and cw is not None and ch is not None:
                x_val = int(cx)
                y_val = int(cy)
                w_val = int(cw)
                h_val = int(ch)
                xs.append(x_val)
                ys.append(y_val)
                x2s.append(x_val + w_val)
                y2s.append(y_val + h_val)

        if xs:
            bbox = [min(xs), min(ys), max(x2s), max(y2s)]
        else:
            # Fallback: use line baseline attributes if no <cmp> found
            bbox = [0, 0, 0, 0]

        blocks.append({
            "bbox": bbox,
            "text": text_attr,
            "confidence": 1.0,
        })
        all_text.append(text_attr)

    # Sort by y-position then x-position for reading order
    blocks_with_idx = sorted(enumerate(blocks), key=lambda t: (t[1]["bbox"][1], t[1]["bbox"][0]))
    sorted_blocks = []
    reading_order = [-1] * len(blocks)
    for rank, (orig_idx, block) in enumerate(blocks_with_idx):
        sorted_blocks.append(block)
        reading_order[orig_idx] = rank

    return {
        "text": "\n".join(all_text),
        "blocks": sorted_blocks,
        "reading_order": reading_order,
        "errors": [],  # IAM is clean copy-text, no intentional errors
    }


def find_xml_for_image(image_name: str) -> Optional[Path]:
    """
    Find the IAM XML file corresponding to an image.

    With Kaggle IAM form images, filenames match XML filenames directly:
      a01-000u.png  <->  a01-000u.xml
    """
    if not IAM_XML_DIR.exists():
        return None

    # Strip extension to get the form ID stem
    stem = Path(image_name).stem  # e.g., 'a01-000u'

    for xml_file in IAM_XML_DIR.rglob("*.xml"):
        if xml_file.stem == stem:
            return xml_file

    return None


# ---------------------------------------------------------------------------
# Skeleton generation (no XML available)
# ---------------------------------------------------------------------------

def generate_skeleton(image_name: str, image_path: Path) -> dict[str, Any]:
    """Generate a skeleton ground truth entry with empty fields."""
    return {
        "image": image_name,
        "text": "",
        "reading_order": [],
        "blocks": [],
        "errors": [],
        "_note": "Ground truth not yet annotated. Populate text, blocks, reading_order manually.",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate ground_truth.json")
    parser.add_argument("--subset", choices=["curated", "all"], default="all",
                        help="Which images to include (default: all)")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH,
                        help=f"Output path (default: {OUTPUT_PATH})")
    args = parser.parse_args()

    # Determine image list
    if args.subset == "curated":
        if not CURATED_MANIFEST.exists():
            print(f"ERROR: Curated manifest not found at {CURATED_MANIFEST}")
            print("Run scripts/curate_test_subset.py first.")
            sys.exit(1)
        curated = json.loads(CURATED_MANIFEST.read_text())
        images = [(e["image"], TEST_DATASET / e["image"]) for e in curated]
    else:
        images = []
        for pat in ("*.jpg", "*.jpeg", "*.png"):
            for p in sorted(TEST_DATASET.glob(pat)):
                if "ground_truth" not in p.name and "manifest" not in p.name:
                    # Skip curated dir symlinks (they point to same files)
                    if "curated/" not in str(p):
                        images.append((p.name, p))

    if not images:
        print("No images found. Run scripts/download_essay_samples.py first.")
        sys.exit(1)

    print(f"Generating ground truth for {len(images)} images ...")

    # Check for IAM XML
    xml_available = IAM_XML_DIR.exists() and any(IAM_XML_DIR.rglob("*.xml"))
    if xml_available:
        xml_files = list(IAM_XML_DIR.rglob("*.xml"))
        print(f"  Found {len(xml_files)} IAM XML files — parsing annotations ...")
    else:
        print(f"  No IAM XML found at {IAM_XML_DIR}")
        print(f"  Generating skeleton — manual annotation required.")
        print(f"  To get IAM XML: register at fki.tic.heia-fr.ch,")
        print(f"  download xml.tgz, extract to {IAM_XML_DIR}")

    # Build ground truth entries
    entries: list[dict[str, Any]] = []
    xml_hits = 0

    for img_name, img_path in images:
        if xml_available:
            xml_file = find_xml_for_image(img_name)
            if xml_file:
                try:
                    entry = parse_iam_xml(xml_file)
                    entry["image"] = img_name
                    entries.append(entry)
                    xml_hits += 1
                    continue
                except Exception as e:
                    print(f"  [warn] Failed to parse {xml_file.name}: {e}")

        # Fallback: skeleton
        entries.append(generate_skeleton(img_name, img_path))

    if xml_available:
        print(f"  Parsed {xml_hits}/{len(images)} images from XML")
    print(f"  {len(entries) - xml_hits} skeleton entries")

    # Write output
    args.output.write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    print(f"\nGround truth written to {args.output}")
    print(f"  Total entries: {len(entries)}")
    print(f"  Schema: image, text, reading_order, blocks[], errors[]")


if __name__ == "__main__":
    main()
