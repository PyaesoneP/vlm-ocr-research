#!/usr/bin/env python3
"""
Generate word-level ground truth from IAM XML annotations.

Extracts per-word bounding boxes from <word> -> <cmp> elements in the IAM XML,
applies the same crop-offset transform used for line-level GT.

Output: benchmark/test_dataset/ground_truth_wordlevel.json

Usage:
    python scripts/generate_wordlevel_gt.py
"""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DATASET = PROJECT_ROOT / "benchmark" / "test_dataset"
IAM_XML_DIR = TEST_DATASET / "iam_xml" / "archive" / "xml"
HANDWRITTEN_MANIFEST = TEST_DATASET / "handwritten_manifest.json"
LINE_GT_PATH = TEST_DATASET / "ground_truth_handwritten.json"
OUTPUT_PATH = TEST_DATASET / "ground_truth_wordlevel.json"

# Must match PADDING in crop_handwritten.py and generate_ground_truth.py
PADDING = 20


def extract_word_bboxes(xml_path: Path) -> list[dict]:
    """Extract per-word bboxes from IAM XML <handwritten-part> <line> <word> elements."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    handwritten = root.find("handwritten-part")
    if handwritten is None:
        return []

    form_height = int(root.get("height", 3542))
    words = []
    lines = handwritten.findall("line")

    for line in lines:
        for word_elem in line.findall("word"):
            text = word_elem.get("text", "")
            if not text:
                continue

            xs, ys, x2s, y2s = [], [], [], []
            for cmp_elem in word_elem.findall("cmp"):
                cx = cmp_elem.get("x")
                cy = cmp_elem.get("y")
                cw = cmp_elem.get("width")
                ch = cmp_elem.get("height")
                if cx is not None and cy is not None and cw is not None and ch is not None:
                    x_val, y_val = int(cx), int(cy)
                    w_val, h_val = int(cw), int(ch)
                    xs.append(x_val)
                    ys.append(y_val)
                    x2s.append(x_val + w_val)
                    y2s.append(y_val + h_val)

            if xs:
                # Signature filter: skip last line words with "Name:" in bottom 15%
                # (same heuristic as line-level GT)
                bbox = [min(xs), min(ys), max(x2s), max(y2s)]
                is_short = len(text.split()) <= 1
                in_bottom = bbox[1] > form_height * 0.85
                if "Name:" in text and is_short and in_bottom:
                    continue

                words.append({
                    "bbox": bbox,
                    "text": text,
                })

    return words


def apply_crop_offset(words: list[dict], crop_origin: tuple[int, int]) -> list[dict]:
    """Offset word bboxes from full-form coordinates to cropped image coordinates."""
    ox = max(0, crop_origin[0] - PADDING)
    oy = max(0, crop_origin[1] - PADDING)
    for w in words:
        b = w["bbox"]
        w["bbox"] = [b[0] - ox, b[1] - oy, b[2] - ox, b[3] - oy]
    return words


def find_xml_for_image(image_name: str) -> Path | None:
    """Find IAM XML matching an image stem."""
    stem = Path(image_name).stem
    for xml_file in IAM_XML_DIR.rglob("*.xml"):
        if xml_file.stem == stem:
            return xml_file
    return None


def main():
    if not IAM_XML_DIR.exists():
        print(f"ERROR: IAM XML directory not found at {IAM_XML_DIR}")
        sys.exit(1)

    if not HANDWRITTEN_MANIFEST.exists():
        print(f"ERROR: Handwritten manifest not found at {HANDWRITTEN_MANIFEST}")
        print("Run scripts/crop_handwritten.py first.")
        sys.exit(1)

    # Load crop offsets from handwritten manifest
    manifest = json.loads(HANDWRITTEN_MANIFEST.read_text())
    crop_offsets = {}
    for item in manifest:
        crop = item.get("crop_bbox", [0, 0, 0, 0])
        crop_offsets[item["image"]] = (crop[0], crop[1])

    # Also load line-level GT for reference (image list + text)
    line_gt = json.loads(LINE_GT_PATH.read_text()) if LINE_GT_PATH.exists() else []
    image_names = [e["image"] for e in line_gt]

    entries = []
    xml_hits = 0

    for img_name in image_names:
        xml_file = find_xml_for_image(img_name)
        if xml_file:
            try:
                words = extract_word_bboxes(xml_file)
                offset = crop_offsets.get(img_name)
                if offset:
                    words = apply_crop_offset(words, offset)
                entries.append({
                    "image": img_name,
                    "words": words,
                    "word_count": len(words),
                    "text": " ".join(w["text"] for w in words),
                })
                xml_hits += 1
            except Exception as e:
                print(f"  [warn] {img_name}: {e}")
                entries.append({"image": img_name, "words": [], "word_count": 0, "text": ""})
        else:
            print(f"  [warn] No XML for {img_name}")
            entries.append({"image": img_name, "words": [], "word_count": 0, "text": ""})

    OUTPUT_PATH.write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    print(f"Word-level ground truth written to {OUTPUT_PATH}")
    print(f"  Images: {len(entries)}")
    print(f"  Parsed from XML: {xml_hits}/{len(entries)}")
    if entries:
        total_words = sum(e["word_count"] for e in entries)
        print(f"  Total words: {total_words} (avg {total_words/len(entries):.0f}/image)")


if __name__ == "__main__":
    main()
