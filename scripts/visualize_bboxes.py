#!/usr/bin/env python3
"""Visualize OCR bboxes on cropped handwritten images."""

import json
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HANDWRITTEN_DIR = PROJECT_ROOT / "benchmark" / "test_dataset" / "handwritten"
GT_FILE = PROJECT_ROOT / "benchmark" / "test_dataset" / "ground_truth_handwritten.json"
BBOX_RESULTS = PROJECT_ROOT / "benchmark" / "results" / "florence2_large_bbox_reading_order.json"
VIZ_DIR = PROJECT_ROOT / "benchmark" / "visualizations"


def draw_boxes(img, blocks, color, label_prefix="", width=3):
    """Draw bounding boxes on image. Returns the draw object."""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()
    for i, b in enumerate(blocks):
        bbox = b.get("bbox", [])
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        # Ensure xyxy format
        if x2 < x1 or y2 < y1:
            x2, y2 = x1 + x2, y1 + y2
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        text = b.get("text", "")[:40]
        label = f"{label_prefix}{i+1}" if label_prefix else f"{i+1}"
        draw.text((x1 + 2, y1 - 16), label, fill=color, font=font)
    return draw


def visualize_one(image_name, output_path):
    """Draw GT (green) and predicted (red) bboxes on one image."""
    # Load image
    img_path = HANDWRITTEN_DIR / image_name
    if not img_path.exists():
        print(f"Image not found: {img_path}")
        return
    img = Image.open(img_path).convert("RGB")

    # Load GT blocks
    with open(GT_FILE) as f:
        gt_list = json.load(f)
    gt_entry = next((g for g in gt_list if g["image"] == image_name), None)
    gt_blocks = gt_entry["blocks"] if gt_entry else []

    # Load predicted blocks
    with open(BBOX_RESULTS) as f:
        res = json.load(f)
    pred_entry = next((r for r in res["images"] if r["image"] == image_name), None)
    pred_blocks = []
    if pred_entry and "sample_blocks" in pred_entry:
        pred_blocks = pred_entry["sample_blocks"]

    w, h = img.size

    # Draw GT in green (thick)
    draw_boxes(img, gt_blocks, color=(0, 200, 0), width=3)

    # Draw predicted in red (thin)
    draw_boxes(img, pred_blocks, color=(255, 40, 40), width=2)

    # Legend
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except (OSError, IOError):
        font = ImageFont.load_default()
    draw.text((10, 10), f"GREEN = Ground Truth ({len(gt_blocks)} blocks)", fill=(0, 200, 0), font=font)
    draw.text((10, 32), f"RED = Florence-2-large ({len(pred_blocks)} blocks)", fill=(255, 40, 40), font=font)

    # Stats
    if pred_entry:
        draw.text((10, 56), f"IoU={pred_entry.get('bbox_mean_iou', '?'):.3f}  τ={pred_entry.get('reading_order_tau', '?'):.3f}",
                  fill=(255, 255, 200), font=font)

    img.save(output_path)
    print(f"Saved: {output_path}")
    print(f"  GT blocks: {len(gt_blocks)}, Pred blocks: {len(pred_blocks)}")
    if pred_entry:
        print(f"  IoU: {pred_entry.get('bbox_mean_iou', '?'):.3f}, τ: {pred_entry.get('reading_order_tau', '?'):.3f}")


def visualize_all():
    """Generate visualizations for all images."""
    with open(BBOX_RESULTS) as f:
        res = json.load(f)

    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    for entry in res["images"]:
        visualize_one(entry["image"], VIZ_DIR / f"bbox_{entry['image']}")


def visualize_side_by_side(image_name, output_path):
    """Side-by-side: original + annotated with GT + annotated with pred."""
    img_path = HANDWRITTEN_DIR / image_name
    if not img_path.exists():
        print(f"Image not found: {img_path}")
        return
    original = Image.open(img_path).convert("RGB")
    gt_img = original.copy()
    pred_img = original.copy()

    with open(GT_FILE) as f:
        gt_list = json.load(f)
    gt_entry = next((g for g in gt_list if g["image"] == image_name), None)
    gt_blocks = gt_entry["blocks"] if gt_entry else []

    with open(BBOX_RESULTS) as f:
        res = json.load(f)
    pred_entry = next((r for r in res["images"] if r["image"] == image_name), None)
    pred_blocks = []
    if pred_entry and "sample_blocks" in pred_entry:
        pred_blocks = pred_entry["sample_blocks"]

    draw_boxes(gt_img, gt_blocks, color=(0, 200, 0), width=3, label_prefix="GT")
    draw_boxes(pred_img, pred_blocks, color=(255, 40, 40), width=2, label_prefix="")

    # Add titles
    for img, title, color in [(gt_img, f"Ground Truth ({len(gt_blocks)} blocks)", (0,200,0)),
                               (pred_img, f"Florence-2-large ({len(pred_blocks)} blocks)", (255,40,40))]:
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        except (OSError, IOError):
            font = ImageFont.load_default()
        draw.text((10, 10), title, fill=color, font=font)

    w, h = original.size
    combined = Image.new("RGB", (w * 3 + 20, h + 40), (30, 30, 30))
    combined.paste(original, (0, 0))
    combined.paste(gt_img, (w + 10, 0))
    combined.paste(pred_img, (w * 2 + 20, 0))

    # Bottom labels
    draw = ImageDraw.Draw(combined)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except (OSError, IOError):
        font = ImageFont.load_default()
    draw.text((10, h + 10), f"Original: {image_name}", fill=(200, 200, 200), font=font)
    if pred_entry:
        stats = f"IoU={pred_entry.get('bbox_mean_iou', '?'):.3f}  τ={pred_entry.get('reading_order_tau', '?'):.3f}  {pred_entry.get('latency_s', '?'):.2f}s"
        draw.text((w + 10, h + 10), stats, fill=(255, 255, 200), font=font)

    combined.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    # Default: side-by-side for a04-039 (the one in the README comparison)
    img = sys.argv[1] if len(sys.argv) > 1 else "a04-039.png"

    # Single image with overlay
    visualize_one(img, VIZ_DIR / f"bbox_overlay_{img}")

    # Side-by-side comparison
    visualize_side_by_side(img, VIZ_DIR / f"bbox_compare_{img}")

    print(f"\nTo generate all 25: python {__file__} --all")
    if "--all" in sys.argv:
        visualize_all()
