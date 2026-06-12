#!/usr/bin/env python3
"""Generate PaddleOCR-VL visualizations from saved results — runs on host, no GPU."""
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw

PROJECT = Path(__file__).resolve().parents[1]
RESULT = PROJECT / "benchmark" / "results" / "paddleocr_vl_handwritten.json"
GT = PROJECT / "benchmark" / "test_dataset" / "ground_truth_handwritten.json"
HW_DIR = PROJECT / "benchmark" / "test_dataset" / "handwritten"
VIZ_DIR = PROJECT / "benchmark" / "visualizations" / "paddleocr"

def draw_boxes(img, blocks, color):
    out = img.copy()
    d = ImageDraw.Draw(out)
    for i, b in enumerate(blocks):
        bb = b.get("bbox", [])
        if len(bb) != 4: continue
        x1, y1, x2, y2 = [int(v) for v in bb]
        if x2 < x1 or y2 < y1: x2, y2 = x1 + x2, y1 + y2
        d.rectangle([x1, y1, x2, y2], outline=color, width=2)
        d.text((x1 + 2, y1 + 2), str(i + 1), fill=color)
    return out

def main():
    result = json.load(open(RESULT))
    gt_data = {e["image"]: e for e in json.load(open(GT))}
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    hw_images = sorted(p.name for p in HW_DIR.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    samples = result.get("sample_outputs", [])

    for i, sample in enumerate(samples[:5]):
        if i >= len(hw_images): break
        name = hw_images[i]
        img_path = HW_DIR / name
        if not img_path.exists(): continue

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        gt_entry = gt_data.get(name, {})
        gt_blocks = gt_entry.get("blocks", [])
        pred_blocks = sample.get("blocks", [])

        gt_panel = draw_boxes(img, gt_blocks, "lime")
        pred_panel = draw_boxes(img, pred_blocks, "red")

        combined = Image.new("RGB", (w * 3 + 16, h + 40), (30, 30, 30))
        combined.paste(img, (0, 0))
        combined.paste(gt_panel, (w + 8, 0))
        combined.paste(pred_panel, (w * 2 + 16, 0))
        d = ImageDraw.Draw(combined)
        d.text((10, h + 10), f"Original: {name}", fill=(200, 200, 200))
        d.text((w + 18, h + 10), f"GT ({len(gt_blocks)} lines)", fill=(0, 220, 0))
        d.text((w * 2 + 26, h + 10), f"PaddleOCR-VL ({len(pred_blocks)} blocks)", fill=(255, 60, 60))

        out_path = VIZ_DIR / f"compare_{name}"
        combined.save(out_path)
        print(f"  Saved: {out_path}")

    print(f"Done. {min(len(samples), 5)} images in {VIZ_DIR}")

if __name__ == "__main__":
    main()
