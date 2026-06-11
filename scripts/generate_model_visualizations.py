#!/usr/bin/env python3
"""
Generate per-model visualization folders for GitHub push.

Each model gets 3-5 representative images showing:
  Original | Ground Truth (green bboxes) | Model prediction (red bboxes)

Usage:
    conda activate florencetf && python scripts/generate_model_visualizations.py florence2
    conda activate aiml && python scripts/generate_model_visualizations.py nemotron
    source .venv/bin/activate && python scripts/generate_model_visualizations.py smoldocling
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

HANDWRITTEN_DIR = PROJECT_ROOT / "benchmark" / "test_dataset" / "handwritten"
GT_FILE = PROJECT_ROOT / "benchmark" / "test_dataset" / "ground_truth_handwritten.json"
VIZ_BASE = PROJECT_ROOT / "benchmark" / "visualizations"
RESULTS_DIR = PROJECT_ROOT / "benchmark" / "results"

# Which images to visualize: best CER, worst CER, median CER, plus 2 more
def select_representative_images(results_path: str, n: int = 5) -> list[dict]:
    """Select best, worst, median, and random images for visualization.
    Returns list of dicts with at least 'image' key.
    Falls back to hardcoded representative images if no per-image data."""
    try:
        with open(results_path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    images = data.get("images", [])
    if not images:
        # Harness format: no per-image data. Return empty.
        return []

    sorted_imgs = sorted(images, key=lambda x: x.get("cer", 1.0))
    selected = []
    if sorted_imgs:
        selected.append(sorted_imgs[0])
    if len(sorted_imgs) > 1:
        selected.append(sorted_imgs[-1])
    if len(sorted_imgs) > 2:
        selected.append(sorted_imgs[len(sorted_imgs) // 2])
    if len(sorted_imgs) > 5:
        selected.append(sorted_imgs[len(sorted_imgs) // 4])
        selected.append(sorted_imgs[3 * len(sorted_imgs) // 4])
    return selected


FALLBACK_IMAGES = ["r06-106.png", "a04-039.png", "f01-058.png", "b06-100.png", "h06-082.png"]


def get_image_list(results_path: str, use_fallback: bool = True) -> list[str]:
    """Get list of image names to visualize, from results or fallback."""
    representatives = select_representative_images(results_path)
    if representatives:
        return [r["image"] for r in representatives]
    if use_fallback:
        return FALLBACK_IMAGES.copy()
    return []


def draw_boxes(img: Image.Image, blocks: list[dict], color: str, label_idx: bool = True) -> Image.Image:
    """Draw bounding boxes on an image copy."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for i, b in enumerate(blocks):
        bbox = b.get("bbox", [])
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        if x2 < x1 or y2 < y1:
            x2, y2 = x1 + x2, y1 + y2
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        if label_idx:
            draw.text((x1 + 2, y1 + 2), str(i + 1), fill=color)
    return out


def generate_florence2_images():
    """Generate Florence-2 visualizations (requires florencetf env)."""
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM

    MODEL_ID = "microsoft/Florence-2-large"
    out_dir = VIZ_BASE / "florence2"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.float16
    ).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Load GT
    with open(GT_FILE) as f:
        gt_data = json.load(f)
    gt_map = {g["image"]: g for g in gt_data}

    # Find CER results to pick representative images
    cer_path = RESULTS_DIR / "florence2_large_handwritten.json"
    image_names = get_image_list(str(cer_path)) if cer_path.exists() else FALLBACK_IMAGES

    prompt = "<OCR_WITH_REGION>"
    for img_name in image_names:
        img_path = HANDWRITTEN_DIR / img_name
        if not img_path.exists():
            continue

        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        gt_entry = gt_map.get(img_name)
        gt_blocks = gt_entry["blocks"] if gt_entry else []

        # Run inference
        inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda", torch.float16)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
        )
        raw = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(raw, task=prompt, image_size=image.size)
        ocr_data = parsed.get("<OCR_WITH_REGION>", {})
        pred_blocks = []
        for quad, label in zip(ocr_data.get("quad_boxes", []), ocr_data.get("labels", [])):
            xs = quad[::2]; ys = quad[1::2]
            pred_blocks.append({
                "bbox": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                "text": label,
            })

        # Generate panels
        gt_img = draw_boxes(image, gt_blocks, "lime")
        pred_img = draw_boxes(image, pred_blocks, "red")

        # Side-by-side
        combined = Image.new("RGB", (w * 3 + 16, h + 40), (30, 30, 30))
        combined.paste(image, (0, 0))
        combined.paste(gt_img, (w + 8, 0))
        combined.paste(pred_img, (w * 2 + 16, 0))

        draw = ImageDraw.Draw(combined)
        draw.text((10, h + 10), f"Original: {img_name}", fill=(200, 200, 200))
        draw.text((w + 18, h + 10), f"Ground Truth ({len(gt_blocks)} lines)", fill=(0, 220, 0))
        draw.text((w * 2 + 26, h + 10), f"Florence-2-large ({len(pred_blocks)} regions)", fill=(255, 60, 60))

        out_path = out_dir / f"compare_{img_name}"
        combined.save(out_path)
        print(f"  Saved: {out_path}")

    print(f"Florence-2: {len(image_names)} images saved to {out_dir}/")


def generate_nemotron_images():
    """Generate Nemotron visualizations (requires aiml env)."""
    out_dir = VIZ_BASE / "nemotron"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from candidates.nemotron_ocr.eval import inference_fn as nemotron_infer
    except ImportError:
        print("ERROR: Nemotron requires `conda activate aiml`")
        return

    with open(GT_FILE) as f:
        gt_data = json.load(f)
    gt_map = {g["image"]: g for g in gt_data}

    image_names = get_image_list(str(RESULTS_DIR / "nemotron_ocr_v2_handwritten.json"))

    for img_name in image_names:
        img_path = HANDWRITTEN_DIR / img_name
        if not img_path.exists():
            continue

        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        gt_entry = gt_map.get(img_name)
        gt_blocks = gt_entry["blocks"] if gt_entry else []

        # Run inference
        try:
            result = nemotron_infer(str(img_path))
            pred_blocks = result.get("blocks", [])
        except Exception as e:
            print(f"  ERROR on {img_name}: {e}")
            continue

        gt_img = draw_boxes(image, gt_blocks, "lime")
        pred_img = draw_boxes(image, pred_blocks, "red")

        combined = Image.new("RGB", (w * 3 + 16, h + 40), (30, 30, 30))
        combined.paste(image, (0, 0))
        combined.paste(gt_img, (w + 8, 0))
        combined.paste(pred_img, (w * 2 + 16, 0))

        draw = ImageDraw.Draw(combined)
        draw.text((10, h + 10), f"Original: {img_name}", fill=(200, 200, 200))
        draw.text((w + 18, h + 10), f"Ground Truth ({len(gt_blocks)} lines)", fill=(0, 220, 0))
        draw.text((w * 2 + 26, h + 10), f"Nemotron OCR v2 ({len(pred_blocks)} blocks)", fill=(255, 60, 60))

        out_path = out_dir / f"compare_{img_name}"
        combined.save(out_path)
        print(f"  Saved: {out_path}")

    print(f"Nemotron: {len(image_names)} images saved to {out_dir}/")


def generate_smoldocling_images():
    """Generate SmolDocling visualizations (requires .venv)."""
    out_dir = VIZ_BASE / "smoldocling"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from candidates.smoldocling.eval import inference_fn as smol_infer
    except ImportError:
        print("ERROR: SmolDocling requires `source .venv/bin/activate`")
        return

    with open(GT_FILE) as f:
        gt_data = json.load(f)
    gt_map = {g["image"]: g for g in gt_data}

    image_names = get_image_list(str(RESULTS_DIR / "smoldocling_handwritten.json"))

    for img_name in image_names:
        img_path = HANDWRITTEN_DIR / img_name
        if not img_path.exists():
            continue

        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        gt_entry = gt_map.get(img_name)
        gt_blocks = gt_entry["blocks"] if gt_entry else []

        try:
            result = smol_infer(str(img_path))
            pred_blocks = result.get("blocks", [])
        except Exception as e:
            print(f"  ERROR on {img_name}: {e}")
            continue

        gt_img = draw_boxes(image, gt_blocks, "lime")
        pred_img = draw_boxes(image, pred_blocks, "red")

        combined = Image.new("RGB", (w * 3 + 16, h + 40), (30, 30, 30))
        combined.paste(image, (0, 0))
        combined.paste(gt_img, (w + 8, 0))
        combined.paste(pred_img, (w * 2 + 16, 0))

        draw = ImageDraw.Draw(combined)
        draw.text((10, h + 10), f"Original: {img_name}", fill=(200, 200, 200))
        draw.text((w + 18, h + 10), f"Ground Truth ({len(gt_blocks)} lines)", fill=(0, 220, 0))
        draw.text((w * 2 + 26, h + 10), f"SmolDocling ({len(pred_blocks)} blocks)", fill=(255, 60, 60))

        out_path = out_dir / f"compare_{img_name}"
        combined.save(out_path)
        print(f"  Saved: {out_path}")

    print(f"SmolDocling: {len(image_names)} images saved to {out_dir}/")


def generate_gt_reference():
    """Generate GT-only reference images."""
    out_dir = VIZ_BASE / "ground_truth"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(GT_FILE) as f:
        gt_data = json.load(f)

    # Best, worst, median CER images from Florence-2 results
    cer_path = RESULTS_DIR / "florence2_large_handwritten.json"
    if cer_path.exists():
        representatives = select_representative_images(str(cer_path))
        image_names = [r["image"] for r in representatives]
    else:
        image_names = ["r06-106.png", "a04-039.png", "f01-058.png"]

    for img_name in image_names:
        img_path = HANDWRITTEN_DIR / img_name
        if not img_path.exists():
            continue

        image = Image.open(img_path).convert("RGB")
        gt_entry = next((g for g in gt_data if g["image"] == img_name), None)
        if not gt_entry:
            continue

        gt_img = draw_boxes(image, gt_entry["blocks"], "lime")
        draw = ImageDraw.Draw(gt_img)
        draw.text((10, 10), "Ground Truth (" + str(len(gt_entry['blocks'])) + " lines)", fill=(0, 220, 0))

        out_path = out_dir / f"gt_{img_name}"
        gt_img.save(out_path)
        print(f"  Saved: {out_path}")

    print(f"GT reference: {len(image_names)} images saved to {out_dir}/")


def generate_monkeyocr_images():
    """Generate MonkeyOCR visualizations using DocLayoutYOLO bboxes (.venv)."""
    import os
    from doclayout_yolo import YOLOv10

    out_dir = VIZ_BASE / "monkeyocr"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load DocLayoutYOLO model
    from huggingface_hub import snapshot_download
    local = snapshot_download("juliozhao/DocLayout-YOLO-DocStructBench")
    weight_file = os.path.join(local, "doclayout_yolo_docstructbench_imgsz1024.pt")
    print(f"Loading DocLayoutYOLO from {weight_file}...")
    model = YOLOv10(weight_file)

    with open(GT_FILE) as f:
        gt_data = json.load(f)
    gt_map = {g["image"]: g for g in gt_data}

    image_names = FALLBACK_IMAGES

    for img_name in image_names:
        img_path = HANDWRITTEN_DIR / img_name
        if not img_path.exists():
            continue

        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        gt_entry = gt_map.get(img_name)
        gt_blocks = gt_entry["blocks"] if gt_entry else []

        # Run DocLayoutYOLO inference
        results = model.predict(image, imgsz=1024, conf=0.15, iou=0.3, verbose=False)
        pred_blocks = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i])
                cls_name = model.names.get(int(boxes.cls[i]), "unknown")
                pred_blocks.append({
                    "bbox": [float(x) for x in xyxy],
                    "text": f"{cls_name} ({conf:.2f})",
                })
        # Sort by y for reading order
        pred_blocks.sort(key=lambda b: b["bbox"][1])

        gt_img = draw_boxes(image, gt_blocks, "lime")
        pred_img = draw_boxes(image, pred_blocks, "red")

        combined = Image.new("RGB", (w * 3 + 16, h + 40), (30, 30, 30))
        combined.paste(image, (0, 0))
        combined.paste(gt_img, (w + 8, 0))
        combined.paste(pred_img, (w * 2 + 16, 0))

        draw_im = ImageDraw.Draw(combined)
        draw_im.text((10, h + 10), f"Original: {img_name}", fill=(200, 200, 200))
        draw_im.text((w + 18, h + 10), f"Ground Truth ({len(gt_blocks)} lines)", fill=(0, 220, 0))
        draw_im.text((w * 2 + 26, h + 10), f"MonkeyOCR DocLayoutYOLO ({len(pred_blocks)} regions)", fill=(255, 60, 60))

        out_path = out_dir / f"compare_{img_name}"
        combined.save(out_path)
        print(f"  Saved: {out_path}")

    print(f"MonkeyOCR: {len(image_names)} images saved to {out_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate per-model visualization folders")
    parser.add_argument("candidate", nargs="?", default="florence2",
                        choices=["florence2", "nemotron", "smoldocling", "monkeyocr", "gt", "all"])
    args = parser.parse_args()

    if args.candidate == "florence2":
        generate_florence2_images()
    elif args.candidate == "nemotron":
        generate_nemotron_images()
    elif args.candidate == "smoldocling":
        generate_smoldocling_images()
    elif args.candidate == "monkeyocr":
        generate_monkeyocr_images()
    elif args.candidate == "gt":
        generate_gt_reference()
    elif args.candidate == "all":
        generate_gt_reference()
        generate_florence2_images()
        print("\n--- Switch to aiml env for Nemotron ---")
        print("conda activate aiml && python scripts/generate_model_visualizations.py nemotron")
        print("\n--- Switch to .venv for SmolDocling ---")
        print("source .venv/bin/activate && python scripts/generate_model_visualizations.py smoldocling")
        print("\n--- MonkeyOCR (uses .venv) ---")
        print("source .venv/bin/activate && python scripts/generate_model_visualizations.py monkeyocr")
