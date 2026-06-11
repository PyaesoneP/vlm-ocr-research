#!/usr/bin/env python3
"""
Evaluate OCR candidates on reading order and bounding box quality.

Uses IAM XML ground truth blocks (line-level bboxes + reading order)
and compares against model outputs with structured bounding boxes.

Models that output bboxes:
  - Florence-2 (<OCR_WITH_REGION>): quad boxes → xyxy bboxes per text region
  - Nemotron OCR v2: bboxes per text region (xywh format)
  - SmolDocling: DocTags bboxes (xyxy format)

Models without bboxes: MonkeyOCR, GOT-OCR2.0 (excluded)

Usage:
    conda activate florencetf && python scripts/eval_bbox_reading_order.py florence2
    conda activate aiml && python scripts/eval_bbox_reading_order.py nemotron
    source .venv/bin/activate && python scripts/eval_bbox_reading_order.py smoldocling
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None  # PaddlePaddle-only envs
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.metrics import (
    compute_block_iou,
    compute_reading_order_from_blocks,
    _normalize_bbox_to_xyxy,
)

HANDWRITTEN_DIR = PROJECT_ROOT / "benchmark" / "test_dataset" / "handwritten"
GT_FILE = PROJECT_ROOT / "benchmark" / "test_dataset" / "ground_truth_handwritten.json"
RESULTS_DIR = PROJECT_ROOT / "benchmark" / "results"


# ---------------------------------------------------------------------------
# Florence-2 inference with OCR_WITH_REGION
# ---------------------------------------------------------------------------

def evaluate_florence2() -> None:
    from transformers import AutoProcessor, AutoModelForCausalLM

    MODEL_ID = "microsoft/Florence-2-large"
    print(f"Loading {MODEL_ID}...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.float16
    ).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    print(f"Loaded in {time.perf_counter()-t0:.1f}s")

    with open(GT_FILE) as f:
        gt_list = json.load(f)
    gt_map = {g["image"]: g for g in gt_list}

    images = sorted(p for p in HANDWRITTEN_DIR.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    print(f"Running on {len(images)} images...\n")

    prompt = "<OCR_WITH_REGION>"
    all_results = []
    total_iou = 0.0
    total_tau = 0.0
    total_time = 0.0
    iou_count = 0
    tau_count = 0

    for i, img_path in enumerate(images):
        name = img_path.name
        gt_entry = gt_map.get(name)
        if not gt_entry:
            print(f"  [{i+1:2d}] {name}: SKIP (no GT)")
            continue

        image = Image.open(img_path).convert("RGB")

        torch.cuda.synchronize()
        t_start = time.perf_counter()

        inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda", torch.float16)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
        )
        raw = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(raw, task=prompt, image_size=image.size)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t_start

        # Extract blocks from OCR_WITH_REGION output
        ocr_data = parsed.get("<OCR_WITH_REGION>", {})
        quad_boxes = ocr_data.get("quad_boxes", [])
        labels = ocr_data.get("labels", [])
        pred_blocks = []
        for quad, label in zip(quad_boxes, labels):
            # Convert 4-corner quad [x1,y1,x2,y2,x3,y3,x4,y4] to xyxy
            xs = quad[::2]
            ys = quad[1::2]
            pred_blocks.append({
                "bbox": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                "text": label,
            })

        # --- Compute bbox IoU ---
        gt_blocks = gt_entry.get("blocks", [])
        iou_result = compute_block_iou(pred_blocks, gt_blocks)

        # --- Compute reading order tau ---
        gt_order = gt_entry.get("reading_order", [])
        tau_result = compute_reading_order_from_blocks(pred_blocks, gt_blocks, gt_order)

        total_time += elapsed
        if iou_result["matched"] > 0:
            total_iou += iou_result["mean_iou"]
            iou_count += 1
        if tau_result["matched_blocks"] >= 2:
            total_tau += tau_result["kendall_tau"]
            tau_count += 1

        all_results.append({
            "image": name,
            "num_pred_blocks": len(pred_blocks),
            "num_gt_blocks": len(gt_blocks),
            "bbox_mean_iou": round(iou_result["mean_iou"], 4),
            "bbox_recall": round(iou_result["recall"], 4),
            "bbox_precision": round(iou_result["precision"], 4),
            "reading_order_tau": round(tau_result["kendall_tau"], 4),
            "reading_order_matched": tau_result["matched_blocks"],
            "latency_s": round(elapsed, 2),
            # Store first 3 blocks for inspection
            "sample_blocks": [
                {"bbox": b["bbox"], "text": b["text"][:80]}
                for b in pred_blocks[:3]
            ],
        })

        print(f"  [{i+1:2d}/{len(images)}] {name:20s}  "
              f"IoU={iou_result['mean_iou']:.3f}  τ={tau_result['kendall_tau']:.3f}  "
              f"{elapsed:.2f}s  ({iou_result['matched']}/{iou_result['total_gt']} matched)")

    n = len(images)
    print(f"\n{'='*60}")
    print(f"FLORENCE-2-LARGE BBOX + READING ORDER RESULTS")
    print(f"  Mean Bbox IoU:     {total_iou/iou_count:.3f}  ({iou_count}/{n} images with matches)")
    print(f"  Mean Kendall's τ:  {total_tau/tau_count:.3f}  ({tau_count}/{n} images)")
    print(f"  Mean Latency:      {total_time/n:.2f}s")
    print(f"{'='*60}")

    output = {
        "candidate": "florence2_large_bbox",
        "model": MODEL_ID,
        "summary": {
            "mean_bbox_iou": round(total_iou / iou_count, 4) if iou_count else 0,
            "mean_kendall_tau": round(total_tau / tau_count, 4) if tau_count else 0,
            "mean_latency_s": round(total_time / n, 2),
            "images_with_bbox_matches": iou_count,
            "images_with_tau": tau_count,
            "total_images": n,
        },
        "images": all_results,
    }

    out_path = RESULTS_DIR / "florence2_large_bbox_reading_order.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


# ---------------------------------------------------------------------------
# Generic evaluation runner (for Nemotron, SmolDocling)
# ---------------------------------------------------------------------------

def run_generic_eval(candidate_name: str, inference_fn, bbox_normalizer=None, num_images: int = 0):
    """
    Generic evaluation loop for models with an inference_fn(image_path) -> dict
    that returns {"blocks": [...], "text": "...", "stage1_latency": ...}.

    bbox_normalizer: optional function to convert model's bbox format to xyxy.
    num_images: if >0, limit to first N images.
    """
    with open(GT_FILE) as f:
        gt_list = json.load(f)
    gt_map = {g["image"]: g for g in gt_list}

    images = sorted(p for p in HANDWRITTEN_DIR.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if num_images > 0:
        images = images[:num_images]
    print(f"Running on {len(images)} images...\n")

    all_results = []
    total_iou = 0.0
    total_tau = 0.0
    total_time = 0.0
    iou_count = 0
    tau_count = 0

    for i, img_path in enumerate(images):
        name = img_path.name
        gt_entry = gt_map.get(name)
        if not gt_entry:
            print(f"  [{i+1:2d}/{len(images)}] {name:20s}  SKIP (no GT)")
            sys.stdout.flush()
            continue

        result = inference_fn(str(img_path))
        elapsed = result.get("stage1_latency", 0)

        pred_blocks = result.get("blocks", [])
        # Normalize bbox format if needed
        if bbox_normalizer and pred_blocks:
            pred_blocks = [
                {**b, "bbox": bbox_normalizer(b.get("bbox", []))}
                for b in pred_blocks
            ]

        gt_blocks = gt_entry.get("blocks", [])
        iou_result = compute_block_iou(pred_blocks, gt_blocks)
        gt_order = gt_entry.get("reading_order", [])
        tau_result = compute_reading_order_from_blocks(pred_blocks, gt_blocks, gt_order)

        total_time += elapsed
        if iou_result["matched"] > 0:
            total_iou += iou_result["mean_iou"]
            iou_count += 1
        if tau_result["matched_blocks"] >= 2:
            total_tau += tau_result["kendall_tau"]
            tau_count += 1

        all_results.append({
            "image": name,
            "num_pred_blocks": len(pred_blocks),
            "num_gt_blocks": len(gt_blocks),
            "bbox_mean_iou": round(iou_result["mean_iou"], 4),
            "bbox_recall": round(iou_result["recall"], 4),
            "bbox_precision": round(iou_result["precision"], 4),
            "reading_order_tau": round(tau_result["kendall_tau"], 4),
            "reading_order_matched": tau_result["matched_blocks"],
            "latency_s": round(elapsed, 2),
            "sample_blocks": [
                {"bbox": b["bbox"], "text": b.get("text", "")[:80]}
                for b in pred_blocks[:3]
            ],
        })

        print(f"  [{i+1:2d}/{len(images)}] {name:20s}  "
              f"IoU={iou_result['mean_iou']:.3f}  τ={tau_result['kendall_tau']:.3f}  "
              f"{elapsed:.2f}s  ({iou_result['matched']}/{iou_result['total_gt']} matched)")
        sys.stdout.flush()

    n = len(images)
    print(f"\n{'='*60}")
    print(f"{candidate_name.upper()} BBOX + READING ORDER RESULTS")
    print(f"  Mean Bbox IoU:     {total_iou/iou_count:.3f}  ({iou_count}/{n} images)")
    print(f"  Mean Kendall's τ:  {total_tau/tau_count:.3f}  ({tau_count}/{n} images)")
    print(f"  Mean Latency:      {total_time/n:.2f}s")
    print(f"{'='*60}")

    output = {
        "candidate": f"{candidate_name}_bbox",
        "summary": {
            "mean_bbox_iou": round(total_iou / iou_count, 4) if iou_count else 0,
            "mean_kendall_tau": round(total_tau / tau_count, 4) if tau_count else 0,
            "mean_latency_s": round(total_time / n, 2),
            "images_with_bbox_matches": iou_count,
            "images_with_tau": tau_count,
            "total_images": n,
        },
        "images": all_results,
    }

    out_path = RESULTS_DIR / f"{candidate_name}_bbox_reading_order.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


# ---------------------------------------------------------------------------
# Nemotron (uses conda aiml env)
# ---------------------------------------------------------------------------

def _nemotron_bbox_to_xyxy(bbox: list) -> list:
    """Convert Nemotron xywh to xyxy format."""
    if len(bbox) != 4:
        return [0, 0, 0, 0]
    # Heuristic: Nemotron uses [x, y, w, h]
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]


def evaluate_nemotron() -> None:
    try:
        from candidates.nemotron_ocr.eval import inference_fn as nemotron_infer
    except ImportError:
        print("ERROR: Nemotron requires conda activate aiml")
        sys.exit(1)
    run_generic_eval("nemotron_ocr_v2", nemotron_infer, bbox_normalizer=_nemotron_bbox_to_xyxy)


# ---------------------------------------------------------------------------
# SmolDocling (uses .venv)
# ---------------------------------------------------------------------------

def evaluate_smoldocling() -> None:
    try:
        from candidates.smoldocling.eval import inference_fn as smoldocling_infer
    except ImportError:
        print("ERROR: SmolDocling requires source .venv/bin/activate")
        sys.exit(1)
    # SmolDocling blocks are already xyxy
    run_generic_eval("smoldocling", smoldocling_infer)


# ---------------------------------------------------------------------------
# DocLayoutYOLO
# ---------------------------------------------------------------------------

def evaluate_doclayout_yolo() -> None:
    """
    Use DocLayoutYOLO for layout detection → actual bounding boxes.

    DocLayoutYOLO is an ONNX/PyTorch-based document layout detection model
    that outputs REGION-level bboxes (not line-level).  It was designed for
    printed documents; on handwritten IAM forms it detects the main content
    block as a single region.

    Setup (before running):
        source .venv/bin/activate
        pip install doclayout_yolo==0.0.2b1
        # Model auto-downloaded from juliozhao/DocLayout-YOLO-DocStructBench
    """
    import os
    from doclayout_yolo import YOLOv10

    # --- DocLayoutYOLO inference function ---
    def doclayout_infer(image_path: str) -> dict:
        import time
        from PIL import Image

        if not hasattr(doclayout_infer, "_model"):
            # Auto-download and cache model
            from huggingface_hub import snapshot_download
            local = snapshot_download(
                "juliozhao/DocLayout-YOLO-DocStructBench",
            )
            weight_file = os.path.join(
                local, "doclayout_yolo_docstructbench_imgsz1024.pt"
            )
            doclayout_infer._model = YOLOv10(weight_file)
            print(f"[doclayout_yolo] Model loaded from {weight_file}")

        model = doclayout_infer._model
        img = Image.open(image_path).convert("RGB")

        t0 = time.perf_counter()
        results = model.predict(img, imgsz=1024, conf=0.15, iou=0.3, verbose=False)
        elapsed = time.perf_counter() - t0

        blocks = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                cls_name = model.names.get(cls_id, str(cls_id))
                blocks.append({
                    "bbox": [float(x) for x in xyxy],
                    "text": f"{cls_name} ({conf:.2f})",
                    "confidence": conf,
                    "class": cls_name,
                })

        # Sort blocks by y-position (top to bottom)
        blocks.sort(key=lambda b: b["bbox"][1])

        return {
            "blocks": blocks,
            "text": "",
            "stage1_latency": elapsed,
            "bbox_source": "doclayout_yolo",
        }

    run_generic_eval("doclayout_yolo", doclayout_infer)


# ---------------------------------------------------------------------------
# PaddleOCR-VL (uses .venv_paddleocr)
# ---------------------------------------------------------------------------

def evaluate_paddleocr(num_images: int = 0) -> None:
    """
    Evaluate PaddleOCR-VL-1.6 bbox and reading order quality.

    PaddleOCR-VL outputs blocks via parsing_res_list with block_bbox
    and block_content.  Bboxes are already in xyxy format.

    Setup: source .venv_paddleocr/bin/activate
    """
    try:
        from candidates.paddleocr_vl.eval import inference_fn as paddleocr_infer
    except ImportError:
        print("ERROR: PaddleOCR-VL requires source .venv_paddleocr/bin/activate")
        sys.exit(1)
    # PaddleOCR-VL blocks are already xyxy
    run_generic_eval("paddleocr_vl", paddleocr_infer, num_images=num_images)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate OCR bbox + reading order")
    parser.add_argument("candidate", choices=["florence2", "nemotron", "smoldocling", "doclayout_yolo", "paddleocr"],
                        help="Which candidate to evaluate")
    parser.add_argument("--num-images", type=int, default=0,
                        help="Limit to first N images (0 = all)")
    args = parser.parse_args()

    if args.candidate == "florence2":
        evaluate_florence2()
    elif args.candidate == "nemotron":
        evaluate_nemotron()
    elif args.candidate == "smoldocling":
        evaluate_smoldocling()
    elif args.candidate == "doclayout_yolo":
        evaluate_doclayout_yolo()
    elif args.candidate == "paddleocr":
        evaluate_paddleocr(args.num_images)
