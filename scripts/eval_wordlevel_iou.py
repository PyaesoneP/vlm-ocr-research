#!/usr/bin/env python3
"""
Word-level IoU evaluation for OCR models that produce per-word bounding boxes.

Computes word IoU (greedy spatial matching) and reading order (Kendall's tau-b)
against ground_truth_wordlevel.json for any model registered in CANDIDATE_CONFIG.

Usage (from appropriate environment):
    source .venv/bin/activate
    python scripts/eval_wordlevel_iou.py easyocr
    python scripts/eval_wordlevel_iou.py tesseract
    python scripts/eval_wordlevel_iou.py doctr
    python scripts/eval_wordlevel_iou.py florence2_large   # conda activate florencetf
    python scripts/eval_wordlevel_iou.py paddleocr_vl       # source .venv_paddleocr/bin/activate
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.metrics import compute_cer_normalized, compute_wer_normalized

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HANDWRITTEN_DIR = PROJECT_ROOT / "benchmark" / "test_dataset" / "handwritten"
WORD_GT_PATH = PROJECT_ROOT / "benchmark" / "test_dataset" / "ground_truth_wordlevel.json"
RESULTS_DIR = PROJECT_ROOT / "benchmark" / "results"
VIZ_BASE = PROJECT_ROOT / "benchmark" / "visualizations"

# ---------------------------------------------------------------------------
# Model registry — same structure as eval_handwritten.py
# ---------------------------------------------------------------------------
CANDIDATE_CONFIG = {
    "easyocr": {
        "module": "candidates.baselines.eval",
        "fn_name": "_easyocr_inference",
        "candidate": "easyocr_wordlevel",
    },
    "tesseract": {
        "module": "candidates.baselines.eval",
        "fn_name": "_tesseract_inference",
        "candidate": "tesseract_wordlevel",
    },
    "doctr": {
        "module": "candidates.baselines.eval",
        "fn_name": "_doctr_inference",
        "candidate": "doctr_wordlevel",
    },
    "florence2_large": {
        "module": "candidates.florence2.eval",
        "fn_name": "inference_fn",
        "candidate": "florence2_large_wordlevel",
    },
    "paddleocr_vl": {
        "module": "candidates.paddleocr_vl.eval",
        "fn_name": "inference_fn",
        "candidate": "paddleocr_vl_wordlevel",
    },
}

# ---------------------------------------------------------------------------
# Word IoU (same algorithm as Qwen scripts)
# ---------------------------------------------------------------------------
def compute_word_iou(pred_blocks: list[dict], gt_words: list[dict]) -> dict:
    valid = [b for b in pred_blocks if b.get("bbox", [0, 0, 0, 0]) != [0, 0, 0, 0]]
    if not valid or not gt_words:
        return {
            "mean_iou": 0.0, "matched": 0, "gt_count": len(gt_words),
            "pred_count": len(valid),
            "recall": 0.0, "precision": 0.0,
        }

    matched_gt = set()
    ious = []

    for pred in valid:
        px1, py1, px2, py2 = pred["bbox"]
        best_iou = 0.0
        best_j = -1

        for j, gt in enumerate(gt_words):
            if j in matched_gt:
                continue
            gx1, gy1, gx2, gy2 = gt["bbox"]
            ix1 = max(px1, gx1)
            iy1 = max(py1, gy1)
            ix2 = min(px2, gx2)
            iy2 = min(py2, gy2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter == 0:
                continue
            area_pred = (px2 - px1) * (py2 - py1)
            area_gt = (gx2 - gx1) * (gy2 - gy1)
            iou = inter / (area_pred + area_gt - inter)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou > 0.05:
            ious.append(best_iou)
            matched_gt.add(best_j)

    return {
        "mean_iou": sum(ious) / len(ious) if ious else 0.0,
        "matched": len(ious),
        "gt_count": len(gt_words),
        "pred_count": len(valid),
        "recall": len(ious) / len(gt_words) if gt_words else 0.0,
        "precision": len(ious) / len(valid) if valid else 0.0,
    }


def compute_kendall_tau(pred_blocks: list[dict], gt_words: list[dict]) -> float:
    """Kendall's tau-b between predicted and GT reading order via spatial matching."""
    valid = [b for b in pred_blocks if b.get("bbox", [0, 0, 0, 0]) != [0, 0, 0, 0]]
    if len(valid) < 2 or len(gt_words) < 2:
        return 1.0

    # Match predicted words to GT words spatially
    matched_gt = set()
    gt_ranks = []
    pred_ranks = []

    # GT order is sequential (extracted top-to-bottom, left-to-right from XML)
    for pi, pred in enumerate(valid):
        px1, py1, px2, py2 = pred["bbox"]
        best_iou = 0.0
        best_j = -1
        for j, gt in enumerate(gt_words):
            if j in matched_gt:
                continue
            gx1, gy1, gx2, gy2 = gt["bbox"]
            ix1 = max(px1, gx1)
            iy1 = max(py1, gy1)
            ix2 = min(px2, gx2)
            iy2 = min(py2, gy2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter == 0:
                continue
            area_pred = (px2 - px1) * (py2 - py1)
            area_gt = (gx2 - gx1) * (gy2 - gy1)
            iou = inter / (area_pred + area_gt - inter)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou > 0.05:
            pred_ranks.append(pi)
            gt_ranks.append(best_j)
            matched_gt.add(best_j)

    if len(gt_ranks) < 2:
        return 1.0

    # Kendall's tau-b
    from scipy.stats import kendalltau
    tau, _ = kendalltau(pred_ranks, gt_ranks)
    return float(tau) if not np.isnan(tau) else 1.0


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def draw_bboxes(img_path: str, pred_blocks: list[dict], gt_words: list[dict], out_path: str):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    # GT in green (behind)
    for gw in gt_words:
        draw.rectangle(gw["bbox"], outline="#00FF00", width=1)
    # Pred in red (on top)
    for block in pred_blocks:
        b = block.get("bbox", [0, 0, 0, 0])
        if b != [0, 0, 0, 0]:
            draw.rectangle(b, outline="#FF0000", width=2)
    img.save(out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/eval_wordlevel_iou.py <model_name>")
        print("Available:", ", ".join(CANDIDATE_CONFIG.keys()))
        sys.exit(1)

    model_name = sys.argv[1]
    if model_name not in CANDIDATE_CONFIG:
        print(f"Unknown model '{model_name}'. Available: {', '.join(CANDIDATE_CONFIG.keys())}")
        sys.exit(1)

    cfg = CANDIDATE_CONFIG[model_name]
    candidate_id = cfg["candidate"]

    # Import inference function
    module = importlib.import_module(cfg["module"])
    inference_fn = getattr(module, cfg.get("fn_name", "inference_fn"))

    # Apply env vars if specified
    for key, val in cfg.get("env", {}).items():
        os.environ[key] = val

    # Load word-level GT
    gt_data = json.loads(WORD_GT_PATH.read_text())
    gt_by_name = {e["image"]: e for e in gt_data}

    # Image list
    images = sorted([
        str(p) for p in HANDWRITTEN_DIR.glob("*.png") if p.suffix.lower() == ".png"
    ])
    if not images:
        print(f"No images in {HANDWRITTEN_DIR}")
        sys.exit(1)

    # Output dirs
    viz_dir = VIZ_BASE / f"{candidate_id}"
    viz_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating {candidate_id} word-level IoU on {len(images)} images...")
    print()

    results = []
    cer_vals, wer_vals, iou_vals, tau_vals = [], [], [], []
    latencies = []

    for idx, img_path in enumerate(images):
        img_name = Path(img_path).name
        gt_entry = gt_by_name.get(img_name, {})
        gt_words = gt_entry.get("words", [])

        # Run inference
        t0 = time.perf_counter()
        try:
            output = inference_fn(img_path)
        except Exception as e:
            print(f"[{idx+1:2d}/{len(images)}] {img_name}: ERROR: {e}")
            continue
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        blocks = output.get("blocks", [])
        text = output.get("text", "")

        # Compute metrics
        gt_text = gt_entry.get("text", "")
        cer = compute_cer_normalized(text, gt_text) if gt_text else 0.0
        wer = compute_wer_normalized(text, gt_text) if gt_text else 0.0
        iou_info = compute_word_iou(blocks, gt_words)
        tau = compute_kendall_tau(blocks, gt_words)

        cer_vals.append(cer)
        wer_vals.append(wer)
        iou_vals.append(iou_info["mean_iou"])
        tau_vals.append(tau)

        # Visualization
        viz_path = viz_dir / f"{Path(img_name).stem}_bboxes.png"
        draw_bboxes(img_path, blocks, gt_words, str(viz_path))

        result = {
            "image": img_name,
            "text": text,
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "word_iou": round(iou_info["mean_iou"], 4),
            "iou_matched": iou_info["matched"],
            "iou_gt_words": iou_info["gt_count"],
            "iou_pred_words": iou_info["pred_count"],
            "iou_recall": round(iou_info["recall"], 4),
            "iou_precision": round(iou_info["precision"], 4),
            "kendall_tau": round(tau, 4),
            "latency_s": round(elapsed, 2),
        }
        results.append(result)

        print(f"[{idx+1:2d}/{len(images)}] {img_name}: "
              f"CER={cer:.4f} WER={wer:.4f} IoU={iou_info['mean_iou']:.3f} "
              f"τ={tau:.3f} words={iou_info['pred_count']} "
              f"latency={elapsed:.1f}s")

    # Aggregate
    avg_cer = sum(cer_vals) / len(cer_vals) if cer_vals else 0
    avg_wer = sum(wer_vals) / len(wer_vals) if wer_vals else 0
    avg_iou = sum(iou_vals) / len(iou_vals) if iou_vals else 0
    avg_tau = sum(tau_vals) / len(tau_vals) if tau_vals else 0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0

    output = {
        "candidate": candidate_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "images": results,
        "aggregate": {
            "cer": round(avg_cer, 4),
            "wer": round(avg_wer, 4),
            "word_iou": round(avg_iou, 4),
            "kendall_tau": round(avg_tau, 4),
            "latency_avg_s": round(avg_lat, 2),
            "num_images": len(results),
        },
    }

    out_path = RESULTS_DIR / f"{candidate_id}_handwritten.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved to {out_path}")
    print(f"Aggregate: CER={avg_cer:.4f} WER={avg_wer:.4f} IoU={avg_iou:.3f} "
          f"τ={avg_tau:.3f} Latency={avg_lat:.1f}s")
    print(f"Visualizations: {viz_dir}/")


if __name__ == "__main__":
    main()
