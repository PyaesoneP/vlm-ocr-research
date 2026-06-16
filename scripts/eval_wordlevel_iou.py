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
    python scripts/eval_wordlevel_iou.py locateanything     # source .venv_locateanything/bin/activate
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

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore
    HAS_TORCH = False

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
    "locateanything": {
        "module": "candidates.locateanything.eval",
        "fn_name": "inference_fn",
        "candidate": "locateanything_wordlevel",
        "notes": (
            "NVLabs Eagle Embodied / LocateAnything-3B. Stage 1 text-localization "
            "candidate; CER/WER are secondary and only computed when output labels "
            "look like actual transcribed words."
        ),
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


def reset_gpu_memory() -> None:
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_peak_vram_mb() -> int:
    if HAS_TORCH and torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated(0) // (1024 * 1024))
    return 0


def synchronize_gpu() -> None:
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate word-level OCR/localization metrics.")
    parser.add_argument("model_name", help=f"Model name: {', '.join(CANDIDATE_CONFIG.keys())}")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images for smoke tests.")
    args = parser.parse_args()

    if not args.model_name:
        print("Usage: python scripts/eval_wordlevel_iou.py <model_name>")
        print("Available:", ", ".join(CANDIDATE_CONFIG.keys()))
        sys.exit(1)

    model_name = args.model_name
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
    if args.limit is not None:
        if args.limit <= 0:
            print("--limit must be positive")
            sys.exit(1)
        images = images[:args.limit]
    if not images:
        print(f"No images in {HANDWRITTEN_DIR}")
        sys.exit(1)

    # Output dirs
    viz_dir = VIZ_BASE / f"{candidate_id}"
    viz_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating {candidate_id} word-level IoU on {len(images)} images...", flush=True)
    print(flush=True)

    results = []
    cer_vals, wer_vals, iou_vals, tau_vals = [], [], [], []
    latencies = []
    vram_peaks = []

    for idx, img_path in enumerate(images):
        img_name = Path(img_path).name
        gt_entry = gt_by_name.get(img_name, {})
        gt_words = gt_entry.get("words", [])

        # Run inference
        print(f"[{idx+1:2d}/{len(images)}] {img_name}: starting inference...", flush=True)
        reset_gpu_memory()
        synchronize_gpu()
        t0 = time.perf_counter()
        try:
            output = inference_fn(img_path)
        except Exception as e:
            print(f"[{idx+1:2d}/{len(images)}] {img_name}: ERROR: {e}", flush=True)
            continue
        synchronize_gpu()
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)
        vram_peak_mb = get_peak_vram_mb()
        vram_peaks.append(vram_peak_mb)

        blocks = output.get("blocks", [])
        text = output.get("text", "")
        text_is_transcription = output.get("text_is_transcription", True)

        # Compute metrics
        gt_text = gt_entry.get("text", "")
        if text_is_transcription and gt_text:
            cer = compute_cer_normalized(text, gt_text)
            wer = compute_wer_normalized(text, gt_text)
            cer_vals.append(cer)
            wer_vals.append(wer)
        else:
            cer = None
            wer = None
        iou_info = compute_word_iou(blocks, gt_words)
        tau = compute_kendall_tau(blocks, gt_words)

        iou_vals.append(iou_info["mean_iou"])
        tau_vals.append(tau)

        # Visualization
        viz_path = viz_dir / f"{Path(img_name).stem}_bboxes.png"
        draw_bboxes(img_path, blocks, gt_words, str(viz_path))

        result = {
            "image": img_name,
            "text": text,
            "text_is_transcription": bool(text_is_transcription),
            "cer": round(cer, 4) if cer is not None else None,
            "wer": round(wer, 4) if wer is not None else None,
            "word_iou": round(iou_info["mean_iou"], 4),
            "iou_matched": iou_info["matched"],
            "iou_gt_words": iou_info["gt_count"],
            "iou_pred_words": iou_info["pred_count"],
            "iou_recall": round(iou_info["recall"], 4),
            "iou_precision": round(iou_info["precision"], 4),
            "kendall_tau": round(tau, 4),
            "latency_s": round(elapsed, 2),
            "vram_peak_mb": vram_peak_mb,
        }
        if "raw_answer" in output:
            result["raw_answer"] = output["raw_answer"]
        results.append(result)

        cer_text = f"{cer:.4f}" if cer is not None else "n/a"
        wer_text = f"{wer:.4f}" if wer is not None else "n/a"
        print(f"[{idx+1:2d}/{len(images)}] {img_name}: "
              f"CER={cer_text} WER={wer_text} IoU={iou_info['mean_iou']:.3f} "
              f"τ={tau:.3f} words={iou_info['pred_count']} "
              f"latency={elapsed:.1f}s vram={vram_peak_mb}MB", flush=True)

    # Aggregate
    avg_cer = sum(cer_vals) / len(cer_vals) if cer_vals else None
    avg_wer = sum(wer_vals) / len(wer_vals) if wer_vals else None
    avg_iou = sum(iou_vals) / len(iou_vals) if iou_vals else 0
    avg_tau = sum(tau_vals) / len(tau_vals) if tau_vals else 0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    peak_vram = max(vram_peaks) if vram_peaks else 0

    output = {
        "candidate": candidate_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "notes": cfg.get("notes", ""),
        "images": results,
        "aggregate": {
            "cer": round(avg_cer, 4) if avg_cer is not None else None,
            "wer": round(avg_wer, 4) if avg_wer is not None else None,
            "word_iou": round(avg_iou, 4),
            "kendall_tau": round(avg_tau, 4),
            "latency_avg_s": round(avg_lat, 2),
            "vram_peak_mb": peak_vram,
            "num_images": len(results),
            "transcription_images": len(cer_vals),
        },
    }

    out_path = RESULTS_DIR / f"{candidate_id}_handwritten.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved to {out_path}", flush=True)
    cer_text = f"{avg_cer:.4f}" if avg_cer is not None else "n/a"
    wer_text = f"{avg_wer:.4f}" if avg_wer is not None else "n/a"
    print(f"Aggregate: CER={cer_text} WER={wer_text} IoU={avg_iou:.3f} "
          f"τ={avg_tau:.3f} Latency={avg_lat:.1f}s VRAM={peak_vram}MB", flush=True)
    print(f"Visualizations: {viz_dir}/", flush=True)


if __name__ == "__main__":
    main()
