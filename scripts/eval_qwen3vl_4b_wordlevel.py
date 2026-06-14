#!/usr/bin/env python3
"""
Qwen3-VL-4B word-level evaluation with IoU and reading order.

Runs local 4B model on all 25 cropped handwriting images with word-level prompt.
Computes: CER, WER, word-to-word IoU, Kendall's tau reading order.
Generates bbox overlay visualizations (red=predicted, green=ground truth).

Usage:
    source .venv/bin/activate
    python scripts/eval_qwen3vl_4b_wordlevel.py
"""

import json
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForImageTextToText, AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.metrics import compute_cer_normalized, compute_wer_normalized

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HANDWRITTEN_DIR = PROJECT_ROOT / "benchmark" / "test_dataset" / "handwritten"
WORD_GT_PATH = PROJECT_ROOT / "benchmark" / "test_dataset" / "ground_truth_wordlevel.json"
RESULTS_DIR = PROJECT_ROOT / "benchmark" / "results"
VIZ_DIR = PROJECT_ROOT / "benchmark" / "visualizations" / "qwen3vl_4b_wordlevel"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
BBOX_PATTERN = re.compile(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\s*")

PROMPT = (
    "Transcribe every individual word in this handwritten text. "
    "For each word, output exactly: [x1, y1, x2, y2] the_word "
    "One word per line. Preserve reading order. Do not group words together."
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_response(content: str, img_w: int, img_h: int) -> list[dict]:
    """Parse model output into word-level blocks with denormalized pixel coordinates."""
    blocks = []
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        match = BBOX_PATTERN.match(line)
        if match:
            bx1, by1, bx2, by2 = map(int, match.groups())
            x1 = int(bx1 / 999 * img_w)
            y1 = int(by1 / 999 * img_h)
            x2 = int(bx2 / 999 * img_w)
            y2 = int(by2 / 999 * img_h)
            if x1 > x2: x1, x2 = x2, x1
            if y1 > y2: y1, y2 = y2, y1
            word = line[match.end():].strip()
            if word:
                blocks.append({"bbox": [x1, y1, x2, y2], "text": word, "confidence": 1.0})
        else:
            blocks.append({"bbox": [0, 0, 0, 0], "text": line, "confidence": 1.0})
    return blocks


# ---------------------------------------------------------------------------
# Word-level IoU
# ---------------------------------------------------------------------------
def compute_word_iou(pred: list[dict], gt_words: list[dict]) -> dict:
    """Greedy word-to-word IoU matching."""
    valid = [b for b in pred if b["bbox"] != [0, 0, 0, 0]]
    if not valid or not gt_words:
        return {"mean_iou": 0, "matched": 0, "gt_count": len(gt_words), "pred_count": len(valid),
                "recall": 0, "precision": 0}

    matched_gt = set()
    ious = []
    for p in valid:
        px1, py1, px2, py2 = p["bbox"]
        best_iou, best_j = 0.0, -1
        for j, gw in enumerate(gt_words):
            if j in matched_gt:
                continue
            gx1, gy1, gx2, gy2 = gw["bbox"]
            ix1, iy1 = max(px1, gx1), max(py1, gy1)
            ix2, iy2 = min(px2, gx2), min(py2, gy2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter == 0:
                continue
            area_p = (px2 - px1) * (py2 - py1)
            area_g = (gx2 - gx1) * (gy2 - gy1)
            iou = inter / (area_p + area_g - inter)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou > 0.05:
            ious.append(best_iou)
            matched_gt.add(best_j)

    return {
        "mean_iou": sum(ious) / len(ious) if ious else 0.0,
        "matched": len(ious),
        "gt_count": len(gt_words),
        "pred_count": len(valid),
        "recall": len(ious) / len(gt_words) if gt_words else 0,
        "precision": len(ious) / len(valid) if valid else 0,
    }


# ---------------------------------------------------------------------------
# Reading order (Kendall's tau)
# ---------------------------------------------------------------------------
def compute_kendall_tau(pred: list[dict], gt_words: list[dict]) -> float:
    """Kendall's tau-b between predicted and GT reading order via spatial matching."""
    valid = [b for b in pred if b["bbox"] != [0, 0, 0, 0]]
    if len(valid) < 2 or len(gt_words) < 2:
        return 1.0

    # Predicted order: sort by y then x
    pred_sorted = sorted(enumerate(valid), key=lambda t: (t[1]["bbox"][1], t[1]["bbox"][0]))

    # GT order is already sequential (extracted top-to-bottom, left-to-right from XML)
    # Match predicted words to GT words spatially (same greedy IoU approach)
    matched_gt = set()
    pred_ranks = []
    gt_ranks = []
    for pred_idx, p in pred_sorted:
        px1, py1, px2, py2 = p["bbox"]
        best_iou, best_j = 0.0, -1
        for j, gw in enumerate(gt_words):
            if j in matched_gt:
                continue
            gx1, gy1, gx2, gy2 = gw["bbox"]
            ix1, iy1 = max(px1, gx1), max(py1, gy1)
            ix2, iy2 = min(px2, gx2), min(py2, gy2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter == 0:
                continue
            area_p = (px2 - px1) * (py2 - py1)
            area_g = (gx2 - gx1) * (gy2 - gy1)
            iou = inter / (area_p + area_g - inter)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou > 0.05:
            pred_ranks.append(pred_idx)
            gt_ranks.append(best_j)
            matched_gt.add(best_j)

    if len(pred_ranks) < 2:
        return 1.0

    n = len(pred_ranks)
    concordant, discordant = 0, 0
    ties_pred, ties_gt = 0, 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            a = pred_ranks[i] - pred_ranks[j]
            b = gt_ranks[i] - gt_ranks[j]
            if a == 0:
                ties_pred += 1
            if b == 0:
                ties_gt += 1
            if a * b > 0:
                concordant += 1
            elif a * b < 0:
                discordant += 1

    total = n * (n - 1) / 2
    tau = (concordant - discordant) / (
        ((total - ties_pred) * (total - ties_gt)) ** 0.5
    ) if total > 0 else 1.0
    return tau


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def draw_bboxes(img_path: str, pred: list[dict], gt_words: list[dict], out_path: str):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for block in pred:
        b = block["bbox"]
        if b != [0, 0, 0, 0]:
            draw.rectangle(b, outline="#FF0000", width=2)
    for gw in gt_words:
        draw.rectangle(gw["bbox"], outline="#00FF00", width=1)
    img.save(out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    images = sorted([str(p) for p in HANDWRITTEN_DIR.glob("*.png")])
    if not images:
        print(f"No images in {HANDWRITTEN_DIR}")
        sys.exit(1)

    gt_data = json.loads(WORD_GT_PATH.read_text())
    gt_by_name = {e["image"]: e for e in gt_data}

    print(f"Loading {MODEL_ID}...")
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)
    print(f"Loaded. Evaluating {len(images)} images...\n")

    results = []
    cer_vals, wer_vals, iou_vals, tau_vals = [], [], [], []

    for idx, img_path in enumerate(images):
        img_name = Path(img_path).name
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT},
        ]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=4096)
        gen_trim = [out[len(inp):] for inp, out in zip(inputs["input_ids"], gen_ids)]
        text = processor.batch_decode(gen_trim, skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False)[0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        blocks = parse_response(text, img_w, img_h)
        clean_text = " ".join(b["text"] for b in blocks if b["bbox"] != [0, 0, 0, 0])

        gt_entry = gt_by_name.get(img_name, {})
        gt_words = gt_entry.get("words", [])
        gt_text = gt_entry.get("text", "")

        cer = compute_cer_normalized(clean_text, gt_text) if gt_text else 0
        wer = compute_wer_normalized(clean_text, gt_text) if gt_text else 0
        iou_info = compute_word_iou(blocks, gt_words)
        tau = compute_kendall_tau(blocks, gt_words)

        cer_vals.append(cer)
        wer_vals.append(wer)
        iou_vals.append(iou_info["mean_iou"])
        tau_vals.append(tau)

        results.append({
            "image": img_name,
            "text": clean_text,
            "cer": round(cer, 4), "wer": round(wer, 4),
            "word_iou": round(iou_info["mean_iou"], 4),
            "iou_recall": round(iou_info["recall"], 4),
            "iou_precision": round(iou_info["precision"], 4),
            "iou_matched": iou_info["matched"],
            "gt_words": iou_info["gt_count"],
            "pred_words": iou_info["pred_count"],
            "kendall_tau": round(tau, 4),
            "latency_s": round(elapsed, 1),
        })

        draw_bboxes(img_path, blocks, gt_words,
                     str(VIZ_DIR / f"{Path(img_name).stem}_bboxes.png"))

        print(f"[{idx+1:2d}/{len(images)}] {img_name}: "
              f"CER={cer:.4f} WER={wer:.4f} IoU={iou_info['mean_iou']:.3f} "
              f"τ={tau:.3f} words={iou_info['pred_count']} "
              f"({iou_info['matched']}/{iou_info['gt_count']} matched) "
              f"latency={elapsed:.0f}s")

    # Aggregate
    output = {
        "candidate": "qwen3_vl_4b_wordlevel",
        "model": MODEL_ID,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "images": results,
        "aggregate": {
            "cer": round(sum(cer_vals) / len(cer_vals), 4) if cer_vals else 0,
            "wer": round(sum(wer_vals) / len(wer_vals), 4) if wer_vals else 0,
            "word_iou": round(sum(iou_vals) / len(iou_vals), 4) if iou_vals else 0,
            "kendall_tau": round(sum(tau_vals) / len(tau_vals), 4) if tau_vals else 0,
            "num_images": len(images),
        },
    }

    out_path = RESULTS_DIR / "qwen3_vl_4b_wordlevel_handwritten.json"
    out_path.write_text(json.dumps(output, indent=2))
    agg = output["aggregate"]
    print(f"\nSaved to {out_path}")
    print(f"Aggregate: CER={agg['cer']:.4f} WER={agg['wer']:.4f} "
          f"IoU={agg['word_iou']:.3f} τ={agg['kendall_tau']:.3f}")
    print(f"Visualizations: {VIZ_DIR}/")


if __name__ == "__main__":
    main()
