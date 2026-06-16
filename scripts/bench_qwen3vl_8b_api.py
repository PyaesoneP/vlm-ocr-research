#!/usr/bin/env python3
"""
Qwen3-VL-8B word-level benchmark via HuggingFace Inference API (novita provider).

Evaluates word-level OCR with bounding boxes on the cropped handwriting dataset.
Saves per-image result JSON and generates bbox overlay visualizations.

Usage:
    source .venv/bin/activate
    python scripts/bench_qwen3vl_8b_api.py
"""

import base64
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

HANDWRITTEN_DIR = PROJECT_ROOT / "benchmark" / "test_dataset" / "handwritten"
GROUND_TRUTH_PATH = PROJECT_ROOT / "benchmark" / "test_dataset" / "ground_truth_wordlevel.json"
RESULTS_DIR = PROJECT_ROOT / "benchmark" / "results"
VIZ_DIR = PROJECT_ROOT / "benchmark" / "visualizations" / "qwen3vl_8b_wordlevel"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()

# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------
import httpx

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
    timeout=httpx.Timeout(300.0, connect=30.0),
)

MODEL_API = "Qwen/Qwen3-VL-8B-Instruct:novita"
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds base delay

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
PROMPT = (
    "Transcribe every individual word in this handwritten text. "
    "For each word, output exactly: [x1, y1, x2, y2] the_word "
    "One word per line. Preserve reading order. Do not group words together."
)

# ---------------------------------------------------------------------------
# Bbox parsing
# ---------------------------------------------------------------------------
BBOX_PATTERN = re.compile(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\s*")


def parse_response(content: str, img_width: int, img_height: int) -> tuple[str, list[dict]]:
    """Parse API response into clean text and word-level blocks.

    Qwen3-VL outputs coordinates in 0-999 normalized bin space (like Qwen2-VL).
    We denormalize to pixel coordinates using the actual image dimensions.
    """
    blocks = []
    clean_words = []
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        match = BBOX_PATTERN.match(line)
        if match:
            bx1, by1, bx2, by2 = map(int, match.groups())
            # Denormalize from 0-999 bin space to pixel coordinates
            x1 = int(bx1 / 999 * img_width)
            y1 = int(by1 / 999 * img_height)
            x2 = int(bx2 / 999 * img_width)
            y2 = int(by2 / 999 * img_height)
            # Ensure x1<x2, y1<y2 (model sometimes reverses order)
            if x1 > x2: x1, x2 = x2, x1
            if y1 > y2: y1, y2 = y2, y1
            word = line[match.end():].strip()
            if word:
                blocks.append({"bbox": [x1, y1, x2, y2], "text": word, "confidence": 1.0})
                clean_words.append(word)
        else:
            clean_words.append(line)
            blocks.append({"bbox": [0, 0, 0, 0], "text": line, "confidence": 1.0})
    return " ".join(clean_words), blocks


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
from benchmark.metrics import compute_cer_normalized, compute_wer_normalized


def compute_word_iou(pred_blocks: list[dict], gt_entry: dict) -> dict:
    """Compute word-to-word IoU with greedy spatial matching.

    Each predicted word is matched to the best-IoU unmatched GT word.
    Returns mean IoU, recall, and precision.
    """
    gt_words = gt_entry.get("words", [])
    if not gt_words:
        return {"mean_iou": 0.0, "matched": 0, "gt_count": 0, "pred_count": len(pred_blocks)}

    # Filter to blocks with valid bboxes
    valid_blocks = [b for b in pred_blocks if b["bbox"] != [0, 0, 0, 0]]
    if not valid_blocks:
        return {"mean_iou": 0.0, "matched": 0, "gt_count": len(gt_words), "pred_count": 0}

    matched_gt = set()
    ious = []

    for pred in valid_blocks:
        px1, py1, px2, py2 = pred["bbox"]
        best_iou = 0.0
        best_gt_idx = -1

        for j, gt in enumerate(gt_words):
            if j in matched_gt:
                continue
            gx1, gy1, gx2, gy2 = gt["bbox"]

            # Intersection
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
                best_gt_idx = j

        if best_iou > 0.05:  # permissive threshold for noisy handwriting bboxes
            ious.append(best_iou)
            matched_gt.add(best_gt_idx)

    return {
        "mean_iou": sum(ious) / len(ious) if ious else 0.0,
        "matched": len(ious),
        "gt_count": len(gt_words),
        "pred_count": len(valid_blocks),
        "recall": len(ious) / len(gt_words) if gt_words else 0.0,
        "precision": len(ious) / len(valid_blocks) if valid_blocks else 0.0,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def draw_bboxes(image_path: str, blocks: list[dict], gt_words: list[dict], output_path: str) -> None:
    """Draw word-level bounding boxes on the image (red=predicted, green=GT)."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Draw ground truth boxes in green (behind predictions)
    for gw in gt_words:
        draw.rectangle(gw["bbox"], outline="#00FF00", width=1)

    # Draw predicted boxes in red (on top)
    for block in blocks:
        bbox = block["bbox"]
        if bbox == [0, 0, 0, 0]:
            continue
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline="#FF0000", width=2)

    img.save(output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    images = sorted(
        [str(p) for p in HANDWRITTEN_DIR.glob("*.png") if p.suffix.lower() == ".png"]
    )
    if not images:
        print(f"No images in {HANDWRITTEN_DIR}")
        sys.exit(1)

    # Load ground truth
    gt_data = json.loads(GROUND_TRUTH_PATH.read_text()) if GROUND_TRUTH_PATH.exists() else []
    gt_by_name = {e["image"]: e for e in gt_data}

    print(f"Evaluating Qwen3-VL-8B word-level on {len(images)} images via HF API (novita)...")
    print(f"Model: {MODEL_API}")
    print(f"Estimated total tokens: ~{len(images) * 5000} (cost < $0.02)")
    print()

    results = []
    cer_vals, wer_vals, iou_vals = [], [], []
    latencies = []

    for idx, img_path in enumerate(images):
        img_name = Path(img_path).name
        img_b64 = base64.b64encode(Path(img_path).read_bytes()).decode()

        t0 = time.perf_counter()
        completion = None
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                completion = client.chat.completions.create(
                    model=MODEL_API,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                            {"type": "text", "text": PROMPT},
                        ],
                    }],
                    max_tokens=4096,
                )
                break
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    wait = RETRY_DELAY * (2 ** (attempt - 1))
                    print(f"  Retry {attempt}/{MAX_RETRIES} in {wait}s: {type(e).__name__}: {e}")
                    time.sleep(wait)
                else:
                    raise last_error

        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        content = completion.choices[0].message.content
        # Get image dimensions for coordinate denormalization
        img = Image.open(img_path)
        img_w, img_h = img.size
        clean_text, blocks = parse_response(content, img_w, img_h)

        # Compute metrics
        gt_entry = gt_by_name.get(img_name, {})
        gt_text = gt_entry.get("text", "")
        cer = compute_cer_normalized(clean_text, gt_text) if gt_text else 0.0
        wer = compute_wer_normalized(clean_text, gt_text) if gt_text else 0.0
        iou_info = compute_word_iou(blocks, gt_entry)

        cer_vals.append(cer)
        wer_vals.append(wer)
        iou_vals.append(iou_info["mean_iou"])

        # Generate visualization (red=predicted, green=GT)
        viz_path = VIZ_DIR / f"{Path(img_name).stem}_bboxes.png"
        gt_words = gt_entry.get("words", [])
        draw_bboxes(img_path, blocks, gt_words, str(viz_path))

        result = {
            "image": img_name,
            "text": clean_text,
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "word_iou": round(iou_info["mean_iou"], 4),
            "iou_matched": iou_info["matched"],
            "iou_gt_words": iou_info["gt_count"],
            "iou_pred_words": iou_info["pred_count"],
            "iou_recall": round(iou_info["recall"], 4),
            "iou_precision": round(iou_info["precision"], 4),
            "word_count": len([b for b in blocks if b["bbox"] != [0, 0, 0, 0]]),
            "latency_s": round(elapsed, 2),
            "tokens": {
                "prompt": completion.usage.prompt_tokens,
                "completion": completion.usage.completion_tokens,
                "total": completion.usage.total_tokens,
            },
        }
        results.append(result)

        print(f"[{idx+1:2d}/{len(images)}] {img_name}: "
              f"CER={cer:.4f} WER={wer:.4f} IoU={iou_info['mean_iou']:.3f} "
              f"words={result['word_count']} latency={elapsed:.1f}s "
              f"tokens={completion.usage.total_tokens}")

    # Aggregate
    avg_cer = sum(cer_vals) / len(cer_vals) if cer_vals else 0
    avg_wer = sum(wer_vals) / len(wer_vals) if wer_vals else 0
    avg_iou = sum(iou_vals) / len(iou_vals) if iou_vals else 0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    total_tokens = sum(r["tokens"]["total"] for r in results)

    output = {
        "candidate": "qwen3_vl_8b_wordlevel_api",
        "model": MODEL_API,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "images": results,
        "aggregate": {
            "cer": round(avg_cer, 4),
            "wer": round(avg_wer, 4),
            "word_iou": round(avg_iou, 4),
            "latency_avg_s": round(avg_lat, 2),
            "total_tokens": total_tokens,
            "num_images": len(images),
        },
    }

    out_path = RESULTS_DIR / "qwen3_vl_8b_wordlevel_handwritten.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved to {out_path}")
    print(f"Aggregate: CER={avg_cer:.4f} WER={avg_wer:.4f} IoU={avg_iou:.3f} "
          f"Latency={avg_lat:.1f}s Tokens={total_tokens}")
    print(f"Visualizations: {VIZ_DIR}/")


if __name__ == "__main__":
    main()
