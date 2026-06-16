#!/usr/bin/env python3
"""
Google Document AI word-level IoU evaluation.

Extracts per-word (token-level) bounding boxes from Document AI OCR
and computes word IoU against ground_truth_wordlevel.json.

Usage:
    set -a && source .env && set +a
    source .venv/bin/activate
    python scripts/eval_docai_wordlevel.py

Cost: 25 pages × $1.50/1K ≈ $0.04
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.metrics import compute_cer_normalized, compute_wer_normalized
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HANDWRITTEN_DIR = PROJECT_ROOT / "benchmark" / "test_dataset" / "handwritten"
WORD_GT_PATH = PROJECT_ROOT / "benchmark" / "test_dataset" / "ground_truth_wordlevel.json"
RESULTS_DIR = PROJECT_ROOT / "benchmark" / "results"
VIZ_DIR = PROJECT_ROOT / "benchmark" / "visualizations" / "docai_wordlevel"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Doc AI client (reuse from baseline.py patterns)
# ---------------------------------------------------------------------------
def _get_docai_client():
    from google.cloud import documentai
    from google.api_core.client_options import ClientOptions

    location = os.environ.get("GCP_LOCATION", "us")
    if location != "us":
        opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
        return documentai.DocumentProcessorServiceClient(client_options=opts)
    return documentai.DocumentProcessorServiceClient()


def _layout_text(doc, layout):
    """Extract text from Document AI layout using text_anchor."""
    if layout.text_anchor and layout.text_anchor.text_segments:
        seg = layout.text_anchor.text_segments[0]
        return doc.text[seg.start_index:seg.end_index] if hasattr(seg, 'start_index') else ""
    return ""


def _normalized_to_absolute(bounding_poly, img_w, img_h):
    """Convert normalized vertices to absolute pixel bbox [x1,y1,x2,y2]."""
    xs, ys = [], []
    for v in bounding_poly.normalized_vertices:
        xs.append(int(v.x * img_w))
        ys.append(int(v.y * img_h))
    return [min(xs), min(ys), max(xs), max(ys)]


def docai_wordlevel_ocr(image_path: str) -> dict:
    """Run Doc AI OCR and extract word-level (token) bboxes."""
    from google.cloud import documentai

    client = _get_docai_client()
    project = os.environ["GCP_PROJECT"]
    location = os.environ.get("GCP_LOCATION", "us")
    processor_id = os.environ["DOCAI_PROCESSOR_ID"]

    image_path = Path(image_path)

    t0 = time.perf_counter()
    with open(image_path, "rb") as f:
        content = f.read()

    name = client.processor_path(project, location, processor_id)
    raw_doc = documentai.RawDocument(content=content, mime_type="image/png")
    request = documentai.ProcessRequest(name=name, raw_document=raw_doc)
    response = client.process_document(request=request)
    doc = response.document
    elapsed = time.perf_counter() - t0

    # Extract word-level (token) bboxes
    blocks = []
    for page in doc.pages:
        for token in page.tokens:
            text = _layout_text(doc, token.layout).strip()
            if text:
                bbox = _normalized_to_absolute(
                    token.layout.bounding_poly,
                    page.image.width,
                    page.image.height,
                )
                blocks.append({
                    "bbox": bbox,
                    "text": text,
                    "confidence": token.layout.confidence,
                })

    return {
        "text": doc.text.strip(),
        "blocks": blocks,
        "stage1_latency": elapsed,
    }


# ---------------------------------------------------------------------------
# Word IoU (same as other eval scripts)
# ---------------------------------------------------------------------------
def compute_word_iou(pred_blocks: list[dict], gt_words: list[dict]) -> dict:
    valid = [b for b in pred_blocks if b.get("bbox", [0, 0, 0, 0]) != [0, 0, 0, 0]]
    if not valid or not gt_words:
        return {
            "mean_iou": 0.0, "matched": 0, "gt_count": len(gt_words),
            "pred_count": len(valid), "recall": 0.0, "precision": 0.0,
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
            ix1 = max(px1, gx1); iy1 = max(py1, gy1)
            ix2 = min(px2, gx2); iy2 = min(py2, gy2)
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


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def draw_bboxes(img_path: str, pred_blocks: list[dict], gt_words: list[dict], out_path: str):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for gw in gt_words:
        draw.rectangle(gw["bbox"], outline="#00FF00", width=1)
    for block in pred_blocks:
        b = block.get("bbox", [0, 0, 0, 0])
        if b != [0, 0, 0, 0]:
            draw.rectangle(b, outline="#FF0000", width=2)
    img.save(out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Verify env
    for var in ["GCP_PROJECT", "DOCAI_PROCESSOR_ID"]:
        if not os.environ.get(var):
            print(f"ERROR: {var} not set. Run: set -a && source .env && set +a")
            sys.exit(1)

    images = sorted([str(p) for p in HANDWRITTEN_DIR.glob("*.png")])
    if not images:
        print(f"No images in {HANDWRITTEN_DIR}")
        sys.exit(1)

    gt_data = json.loads(WORD_GT_PATH.read_text())
    gt_by_name = {e["image"]: e for e in gt_data}

    print(f"Evaluating Google Document AI word-level IoU on {len(images)} images...")
    print(f"Estimated cost: ~${len(images) * 0.0015:.3f}")
    print()

    results = []
    cer_vals, wer_vals, iou_vals = [], [], []
    latencies = []

    for idx, img_path in enumerate(images):
        img_name = Path(img_path).name
        gt_entry = gt_by_name.get(img_name, {})
        gt_words = gt_entry.get("words", [])

        try:
            output = docai_wordlevel_ocr(img_path)
        except Exception as e:
            print(f"[{idx+1:2d}/{len(images)}] {img_name}: ERROR: {e}")
            continue

        elapsed = output["stage1_latency"]
        latencies.append(elapsed)
        blocks = output["blocks"]
        text = output["text"]

        gt_text = gt_entry.get("text", "")
        cer = compute_cer_normalized(text, gt_text) if gt_text else 0.0
        wer = compute_wer_normalized(text, gt_text) if gt_text else 0.0
        iou_info = compute_word_iou(blocks, gt_words)

        cer_vals.append(cer)
        wer_vals.append(wer)
        iou_vals.append(iou_info["mean_iou"])

        viz_path = VIZ_DIR / f"{Path(img_name).stem}_bboxes.png"
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
            "latency_s": round(elapsed, 2),
        }
        results.append(result)

        print(f"[{idx+1:2d}/{len(images)}] {img_name}: "
              f"CER={cer:.4f} WER={wer:.4f} IoU={iou_info['mean_iou']:.3f} "
              f"words={iou_info['pred_count']} latency={elapsed:.1f}s")

    avg_cer = sum(cer_vals) / len(cer_vals) if cer_vals else 0
    avg_wer = sum(wer_vals) / len(wer_vals) if wer_vals else 0
    avg_iou = sum(iou_vals) / len(iou_vals) if iou_vals else 0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0

    output = {
        "candidate": "google_docai_wordlevel",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "images": results,
        "aggregate": {
            "cer": round(avg_cer, 4),
            "wer": round(avg_wer, 4),
            "word_iou": round(avg_iou, 4),
            "latency_avg_s": round(avg_lat, 2),
            "num_images": len(results),
        },
    }

    out_path = RESULTS_DIR / "google_docai_wordlevel_handwritten.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved to {out_path}")
    print(f"Aggregate: CER={avg_cer:.4f} WER={avg_wer:.4f} IoU={avg_iou:.3f} "
          f"Latency={avg_lat:.1f}s")
    print(f"Visualizations: {VIZ_DIR}/")


if __name__ == "__main__":
    main()
