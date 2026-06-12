#!/usr/bin/env python3
"""
Evaluate Google Document AI on handwritten-only cropped images:
block-level bounding box IoU and reading-order tau.

Uses the ground_truth_handwritten.json (cropped coordinates) and
runs Document AI OCR only (no Gemini stage) on the handwritten crops.

Usage:
    source .venv/bin/activate
    set -a && source .env && set +a
    python scripts/eval_baseline_bbox.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.metrics import (
    compute_block_iou,
    compute_reading_order_from_blocks,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GCP_PROJECT = os.environ.get("GCP_PROJECT", "")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us")
DOCAI_PROCESSOR_ID = os.environ.get("DOCAI_PROCESSOR_ID", "")

HANDWRITTEN_DIR = PROJECT_ROOT / "benchmark" / "test_dataset" / "handwritten"
GT_FILE = PROJECT_ROOT / "benchmark" / "test_dataset" / "ground_truth_handwritten.json"
RESULTS_DIR = PROJECT_ROOT / "benchmark" / "results"

GEMINI_RPM = int(os.environ.get("GEMINI_RPM", "5"))


# ---------------------------------------------------------------------------
# Document AI client
# ---------------------------------------------------------------------------

def _get_docai_client():
    from google.cloud import documentai
    from google.api_core.client_options import ClientOptions

    opts = ClientOptions(
        api_endpoint=f"{GCP_LOCATION}-documentai.googleapis.com"
    ) if GCP_LOCATION != "us" else None
    if opts:
        return documentai.DocumentProcessorServiceClient(client_options=opts)
    return documentai.DocumentProcessorServiceClient()


def _mime_type(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".bmp": "image/bmp", ".tiff": "image/tiff",
    }.get(ext, "image/png")


def _layout_text(doc, layout) -> str:
    text = ""
    for segment in layout.text_anchor.text_segments:
        start = int(segment.start_index) if segment.start_index else 0
        end = int(segment.end_index) if segment.end_index else len(doc.text)
        text += doc.text[start:end]
    return text


def docai_ocr(image_path: str | Path) -> dict:
    """Run Document AI OCR on a single image. Returns blocks + text."""
    from google.cloud import documentai

    client = _get_docai_client()
    image_path = Path(image_path)

    t0 = time.perf_counter()
    with open(image_path, "rb") as f:
        content = f.read()

    raw_doc = documentai.RawDocument(content=content, mime_type=_mime_type(image_path))
    name = client.processor_path(GCP_PROJECT, GCP_LOCATION, DOCAI_PROCESSOR_ID)
    request = documentai.ProcessRequest(name=name, raw_document=raw_doc)
    response = client.process_document(request=request)
    doc = response.document
    elapsed = time.perf_counter() - t0

    blocks = []
    for page in doc.pages:
        for block in page.blocks:
            text = _layout_text(doc, block.layout)
            # Use normalized_vertices (0-1 range), NOT .vertices (absolute)
            xs = [v.x * page.image.width for v in block.layout.bounding_poly.normalized_vertices]
            ys = [v.y * page.image.height for v in block.layout.bounding_poly.normalized_vertices]
            if xs:
                bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
            else:
                bbox = [0, 0, 0, 0]
            blocks.append({
                "bbox": bbox,
                "text": text,
                "confidence": block.layout.confidence,
            })

    return {
        "text": doc.text,
        "blocks": blocks,
        "stage1_latency": elapsed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not GCP_PROJECT or not DOCAI_PROCESSOR_ID:
        print("ERROR: Set GCP_PROJECT and DOCAI_PROCESSOR_ID in .env")
        sys.exit(1)

    # Load GT (cropped coordinates)
    with open(GT_FILE) as f:
        gt_list = json.load(f)
    gt_map = {g["image"]: g for g in gt_list}

    images = sorted(
        p for p in HANDWRITTEN_DIR.glob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    print(f"Evaluating Document AI bbox + reading order on {len(images)} handwritten images...\n")
    print(f"  Project: {GCP_PROJECT}")
    print(f"  Processor: {DOCAI_PROCESSOR_ID}")
    print(f"  Location: {GCP_LOCATION}")

    all_results = []
    total_iou = 0.0
    total_tau = 0.0
    total_time = 0.0
    iou_count = 0
    tau_count = 0

    rate_limit_interval = 60.0 / GEMINI_RPM
    last_call = 0.0

    for i, img_path in enumerate(images):
        name = img_path.name
        gt_entry = gt_map.get(name)
        if not gt_entry:
            print(f"  [{i+1:2d}/{len(images)}] {name:20s}  SKIP (no GT)")
            continue

        # Rate limit
        elapsed_since = time.perf_counter() - last_call
        if elapsed_since < rate_limit_interval:
            time.sleep(rate_limit_interval - elapsed_since)

        result = docai_ocr(str(img_path))
        last_call = time.perf_counter()

        elapsed = result["stage1_latency"]
        pred_blocks = result.get("blocks", [])
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
                {"bbox": b["bbox"], "text": b["text"][:80]}
                for b in pred_blocks[:3]
            ],
        })

        print(f"  [{i+1:2d}/{len(images)}] {name:20s}  "
              f"IoU={iou_result['mean_iou']:.3f}  tau={tau_result['kendall_tau']:.3f}  "
              f"{elapsed:.2f}s  ({iou_result['matched']}/{iou_result['total_gt']} matched)")

    n = len(images)
    print(f"\n{'='*60}")
    print(f"DOCUMENT AI BBOX + READING ORDER RESULTS")
    print(f"  Mean Bbox IoU:     {total_iou/iou_count:.3f}  ({iou_count}/{n} images)")
    print(f"  Mean Kendall's tau:  {total_tau/tau_count:.3f}  ({tau_count}/{n} images)")
    print(f"  Mean Latency:      {total_time/n:.2f}s")
    print(f"{'='*60}")

    output = {
        "candidate": "baseline_google_docai_bbox",
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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "baseline_google_docai_layout.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
