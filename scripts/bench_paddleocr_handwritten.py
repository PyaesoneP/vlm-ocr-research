#!/usr/bin/env python3
"""
Standalone PaddleOCR-VL benchmark for handwritten images.
Zero project imports — uses PaddleOCRVL directly to avoid import conflicts.

Usage (inside Docker):
    docker run --rm --gpus all --network host --shm-size=8g \
      -e PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
      -e PYTHONUNBUFFERED=1 \
      -v paddlex_models:/home/paddleocr/.paddlex \
      -v /home/pyaes/vlm-ocr-research/benchmark/test_dataset:/data:ro \
      -v /home/pyaes/vlm-ocr-research/benchmark/results:/results \
      ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-nvidia-gpu-sm120-offline \
      python3 -u /scripts/bench_paddleocr_handwritten.py

Or from host:
    python3 scripts/bench_paddleocr_handwritten.py  # requires .venv_paddleocr
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

IMAGES_DIR = Path(os.environ.get("DATA_DIR", "/data/handwritten"))
GT_PATH = Path(os.environ.get("GT_PATH", "/data/ground_truth_handwritten.json"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "/results"))
CANDIDATE_NAME = "paddleocr_vl_handwritten"


# ---------------------------------------------------------------------------
# Minimal CER / WER (no project imports)
# ---------------------------------------------------------------------------

def _edit_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(
                prev[j + 1] + 1,          # deletion
                curr[j] + 1,              # insertion
                prev[j] + (c1 != c2),     # substitution
            ))
        prev = curr
    return prev[-1]


def normalize_text(text: str) -> str:
    import re
    text = text.replace("\n", " ").replace("\r", " ")
    return re.sub(r"\s+", " ", text).strip()


def compute_cer(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref:
        return 1.0 if hyp else 0.0
    return _edit_distance(ref, hyp) / len(ref)


def compute_wer(reference: str, hypothesis: str) -> float:
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    if not ref_words:
        return 1.0 if hyp_words else 0.0
    return _edit_distance(ref_words, hyp_words) / len(ref_words)


# ---------------------------------------------------------------------------
# GPU info
# ---------------------------------------------------------------------------

def get_gpu_name() -> str:
    try:
        import paddle
        return paddle.device.cuda.get_device_name(0)
    except Exception:
        return "Unknown GPU"


def get_vram_total_mb() -> int:
    try:
        import paddle
        return paddle.device.cuda.get_device_properties(0).total_global_memory // (1024 * 1024)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    # --- Discover images ---
    images = sorted([
        str(p) for p in IMAGES_DIR.glob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])
    if not images:
        print(f"ERROR: No images in {IMAGES_DIR}", file=sys.stderr)
        return 1

    # --- Load ground truth ---
    gt_data: dict[str, str] = {}
    if GT_PATH.exists():
        for entry in json.loads(GT_PATH.read_text()):
            name = entry.get("image", "")
            gt_data[name] = entry.get("text", "")
        print(f"Loaded ground truth: {len(gt_data)} entries")
    else:
        print(f"WARNING: No ground truth at {GT_PATH} — CER/WER will be skipped")

    # --- Load pipeline ---
    print("Loading PaddleOCR-VL pipeline ...")
    from paddleocr import PaddleOCRVL
    pipeline = PaddleOCRVL(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )
    print("Pipeline loaded.")

    import paddle
    gpu_name = get_gpu_name()
    vram_total = get_vram_total_mb()

    # --- Warmup ---
    print(f"Warmup on {Path(images[0]).name} ...")
    paddle.device.synchronize()
    pipeline.predict(images[0])
    paddle.device.synchronize()
    print("Warmup complete.")

    # --- Benchmark ---
    latencies: list[float] = []
    outputs: list[dict] = []
    image_names: list[str] = []

    print(f"\nBenchmarking {len(images)} images ...")
    for i, img_path in enumerate(images):
        fname = Path(img_path).name

        paddle.device.synchronize()
        t0 = time.perf_counter()
        output = pipeline.predict(img_path)
        paddle.device.synchronize()
        elapsed = time.perf_counter() - t0

        latencies.append(elapsed)

        # Parse blocks and full text
        blocks = []
        full_text = ""
        if output:
            for res in output:
                if hasattr(res, "json") and res.json:
                    inner = res.json.get("res", res.json) if isinstance(res.json, dict) else {}
                    for item in inner.get("parsing_res_list", []):
                        text = item.get("block_content", "")
                        bbox_raw = item.get("block_bbox", "")
                        if text:
                            if isinstance(bbox_raw, list):
                                bbox = [float(x) for x in bbox_raw[:4]]
                            elif isinstance(bbox_raw, str):
                                try:
                                    bbox = [int(float(x.strip())) for x in bbox_raw.strip("[]").split(",")]
                                except (ValueError, AttributeError):
                                    bbox = [0, 0, 0, 0]
                            else:
                                bbox = [0, 0, 0, 0]
                            blocks.append({
                                "bbox": bbox if len(bbox) == 4 else [0, 0, 0, 0],
                                "text": str(text),
                                "confidence": 1.0,
                            })
                            full_text += str(text) + " "
                if not full_text and hasattr(res, "markdown") and res.markdown:
                    md = res.markdown
                    full_text = md.get("markdown_texts", str(md)) if isinstance(md, dict) else str(md)
                if not full_text and hasattr(res, "text") and res.text:
                    full_text = str(res.text)

        outputs.append({"text": full_text.strip(), "blocks": blocks})
        image_names.append(fname)

        print(f"  [{i+1:2d}/{len(images)}] {fname}: {elapsed:.2f}s  text={full_text[:60]}...")

    # --- Aggregate ---
    n = len(latencies)
    avg_latency = sum(latencies) / n
    std_latency = (
        (sum((x - avg_latency) ** 2 for x in latencies) / (n - 1)) ** 0.5
    ) if n > 1 else 0.0

    # --- CER / WER ---
    cer_values: list[float] = []
    wer_values: list[float] = []
    for fname, out in zip(image_names, outputs):
        ref = gt_data.get(fname, "")
        if ref:
            cer_values.append(compute_cer(ref, out["text"]))
            wer_values.append(compute_wer(ref, out["text"]))

    avg_cer = sum(cer_values) / len(cer_values) if cer_values else 0.0
    avg_wer = sum(wer_values) / len(wer_values) if wer_values else 0.0

    # --- Build result ---
    result = {
        "candidate_name": CANDIDATE_NAME,
        "model_version": "PaddleOCR-VL-1.6-0.9B",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu_name": gpu_name,
        "vram_total_mb": vram_total,
        "latency_total_avg": avg_latency,
        "latency_total_std": std_latency,
        "latency_stage1_ocr": avg_latency,
        "latency_stage2_errors": 0.0,
        "cer": avg_cer,
        "wer": avg_wer,
        "reading_order_tau": 0.0,
        "error_detection_f1": 0.0,
        "bounding_box_iou": 0.0,
        "vram_peak_mb": 0,
        "throughput_ppm": 60.0 / avg_latency if avg_latency > 0 else 0.0,
        "setup_complexity": 0,
        "flexibility": 0,
        "sample_outputs": outputs[:3],
        "notes": "PaddleOCR-VL-1.6 on cropped handwritten regions. "
                 "Standalone benchmark (no harness imports).",
    }

    # --- Save ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{CANDIDATE_NAME}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out_path}")

    # --- Summary ---
    print(f"\n{'='*50}")
    print(f"PaddleOCR-VL Handwritten Benchmark")
    print(f"  Images:  {n}")
    print(f"  Avg Latency: {avg_latency:.2f}s ± {std_latency:.2f}s")
    print(f"  Throughput:  {result['throughput_ppm']:.1f} ppm")
    print(f"  CER:  {avg_cer:.4f}")
    print(f"  WER:  {avg_wer:.4f}")
    print(f"{'='*50}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
