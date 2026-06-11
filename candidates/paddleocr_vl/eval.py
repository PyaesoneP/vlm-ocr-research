"""
PaddleOCR-VL-1.6 — Candidate evaluation script.

PaddleOCR-VL is a 0.9B VLM specifically designed for document parsing.
It achieves 96.3% on OmniDocBench v1.6 and supports structured output
in Markdown/JSON with bounding boxes.

Model: https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6

Usage:
    python candidates/paddleocr_vl/eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from candidates import run_candidate

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.6"
CANDIDATE_NAME = "paddleocr_vl"
TEST_DATASET = PROJECT_ROOT / "benchmark" / "test_dataset"
GROUND_TRUTH = TEST_DATASET / "ground_truth.json"


# ---------------------------------------------------------------------------
# Inference function
# ---------------------------------------------------------------------------

def inference_fn(image_path: str) -> dict:
    """
    Run PaddleOCR-VL-1.6 on a single handwritten essay image.

    Returns a dict conforming to the harness output spec:
        {text, blocks, stage1_latency, ...}
    """
    import time
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForVision2Seq

    # --- Lazy-load model (cached at module level) ---
    if not hasattr(inference_fn, "_model"):
        print(f"[{CANDIDATE_NAME}] Loading model {MODEL_ID} ...")
        inference_fn._processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=True)
        inference_fn._model = AutoModelForVision2Seq.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        print(f"[{CANDIDATE_NAME}] Model loaded.")

    processor = inference_fn._processor
    model = inference_fn._model
    device = model.device

    # --- Inference ---
    image = Image.open(image_path).convert("RGB")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=2048)

    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # --- Parse output ---
    # PaddleOCR-VL outputs Markdown/JSON by default.
    # For now, return text with placeholder blocks.
    # TODO: Parse structured PaddleOCR-VL output into blocks/bboxes.
    return {
        "text": output_text,
        "blocks": [],
        "stage1_latency": elapsed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    images = sorted([
        str(p) for p in (TEST_DATASET / "curated").glob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])

    if not images:
        print(f"No test images in {TEST_DATASET}. Add handwritten essay samples.")
        sys.exit(1)

    result = run_candidate(
        candidate_name=CANDIDATE_NAME,
        inference_fn=inference_fn,
        test_images=images,
        ground_truth=GROUND_TRUTH if GROUND_TRUTH.exists() else None,
        notes="PaddleOCR-VL-1.6 — 0.9B document VLM.",
    )

    print(f"[{CANDIDATE_NAME}] Done. Avg latency: {result.latency_total_avg:.2f}s")
