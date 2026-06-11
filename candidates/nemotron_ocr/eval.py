"""
NVIDIA Nemotron OCR v2 — Candidate evaluation script.

Nemotron OCR v2 is a production-grade OCR model with three components:
  - Text Detector (RegNetX-8GF backbone)
  - Text Recognizer (Transformer-based)
  - Relational Model (reading order & layout)

Includes built-in reading order prediction — a key differentiator for
handwritten essays without ruled lines.

Model: https://huggingface.co/nvidia/nemotron-ocr-v2

Environment: Requires `conda activate aiml` (CUDA 13.0 + PyTorch 2.12).
The package compiles a C++ CUDA extension that must match system nvcc.

Usage:
    conda activate aiml
    python candidates/nemotron_ocr/eval.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# --- Environment guard ---
if "CONDA_DEFAULT_ENV" not in os.environ or os.environ["CONDA_DEFAULT_ENV"] != "aiml":
    print(
        "[nemotron_ocr_v2] ERROR: This script requires the 'aiml' conda environment.\n"
        "  conda activate aiml\n"
        "  python candidates/nemotron_ocr/eval.py"
    )
    sys.exit(1)

from candidates import run_candidate

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CANDIDATE_NAME = "nemotron_ocr_v2"
TEST_DATASET = PROJECT_ROOT / "benchmark" / "test_dataset"
GROUND_TRUTH = TEST_DATASET / "ground_truth.json"


# ---------------------------------------------------------------------------
# Inference function
# ---------------------------------------------------------------------------

def inference_fn(image_path: str) -> dict:
    """
    Run Nemotron OCR v2 on a handwritten essay image.

    Uses the nemotron_ocr Python package which includes:
      - detector-only mode (~37% less VRAM)
      - skip_relational mode (~35% less VRAM, no reading order)
      - verbose_post mode (per-phase timing)

    For the research phase, use the English v2 variant (word-level).
    For deployment, the multilingual variant adds Chinese/Japanese/Korean/Russian.
    """
    import time
    import torch

    if not hasattr(inference_fn, "_ocr"):
        print(f"[{CANDIDATE_NAME}] Loading Nemotron OCR v2 (EN) ...")
        try:
            from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2
        except ImportError:
            raise ImportError(
                "Nemotron OCR v2 not installed.\n"
                "Clone https://huggingface.co/nvidia/nemotron-ocr-v2 and:\n"
                "  pip install --no-build-isolation -v .\n"
                "Requires Python 3.12, CUDA toolkit, and PyTorch with matching CUDA version."
            )
        inference_fn._ocr = NemotronOCRV2(lang="en", model_dir="/tmp/nemotron-ocr-v2/v2_english")
        print(f"[{CANDIDATE_NAME}] Model loaded.")

    ocr = inference_fn._ocr

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    # Run full pipeline: detect + recognize + relational (reading order)
    predictions = ocr(str(image_path))

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # --- Parse Nemotron output ---
    blocks = []
    full_text_parts = []

    # Get image dimensions for coordinate denormalization (Nemotron outputs 0-1)
    from PIL import Image as PILImage
    img = PILImage.open(image_path)
    img_w, img_h = img.size

    for pred in predictions:
        bbox = [
            int(pred["left"] * img_w),
            int(pred["upper"] * img_h),
            int(pred["right"] * img_w),
            int(pred["lower"] * img_h),
        ]
        blocks.append({
            "bbox": bbox,
            "text": pred["text"],
            "confidence": float(pred["confidence"]),
        })
        full_text_parts.append(pred["text"])

    return {
        "text": "\n".join(full_text_parts),
        "blocks": blocks,
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
        num_runs=1,
        notes="NVIDIA Nemotron OCR v2 (EN) — detector + recognizer + relational model. Built in aiml conda env (CUDA 13.0 + PyTorch 2.12).",
    )

    print(f"[{CANDIDATE_NAME}] Done. Avg latency: {result.latency_total_avg:.2f}s")
