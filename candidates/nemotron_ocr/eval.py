"""
NVIDIA Nemotron OCR v2 — Candidate evaluation script.

Nemotron OCR v2 is a production-grade OCR model with three components:
  - Text Detector (RegNetX-8GF backbone)
  - Text Recognizer (Transformer-based)
  - Relational Model (reading order & layout)

Includes built-in reading order prediction — a key differentiator for
handwritten essays without ruled lines.

Model: https://huggingface.co/nvidia/nemotron-ocr-v2

Usage:
    python candidates/nemotron_ocr/eval.py
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
        inference_fn._ocr = NemotronOCRV2(lang="en")
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

    for pred in predictions:
        bbox = [
            int(pred["left"]),
            int(pred["upper"]),
            int(pred["right"]),
            int(pred["lower"]),
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
        str(p) for p in TEST_DATASET.glob("*")
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
        notes="NVIDIA Nemotron OCR v2 (EN) — detector + recognizer + relational model.",
    )

    print(f"[{CANDIDATE_NAME}] Done. Avg latency: {result.latency_total_avg:.2f}s")
