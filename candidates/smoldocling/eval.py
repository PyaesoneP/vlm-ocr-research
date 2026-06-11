"""
SmolDocling / granite-docling-258M — Candidate evaluation script.

SmolDocling is an ultra-compact (256M) VLM for document conversion that
preserves layout, bounding boxes, and reading structure via DocTags.
granite-docling-258M is the successor, built on IBM Granite.

Models:
  - docling-project/SmolDocling-256M-preview
  - ibm-granite/granite-docling-258M (successor, recommended)

Usage:
    python candidates/smoldocling/eval.py
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

# Use the successor model
MODEL_ID = "ibm-granite/granite-docling-258M"
CANDIDATE_NAME = "granite_docling"
TEST_DATASET = PROJECT_ROOT / "benchmark" / "test_dataset"
GROUND_TRUTH = TEST_DATASET / "ground_truth.json"


# ---------------------------------------------------------------------------
# Inference function
# ---------------------------------------------------------------------------

def inference_fn(image_path: str) -> dict:
    """
    Run granite-docling-258M on a handwritten essay image.

    Uses DocTags for structured output — preserves text, bounding boxes,
    and reading order in a compact token format.
    """
    import time
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForVision2Seq

    if not hasattr(inference_fn, "_model"):
        print(f"[{CANDIDATE_NAME}] Loading {MODEL_ID} ...")
        inference_fn._processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)
        inference_fn._model = AutoModelForVision2Seq.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        print(f"[{CANDIDATE_NAME}] Model loaded.")

    processor = inference_fn._processor
    model = inference_fn._model
    device = model.device

    image = Image.open(image_path).convert("RGB")

    # Docling uses "Convert this page to docling." prompt
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Convert this page to docling."},
            ],
        }
    ]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        text=prompt, images=[image], return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=4096)

    doctags = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # --- Parse DocTags into blocks ---
    # DocTags uses <loc_xxx> tokens for bounding boxes.
    # Full parsing requires Docling library; here we return raw doctags.
    # For production: integrate with `docling` package for structured output.
    return {
        "text": doctags,
        "blocks": [],   # TODO: integrate docling for DocTags → blocks
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
        notes="granite-docling-258M — ultra-compact 256M document VLM (successor to SmolDocling).",
    )

    print(f"[{CANDIDATE_NAME}] Done. Avg latency: {result.latency_total_avg:.2f}s")
