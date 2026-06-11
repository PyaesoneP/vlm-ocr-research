"""
GOT-OCR2.0 — Candidate evaluation script.

GOT-OCR2.0 (General OCR Theory) is a unified end-to-end OCR model supporting
plain text, formatted text, and fine-grained OCR with bounding boxes.  Built
on Qwen and supports GGUF/llama.cpp for quantized inference.

Model: https://huggingface.co/ucaslcl/GOT-OCR2_0
GGUF:  https://huggingface.co/MosRat/got.cpp (community)

Usage:
    python candidates/got_ocr/eval.py
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

MODEL_ID = "stepfun-ai/GOT-OCR-2.0-hf"  # HuggingFace transformers version
CANDIDATE_NAME = "got_ocr2"
TEST_DATASET = PROJECT_ROOT / "benchmark" / "test_dataset"
GROUND_TRUTH = TEST_DATASET / "ground_truth.json"


# ---------------------------------------------------------------------------
# Inference function
# ---------------------------------------------------------------------------

def inference_fn(image_path: str) -> dict:
    """
    Run GOT-OCR2.0 on a handwritten essay image.

    Uses the HuggingFace transformers integration.
    For quantized inference, use the llama.cpp / GGUF version instead.
    """
    import re
    import time
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText

    if not hasattr(inference_fn, "_model"):
        print(f"[{CANDIDATE_NAME}] Loading {MODEL_ID} ...")
        inference_fn._processor = AutoProcessor.from_pretrained(
            MODEL_ID, trust_remote_code=True, use_fast=True
        )
        inference_fn._model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        print(f"[{CANDIDATE_NAME}] Model loaded.")

    processor = inference_fn._processor
    model = inference_fn._model
    device = model.device

    image = Image.open(image_path).convert("RGB")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    # GOT-OCR2.0: pass image only (model auto-detects OCR task)
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=2048)

    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # --- Clean up chat template tokens from output ---
    # GOT-OCR2.0 emits system messages, role markers, and metadata before OCR text.
    # The pattern is: "You should follow the instructions...\nuser\n\n OCR: assistant\n[metadata]\n[ACTUAL TEXT]"
    clean_text = generated_text
    # Strip everything up to and including "OCR:" or "OCR: assistant" line
    clean_text = re.sub(r'^.*?OCR:\s*(assistant\s*)?\n', '', clean_text, count=1, flags=re.DOTALL)
    # If the above didn't catch it, try simpler patterns
    if clean_text == generated_text:
        for prefix in ["OCR: assistant\n", "OCR:\n", "assistant\n"]:
            if clean_text.startswith(prefix):
                clean_text = clean_text[len(prefix):].strip()
                break
    # Remove "Sentence Database" header line
    clean_text = re.sub(r'^Sentence Database\s*\n', '', clean_text)
    # Remove IAM form ID line (e.g., "A04-039 \n")
    clean_text = re.sub(r'^[A-Ka-k]\d{2}-\d{3}[a-z]?\s*\n', '', clean_text)
    # Remove any trailing role markers
    clean_text = re.sub(r'\n\s*(assistant|user|system)\s*$', '', clean_text, flags=re.IGNORECASE)
    clean_text = clean_text.strip()

    # --- Parse blocks from text (line-based heuristic) ---
    blocks = []
    lines = clean_text.split("\n")
    for line in lines:
        line = line.strip()
        if line:
            blocks.append({
                "bbox": [0, 0, 0, 0],  # GOT plain OCR doesn't provide bboxes
                "text": line,
                "confidence": 1.0,
            })

    return {
        "text": clean_text,
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
        notes="GOT-OCR2.0 — unified end-to-end OCR model (Qwen-based).",
    )

    print(f"[{CANDIDATE_NAME}] Done. Avg latency: {result.latency_total_avg:.2f}s")
