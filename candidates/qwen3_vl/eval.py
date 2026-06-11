"""
Qwen3-VL — Candidate evaluation script.

Qwen3-VL is the latest Qwen vision-language model generation with:
  - Expanded OCR: 32 languages, robust in low light/blur/tilt
  - 256K native context, improved long-document parsing
  - 2D/3D spatial grounding
  - DeepStack for fine-grained visual details

Models evaluated:
  - Qwen/Qwen3-VL-4B-Instruct  (BF16 fits 12 GB)
  - Qwen/Qwen3-VL-8B-Instruct  (requires INT4 for 12 GB)

Usage:
    python candidates/qwen3_vl/eval.py
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

# Switch to 8B by changing MODEL_ID (add --load-in-4bit for 12 GB VRAM)
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
CANDIDATE_NAME = "qwen3_vl_4b"
TEST_DATASET = PROJECT_ROOT / "benchmark" / "test_dataset"
GROUND_TRUTH = TEST_DATASET / "ground_truth.json"
LOAD_IN_4BIT = False  # Set True for 8B model on 12 GB VRAM


# ---------------------------------------------------------------------------
# Inference function
# ---------------------------------------------------------------------------

def inference_fn(image_path: str) -> dict:
    """
    Run Qwen3-VL on a handwritten essay image.

    Two-stage approach:
      Stage 1: OCR — transcribe text and localize each line with bboxes
      Stage 2: Error detection — identify writing errors in the transcription
    """
    import time
    import torch
    from PIL import Image
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    if not hasattr(inference_fn, "_model"):
        print(f"[{CANDIDATE_NAME}] Loading {MODEL_ID} ...")

        load_kwargs = {
            "dtype": "auto",
            "device_map": "auto",
        }
        if LOAD_IN_4BIT:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        inference_fn._model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID, **load_kwargs
        )
        inference_fn._processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)
        print(f"[{CANDIDATE_NAME}] Model loaded.")

    model = inference_fn._model
    processor = inference_fn._processor
    device = model.device

    image = Image.open(image_path).convert("RGB")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    # --- Stage 1: OCR with bounding boxes ---
    ocr_prompt = (
        "Transcribe all handwritten text in this image. "
        "For each line of text, provide the bounding box coordinates "
        "as [x1, y1, x2, y2] and the transcribed text. "
        "Preserve the reading order."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": ocr_prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=4096)

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    ocr_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return {
        "text": ocr_text,
        "blocks": [],   # TODO: parse structured bbox output from VLM response
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

    quant_note = " (INT4)" if LOAD_IN_4BIT else ""
    result = run_candidate(
        candidate_name=CANDIDATE_NAME,
        inference_fn=inference_fn,
        test_images=images,
        ground_truth=GROUND_TRUTH if GROUND_TRUTH.exists() else None,
        notes=f"Qwen3-VL-4B-Instruct{quant_note} — latest Qwen VLM with expanded OCR.",
    )

    print(f"[{CANDIDATE_NAME}] Done. Avg latency: {result.latency_total_avg:.2f}s")
