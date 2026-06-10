"""
Florence-2 — Candidate evaluation script.

Florence-2 is Microsoft's lightweight vision foundation model (0.23B/0.77B).
It supports prompt-based OCR with bounding boxes via the "<OCR>" and
"<OCR_WITH_REGION>" tasks.

Model: https://huggingface.co/microsoft/Florence-2-large

Usage:
    python candidates/florence2/eval.py
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

MODEL_ID = "microsoft/Florence-2-large"
CANDIDATE_NAME = "florence2"
TEST_DATASET = PROJECT_ROOT / "benchmark" / "test_dataset"
GROUND_TRUTH = TEST_DATASET / "ground_truth.json"


# ---------------------------------------------------------------------------
# Inference function
# ---------------------------------------------------------------------------

def inference_fn(image_path: str) -> dict:
    import time
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForCausalLM

    if not hasattr(inference_fn, "_model"):
        print(f"[{CANDIDATE_NAME}] Loading {MODEL_ID} ...")
        inference_fn._model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        inference_fn._processor = AutoProcessor.from_pretrained(
            MODEL_ID, trust_remote_code=True
        )
        print(f"[{CANDIDATE_NAME}] Model loaded.")

    model = inference_fn._model
    processor = inference_fn._processor
    device = model.device

    image = Image.open(image_path).convert("RGB")

    # --- OCR with regions ---
    prompt = "<OCR_WITH_REGION>"

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(
        device, torch.float16 if torch.cuda.is_available() else torch.float32
    )

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=4096,
            num_beams=3,
            do_sample=False,
        )

    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=False
    )[0]

    parsed = processor.post_process_generation(
        generated_text,
        task="<OCR_WITH_REGION>",
        image_size=(image.width, image.height),
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # --- Parse Florence-2 output ---
    blocks = []
    full_text_parts = []
    ocr_data = parsed.get("<OCR_WITH_REGION>", {})

    for quad, label in zip(
        ocr_data.get("quad_boxes", []), ocr_data.get("labels", [])
    ):
        # quad_boxes are [x1,y1,x2,y2,x3,y3,x4,y4] (4 corners)
        xs = quad[::2]
        ys = quad[1::2]
        bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
        blocks.append({
            "bbox": bbox,
            "text": label,
            "confidence": 1.0,  # Florence-2 doesn't expose per-word confidence
        })
        full_text_parts.append(label)

    return {
        "text": " ".join(full_text_parts),
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
        notes="Florence-2-large — 0.77B Microsoft vision foundation model.",
    )

    print(f"[{CANDIDATE_NAME}] Done. Avg latency: {result.latency_total_avg:.2f}s")
