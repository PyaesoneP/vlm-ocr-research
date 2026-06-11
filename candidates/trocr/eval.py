"""
TrOCR (Handwritten) — Candidate evaluation script.

TrOCR is Microsoft's Transformer-based OCR model, fine-tuned on the IAM
handwriting dataset.  Note: TrOCR works on *single text-line images*, so
it needs a separate text detector (e.g., CRAFT, DBNet) for full-page input.

Models:
  - microsoft/trocr-base-handwritten  (0.3B)
  - microsoft/trocr-large-handwritten (0.6B)

Usage:
    python candidates/trocr/eval.py
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

# Switch between base and large by changing MODEL_ID
MODEL_ID = "microsoft/trocr-large-handwritten"
CANDIDATE_NAME = "trocr_large_handwritten"
TEST_DATASET = PROJECT_ROOT / "benchmark" / "test_dataset"
GROUND_TRUTH = TEST_DATASET / "ground_truth.json"


# ---------------------------------------------------------------------------
# Inference function
# ---------------------------------------------------------------------------

def inference_fn(image_path: str) -> dict:
    """
    Run TrOCR on a full handwritten page.

    Since TrOCR is line-level, this implementation:
      1. Uses a simple heuristic text-line detector (vertical projection)
      2. Runs TrOCR on each detected line
      3. Assembles results

    For production use, replace the heuristic detector with CRAFT or DBNet.
    """
    import time
    import torch
    import numpy as np
    from PIL import Image
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    if not hasattr(inference_fn, "_model"):
        print(f"[{CANDIDATE_NAME}] Loading {MODEL_ID} ...")
        inference_fn._processor = TrOCRProcessor.from_pretrained(MODEL_ID)
        inference_fn._model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)
        if torch.cuda.is_available():
            inference_fn._model = inference_fn._model.to("cuda")
        inference_fn._model.eval()
        print(f"[{CANDIDATE_NAME}] Model loaded.")

    processor = inference_fn._processor
    model = inference_fn._model
    device = next(model.parameters()).device

    image = Image.open(image_path).convert("RGB")
    img_array = np.array(image.convert("L"))  # Grayscale for detection

    # --- Heuristic line detection ---
    lines = _detect_text_lines(img_array)
    if not lines:
        # Fallback: treat whole image as one line
        lines = [(0, 0, image.width, image.height)]

    blocks = []
    full_text_parts = []

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for (x1, y1, x2, y2) in lines:
        line_img = image.crop((x1, y1, x2, y2))
        pixel_values = processor(images=line_img, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
        line_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        blocks.append({
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "text": line_text,
            "confidence": 1.0,
        })
        full_text_parts.append(line_text)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return {
        "text": "\n".join(full_text_parts),
        "blocks": blocks,
        "stage1_latency": elapsed,
    }


# ---------------------------------------------------------------------------
# Heuristic line detector (placeholder — replace with CRAFT/DBNet)
# ---------------------------------------------------------------------------

def _detect_text_lines(
    gray: "np.ndarray", min_line_height: int = 12, gap_threshold: float = 0.95
) -> list[tuple[int, int, int, int]]:
    """
    Detect text lines using horizontal projection profile.

    Returns list of (x1, y1, x2, y2) bounding boxes.
    """
    # Binarize
    binary = (gray < 128).astype(np.uint8) * 255

    # Horizontal projection (row-wise sum of black pixels)
    h_proj = binary.sum(axis=1) / 255

    # Find rows with text
    in_line = False
    line_start = 0
    lines = []

    height = len(h_proj)
    for y in range(height):
        has_text = h_proj[y] > (binary.shape[1] * (1 - gap_threshold))

        if has_text and not in_line:
            line_start = y
            in_line = True
        elif not has_text and in_line:
            if y - line_start >= min_line_height:
                lines.append((0, line_start, binary.shape[1], y))
            in_line = False

    # Last line
    if in_line and height - line_start >= min_line_height:
        lines.append((0, line_start, binary.shape[1], height))

    return lines


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
        notes="TrOCR-large fine-tuned on IAM handwriting dataset.",
    )

    print(f"[{CANDIDATE_NAME}] Done. Avg latency: {result.latency_total_avg:.2f}s")
