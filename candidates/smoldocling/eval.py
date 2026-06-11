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
MODEL_ID = "docling-project/SmolDocling-256M-preview"
CANDIDATE_NAME = "smoldocling"
TEST_DATASET = PROJECT_ROOT / "benchmark" / "test_dataset"
GROUND_TRUTH = TEST_DATASET / "ground_truth.json"


# ---------------------------------------------------------------------------
# Inference function
# ---------------------------------------------------------------------------

def inference_fn(image_path: str) -> dict:
    """
    Run SmolDocling-256M on a handwritten essay image.

    Uses AutoModelForMultimodalLM + chat template as per official API.
    Output is DocTags format — preserves text, bounding boxes, reading order.
    """
    import time
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForMultimodalLM

    if not hasattr(inference_fn, "_model"):
        print(f"[{CANDIDATE_NAME}] Loading {MODEL_ID} ...")
        inference_fn._model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        inference_fn._processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)
        print(f"[{CANDIDATE_NAME}] Model loaded.")

    model = inference_fn._model
    processor = inference_fn._processor
    device = model.device

    image = Image.open(image_path).convert("RGB")

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

    # Trim input tokens to get only generated output (official API pattern)
    doctags = processor.decode(
        generated_ids[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # --- Parse DocTags into blocks ---
    # DocTags format: coordinates in 0-999 bin space (<loc_N> tokens)
    #   - Header lines: x1>y1>x2>y2>text
    #   - Tagged blocks: <text>x1>y1>x2>y2>content</text>
    #   - Other tags: <table>x1>y1>x2>y2>...</table>, etc.
    import re

    blocks = []
    full_text_parts = []

    # Get image dimensions for coordinate denormalization
    img = Image.open(image_path)
    img_w, img_h = img.size

    def denorm_coords(x1, y1, x2, y2):
        """Convert 0-999 normalized coords to pixel coords."""
        return [
            int(int(x1) / 999 * img_w),
            int(int(y1) / 999 * img_h),
            int(int(x2) / 999 * img_w),
            int(int(y2) / 999 * img_h),
        ]

    # Pattern 1: Tagged blocks with coordinates e.g. <text>68>44>433>88>content</text>
    tagged_pattern = re.compile(
        r'<(text|table|code|formula|chart|figure|caption|list|header|footer|section|title)>\s*'
        r'(\d+)>(\d+)>(\d+)>(\d+)>(.*?)</\1>',
        re.DOTALL
    )

    # Pattern 2: Bare coordinate lines e.g. "72>29>183>41>Sentence Database"
    bare_pattern = re.compile(r'^(\d+)>(\d+)>(\d+)>(\d+)>(.+)$', re.MULTILINE)

    # First extract tagged blocks
    for match in tagged_pattern.finditer(doctags):
        tag_type = match.group(1)
        x1, y1, x2, y2 = match.group(2), match.group(3), match.group(4), match.group(5)
        content = match.group(6).strip()
        if content:
            blocks.append({
                "bbox": denorm_coords(x1, y1, x2, y2),
                "text": content,
                "confidence": 1.0,
                "type": tag_type,
            })
            full_text_parts.append(content)

    # Then extract bare coordinate lines (headers/titles without wrappers)
    for match in bare_pattern.finditer(doctags):
        # Skip if this line is inside a tagged block we already captured
        line_start = match.start()
        inside_tagged = any(
            m.start() <= line_start < m.end() for m in tagged_pattern.finditer(doctags)
        )
        if not inside_tagged:
            x1, y1, x2, y2 = match.group(1), match.group(2), match.group(3), match.group(4)
            content = match.group(5).strip()
            if content:
                blocks.append({
                    "bbox": denorm_coords(x1, y1, x2, y2),
                    "text": content,
                    "confidence": 1.0,
                    "type": "header",
                })
                full_text_parts.append(content)

    # Sort blocks by reading position (top-to-bottom, left-to-right)
    blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))

    return {
        "text": "\n".join(full_text_parts) if full_text_parts else doctags,
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
        num_runs=1,  # Single pass: 25 images at ~63s each
        notes="SmolDocling-256M-preview — 256M DocTags VLM via AutoModelForMultimodalLM (transformers 5.x).",
    )

    print(f"[{CANDIDATE_NAME}] Done. Avg latency: {result.latency_total_avg:.2f}s")
