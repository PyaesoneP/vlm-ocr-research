"""
PaddleOCR-VL-1.6 — Candidate evaluation script.

PaddleOCR-VL is a 0.9B VLM specifically designed for document parsing.
It achieves 96.3% on OmniDocBench v1.6 and supports structured output
in Markdown/JSON with bounding boxes.

Model: https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6

On NVIDIA Blackwell GPUs (sm_120), PaddleOCR-VL uses the PaddlePaddle
native inference engine (NOT HuggingFace transformers).  The PaddlePaddle
framework has dedicated CUDA 12.9+ support for Blackwell.

Reference: https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL-NVIDIA-Blackwell.html

Usage:
    source .venv/bin/activate
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

CANDIDATE_NAME = "paddleocr_vl"
TEST_DATASET = PROJECT_ROOT / "benchmark" / "test_dataset"
GROUND_TRUTH = TEST_DATASET / "ground_truth.json"


# ---------------------------------------------------------------------------
# Inference function
# ---------------------------------------------------------------------------

def inference_fn(image_path: str) -> dict:
    """
    Run PaddleOCR-VL-1.6 via the native PaddleOCR pipeline API.

    On Blackwell GPUs, PaddleOCR-VL uses PaddlePaddle's native inference engine
    with dedicated sm_120 support.  This is the ONLY supported path for Blackwell;
    the HuggingFace transformers path crashes with a rope_type KeyError.

    Returns a dict conforming to the harness output spec:
        {text, blocks, stage1_latency, ...}
    """
    import time

    # --- Lazy-load the PaddleOCR-VL pipeline (cached at module level) ---
    if not hasattr(inference_fn, "_pipeline"):
        print(f"[{CANDIDATE_NAME}] Loading PaddleOCR-VL-1.6 (PaddlePaddle native) ...")
        from paddleocr import PaddleOCRVL

        inference_fn._pipeline = PaddleOCRVL(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
        print(f"[{CANDIDATE_NAME}] Pipeline loaded.")

    pipeline = inference_fn._pipeline

    # --- Inference ---
    import paddle
    paddle.device.synchronize()
    t0 = time.perf_counter()

    output = pipeline.predict(str(image_path))

    paddle.device.synchronize()
    elapsed = time.perf_counter() - t0

    # --- Parse PaddleOCR-VL structured output ---
    # PaddleOCRVL.predict() returns list of result objects with:
    #   res.json['res']['parsing_res_list'] → blocks with block_bbox, block_content
    #   res.markdown['markdown_texts']        → full markdown text
    #   res.json['res']['layout_det_res']     → layout detection results
    blocks = []
    full_text = ""

    if output:
        for res in output:
            # Extract from JSON (has bboxes and layout)
            if hasattr(res, "json") and res.json:
                j = res.json
                # res.json is {"res": {...}}
                inner = j.get("res", j) if isinstance(j, dict) else {}
                parsing_list = inner.get("parsing_res_list", [])
                for item in parsing_list:
                    text = item.get("block_content", "")
                    bbox_str = item.get("block_bbox", "")
                    if text and bbox_str:
                        # bbox is stored as string "[x1, y1, x2, y2]"
                        try:
                            bbox = [int(float(x.strip())) for x in bbox_str.strip("[]").split(",")]
                        except (ValueError, AttributeError):
                            bbox = [0, 0, 0, 0]
                        blocks.append({
                            "bbox": bbox if len(bbox) == 4 else [0, 0, 0, 0],
                            "text": str(text),
                            "confidence": 1.0,
                        })
                        full_text += str(text) + " "

            # Fallback: markdown text
            if not full_text and hasattr(res, "markdown") and res.markdown:
                md = res.markdown
                if isinstance(md, dict):
                    full_text = md.get("markdown_texts", str(md))
                else:
                    full_text = str(md)

            # Fallback: plain text
            if not full_text and hasattr(res, "text") and res.text:
                full_text = str(res.text)

    return {
        "text": full_text.strip(),
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
        notes="PaddleOCR-VL-1.6 — 0.9B document VLM (PaddlePaddle native).",
    )

    print(f"[{CANDIDATE_NAME}] Done. Avg latency: {result.latency_total_avg:.2f}s, CER: {result.cer:.4f}")

    print(f"[{CANDIDATE_NAME}] Done. Avg latency: {result.latency_total_avg:.2f}s")
