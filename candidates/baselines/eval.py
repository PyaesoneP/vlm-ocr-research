"""
Traditional OCR baselines — EasyOCR, docTR, Tesseract.

These provide reference points for comparison against VLM-based approaches.

Usage:
    EASYOCR_ENABLED=1 python candidates/baselines/eval.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from candidates import run_candidate

TEST_DATASET = PROJECT_ROOT / "benchmark" / "test_dataset"
GROUND_TRUTH = TEST_DATASET / "ground_truth.json"

# ---------------------------------------------------------------------------
# EasyOCR
# ---------------------------------------------------------------------------

def _easyocr_inference(image_path: str) -> dict:
    import time
    import torch
    import easyocr

    if not hasattr(_easyocr_inference, "_reader"):
        print("[easyocr] Loading model ...")
        _easyocr_inference._reader = easyocr.Reader(
            ["en"], gpu=torch.cuda.is_available()
        )
        print("[easyocr] Model loaded.")

    reader = _easyocr_inference._reader

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    results = reader.readtext(str(image_path))

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    blocks = []
    full_text = []
    for (bbox, text, conf) in results:
        # bbox is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        blocks.append({
            "bbox": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
            "text": text,
            "confidence": float(conf),
        })
        full_text.append(text)

    return {
        "text": " ".join(full_text),
        "blocks": blocks,
        "stage1_latency": elapsed,
    }


# ---------------------------------------------------------------------------
# Tesseract
# ---------------------------------------------------------------------------

def _tesseract_inference(image_path: str) -> dict:
    import time
    from PIL import Image

    try:
        import pytesseract
    except ImportError:
        raise ImportError("pytesseract required: pip install pytesseract")

    image = Image.open(image_path)

    t0 = time.perf_counter()

    # Get bounding boxes
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    elapsed = time.perf_counter() - t0

    blocks = []
    full_text = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if not text:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        blocks.append({
            "bbox": [x, y, x + w, y + h],
            "text": text,
            "confidence": float(data["conf"][i]) / 100.0,
        })
        full_text.append(text)

    return {
        "text": " ".join(full_text),
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

    # --- EasyOCR ---
    if os.environ.get("EASYOCR_ENABLED"):
        try:
            result = run_candidate(
                candidate_name="easyocr",
                inference_fn=_easyocr_inference,
                test_images=images,
                ground_truth=GROUND_TRUTH if GROUND_TRUTH.exists() else None,
                notes="EasyOCR — 80+ language support, CRNN-based.",
            )
            print(f"[easyocr] Done. Avg latency: {result.latency_total_avg:.2f}s")
        except ImportError as e:
            print(f"[easyocr] Skipped: {e}")

    # --- Tesseract ---
    if os.environ.get("TESSERACT_ENABLED"):
        try:
            result = run_candidate(
                candidate_name="tesseract",
                inference_fn=_tesseract_inference,
                test_images=images,
                ground_truth=GROUND_TRUTH if GROUND_TRUTH.exists() else None,
                notes="Tesseract 5 — traditional OCR engine.",
            )
            print(f"[tesseract] Done. Avg latency: {result.latency_total_avg:.2f}s")
        except ImportError as e:
            print(f"[tesseract] Skipped: {e}")
