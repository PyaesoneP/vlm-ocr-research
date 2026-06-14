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
HANDWRITTEN_DIR = TEST_DATASET / "handwritten"
GROUND_TRUTH = TEST_DATASET / "ground_truth_handwritten.json"


def _get_test_images() -> list[str]:
    """Return test image paths — handwritten if available, else curated fallback."""
    img_dir = HANDWRITTEN_DIR if HANDWRITTEN_DIR.exists() else TEST_DATASET / "curated"
    return sorted([
        str(p) for p in img_dir.glob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])


def _get_ground_truth():
    """Return ground truth path if it exists, else None."""
    return GROUND_TRUTH if GROUND_TRUTH.exists() else None

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
# docTR
# ---------------------------------------------------------------------------

def _doctr_inference(image_path: str) -> dict:
    """Run docTR (PyTorch backend) on a handwritten image.

    Uses the high-level ocr_predictor API which combines detection + recognition.
    Install: pip install python-doctr[torch]
    """
    import time

    try:
        from doctr.models import ocr_predictor
    except ImportError:
        raise ImportError("python-doctr[torch] required: pip install python-doctr[torch]")

    if not hasattr(_doctr_inference, "_model"):
        print("[doctr] Loading model ...")
        _doctr_inference._model = ocr_predictor(
            det_arch="db_resnet50",
            reco_arch="crnn_vgg16_bn",
            pretrained=True,
        )
        print("[doctr] Model loaded.")

    model = _doctr_inference._model

    t0 = time.perf_counter()
    # docTR expects numpy arrays; load image and convert
    import numpy as np
    from PIL import Image
    img = np.array(Image.open(image_path).convert("RGB"))
    result = model([img])
    elapsed = time.perf_counter() - t0

    blocks = []
    full_text_parts = []
    # Get image dimensions for coordinate denormalization
    from PIL import Image
    img = Image.open(image_path)
    img_w, img_h = img.size

    page = result.pages[0]
    for block in page.blocks:
        for line in block.lines:
            for word in line.words:
                # word.geometry is ((x1,y1),(x2,y2)) relative coords [0,1]
                (x1, y1), (x2, y2) = word.geometry
                bbox = [
                    int(x1 * img_w),
                    int(y1 * img_h),
                    int(x2 * img_w),
                    int(y2 * img_h),
                ]
                text = word.value
                blocks.append({
                    "bbox": bbox,
                    "text": text,
                    "confidence": float(word.confidence),
                })
                full_text_parts.append(text)

    return {
        "text": " ".join(full_text_parts),
        "blocks": blocks,
        "stage1_latency": elapsed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    images = _get_test_images()

    if not images:
        print(f"No test images found. Add handwritten essay samples.")
        sys.exit(1)

    gt_path = _get_ground_truth()

    # --- EasyOCR ---
    if os.environ.get("EASYOCR_ENABLED"):
        try:
            result = run_candidate(
                candidate_name="easyocr",
                inference_fn=_easyocr_inference,
                test_images=images,
                ground_truth=gt_path,
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
                ground_truth=gt_path,
                notes="Tesseract 5 — traditional OCR engine.",
            )
            print(f"[tesseract] Done. Avg latency: {result.latency_total_avg:.2f}s")
        except ImportError as e:
            print(f"[tesseract] Skipped: {e}")

    # --- docTR ---
    if os.environ.get("DOCTR_ENABLED"):
        try:
            result = run_candidate(
                candidate_name="doctr",
                inference_fn=_doctr_inference,
                test_images=images,
                ground_truth=gt_path,
                notes="docTR — PyTorch detection + recognition, db_resnet50 + crnn_vgg16_bn.",
            )
            print(f"[doctr] Done. Avg latency: {result.latency_total_avg:.2f}s")
        except ImportError as e:
            print(f"[doctr] Skipped: {e}")
