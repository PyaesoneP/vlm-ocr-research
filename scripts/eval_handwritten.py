#!/usr/bin/env python3
"""
Evaluate OCR candidates on cropped (handwriting-only) IAM images.

Each candidate's existing eval script is reused — only the image directory and
ground truth file are overridden to point at the cropped handwritten region.

Usage:
    python scripts/eval_handwritten.py smoldocling
    python scripts/eval_handwritten.py nemotron_ocr_v2  # requires conda aiml
    python scripts/eval_handwritten.py got_ocr2
    python scripts/eval_handwritten.py monkeyocr          # requires llama-server
    python scripts/eval_handwritten.py --all               # all completed candidates
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.harness import BenchmarkHarness
from candidates import run_candidate

# ---------------------------------------------------------------------------
# Paths (handwriting-only)
# ---------------------------------------------------------------------------

HANDWRITTEN_IMAGES_DIR = PROJECT_ROOT / "benchmark" / "test_dataset" / "handwritten"
HANDWRITTEN_GT = PROJECT_ROOT / "benchmark" / "test_dataset" / "ground_truth_handwritten.json"
RESULTS_DIR = PROJECT_ROOT / "benchmark" / "results"

CANDIDATE_CONFIG = {
    "smoldocling": {
        "module": "candidates.smoldocling.eval",
        "notes": "SmolDocling-256M on cropped handwritten region. "
                 "DocTags output may be affected by partial-page input.",
    },
    "nemotron_ocr_v2": {
        "module": "candidates.nemotron_ocr.eval",
        "notes": "Nemotron OCR v2 on cropped handwritten region. "
                 "Requires `conda activate aiml`.",
    },
    "got_ocr2": {
        "module": "candidates.got_ocr.eval",
        "notes": "GOT-OCR2.0 on cropped handwritten region.",
    },
    "monkeyocr": {
        "module": "candidates.monkeyocr.eval",
        "notes": "MonkeyOCR on cropped handwritten region. "
                 "Requires llama-server running on port 8080.",
    },
    "florence2_large": {
        "module": "candidates.florence2.eval",
        "notes": "Florence-2-large on cropped handwritten region. "
                 "Requires `conda activate florencetf` (transformers 4.40.0).",
    },
    "paddleocr_vl": {
        "module": "candidates.paddleocr_vl.eval",
        "notes": "PaddleOCR-VL-1.6 on cropped handwritten region. "
                 "Requires `source .venv_paddleocr/bin/activate` and LD_LIBRARY_PATH=/usr/lib/wsl/lib.",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_handwritten_images() -> list[str]:
    """Return sorted list of cropped image paths."""
    if not HANDWRITTEN_IMAGES_DIR.exists():
        print(f"ERROR: Cropped images not found at {HANDWRITTEN_IMAGES_DIR}")
        print("Run: python scripts/crop_handwritten.py")
        sys.exit(1)

    images = sorted([
        str(p) for p in HANDWRITTEN_IMAGES_DIR.glob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])
    if not images:
        print(f"ERROR: No images in {HANDWRITTEN_IMAGES_DIR}")
        sys.exit(1)
    return images


def eval_candidate(name: str) -> None:
    """Evaluate one candidate on handwritten-only images."""
    if name not in CANDIDATE_CONFIG:
        print(f"Unknown candidate: {name}")
        print(f"Available: {list(CANDIDATE_CONFIG.keys())}")
        sys.exit(1)

    config = CANDIDATE_CONFIG[name]
    images = get_handwritten_images()

    print(f"[eval_handwritten] {name} — {len(images)} cropped images")
    print(f"  GT: {HANDWRITTEN_GT}")

    # MonkeyOCR: check server
    if name == "monkeyocr":
        import urllib.request
        try:
            urllib.request.urlopen("http://localhost:8080/health", timeout=5)
        except Exception:
            print("[monkeyocr] ERROR: llama-server not reachable at localhost:8080")
            print("  Start: cd /tmp/llama-b9596 && ./llama-server --hf-repo ... --ctx-size 8192 --image-min-tokens 1024")
            sys.exit(1)

    # Import the candidate's inference_fn
    module = importlib.import_module(config["module"])

    result = run_candidate(
        candidate_name=f"{name}_handwritten",
        inference_fn=module.inference_fn,
        test_images=images,
        ground_truth=HANDWRITTEN_GT if HANDWRITTEN_GT.exists() else None,
        num_warmup=1,
        num_runs=1,
        notes=config["notes"],
    )

    # Save result
    harness = BenchmarkHarness(output_dir=RESULTS_DIR)
    harness.save_result(result)
    print(f"[eval_handwritten] {name} done. "
          f"Latency: {result.latency_total_avg:.2f}s, "
          f"CER: {result.cer:.3f}, WER: {result.wer:.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate OCR on handwritten-only images")
    parser.add_argument("candidate", nargs="?", default=None,
                        help=f"Candidate name: {list(CANDIDATE_CONFIG.keys())}")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all completed candidates")
    args = parser.parse_args()

    if args.all:
        for name in CANDIDATE_CONFIG:
            try:
                eval_candidate(name)
            except Exception as e:
                print(f"[eval_handwritten] {name} FAILED: {e}")
                import traceback
                traceback.print_exc()
    elif args.candidate:
        eval_candidate(args.candidate)
    else:
        parser.print_help()
