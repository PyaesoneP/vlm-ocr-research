"""
Candidate evaluation scaffold.

Each candidate directory under `candidates/` should contain an `eval.py` script
that:
  1. Loads the model
  2. Defines an `inference_fn(image_path) -> dict` conforming to the harness
     output spec
  3. Calls `BenchmarkHarness.run(...)` and saves the result

This file provides a common entry-point template and helper utilities.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.harness import BenchmarkHarness, BenchmarkResult
from benchmark.metrics import compute_cer, compute_wer


# ---------------------------------------------------------------------------
# Expected inference_fn output schema
# ---------------------------------------------------------------------------

EXPECTED_OUTPUT_SCHEMA = """
{
    "text": str,                      # Full page transcription
    "blocks": [                       # Per-text-block details
        {
            "bbox": [x1, y1, x2, y2],
            "text": str,
            "confidence": float,
            "reading_order": int,     # 0-based position in reading order
        }
    ],
    "stage1_latency": float,          # Seconds (OCR/transcription only)
    "reading_order": [int, ...],      # Rank per block (optional, for scoring)
    "errors": [                       # Detected writing errors
        {
            "type": "capitalization|spelling|grammar|punctuation|structural",
            "bbox": [x1, y1, x2, y2],
            "description": str,
        }
    ],
    "stage2_latency": float,          # Seconds (error detection/feedback only)
}
"""


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_candidate(
    candidate_name: str,
    inference_fn,
    test_images: list[str | Path],
    ground_truth: str | Path | None = None,
    num_warmup: int = 2,
    num_runs: int = 10,
    notes: str = "",
) -> BenchmarkResult:
    """
    Run a full benchmark for one candidate.

    Parameters
    ----------
    candidate_name : str
        Shorthand identifier (e.g. "paddleocr_vl").
    inference_fn : callable
        Function image_path -> output dict (see EXPECTED_OUTPUT_SCHEMA).
    test_images : list
        Paths to test images.
    ground_truth : path, optional
        Path to ground-truth JSON.
    num_warmup, num_runs : int
        Warmup and measurement run counts.
    notes : str
        Free-form notes stored in the result.

    Returns
    -------
    BenchmarkResult
    """
    harness = BenchmarkHarness()
    result = harness.run(
        candidate_name=candidate_name,
        inference_fn=inference_fn,
        test_images=test_images,
        ground_truth=ground_truth,
        num_warmup=num_warmup,
        num_runs=num_runs,
    )
    result.notes = notes
    harness.save_result(result)
    return result
