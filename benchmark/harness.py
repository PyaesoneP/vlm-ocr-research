"""
Shared evaluation harness for VLM-OCR candidate benchmarking.

Provides a consistent interface for running OCR/VLM inference and collecting
metrics (latency, VRAM, output structure) across all candidate models.

Usage (per candidate):
    from benchmark.harness import BenchmarkHarness, BenchmarkResult

    harness = BenchmarkHarness(output_dir="benchmark/results")
    result = harness.run(
        candidate_name="paddleocr_vl",
        inference_fn=my_inference_function,
        test_images=["benchmark/test_dataset/sample_01.jpg", ...],
        ground_truth="benchmark/test_dataset/ground_truth.json",
    )
    harness.save_result(result)
"""

from __future__ import annotations

import json
import time
import tracemalloc
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional

import torch


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StageMetrics:
    """Per-stage latency and memory breakdown."""
    latency_seconds: float
    vram_peak_mb: int


@dataclass
class BenchmarkResult:
    """Complete result for a single candidate evaluation."""
    candidate_name: str
    model_version: str
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    # Hardware
    gpu_name: str = ""
    vram_total_mb: int = 0

    # Latency (seconds)
    latency_total_avg: float = 0.0
    latency_total_std: float = 0.0
    latency_stage1_ocr: float = 0.0       # transcription + localization
    latency_stage2_errors: float = 0.0    # error detection + feedback

    # Accuracy
    cer: float = 0.0                      # Character Error Rate
    wer: float = 0.0                      # Word Error Rate
    reading_order_tau: float = 0.0        # Kendall's tau
    error_detection_f1: float = 0.0       # Macro-averaged F1 across error types
    bounding_box_iou: float = 0.0         # Mean IoU for error bboxes

    # Resource
    vram_peak_mb: int = 0
    throughput_ppm: float = 0.0           # Pages per minute

    # Qualitative scores (1–5)
    setup_complexity: int = 0
    flexibility: int = 0

    # Raw outputs (for inspection)
    sample_outputs: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class BenchmarkHarness:
    """Orchestrates evaluation of a single OCR/VLM candidate."""

    def __init__(self, output_dir: str | Path = "benchmark/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # GPU helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_gpu_info() -> tuple[str, int]:
        """Return (gpu_name, vram_total_mb)."""
        if not torch.cuda.is_available():
            return "CPU", 0
        name = torch.cuda.get_device_name(0)
        total_mb = torch.cuda.get_device_properties(0).total_mem // (1024 * 1024)
        return name, total_mb

    @staticmethod
    def reset_gpu_memory() -> None:
        """Clear CUDA cache and reset peak memory stats."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    @staticmethod
    def get_peak_vram_mb() -> int:
        """Return peak VRAM usage in MB since last reset."""
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.max_memory_allocated(0) // (1024 * 1024)

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    @staticmethod
    def timed_call(fn: Callable, *args: Any, **kwargs: Any) -> tuple[Any, float]:
        """Call fn and return (result, elapsed_seconds)."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        return result, elapsed

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(
        self,
        candidate_name: str,
        inference_fn: Callable[[str | Path], dict[str, Any]],
        test_images: list[str | Path],
        ground_truth: Optional[str | Path] = None,
        num_warmup: int = 2,
        num_runs: int = 10,
    ) -> BenchmarkResult:
        """
        Evaluate a candidate model.

        Parameters
        ----------
        candidate_name : str
            Identifier (e.g. "paddleocr_vl").
        inference_fn : callable
            Function that takes an image path and returns a dict:
                {
                    "text": str,                     # full transcription
                    "blocks": [                      # per-text-block info
                        {
                            "bbox": [x1,y1,x2,y2],
                            "text": str,
                            "confidence": float,
                            "reading_order": int,
                        },
                        ...
                    ],
                    "stage1_latency": float,          # seconds (optional)
                    "stage2_latency": float,          # seconds (optional)
                }
        test_images : list
            Paths to test images.
        ground_truth : path, optional
            Path to ground-truth JSON file.
        num_warmup : int
            Warmup runs (excluded from metrics).
        num_runs : int
            Measured runs for averaging.

        Returns
        -------
        BenchmarkResult
        """
        gpu_name, vram_total = self.get_gpu_info()

        result = BenchmarkResult(
            candidate_name=candidate_name,
            model_version="",
            gpu_name=gpu_name,
            vram_total_mb=vram_total,
        )

        # --- Warmup ---
        for i in range(min(num_warmup, len(test_images))):
            self.reset_gpu_memory()
            inference_fn(str(test_images[i]))

        # --- Measured runs ---
        latencies: list[float] = []
        stage1_latencies: list[float] = []
        stage2_latencies: list[float] = []
        vram_peaks: list[int] = []
        all_outputs: list[dict[str, Any]] = []

        for i in range(num_runs):
            img_path = str(test_images[i % len(test_images)])
            self.reset_gpu_memory()

            output, elapsed = self.timed_call(inference_fn, img_path)
            latencies.append(elapsed)
            vram_peaks.append(self.get_peak_vram_mb())
            all_outputs.append(output)

            if "stage1_latency" in output:
                stage1_latencies.append(output["stage1_latency"])
            if "stage2_latency" in output:
                stage2_latencies.append(output["stage2_latency"])

        # --- Aggregate ---
        n = len(latencies)
        result.latency_total_avg = sum(latencies) / n
        result.latency_total_std = (
            (sum((x - result.latency_total_avg) ** 2 for x in latencies) / n) ** 0.5
        )
        if stage1_latencies:
            result.latency_stage1_ocr = sum(stage1_latencies) / len(stage1_latencies)
        if stage2_latencies:
            result.latency_stage2_errors = sum(stage2_latencies) / len(stage2_latencies)

        result.vram_peak_mb = max(vram_peaks) if vram_peaks else 0
        result.throughput_ppm = 60.0 / result.latency_total_avg if result.latency_total_avg > 0 else 0.0
        result.sample_outputs = all_outputs[:3]  # Keep first 3 for inspection

        # --- Ground-truth evaluation (if provided) ---
        if ground_truth:
            self._evaluate_accuracy(result, all_outputs, Path(ground_truth))

        return result

    def _evaluate_accuracy(
        self,
        result: BenchmarkResult,
        outputs: list[dict[str, Any]],
        gt_path: Path,
    ) -> None:
        """Compute CER, WER, reading-order tau, error F1, IoU from ground truth."""
        from benchmark.metrics import (
            compute_cer,
            compute_error_detection_f1,
            compute_iou,
            compute_reading_order_tau,
            compute_wer,
        )

        if not gt_path.exists():
            result.notes += "[warn] ground-truth file not found; accuracy metrics skipped. "
            return

        gt_data = json.loads(gt_path.read_text())

        # Accumulate across samples
        cer_vals, wer_vals, tau_vals, f1_vals, iou_vals = [], [], [], [], []

        for idx, output in enumerate(outputs):
            gt = gt_data[idx] if idx < len(gt_data) else None
            if gt is None:
                continue

            pred_text = output.get("text", "")
            gt_text = gt.get("text", "")
            cer_vals.append(compute_cer(pred_text, gt_text))
            wer_vals.append(compute_wer(pred_text, gt_text))

            if "reading_order" in output and "reading_order" in gt:
                tau_vals.append(compute_reading_order_tau(output["reading_order"], gt["reading_order"]))

            if "errors" in output and "errors" in gt:
                f1_vals.append(compute_error_detection_f1(output["errors"], gt["errors"]))
                iou_vals.append(compute_iou(output["errors"], gt["errors"]))

        if cer_vals:
            result.cer = sum(cer_vals) / len(cer_vals)
        if wer_vals:
            result.wer = sum(wer_vals) / len(wer_vals)
        if tau_vals:
            result.reading_order_tau = sum(tau_vals) / len(tau_vals)
        if f1_vals:
            result.error_detection_f1 = sum(f1_vals) / len(f1_vals)
        if iou_vals:
            result.bounding_box_iou = sum(iou_vals) / len(iou_vals)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_result(self, result: BenchmarkResult) -> Path:
        """Save result as JSON in the output directory."""
        out_path = self.output_dir / f"{result.candidate_name}.json"
        out_path.write_text(json.dumps(asdict(result), indent=2, default=str))
        return out_path

    @staticmethod
    def load_result(path: str | Path) -> BenchmarkResult:
        """Load a previously saved BenchmarkResult from JSON."""
        data = json.loads(Path(path).read_text())
        return BenchmarkResult(**data)
