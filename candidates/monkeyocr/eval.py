"""
MonkeyOCR-pro-1.2B — Candidate evaluation script via llama.cpp server.

MonkeyOCR is a 1.2B vision-language model (Qwen2-VL based) fine-tuned for
challenging OCR. Uses pre-built llama.cpp binaries with Vulkan GPU backend
running as llama-server with OpenAI-compatible API.

Model: dinhquangson/MonkeyOCR-pro-1.2B-Vision-GGUF
Backend: llama.cpp llama-server (Vulkan GPU)
Usage:
    # Start server first:
    #   cd /tmp/llama-b9596 && ./llama-server \
    #     --hf-repo dinhquangson/MonkeyOCR-pro-1.2B-Vision-GGUF \
    #     --hf-file MonkeyOCR-pro-1.2B-Recognition.gguf \
    #     --host 0.0.0.0 --port 8080 --n-gpu-layers 99 --ctx-size 4096
    python candidates/monkeyocr/eval.py
"""

from __future__ import annotations

import base64
import json
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from candidates import run_candidate

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SERVER_URL = "http://localhost:8080/v1/chat/completions"
CANDIDATE_NAME = "monkeyocr"
TEST_DATASET = PROJECT_ROOT / "benchmark" / "test_dataset"
GROUND_TRUTH = TEST_DATASET / "ground_truth.json"

OCR_PROMPT = (
    "Extract all visible text from this handwritten document. "
    "Output only the transcription, no commentary."
)


# ---------------------------------------------------------------------------
# Inference function
# ---------------------------------------------------------------------------

def inference_fn(image_path: str) -> dict:
    """
    Send image to llama-server API for OCR transcription.

    llama-server provides an OpenAI-compatible /v1/chat/completions endpoint
    that natively handles Qwen2-VL multimodal input.
    """
    # Read and encode image
    img_bytes = Path(image_path).read_bytes()
    img_b64 = base64.b64encode(img_bytes).decode()

    payload = {
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": OCR_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            ],
        }],
        "max_tokens": 2048,
        "temperature": 0.0,
    }

    t0 = time.perf_counter()

    req = urllib.request.Request(
        SERVER_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=120)
    data = json.loads(resp.read())

    elapsed = time.perf_counter() - t0
    text = data["choices"][0]["message"]["content"].strip()

    return {
        "text": text,
        "blocks": [],
        "stage1_latency": elapsed,
        "reading_order": [],
        "errors": [],
        "stage2_latency": 0.0,
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
        print(f"No test images in {TEST_DATASET / 'curated'}. Add handwritten essay samples.")
        sys.exit(1)

    # Quick connectivity check
    try:
        urllib.request.urlopen("http://localhost:8080/health", timeout=5)
    except Exception:
        print("[monkeyocr] ERROR: llama-server not reachable at localhost:8080")
        print("  Start it first: cd /tmp/llama-b9596 && ./llama-server --hf-repo ...")
        sys.exit(1)

    result = run_candidate(
        candidate_name=CANDIDATE_NAME,
        inference_fn=inference_fn,
        test_images=images,
        ground_truth=GROUND_TRUTH if GROUND_TRUTH.exists() else None,
        num_warmup=1,
        num_runs=1,
        notes="llama.cpp b9596 llama-server (Vulkan). Qwen2-VL GGUF Q4_K_M. "
              "CPU-only fallback (Vulkan not detected on RTX 5070 Ti).",
    )

    from benchmark.harness import BenchmarkHarness

    harness = BenchmarkHarness(output_dir=PROJECT_ROOT / "benchmark" / "results")
    harness.save_result(result)
    print(f"[{CANDIDATE_NAME}] Done. Avg latency: {result.latency_total_avg:.2f}s")

if __name__ == "__main__":
    images = sorted([
        str(p) for p in (TEST_DATASET / "curated").glob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])

    if not images:
        print(f"No test images in {TEST_DATASET / 'curated'}. Add handwritten essay samples.")
        sys.exit(1)

    result = run_candidate(
        candidate_name=CANDIDATE_NAME,
        inference_fn=inference_fn,
        test_images=images,
        ground_truth=GROUND_TRUTH if GROUND_TRUTH.exists() else None,
        num_warmup=1,
        num_runs=1,
        notes="GGUF Q4_K_M quant via llama-cpp-python CUDA (n_gpu_layers=-1). "
              "llama.cpp vision does not return bounding boxes.",
    )

    from benchmark.harness import BenchmarkHarness

    harness = BenchmarkHarness(output_dir=PROJECT_ROOT / "benchmark" / "results")
    harness.save_result(result)
    print(f"[{CANDIDATE_NAME}] Done. Avg latency: {result.latency_total_avg:.2f}s")
