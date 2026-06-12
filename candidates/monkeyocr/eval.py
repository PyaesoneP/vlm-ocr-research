"""
MonkeyOCR-pro-1.2B — Candidate evaluation script via llama.cpp server.

MonkeyOCR is a 1.2B vision-language model (Qwen2-VL based) fine-tuned for
challenging OCR. Uses llama.cpp llama-server built from source with CUDA
GPU backend, providing an OpenAI-compatible API.

Model: dinhquangson/MonkeyOCR-pro-1.2B-Vision-GGUF
Backend: llama.cpp llama-server (CUDA GPU, built from source)

⚠ Bounding box output: MonkeyOCR via llama-server returns text only.
The underlying Qwen2-VL model supports grounding, but the llama.cpp chat
API provides text-only output and MonkeyOCR's OCR fine-tuning removes
grounding capabilities.  No native bbox output is available.

For bbox support, consider: Florence-2, SmolDocling, or Nemotron OCR v2.

GPU Setup (build llama.cpp from source with CUDA):
    git clone https://github.com/ggerganov/llama.cpp.git
    cd llama.cpp
    cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j$(nproc) --target llama-server

Usage:
    # Start server (GPU-accelerated with CUDA):
    cd llama.cpp/build/bin && LD_LIBRARY_PATH=. ./llama-server \\
      -hf dinhquangson/MonkeyOCR-pro-1.2B-Vision-GGUF \\
      --host 0.0.0.0 --port 8080 -ngl 99 -c 8192 \\
      --mmproj-offload --image-min-tokens 1024

    # Then evaluate:
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
    "Transcribe every word visible on this page, both printed and handwritten. "
    "Do not repeat yourself. Stop when you reach the end of the visible text."
)


# ---------------------------------------------------------------------------
# Inference function
# ---------------------------------------------------------------------------

def inference_fn(image_path: str) -> dict:
    """
    Send image to llama-server API for OCR transcription.

    llama-server provides an OpenAI-compatible /v1/chat/completions endpoint
    that natively handles Qwen2-VL multimodal input.

    Returns ``blocks`` with empty bboxes (``[0,0,0,0]``) — this model
    does not support native bounding box output via llama-server.
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
        "max_tokens": 4096,
        "temperature": 0.0,
        "repeat_penalty": 1.1,
        "stop": ["\n\n\n", "Name:"],  # prevent form-field hallucination loops
    }

    t0 = time.perf_counter()

    req = urllib.request.Request(
        SERVER_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=300)
    data = json.loads(resp.read())

    elapsed = time.perf_counter() - t0
    text = data["choices"][0]["message"]["content"].strip()

    # --- Parse blocks (text-only, no bboxes) ---
    blocks = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            blocks.append({
                "bbox": [0, 0, 0, 0],  # MonkeyOCR does not output bboxes
                "text": line,
                "confidence": 1.0,
            })

    return {
        "text": text,
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
        print(f"No test images in {TEST_DATASET / 'curated'}. Add handwritten essay samples.")
        sys.exit(1)

    # Quick connectivity check
    try:
        urllib.request.urlopen("http://localhost:8080/health", timeout=5)
    except Exception:
        print("[monkeyocr] ERROR: llama-server not reachable at localhost:8080")
        print("  Build and start it first (see docstring for GPU setup instructions).")
        sys.exit(1)

    result = run_candidate(
        candidate_name=CANDIDATE_NAME,
        inference_fn=inference_fn,
        test_images=images,
        ground_truth=GROUND_TRUTH if GROUND_TRUTH.exists() else None,
        num_warmup=1,
        num_runs=1,
        notes="llama.cpp b9596 llama-server. Qwen2-VL GGUF Q4_K_M. "
              "Server: ctx-size=8192, image-min-tokens=1024. "
              "CPU-only (Vulkan not detected on RTX 5070 Ti). "
              "Text-only output — no native bbox support.",
    )

    from benchmark.harness import BenchmarkHarness

    harness = BenchmarkHarness(output_dir=PROJECT_ROOT / "benchmark" / "results")
    harness.save_result(result)
    print(f"[{CANDIDATE_NAME}] Done. Avg latency: {result.latency_total_avg:.2f}s")
