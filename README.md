# VLM-OCR Research: Handwritten English Essay Feedback

[![Status: Research](https://img.shields.io/badge/status-research-blue)](.)
[![Hardware: RTX 5070 Ti](https://img.shields.io/badge/hardware-RTX%205070%20Ti%20(12GB)-green)](.)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-orange)](./LICENSE)

Empirical evaluation of open-source OCR models and Vision-Language Models (VLMs) for a **handwritten English essay feedback system**. The pipeline must (1) transcribe difficult handwriting, (2) detect and localize writing errors with bounding boxes, (3) determine reading order on unruled paper, and (4) generate natural-language feedback, matching or beating a commercial cloud pipeline (Google Document AI + Gemini) on accuracy while running fully local, at lower latency and zero marginal cost. Every finding in this document is implemented and measured locally; nothing is theoretical.

---

## Contents

- [Status at a Glance](#status-at-a-glance)
- [Cloud Reference Pipeline](#cloud-reference-pipeline)
- [Constraints](#constraints)
- [Environments](#environments)
  - [PaddleOCR-VL on Blackwell](#paddleocr-vl-on-blackwell-nvidia-sm_120)
  - [MonkeyOCR GPU Setup](#monkeyocr-gpu-setup-llamacpp-from-source)
- [Project Structure](#project-structure)
- [Candidate Models](#candidate-models)
  - [Tier 1: Most Promising](#tier-1-most-promising)
  - [Tier 2: Baselines & Comparison](#tier-2-baselines--comparison)
- [Architecture Insight](#architecture-insight)
- [Research Phases](#research-phases)
  - [Phase 1: Environment Setup & Baseline](#phase-1-environment-setup--baseline-complete)
  - [Phase 2: Tier-1 Candidate Evaluation](#phase-2-tier-1-candidate-evaluation-complete-66-evaluated)
  - [Handwriting-Only Re-Evaluation](#handwriting-only-re-evaluation-xml-guided-cropping)
  - [Phase 3-8](#phase-3-tier-2-candidate-evaluation)
- [Metrics Framework](#metrics-framework)
- [How Each Metric Is Computed](#how-each-metric-is-computed)
  - [CER](#cer-character-error-rate)
  - [WER](#wer-word-error-rate)
  - [Bounding Box IoU](#bounding-box-iou)
  - [Reading Order: Kendall's τ-b](#reading-order-kendalls-%CF%84-b)
- [Results (Live)](#results-live)
- [Further Considerations](#further-considerations)
- [License](#license)

---

## Status at a Glance

**Phase 3 complete — 13/13 candidates evaluated.** Cloud baseline beaten on every metric.

### Handwriting CER Leaderboard

| # | Model | CER | Type | Notes |
|---|---|---|---|---|
| 1 | **Hunyuan VL** | **0.015** | Text-only | Manual (lmarena), 5 images |
| 2 | **Qwen3-VL-4B** | **0.022** | Line bbox | Local, 14s/image, 9.6 GB VRAM |
| 3 | **Qwen3-VL-8B** | **0.035** | Word bbox | API (novita), 27s/image |
| 4 | **PaddleOCR-VL-1.6** | **0.045** | Layout bbox | Docker-only on Blackwell, 32s |
| 5 | **Florence-2-large** | **0.061** | Line bbox | 1.05s/image, 2.0 GB VRAM |
| 6 | GOT-OCR2.0 | 0.088 | Text-only | 2.5s/image |
| 7 | Google Doc AI | 0.108 | Word bbox | Cloud, 3.6s/image |

### Word-Level IoU (7 models, 1891 GT words)

| Model | Word IoU | CER | Notes |
|---|---|---|---|
| Tesseract 5 | **0.812** | 0.443 | Best bbox precision, unusable CER |
| **Qwen3-VL-8B** | **0.722** | **0.035** | Best CER/IoU balance |
| Qwen3-VL-4B | 0.718 | 0.049 | Local, ~80s/image |
| Google Doc AI | 0.611 | 0.108 | Cloud baseline |
| EasyOCR | 0.597 | 0.625 | |
| docTR | 0.581 | 0.275 | |
| Florence-2-large | 0.176* | 0.061 | Line bboxes only |

### Verdict

- **Best automatable model:** Qwen3-VL-8B (CER 0.035 + word IoU 0.722 via API; 4B runs locally at CER 0.049 + IoU 0.718)
- **Best overall speed/accuracy:** Florence-2-large (CER 0.061 @ 1.05s, but line bboxes only)
- **Cloud baseline beaten:** Doc AI loses on both CER (0.108 vs 0.035) and word IoU (0.611 vs 0.722)
- **Stage 2 (error detection) is now the bottleneck** — OCR is solved; Gemini at 12.4s is the open problem

> **2026-06-12, Evaluation bug fixes applied.** A code review identified several metric bugs (τ always 1.0, single-image CERs, Nemotron bbox double-conversion, missing crop-offset transform). All numbers above reflect corrected evaluations on the full 25-image dataset. See commit history for details.

> **Read this before interpreting any numbers:** IAM evaluation forms contain the same text both **machine-printed** and **handwritten**, so full-form CER/WER cannot reveal which region a model actually read. Only the [handwriting-only re-evaluation](#handwriting-only-re-evaluation-xml-guided-cropping) on XML-cropped images measures true handwriting recognition. Full-form Phase 2 numbers are retained as printed-text OCR benchmarks. Details: [IAM Dataset Structure](#iam-dataset-structure).

**Next up:** Phase 4 (pipeline assembly), Phase 5 (reading-order deep-dive), and Phase 6 (error-detection accuracy). All 13 candidates evaluated. Hunyuan VL (CER 0.015) is new #1 by CER but manual-only. Qwen3-VL-8B (word CER 0.035, IoU 0.722) is best automatable model for Stage 1+2.

---

## Cloud Reference Pipeline

| Metric | Reference (benchmarked independently, see `benchmark/baseline.py`) |
|--------|-----------------------------------------------------------------------|
| End-to-end latency | **16.7s** (Doc AI 2.8s + Gemini 3.5 Flash 12.4s) |
| Cost | Document AI: $1.50-$30/1K pages + Gemini API per-token |
| Architecture | Cloud-only, proprietary |
| Model | `gemini-3.5-flash` via Vertex AI (`location="global"`) |

## Constraints

- **Research phase:** RTX 5070 Ti mobile (12 GB VRAM). Models must fit locally (quantization allowed).
- **Production scenario:** The final recommendation assumes larger compute budgets and permits cloud APIs alongside local models.
- **Transformers version:** 5.8.1 (upgraded from 4.57.6 for SmolDocling `AutoModelForMultimodalLM` support and PaddleOCR-VL compatibility).

## Environments

Four separate Python environments are required due to conflicting CUDA/transformers/PaddlePaddle versions:

| Environment | Type | PyTorch | CUDA | Used for |
|---|---|---|---|---|
| `.venv/` | venv | 2.11.0+cu130 | 13.0 | SmolDocling, GOT-OCR2.0, MonkeyOCR, DocLayoutYOLO, Qwen3-VL, TrOCR, baselines |
| `aiml` | conda | 2.12.0+cu130 | 13.0 | Nemotron OCR v2 (CUDA toolkit must match PyTorch for the C++ extension build) |
| `florencetf` | conda | 2.11.0+cu130 | 13.0 | Florence-2 base/large (requires transformers 4.40.0, incompatible with transformers 5.x) |
| `.venv_paddleocr` | venv | - (PaddlePaddle 3.4.0+) | 12.9 | PaddleOCR-VL-1.6 (PaddlePaddle bundles its own NCCL/cuBLAS, conflicting with PyTorch's CUDA 13.0 stack) |

**Why four environments:** Nemotron OCR v2 needs `aiml` conda for its C++ CUDA extension build. Florence-2 needs `florencetf` for transformers 4.40.0. PaddleOCR-VL needs `.venv_paddleocr` because PaddlePaddle 3.4.0+cu129 bundles its own CUDA 12.9 NCCL/cuBLAS libraries that conflict with PyTorch's CUDA 13.0 stack. The main `.venv` handles all other models.

**Activation:**
```bash
# For SmolDocling, GOT-OCR2.0, MonkeyOCR, Qwen3-VL, TrOCR:
source .venv/bin/activate

# For Nemotron OCR v2:
conda activate aiml

# For Florence-2 (base or large):
conda activate florencetf

# For PaddleOCR-VL-1.6:
source .venv_paddleocr/bin/activate
```

### PaddleOCR-VL on Blackwell (NVIDIA sm_120)

PaddleOCR-VL uses the **PaddlePaddle native inference engine**, NOT HuggingFace transformers. The standard `paddlepaddle-gpu` from PyPI (3.2.1) does **not** include Blackwell (sm_120) support and fails with `RuntimeError: Unsupported GPU architecture`.

**Two options exist, but only Docker is reliable on WSL2:**

| Path | PaddlePaddle | Works? | Notes |
|---|---|---|---|
| `.venv_paddleocr` (native) | 3.3.1 (PyPI) | x | Below 3.4.0+ threshold for sm_120; hangs at `paddle.to_tensor()` |
| **Docker** (`sm120-offline` image) | 3.4.0+ (prebuilt) | yes | Only reliable path on WSL2 + Blackwell |

**Python API:** `from paddleocr import PaddleOCRVL` (NOT `PaddleOCR` or HuggingFace `AutoModel`).

Reference: [PaddleOCR-VL NVIDIA Blackwell Tutorial](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL-NVIDIA-Blackwell.html)

#### PaddleOCR-VL Docker Workflow

Running PaddleOCR-VL successfully requires understanding five distinct failure modes. The last line printed before a freeze identifies which one you have:

| # | Last line printed | Cause | Fix |
|---|---|---|---|
| 1 | `Checking connectivity to the model hosters...` | Pings HF/BOS/ModelScope to pick download source; stalls on bad routes | `-e PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` |
| 2 | `Fetching N files...` | BOS downloads (~2 GB) stall outside China; `--rm` re-downloads every run | `-v paddlex_models:/home/paddleocr/.paddlex` (named volume) |
| 3 | After `generation_config.json`, before `Latency:` | JIT kernel compilation ("No ccache found") or `/dev/shm` exhaustion (Docker defaults to 64 MB) | `--shm-size=8g` + persistent container (drop `--rm`) |
| 4 | Worked before, now hangs immediately | Stuck GPU state, zombie python holds VRAM, next CUDA init blocks (WSL2: VRAM leaks between `--rm` containers) | `wsl --shutdown` from PowerShell; set NVIDIA Control Panel → **CUDA - Sysmem Fallback Policy → Prefer No Sysmem Fallback** |
| 5 | Prints `Latency:` and results, never exits | Known Paddle inference teardown hang | `import os; os._exit(0)` at end of script |

**Critical WSL2-specific issues:**
- VRAM is NOT freed between `docker run --rm` containers (WDDM driver leak). Each subsequent run has 8+ GB of zombie allocations, forcing Paddle's weight uploads into shared GPU memory (system RAM over PCIe), a ~100× slowdown that looks like a freeze, not a deadlock.
- Fix: `wsl --shutdown` from PowerShell between evaluation runs.
- NVIDIA Control Panel → Manage 3D Settings → **CUDA - Sysmem Fallback Policy → Prefer No Sysmem Fallback** converts this failure mode from silent 100× slowdown to fast, visible OOM errors.

**Import conflict (discovered 2026-06-12):** Importing the project's `candidates` Python package (which imports `benchmark.harness` + `benchmark.metrics`) before calling `PaddleOCRVL.predict()` triggers WDDM sysmem fallback even on a clean GPU. The exact mechanism is unconfirmed but the workaround is clear: use a standalone benchmark script with **zero project imports** and **data-only Docker mounts** (mount only `benchmark/test_dataset` and `benchmark/results`, never the full project tree).

**Working benchmark command:**
```bash
# Pull once (Chinese registry — be patient):
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-nvidia-gpu-sm120-offline

# Run benchmark (standalone script, data-only mounts):
docker run --rm --gpus all --network host --shm-size=8g \
  -e PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
  -e PYTHONUNBUFFERED=1 \
  -v paddlex_models:/home/paddleocr/.paddlex \
  -v /home/pyaes/vlm-ocr-research/benchmark/test_dataset:/data:ro \
  -v /home/pyaes/vlm-ocr-research/benchmark/results:/results \
  -v /home/pyaes/vlm-ocr-research/scripts/bench_paddleocr_handwritten.py:/scripts/bench_paddleocr_handwritten.py:ro \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-nvidia-gpu-sm120-offline \
  python3 -u /scripts/bench_paddleocr_handwritten.py
```

**For rapid iteration (persistent container, keeps JIT cache warm):**
```bash
docker run -d --name paddleval --gpus all --network host --shm-size=8g \
  -e PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True -e PYTHONUNBUFFERED=1 \
  -v paddlex_models:/home/paddleocr/.paddlex \
  -v /home/pyaes/vlm-ocr-research/benchmark/test_dataset:/data:ro \
  -v /home/pyaes/vlm-ocr-research/benchmark/results:/results \
  -v /home/pyaes/vlm-ocr-research/scripts:/scripts:ro \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-nvidia-gpu-sm120-offline \
  sleep infinity

docker exec paddleval python3 -u /scripts/bench_paddleocr_handwritten.py   # run
docker exec paddleval python3 -u /scripts/bench_paddleocr_handwritten.py   # re-run (fast — JIT cached)
docker stop paddleval   # clean teardown (SIGTERM, not kill)
```

**Never use `docker kill`**, SIGKILL during active CUDA work leaks VRAM via dxgkrnl on WSL2. Always prefer `docker stop` (SIGTERM) for clean teardown.

### MonkeyOCR GPU Setup (llama.cpp from source)

The pre-built ggml-org/llama.cpp binaries are CPU-only. GPU acceleration requires
building from source with CUDA. The built binary lives at
`/tmp/llama.cpp/build/bin/llama-server`.

```bash
# One-time build (requires cmake, CUDA toolkit):
git clone https://github.com/ggerganov/llama.cpp.git /tmp/llama.cpp
cd /tmp/llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target llama-server

# Start the server (GPU-accelerated):
cd /tmp/llama.cpp/build/bin && LD_LIBRARY_PATH=. ./llama-server \
  -hf dinhquangson/MonkeyOCR-pro-1.2B-Vision-GGUF \
  --host 0.0.0.0 --port 8080 -ngl 99 -c 8192 \
  --mmproj-offload --image-min-tokens 1024

# Verify:
curl -s http://localhost:8080/health  # → {"status":"ok"}
```

**Details:** The `-ngl 99` flag offloads all model layers to GPU. `--mmproj-offload`
puts the multimodal vision projector on GPU (critical for image encoding speed).
Without GPU: 54s/image. With CUDA: 4.27s/image (10× faster).

See `candidates/monkeyocr/eval.py` for the full eval script header docs.

---

## Project Structure

```text
vlm-ocr-research/
├── README.md                          # This file: research plan & live results
├── benchmark/
│   ├── harness.py                     # Shared evaluation harness
│   ├── metrics.py                     # CER, WER, IoU, reading-order scoring
│   ├── test_dataset/                  # Handwritten essay samples + ground truth
│   ├── results/                       # Per-candidate JSON result files
│   └── visualizations/                # Bbox overlay & comparison images
├── candidates/
│   ├── paddleocr_vl/                  # PaddleOCR-VL-1.6 (deprecated: use scripts/bench_paddleocr_handwritten.py)
│   ├── got_ocr/                       # GOT-OCR2.0
│   ├── nemotron_ocr/                  # NVIDIA Nemotron OCR v2
│   ├── florence2/                     # Microsoft Florence-2
│   ├── smoldocling/                   # SmolDocling / granite-docling
│   ├── trocr/                         # Microsoft TrOCR (handwritten)
│   ├── qwen3_vl/                      # Qwen3-VL (4B & 8B)
│   └── baselines/                     # EasyOCR, docTR, Tesseract
├── scripts/
│   ├── bench_paddleocr_handwritten.py  # Standalone PaddleOCR-VL Docker benchmark
│   └── ...
└── pipeline/                          # Final assembled two-stage pipeline
```

---

## Candidate Models

### Tier 1: Most Promising

| # | Candidate | Params | Key Advantage | VRAM Est. |
| --- | --- | --- | --- | --- |
| 1 | **PaddleOCR-VL-1.6** † | 0.9 B | **#1 CER (0.045)**, beats <5% target. SOTA doc VLM (96.3% OmniDocBench), built-in layout & structure. Avg 31.94s on 12 GB (bimodal: 3.7-61.9s due to VRAM limit). | ~8 GB |
| 2 | **GOT-OCR2.0** | ~7 B (Qwen-based) | Unified end-to-end OCR, GGUF available. No bbox output. | ~6-8 GB (INT4) |
| 3 | **NVIDIA Nemotron OCR v2** | 54 M EN / 84 M Multi | **Built-in reading order** (relational model), bounding boxes, production-grade | ~1-2 GB |
| 4 | **Florence-2-large** | 0.77 B | Microsoft foundation model, prompt-based OCR + boxes, multi-task | ~1.5 GB |
| 5 | **granite-docling-258M** | 256 M | Ultra-compact document VLM, successor to SmolDocling, DocTags format | ~1 GB |
| 6 | **MonkeyOCR** | ~1.2 B | Document parsing with SRR triplet (structure-recognition-relation). Built-in layout detection (DocLayoutYOLO) for bboxes. **Does not support handwritten content** (per official limitations). | ~6 GB |

† Blackwell requires PaddlePaddle 3.4.0+ from the cu129 index or the official Docker image, see [PaddleOCR-VL on Blackwell](#paddleocr-vl-on-blackwell-nvidia-sm_120).

- **Phase 2-3 status:** All 13 candidates evaluated. See [Status at a Glance](#status-at-a-glance) for the ranked leaderboard. Hunyuan VL leads CER (0.015, manual), Qwen3-VL-8B is best automatable model (CER 0.035 + word IoU 0.722).

### Tier 2: Baselines & Comparison

| # | Candidate | Params | Purpose |
| --- | --- | --- | --- |
| 7 | **TrOCR (base + large, handwritten)** | 0.3 B / 0.6 B | IAM-finetuned HTR baseline: line-level only, needs text detector |
| 8 | **Qwen3-VL-4B-Instruct** | 4 B | Latest Qwen VLM, expanded OCR (32 languages), robust in low light/blur/tilt, strong document parsing. BF16 fits 12 GB (~8 GB). |
| 9 | **Qwen3-VL-8B-Instruct (INT4)** | 8 B (quantized) | Flagship Qwen VLM: 256K context, advanced spatial perception, 2D/3D grounding. INT4 needed for 12 GB (~6-8 GB), currently blocked by bitsandbytes (see [Blocker Log](#blocker-log)). BF16 on larger GPU or cloud API for production use. |
| 10 | **Hunyuan VL** | ~4 B | Tencent VLM with multi-resolution architecture, strong on document benchmarks, potential single-model Stage 1+2 candidate (~8 GB) |
| 11 | **EasyOCR** | ~50 M | Popular easy-to-use OCR; handwriting support on roadmap |
| 12 | **docTR** | varies | Modular PyTorch detection + recognition, good for documents |
| 13 | **Tesseract 5** | N/A | Traditional baseline for comparison |

---

## Architecture Insight

No single model handles everything. A **two-stage pipeline** is needed:

```text
┌──────────────────────────────────────────────────────┐
│ Stage 1: Transcription + Localization (OCR/VLM)      │
│  Image → { text, bounding boxes, reading order }     │
│  Includes sentence segmentation on unruled paper     │
└──────────────────────────┬───────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────┐
│ Stage 2: Error Detection & Feedback (LLM/VLM)        │
│  { text + bboxes + image } → { error bboxes + NL fb }│
└──────────────────────────────────────────────────────┘
```

**Sentence segmentation:** On unruled paper, assigning a word to the line above or below is non-trivial when spacing is ambiguous. Robust line/word assignment is therefore a first-class evaluation criterion (see Phase 5).

**Stage 2 candidates:** Qwen3-VL-4B, Qwen3-VL-8B (INT4), SmolVLM-Instruct (2 B), granite-docling-258M, Hunyuan VL, Gemini 3.5 Flash (cloud comparison).

**Key research question:** can a single VLM (e.g., Qwen3-VL) handle both stages end-to-end?

---

## Research Phases

### Phase 1: Environment Setup & Baseline (Complete)

1. Set up Python environment (CUDA 13.0, PyTorch 2.11.0, RTX 5070 Ti 12 GB). → `scripts/setup.sh`
2. Collect handwritten essay samples. → `benchmark/test_dataset/` (1,539 IAM forms from Kaggle, 25 curated).
3. Measure Google Document AI + Gemini baseline latency. → `benchmark/baseline.py`, **16.7s total (Doc AI 2.8s + Gemini 3.5 Flash 12.4s)**.
4. Build shared evaluation harness. → `benchmark/harness.py` + `benchmark/metrics.py`
5. Define metric collection JSON schema. → `benchmark/test_dataset/ground_truth.json`

**Validation:** `bash scripts/phase1_validate.sh`, 27/27 checks pass, 0 failures.

#### Phase 1 Findings

| Finding | Detail |
|---|---|
| **Baseline is 16.7s** | 21% faster than previously reported 21.1s (fixed: rate-limit sleep was included in latency). Local models must beat 16.7s. |
| **Stage 2 dominates latency** | Gemini error detection is 12.4s (74% of total). Replacing it with a local model is the highest-impact optimization. |
| **Document AI is fast** | OCR alone is 2.8s. The cloud OCR stage is not the bottleneck. |
| **Gemini needs `location="global"`** | `gemini-3.5-flash` is a preview model only available in the `global` region on Vertex AI. Using `asia-southeast1` returns 404. |
| **Blackwell needs nightly PyTorch** | RTX 5070 Ti (sm_120) unsupported by stable PyTorch 2.6. Uses 2.11.0+cu130. |
| **bitsandbytes blocked** | INT4 quantization unavailable for Blackwell, blocks Qwen3-VL-8B evaluation. |
| **Free-tier API keys inadequate** | 20 req/day + 5 RPM limits make API keys unusable for benchmarking. Vertex AI via ADC resolved this. |

#### How to Run the Baseline

```bash
# 1. Authenticate with Google Cloud
cd vlm-ocr-research
gcloud auth application-default login

# 2. Source the environment
set -a && source .env && set +a
# Required in .env:
#   GCP_PROJECT=<your-gcp-project-id>
#   GCP_LOCATION=<doc-ai-region>           # Document AI processor region
#   DOCAI_PROCESSOR_ID=<your-processor-id>
#   GEMINI_MODEL=gemini-3.5-flash           # Preview model, needs location="global" in code

# 3. Run the baseline (5 curated images)
.venv/bin/python -u benchmark/baseline.py
```

**Critical:** Two auth paths supported:
```python
# API Key (Vertex AI) — simplest, requires GOOGLE_API_KEY or GEMINI_API_KEY env var
client = genai.Client(api_key="...")
response = client.models.generate_content(model="gemini-3.5-flash", contents="...")

# ADC (Vertex AI) — requires gcloud auth, uses location="global" for preview models
from google.genai.types import HttpOptions
client = genai.Client(
    vertexai=True, project="...", location="global",
    http_options=HttpOptions(api_version="v1"),
)
```
Handled in `baseline.py` `_get_gemini_client()`. Prefers API key, falls back to Vertex AI.

#### Challenges Encountered

1. **PyTorch on Blackwell.** The RTX 5070 Ti has compute capability sm_120. PyTorch 2.6 stable only supports up to sm_90. Installing the stable wheel produced CUDA compatibility warnings and `total_mem` → `total_memory` attribute errors. Fixed by switching to PyTorch 2.11.0+cu130.

2. **Gemini API key quota.** Free-tier limits (20 req/day, 5 RPM) caused 429 errors across all Flash models and 503 errors during demand spikes. Added retry logic with exponential backoff and rate limiting, but the daily quota was fundamentally too low for 12 sequential calls. Resolved by switching to Vertex AI via ADC (`gcloud auth application-default login`), which has production-tier quota.

3. **Document AI regional endpoint.** The processor is deployed in `asia-southeast1`. The default client connects to the global endpoint and rejected the regional processor. Fixed by configuring a regional API endpoint (`asia-southeast1-documentai.googleapis.com`) via `ClientOptions`.

4. **Ground truth without IAM XML.** The IAM metadata download requires authentication. Without XML annotations, the ground truth generator produced skeleton entries. CER/WER scoring requires IAM registration or manual annotation. (Resolved in Phase 2A via the Kaggle forms dataset with matching XMLs.)

5. **harness.py API change.** `torch.cuda.get_device_properties(0).total_mem` was renamed to `.total_memory` in PyTorch 2.12. Fixed with a one-line change.

#### Phase 1 Artifacts

```
scripts/setup.sh                     # Environment bootstrap
scripts/validate_env.py              # Standalone GPU validation
scripts/download_essay_samples.py    # IAM manifest builder
scripts/curate_test_subset.py        # Diversity-based subset selector
scripts/generate_ground_truth.py     # IAM XML parser + skeleton fallback
scripts/phase1_validate.sh           # 27-check integration suite
benchmark/test_dataset/DATASET.md    # Dataset provenance doc
benchmark/test_dataset/ground_truth.json  # 20-entry skeleton
benchmark/test_dataset/curated_manifest.json  # Subset metadata
benchmark/results/baseline_google_docai_fullform.json  # 16.7s baseline
docs/METHODOLOGY.md                  # Full research design doc
.env.example                         # GCP credential template
```

### Phase 2: Tier-1 Candidate Evaluation (Complete, 6/6 evaluated)

> **Methodological finding (2025-06-11):** IAM forms contain the writing prompt **machine-printed** and the writer's **handwritten copy** of the same text. The ground truth is therefore textually identical for both regions, so **full-form CER/WER cannot distinguish whether a model read the printed text or the handwriting.** Models that emit the form header ("Sentence Database"), Nemotron OCR and MonkeyOCR, are confirmed to be reading the printed region. SmolDocling and GOT-OCR2.0 skip the header, so their source is ambiguous. **The full-form Phase 2 numbers below are printed-text OCR benchmarks; authoritative handwriting accuracy comes from the [handwriting-only re-evaluation](#handwriting-only-re-evaluation-xml-guided-cropping).** Details: [IAM Dataset Structure](#iam-dataset-structure).

For each of the 6 candidates, the following were measured:

* End-to-end latency (image → structured text with bounding boxes).
* CER / WER on handwritten text (whitespace-normalized).
* Reading order accuracy (Kendall's tau vs. ground truth).
* Bounding box quality (visual inspection + IoU where ground truth exists).
* VRAM peak usage.
* Setup complexity (1-5 scale; 1 = easiest).
* Flexibility (1-5 scale; 5 = handles varied handwriting, layouts, noise best).

**Dataset:** 1,539 IAM handwriting forms from Kaggle (`gwachatkozah/iam-forms-dataset`) with matching XML annotations providing line-level text, bounding boxes, and reading order. Curated subset of 25 forms used for evaluation. Ground truth populated via `parse_iam_xml()` extracting `<cmp>`-level coordinates.

**CER/WER:** Computed with whitespace normalization (newlines → spaces, collapsed whitespace). This prevents line-break formatting differences from artificially inflating error rates.

### IAM Dataset Structure

Every IAM form image has four sections stacked vertically, separated by horizontal lines:

```
┌──────────────────────────────────────┐
│ 1. HEADER (printed)                  │  "Sentence Database" + form ID (e.g., "A04-039")
├──────────────────────────────────────┤
│ 2. PRINTED PROMPT (machine-printed)  │  Paragraph of serif text — the "answer key"
├──────────────────────────────────────┤
│ 3. HANDWRITTEN COPY (cursive/print)  │  Writer's handwritten copy of the printed prompt (~6 lines)
├──────────────────────────────────────┤
│ 4. FOOTER                            │  "Name:" label + handwritten signature
└──────────────────────────────────────┘
```

**Implication for evaluation:** The IAM XML annotations contain identical text in both `<machine-printed-part>` and `<handwritten-part>`. Our ground truth (extracted from `<handwritten-part>` via `parse_iam_xml()`) is textually identical to the printed prompt. This means:

- **Full-form CER/WER measures text correctness, NOT handwriting recognition ability.**
- A model that reads the easy machine-printed text (section 2) scores identically to one that reads the hard handwriting (section 3).
- Models that output "Sentence Database" or form IDs are confirmed to be reading sections 1-2 (printed), not section 3 (handwriting): **Nemotron OCR** and **MonkeyOCR** both do this.
- Models that skip the header have an ambiguous source: **SmolDocling** and **GOT-OCR2.0** may be reading handwriting, or may simply filter headers.

Handwriting-specific evaluation therefore requires isolating section 3 via bounding-box cropping (done below) or using a handwriting-only dataset.

#### Phase 2A: Prerequisites

- Removed old renamed images (`iam_page_*.jpg`), replaced with Kaggle form images (original IAM filenames matching XML).
- Fixed `parse_iam_xml()` to compute bboxes from `<cmp>` child elements (not `<line>` attributes).
- Fixed `find_xml_for_image()` for direct stem matching (`a01-000u.png` ↔ `a01-000u.xml`).
- Regenerated `ground_truth.json`: 1,539/1,539 entries with real text, bboxes, and reading order.
- Created new 25-form curated subset spanning 21 writer prefixes.
- Updated harness to match ground truth by image name (not index).
- Added `normalize_ocr_text()` and `compute_cer_normalized()` / `compute_wer_normalized()` to metrics.

#### Phase 2B-2G: Candidate Evaluations (full forms)

| Candidate | Status | Text Source | Key Findings |
|---|---|---|---|
| **GOT-OCR2.0** | Complete | Ambiguous (no header output) | 35.9s avg, 3.4 GB VRAM. Transcribes full form (printed + handwritten). No native bbox output, format mode produces text formatting only. Verified against HF transformers and original implementation. |
| **Florence-2-large** | Complete (separate env) | **Handwriting** (no header in output) | **Was #1 at Phase 2.** CER 0.061, 1.05s avg (fastest GPU model). 770M params, 2.0 GB VRAM. Now #5 behind Hunyuan, Qwen-4B, Qwen-8B, PaddleOCR-VL. Base model (230M): CER 0.187, 1.2s. |
| **PaddleOCR-VL-1.6** | Complete (Docker) | Handwriting-only (cropped) | **#1 by CER (0.045).** Evaluated via standalone `bench_paddleocr_handwritten.py` in Docker (data-only mount). Avg 31.94s (3.7-61.9s bimodal). `.venv_paddleocr` native path confirmed broken on Blackwell; Docker `sm120-offline` image is the only working path. Built-in layout + bboxes + structure. See [PaddleOCR-VL Docker Workflow](#paddleocr-vl-docker-workflow). |
| **SmolDocling-256M** | Complete | Ambiguous (skips header) | 12.2s avg latency (42% faster than baseline), 0.8 GB VRAM, CER 1.47, WER 1.50. DocTags output with bbox parsing via regex. Hallucinates/repeats on handwriting. `AutoModelForMultimodalLM` + transformers 5.x. |
| **Nemotron OCR v2** | Complete | **Printed text** (headers in output) | **Fastest candidate.** 0.09s avg (234× faster than baseline), 0.6 GB VRAM, CER 1.17, WER 1.24. Outputs form headers ("Sentence Database") + form IDs → confirmed reading printed text. 7 text regions per page with bboxes + reading order via relational model. Built in `aiml` conda env (CUDA 13.0 + PyTorch 2.12). CUDA extension compiled without issues. |
| **MonkeyOCR** | Complete (GPU, GGUF) | Evaluated on cropped handwriting | 4.27s avg (GPU via CUDA llama-server build), handwriting CER 0.566, WER 0.240. GGUF/llama-server path = text recognition only. 1.2B params, ~2 GB VRAM with full GPU offload. Built from ggerganov/llama.cpp with -DGGML_CUDA=ON for RTX 5070 Ti. Official pipeline has layout detection + reading order via DocLayoutYOLO + layoutreader + VLM. |

### Handwriting-Only Re-Evaluation (XML-Guided Cropping)

After the methodological finding above, all images were cropped to the handwritten region using IAM XML `<handwritten-part>` `<cmp>` bounding boxes. Each image was cropped to the union bounding box of all handwritten lines (excluding the signature footer), with 20px padding. This physically removes the printed header/prompt, forcing models to read only handwriting. **These are the authoritative accuracy numbers.**

#### Ground Truth Bounding Box Methodology

Ground truth line-level bounding boxes are derived from the IAM XML annotations with a three-step coordinate transform:

**1. Source data, IAM XML `<cmp>` elements**

Each handwritten word in the XML contains one or more `<cmp>` (connected component) elements with pixel coordinates in the **original full-form coordinate space** (2479 × 3542 px):

```xml
<word id="a01-000u-00-00" text="A">
  <cmp x="408" y="768" width="27" height="51" />
</word>
<!-- ... later in the same line ... -->
<word id="a01-000u-00-06" text="from">
  <cmp x="1896" y="757" width="55" height="72" />
  <cmp x="1955" y="781" width="114" height="33" />
</word>
```

For each handwritten `<line>`, the bounding box is computed as:

| Bound | Source |
|---|---|
| `x1` | `x` of the **first `<cmp>`** of the first word |
| `y1` | `asy` (ascender y) attribute of the `<line>` |
| `x2` | `x + width` of the **last `<cmp>`** of the last word |
| `y2` | `dsy` (descender y) attribute of the `<line>` |

The `asy` and `dsy` line attributes are used for vertical bounds because they provide consistent line height across all words in a line, which is more reliable than per-cmp y values that can drift between characters.

**2. Crop offset computation**

For each form, the handwritten region's union bounding box is computed from ALL `<cmp>` elements across all handwritten lines (excluding signature footer lines detected by a bottom-15% heuristic or "Name:" text). The crop origin with 20px padding is:

```
crop_x = max(0, min(all_cmp_x) - 20)
crop_y = max(0, min(all_asy)  - 20)   # ascender y of first line
```

**Example, form `a01-000u`:**

| Metric | Value |
|---|---|
| Form dimensions | 2479 × 3542 px |
| First cmp | `x=363, y=739` (first character of first handwritten line) |
| Last cmp | `x=2414, y=1913` (last character of last handwritten line) |
| Union bbox | `x=[363, 2414]`, `y=[739, 1913]` |
| Crop origin (with 20px padding) | `(343, 719)` |
| Cropped image size | ~2071 × 1214 px |

The crop removes:
- **Top 719px**, form header ("Sentence Database") + printed prompt
- **Bottom 1629px**, empty space + footer below the handwriting

**3. Coordinate transform to cropped image space**

Each line's bounding box is then offset to the cropped image coordinate system:

```
cropped_x1 = max(0, original_x1 - crop_x)
cropped_y1 = max(0, original_y1 - crop_y)
cropped_x2 = min(img_width,  original_x2 - crop_x)
cropped_y2 = min(img_height, original_y2 - crop_y)
```

**Example, first line of `a04-039`:**

| Coordinate | Original (full form) | Offset (minus crop) | Cropped |
|---|---|---|---|
| `x1` | 364 | 364 − 343 = 21 → padded | 57 |
| `y1` | 912 | 912 − 891 = 21 | 20 |
| `x2` | 1985 | 1985 − 343 = 1642 | 1678 |
| `y2` | 1040 | 1040 − 891 = 149 | 148 |

**Result:** Each GT bbox tightly bounds the handwritten text from the first character to the last character of that line, in the cropped image coordinate space. Boxes span 1565-1858px wide on ~1880px-wide images, with a ~20-60px left margin (empty space inherent in IAM scans). The `reading_order` field is always `[0, 1, 2, ...]` (sequential) since IAM forms are single-column.

See `scripts/generate_ground_truth.py` for the generation code and `scripts/crop_handwritten.py` for the crop logic. Note that all bbox IoU and reading-order (τ) scores below are computed in this cropped coordinate space.

#### Handwriting-Only Results (authoritative, sorted by CER)

| Candidate | Printed CER (full form) | **Handwriting CER** | Bbox IoU | Read Order τ | Latency (cropped) | Verdict |
|---|---|---|---|---|---|---|
| **Florence-2-large** | - | **0.061** | **0.76** (line) | **1.00** | 1.05s (text) / 1.49s (bbox) | **Best overall model** (accuracy + speed + line bboxes). 2.0 GB VRAM. PaddleOCR-VL has better CER (0.045) but 30× slower. Word-level IoU is 0.176 (line bboxes only, see word-level table). |
| **PaddleOCR-VL-1.6** | - | **0.045** | block-level† | block-level† | 31.94s avg (3.7-61.9s) | #4 by CER. Beats the <5% CER target. SOTA doc VLM (96.3% OmniDocBench), built-in layout + bboxes + structure. Bimodal latency: ~4s for short text, ~30-62s for long text (12 GB VRAM limit triggers WDDM sysmem fallback). Docker-only on Blackwell. |
| **Google Doc AI** (cloud) | 0.08 | **0.095** (line) / **0.108** (word) | **0.58** (line) / **0.611** (word) | **0.91** | 2.19s (line) / 3.6s (word) | Cloud baseline. Word-level CER 0.108, WER 0.315, word IoU 0.611. Beaten by Qwen VLMs on both CER (0.035) and word IoU (0.72). |
| **SmolDocling-256M** | 1.47 | 0.107 | 0.24 | 1.00 | 5.37s | Best structured output (DocTags + bboxes), but weak localization (coarse blocks). |
| **Nemotron OCR v2** | 1.17 | 0.214 | 0.28 | 0.89 | 0.07s | Fastest (239× baseline). Recognizer struggles with handwriting (CER 0.214) but detector localizes ~40% of line-level blocks. Reading order τ informative (0.89, not uniformly 1.0). |
| **GOT-OCR2.0** | 2.93 | 0.088 | ‡ | - | 2.53s | No bbox output. |
| **PaddleOCR PP-DocLayout-L** (layout only) | - | - | 0.08 † | **0.92** | **0.07s** | **Best layout detector tested.** 3-9 regions per full-form page (text, header, paragraph_title). Solid reading order (τ = 0.92) vs DocLayoutYOLO's τ = -0.17. Runs in the PaddleOCR Docker image alongside PaddleOCR-VL. |
| **DocLayoutYOLO** (layout only) | - | - | 0.12 † | -0.17 | 0.10s | Layout detection only (no OCR). Region-level blocks cannot resolve line-level reading order (τ = -0.17). Runs on Blackwell without PaddlePaddle. |

† All layout-detector IoU scores reflect region-level vs line-level GT granularity mismatch, not detection failure. PD-DocLayout-L detects 3-9 regions (τ = 0.92), DocLayoutYOLO detects 1-3 regions (τ = -0.17). PaddleOCR-VL outputs document-level blocks (1 per cropped image). See [Model Bounding Box Generation Methods](#model-bounding-box-generation-methods) for details.
‡ GOT-OCR2.0 has no native bbox output in any mode, excluded from IoU comparison.

**Key findings:**

- **Florence-2-large is the best overall model** for deployment speed/accuracy balance (CER 0.061, 1.05s, IoU 0.76, τ = 1.00). However, see [Status at a Glance](#status-at-a-glance) for current rankings — Phase 3 models (Hunyuan, Qwen VLMs, PaddleOCR-VL) all have better CER at higher latency.
- **PaddleOCR-VL-1.6** (CER 0.045) was the first model to beat the <5% CER target. Bimodal latency (3.7-61.9s) due to 12 GB VRAM WDDM sysmem fallback.
- **GOT-OCR2.0** (CER 0.088) at 2.53s, dramatically better than previously reported 4.32. Text-only.
- **Nemotron's speed is unmatched** (0.07s, 239× baseline) but handwriting CER 0.214.
- **Reading order is trivially perfect for line-level models** on single-column IAM forms (τ = 1.00). Multi-column/unruled reading order remains untested (Phase 5).
See `scripts/crop_handwritten.py` and `scripts/eval_handwritten.py` for the methodology.

#### Model Bounding Box Generation Methods

Each model produces bounding boxes differently:

| Model | Bbox Source | Format | Conversion | Evaluation Script |
|---|---|---|---|---|
| **Florence-2** | `<OCR_WITH_REGION>` prompt returns quad boxes + labels | `[x1,y1,x2,y2,x3,y3,x4,y4]` 4-corner quad | `xyxy = [min(xs), min(ys), max(xs), max(ys)]`, min/max over 4 corners | `scripts/eval_bbox_reading_order.py florence2` (requires `florencetf` env) |
| **Nemotron OCR v2** | Detector (RegNetX-8GF) outputs text regions per page | `[x, y, w, h]`, origin + dimensions | `xyxy = [x, y, x+w, y+h]` | `scripts/eval_bbox_reading_order.py nemotron` (requires `aiml` env) |
| **SmolDocling** | DocTags tokens (`<loc_N>` with 0-999 normalized coords) parsed to pixel bboxes | `[x1, y1, x2, y2]`, already xyxy | No conversion needed. Coordinates denormalized from 0-999 bin space via regex parser. | `scripts/eval_bbox_reading_order.py smoldocling` (requires `.venv`) |
| **GOT-OCR2.0** | No native bbox output | - | Model does not output bboxes. Format mode produces formatted text (line breaks), not spatial coordinates. Fine-grained mode takes a bbox as *input*, not output. Verified against HF transformers `stepfun-ai/GOT-OCR-2.0-hf` and original `ucaslcl/GOT-OCR2_0` source. | - |
| **MonkeyOCR (official pipeline)** | DocLayoutYOLO → layout bboxes | `[x1, y1, x2, y2]`, already xyxy | No conversion needed. Detected regions: plain text, figure, formula, table, etc. Layout detection via DocLayoutYOLO (PyTorch/ONNX, no PaddlePaddle needed). | `scripts/eval_bbox_reading_order.py doclayout_yolo` (requires `.venv` + `pip install doclayout_yolo`) |
| **Google Doc AI** | Cloud API returns structured document with block-level bboxes | `[x1, y1, x2, y2]` | Already xyxy | `benchmark/baseline.py` |
| **PaddleOCR-VL** | `res.json["res"]["parsing_res_list"]`, document-level blocks with `block_bbox` + `block_content`. Also outputs `layout_det_res` for region classification (text/table/figure/etc.) and `res.markdown["markdown_texts"]` for structured markdown. | `[x1, y1, x2, y2]` string or list | String parsed via `strip("[]").split(",")` | `scripts/bench_paddleocr_handwritten.py` (Docker) |
| **PaddleOCR PP-DocLayout-L** | `LayoutDetection(model_name="PP-DocLayout-L")`, dedicated layout detection module (RT-DETR-L). Outputs per-region bboxes with class labels (text, header, table, figure, etc.) and confidence scores. Separate from PaddleOCR-VL, uses detection model, not VLM. | `[xmin, ymin, xmax, ymax]`, already xyxy | No conversion needed | Docker inline (see [PP-DocLayout-L Layout Detection](#paddleocr-pp-doclayout-l-layout-detection)) |

**Visual inspection:** Per-model comparison images are in `benchmark/visualizations/{model}/` (5 representative images each). Models: `florence2`, `nemotron`, `smoldocling`, `monkeyocr` (DocLayoutYOLO), `docai` (Google Document AI), `ground_truth`. Generated via `scripts/generate_model_visualizations.py` (local models) and inline script (cloud baseline).

#### Handwriting Output Comparison (a04-039.png)

```
GROUND TRUTH (handwritten lines, 470 chars)
──────────────────────────────────────────────────────────────────────
In VIENNA, before flying off to Moscow,
Mr. Khrushchov said he hoped his weekend
talks with President Kennedy would # help "to
establish an enduring peace between nations."
Replying to a farewell speech from Austrian President
Schaerf, the Soviet Premier thanked Austria for
the hospitality and welcome he had received.
"The Soviet Union has always striven and is
striving to safeguard an enduring peace for
the peoples, to secure an early solution of the

GOOGLE DOC AI   (CER=0.095 line / 0.108 word, 3.6s word-level OCR)
──────────────────────────────────────────────────────────────────────
In Vienna, before flying off to Moscow,
во
Mr. Khrushchou said he hoped his weekend
talks with President Kennedy would help
" to
"
establish an
onduring peace between nations.
Replying to
a
farewell speech from Austrian President
Schaerf
the Soviet Premier thanked Austria for
I
the hospitality and welcome he had received.
"The Soviet Union has always striven and is
striving to safeguard an onduring peace for
the peoples
to
secure
an
early solution of the

MONKEYOCR   (CER=0.57 avg, 4.27s GPU)
──────────────────────────────────────────────────────────────────────
In Vienna, before flying off to Moscow,

Mr. Khrushchov said he hoped his weekend talks with President Kennedy
would help "to establish an onduring peace between nations." Replacing
to a farewell speech from Jussinck President

Schoerd, the Soviet Premier thanks to Jussinck for the hospitality and
welcome he had received.

"The Soviet Union has always shown and is striving to safeguard an
onduring peace for the people, to secure an early solution of the

SMOLDOCLING   (CER=0.107 avg, 5.37s)
──────────────────────────────────────────────────────────────────────
In Vienna, before flying off to Moscow, Mr. Khrushov said he hoped his
weekend talkers with Presidet Kennedys would be help " to establish an
onduring peace between nations. " Replying to a farewell speech from
author Presidet Shaard, he Soulet Premier thanked ushisa for the
hospitally and welcome he had received. 4 The Soulet Union has always
Join a shon and is stiving to safeguard an onduring peace for the
peoples, to secure an early soluion of the

NEMOTRON   (CER=0.214 avg, 0.07s)
──────────────────────────────────────────────────────────────────────
4 The Souiet unioh ha always strven and is striving to safeguard an ou
ondaring peace for the peoples) , to Decure an early sole hor of the
In Vienna, before juying off to Morcow, Mr. khrushchov said he hopect
his weekend talks with Preideww Kennedy would & help 4 to eslaseish an
onduring peaie Selweeh natiows " Replbis so a were speech from Aushiag
Prestdoct Schoerd 1 the Sovret Preneirr thanked Ausiia for the
hospilality and welcome he had roeived.

FLORENCE-2-LARGE   (CER=0.061 avg, 1.05s)
──────────────────────────────────────────────────────────────────────
In Vienna, before flying off to Moscow,
Mr. Khrushdov said he hoped his weekend
talks with President Kennedy would be help 4 to
establish an enduring peace between nations."
Replying to a farewell speech from Soviet President
Schoerd, the Soviet Premier thanked husbia for
the hospitality and welcome he had received.
"The Soviet Union has always shivra and is
striving to safeguard an ordinary peace for
he people, ho secure an early solution of the

GOT-OCR2.0   (CER=0.088 avg, 2.53s)
──────────────────────────────────────────────────────────────────────
In Vienna, before flying off to Moscow,
Mr.Khushchov said he hoped his weekend
talks with President Kennedy would be help "to
eslasish an obduing peace selween nations."
Replying to a farvwell speech from Hushian President
Schoerd, the Soviet Premier thankeal Hushia for
the hospitality and welcome he had received.
"The Soviet Union has always shown and is
strolling to safeguard an ordinary peace for
he peoples,he secure an early suthion of the
```

### Phase 3: Tier-2 Candidate Evaluation (Complete, 13/13 candidates)

Evaluate comparison baselines and larger models for reference on the
authoritative cropped-handwriting dataset (25 images, `ground_truth_handwritten.json`).
> **2026-06-14, Critical bug found.** The existing `candidates/qwen3_vl/eval.py` uses
> `Qwen3VLForConditionalGeneration` which is the WRONG class. The official
> [Qwen3-VL README](https://github.com/QwenLM/Qwen3-VL) shows the correct API is
> `AutoModelForImageTextToText` + `AutoProcessor`. This must be fixed before evaluation.
**Candidates, status, and plan:**
| # | Candidate | Params | Eval Script | Status | Notes |
|---|---|---|---|---|---|
| 7 | TrOCR (base + large) | 0.3B / 0.6B | `candidates/trocr/eval.py` | Exists, needs path update | IAM-finetuned, line-level only. Heuristic line detector may need CRAFT fallback on cropped images. |
| 8 | Qwen3-VL-4B-Instruct | 4B | `candidates/qwen3_vl/eval.py` | **BUG: wrong model class** | Fix `Qwen3VLForConditionalGeneration` → `AutoModelForImageTextToText`. BF16 fits 12 GB (~8 GB). |
| 9 | Qwen3-VL-8B-Instruct (INT4) | 8B | same as #8 | **BLOCKED** | bitsandbytes lacks Blackwell sm_120 kernels. Cloud API fallback noted. |
| 10 | Hunyuan VL | ~4B | none | **Manual only → Complete** | No API/HF access; lmarena chat only. 5-image eval done: CER 0.015 (#1 overall). |
| 11 | EasyOCR | ~50M | `candidates/baselines/eval.py` | Exists, needs path update | `easyocr.Reader(["en"])` already implemented. |
| 12 | docTR | varies | `candidates/baselines/eval.py` | **Needs new function** | `doctr.models.ocr_predictor(pretrained=True)`. PyTorch backend. |
| 13 | Tesseract 5 | N/A | `candidates/baselines/eval.py` | Exists, needs path update | `pytesseract.image_to_data()` already implemented. |
**Implementation steps:**
1. ✓ **Fix Qwen3-VL eval:** Replaced `Qwen3VLForConditionalGeneration` → `AutoModelForImageTextToText`. Updated to handwritten images + `ground_truth_handwritten.json`. (done 2026-06-14)
2. ✓ **Update TrOCR eval:** Switched to handwritten images + `ground_truth_handwritten.json`. Added `TROCR_MODEL` env var for base/large switching. (done 2026-06-14)
3. ✓ **Update baselines eval:** Added `_doctr_inference` (`python-doctr[torch]`), switched to handwritten images via `_get_test_images()` helper. (done 2026-06-14)
4. ✓ **Register in `eval_handwritten.py`:** Added `trocr_large`, `trocr_base`, `qwen3_vl_4b`, `easyocr`, `tesseract`, `doctr` with `fn_name` support for baselines. (done 2026-06-14)
5. ✓ **Run evaluations (6/6 automated complete):** TrOCR-large, TrOCR-base, EasyOCR, Tesseract, docTR, Qwen3-VL-4B done. Qwen3-VL-4B re-evaluated with inline bbox parsing fix. (done 2026-06-14)
6. ✓ **Manual Hunyuan VL:** 5-image eval via lmarena chat. CER **0.015** — new #1 by CER, beats Qwen3-VL-4B (0.022). Text-only, no bbox. (done 2026-06-14)
7. ✓ **Document:** Phase 3 results table added. Per-candidate findings below. (done 2026-06-14)

**Phase 3 Handwriting-Only Results (CER-sorted, cropped images, 25 images)**

| Candidate | Handwriting CER | WER | Latency (cropped) | VRAM | Bbox | Notes |
|---|---|---|---|---|---|---|
| **Hunyuan VL** | **0.015** | — | ~5-10s (chat) | N/A (cloud) | Text-only | **#1 by CER overall.** 5-image manual eval via lmarena. No bbox output. Manual only — not automatable. |
| **Qwen3-VL-4B** | **0.022** | 0.054 | 14.00s | 9.6 GB | Line-level (parsed) | **#1 automatable by CER.** Prompt-based OCR with inline bbox parsing. Fixed model class bug. |
| **TrOCR-large** | **0.244** | 0.506 | 1.35s | 2.2 GB | Heuristic (10 lines) | IAM-finetuned, line-level model. Heuristic vertical-projection detector. |
| **docTR** | **0.272** | 0.757 | 1.09s | ~1 GB | Word-level (native) | db_resnet50 + crnn_vgg16_bn. Word-level bboxes, moderate accuracy on handwriting. |
| **TrOCR-base** | **0.380** | 0.681 | 1.22s | ~1.5 GB | Heuristic (10 lines) | Smaller variant, ~50% worse CER than large. |
| **Tesseract 5** | **0.439** | 0.862 | 0.50s | N/A (CPU) | Word-level (native) | Traditional OCR engine. Not designed for handwriting. |
| **EasyOCR** | **0.628** | 1.086 | 0.88s | ~0.5 GB | Word-level (native) | CRNN-based. Struggles with handwriting (roadmap feature). |
| **Qwen3-VL-8B-INT4** | BLOCKED | BLOCKED | BLOCKED | — | — | bitsandbytes lacks Blackwell sm_120 kernels. Evaluated via HF novita API in word-level eval (CER 0.035, IoU 0.722). |

*> For reference: PaddleOCR-VL-1.6 (CER 0.045), Florence-2-large (CER 0.061), GOT-OCR2.0 (CER 0.088) — see handwriting-only table for full rankings.*

**Key findings:**
- **Hunyuan VL is the #1 model by CER (0.015)** — manual evaluation only (5 images via lmarena), beats Qwen3-VL-4B (0.022). Text-only, no bbox. Not automatable.
- **Qwen3-VL-4B is the #1 automatable model by CER (0.022)** — 2.1× better than PaddleOCR-VL-1.6 (0.045). Prompt-based OCR with inline bbox parsing. However, at 14.0s/page and 9.6 GB VRAM, it's 13× slower than Florence-2-large (1.05s). The corrected model class (`AutoModelForImageTextToText`) and bbox-stripped text parsing were critical — the initial run with bbox tokens in the text produced a misleading CER of 0.597.
- **No other Tier-2 model competes with Tier-1 on handwriting.** TrOCR-large (CER 0.244) is 11× worse than Qwen3-VL-4B.
- **Florence-2-large remains best overall** for deployment (CER 0.061, 1.05s, IoU 0.76, 2.0 GB VRAM) — the best accuracy/speed balance.
- **TrOCR-large > TrOCR-base** (0.244 vs 0.380): the larger model is worth the extra VRAM.
- **Traditional OCR engines (EasyOCR, Tesseract) are not viable for handwriting** — CER > 0.4.
- **docTR is the best open-source traditional OCR** for handwriting at CER 0.272, with native word-level bboxes.

#### Qwen3-VL-4B Word-Level Evaluation (2026-06-14)

Beyond line-level CER, we evaluated Qwen3-VL-4B on **per-word bounding box accuracy** and **reading order** using word-level prompts. Ground truth was generated from IAM XML word-level annotations (1891 words across 25 images, with 20px padding applied to crop offset).

| Metric | Value | Notes |
|---|---|---|
| **Word-level CER** | **0.049** | Slightly higher than line-level (0.022) — word-by-word prompting introduces minor word-splitting errors |
| **WER** | 0.242 | Word-level WER is naturally higher due to per-word evaluation granularity |
| **Word IoU** | **0.718** | Greedy spatial matching at IoU ≥ 0.05 threshold. VLM-predicted bboxes are typically larger than GT, which penalizes IoU even when boxes are visually well-aligned |
| **Reading Order τ** | **1.000** | Perfect on single-column IAM forms. Kendall's tau-b against GT word sequence |
| **Word Recall** | ~93% | Most GT words matched by at least one predicted bbox |
| **Latency** | ~80s/image | Significantly slower than line-level (14s) due to per-word output generation |

**Visualizations:** 25 annotated images saved to `benchmark/visualizations/qwen3vl_4b_wordlevel/` (red = predicted bbox + text, green = GT bbox + text). Result JSON: `benchmark/results/qwen3_vl_4b_wordlevel_handwritten.json`.

**Key insight:** Qwen3-VL-4B can produce usable word-level bounding boxes via prompting alone (no fine-tuning). IoU of 0.72 is solid for a generalist VLM — the main gap vs. specialist detectors (Florence-2 IoU 0.76) is that VLMs tend to predict slightly oversized boxes. The reading order is flawless on structured single-column forms.

#### Qwen3-VL-8B Word-Level Evaluation (2026-06-14)

Evaluated via HF Inference API (novita provider) — the 8B model cannot run locally due to bitsandbytes lacking Blackwell sm_120 kernels. Same word-level prompt, ground truth, and metrics as the 4B evaluation.

| Metric | 8B (API) | 4B (local) | Notes |
|---|---|---|---|
| **Word-level CER** | **0.035** | 0.049 | 8B is 29% more accurate at word level |
| **WER** | **0.223** | 0.242 | |
| **Word IoU** | **0.722** | 0.718 | Essentially tied on spatial accuracy |
| **Word Recall** | ~99% | ~93% | 8B matches nearly every GT word |
| **Latency** | 26.8s/image | ~80s/image | API inference is faster than local 4B generation |
| **Total tokens** | 108,640 | N/A | ~4,300 tokens/image |

**Visualizations:** 25 annotated images (red=pred, green=GT) at `benchmark/visualizations/qwen3vl_8b_wordlevel/`. Result JSON: `benchmark/results/qwen3_vl_8b_wordlevel_handwritten.json`.

**Key insight:** The 8B model improves on the already-strong 4B across every metric — CER drops from 0.049 → 0.035 (29% better), word recall goes from 93% → 99%, and IoU is essentially tied (0.722 vs 0.718). The API latency (26.8s/image) is surprisingly lower than local 4B (~80s/image), likely due to better hardware on the API side. At 0.035 CER, Qwen3-VL-8B achieves the best word-level accuracy of any model evaluated. The practical limitation is API cooldown (~5 min between requests), making full-dataset benchmarks slow but individual image inference fast.

**Decisions:**
- **Direct handwritten evaluation only:** skip full-form; Phase 2 proved it's confounded by printed text.
- **Qwen3-VL-8B local blocked** by bitsandbytes on Blackwell; API evaluation via novita provider is the fallback.
- **Hunyuan VL manual only:** no API access; lmarena chat is the only path.
- **docTR uses PyTorch backend** (`python-doctr[torch]`) to avoid adding TensorFlow to `.venv`.

#### Word-Level IoU Cross-Model Comparison (2026-06-14)

All models evaluated against the same word-level ground truth (1891 words from IAM XML, 25 images). IoU computed via greedy spatial matching at threshold 0.05.

| Model | Word IoU | CER | WER | τ | Latency | Bbox Source |
|---|---|---|---|---|---|---|
| **Tesseract 5** | **0.812** | 0.443 | 0.832 | 0.993 | 0.5s | Native word-level engine |
| **Qwen3-VL-8B** | **0.722** | 0.035 | 0.223 | — | 26.8s (API) | Prompted word-level |
| **Qwen3-VL-4B** | **0.718** | 0.049 | 0.242 | 1.000 | ~80s | Prompted word-level |
| EasyOCR | 0.597 | 0.625 | 1.007 | 0.760 | 1.1s | Native word-level |
| docTR | 0.581 | 0.275 | 0.737 | 0.999 | 1.2s | Native word-level |
| **Google Doc AI** | **0.611** | 0.108 | 0.315 | — | 3.6s | Native word-level (cloud) |
| Florence-2-large | 0.176* | 0.091 | 0.341 | 1.000 | 1.7s | Line-level only* |

> \*Florence-2 outputs line-level bboxes (~10 per page), not word-level. Its previously reported IoU of 0.76 was line-level vs. line-level GT. At word granularity, IoU drops to 0.176 because single line boxes can't match individual word GT annotations. PaddleOCR-VL also outputs layout-level blocks (not word-level), so word IoU is not meaningful for it.

**Key takeaways:**
- **Tesseract wins on bbox precision** (0.812) — its traditional word-level engine produces the tightest bounding boxes. But CER is terrible (0.443), making it unusable for transcription.
- **Qwen3-VL models have the best CER/IoU balance** — 0.72 IoU with 0.035-0.049 CER. This is the only approach that delivers both accurate transcription AND usable word localization.
- **Traditional OCR engines (EasyOCR, docTR) have moderate IoU (0.58-0.60)** with moderate-to-poor CER (0.28-0.63). They can detect where words are but can't reliably read them.
- **No traditional OCR engine achieves CER < 0.27** on handwriting. The VLMs are 5-8× more accurate at transcription.
- **Reading order is a solved problem for single-column forms** — all models achieve τ > 0.99 except EasyOCR (0.76, which has top-to-bottom confusion).
- **For Stage 2 error detection** (Phase 6), per-word bounding boxes are critical — Tesseract or Qwen VLMs would be the only viable candidates.

### Phase 4: Two-Stage Pipeline Architecture Design

Test combinations of best Stage 1 + Stage 2 models. Measure:

* Can a single VLM do both stages end-to-end?
* Is separate OCR + small LLM more efficient than one large VLM?
* Latency breakdown between stages.

### Phase 5: Reading Order Deep-Dive

The hardest sub-problem for unruled handwritten text (note: Phase 2's τ = 1.00 scores only confirm single-column ordering, they say nothing about this harder case):

1. **Nemotron OCR v2 relational model:** built-in reading order prediction.
2. **PaddleOCR PP-StructureV3:** layout-aware structured output.
3. **Heuristic post-processing:** y-coordinate line grouping + x-coordinate sorting.
4. **VLM-based reading order:** ask Qwen3-VL or Florence-2 to output order.
5. **Spatial relationship models:** fine-tune a small model on reading order.

**Metric:** Kendall's tau vs. manual annotation on 5-10 samples.

### Phase 6: Error Detection Accuracy

Per error-type evaluation:

| Error Type | Detection Method | Metrics |
| --- | --- | --- |
| Capitalization | Rule-based + VLM image verification | Precision, Recall, F1 |
| Spelling | Dictionary + context (VLM/LLM) | Precision, Recall, F1 |
| Grammar | LLM analysis of transcription | Precision, Recall, F1 |
| Punctuation | Rule-based + VLM verification | Precision, Recall, F1 |
| Structural | Layout analysis (indentation, margins) | Precision, Recall, F1 |

**Bounding box accuracy:** IoU between predicted and actual error locations.

### Phase 7: Auditability Strategy

Evaluate three approaches:

1. **Per-word image crops:** store cropped image of each word with transcript.
2. **Annotated overlay:** draw recognized text over original with confidence highlighting.
3. **Side-by-side storage:** original image + structured JSON output.

Compare storage overhead, visual verifiability, and implementation complexity.

### Phase 8: Final Pipeline Assembly & Benchmark

1. Assemble best Stage 1 + Stage 2 combination.
2. Run full end-to-end benchmark against baseline.
3. Document final architecture, latency breakdown, and accuracy scores.
4. Compare total cost of ownership (local GPU amortization vs. cloud API).

---

## Metrics Framework

| Metric | How Measured | Target |
| --- | --- | --- |
| **Latency (total)** | Wall-clock time image → feedback, avg of 10 runs | **< 16.7s** (beat measured baseline; original budget was < 40s) |
| **Latency (Stage 1)** | OCR/transcription only | < 15s * |
| **Latency (Stage 2)** | Error detection + feedback | < 25s * |
| **CER** | Character Error Rate vs. ground truth | < 5% |
| **WER** | Word Error Rate vs. ground truth | < 10% |
| **Reading Order Acc.** | Kendall's tau vs. manual annotation | > 0.85 |
| **Error Detection F1** | Per error type, macro-averaged | > 0.75 |
| **Bounding Box IoU** | Mean IoU for error bounding boxes | > 0.6 |
| **VRAM Peak** | `nvidia-smi` monitoring during inference | < 11 GB (research) |
| **Setup Complexity** | 1-5 scale: install steps, config, docs quality | Lower is better |
| **Flexibility** | 1-5 scale: handles varied handwriting, layouts, noise | Higher is better |
| **Throughput** | Pages per minute (batch where applicable) | > 1 ppm |
| **Cost (cloud)** | If any cloud API component used | Documented for comparison |

\* Stage budgets derive from the original 40s end-to-end budget and predate the measured baseline; treat 21.1s as the operative end-to-end target.

---

## How Each Metric Is Computed

All metrics use a **greedy spatial matching** step first: each predicted block is matched to the best-IoU unmatched ground-truth block. Only matched pairs contribute to IoU and reading-order scores.

### CER (Character Error Rate)

CER measures **character-level** transcription accuracy on whitespace-normalized text.

```
1. normalize_text(): replace \n and \r with space, collapse
   multiple whitespace to single space, strip

2. Levenshtein distance: minimum number of single-character
   insertions, deletions, or substitutions to turn prediction
   into ground truth

3. CER = edit_distance / len(ground_truth)
```

| Edge case | Result |
|---|---|
| Both empty | 0.0 (perfect) |
| GT empty, pred non-empty | 1.0 (completely wrong) |
| Perfect match | 0.0 |

**Implementation:** space-optimized DP (two-row `prev`/`curr` arrays, O(n) memory).

**Worked example**, one word: `Khrushchov` (GT, 10 chars) vs `Khrushdov` (pred, 9 chars):

| Operation | Cost | After |
|---|---|---|
| Substitute `c` → `d` | 1 | `Khrushdov` |
| Delete `h` | 1 | `Khrushdov` |

Distance = 2, CER = 2/10 = **0.20** (20% of characters wrong on this word).

**Aggregation:** per-image CER → arithmetic mean across all 25 images. Every image contributes; no exclusions.

### WER (Word Error Rate)

Identical Levenshtein algorithm but the unit is a **word token**, not a character.

```
1. Same normalize_text() → whitespace-normalized string
2. Tokenize via .split() on whitespace → list of word tokens
3. Levenshtein distance on word lists
4. WER = edit_distance / len(gt_words)
```

**Same word example:** `Khrushchov` → `Khrushdov` is 1 wrong word out of 1 (WER = **1.0**), WER penalizes the entire word, CER penalizes the 2 characters within it. This is why WER is always >= CER: every wrong word contributes at least 1 character error.

**Aggregation:** per-image WER → arithmetic mean across all images.

### CER vs WER as a Diagnostic

The ratio WER/CER reveals the **type** of errors:

| Pattern | WER/CER | Meaning |
|---|---|---|
| PaddleOCR-VL (0.045 / 0.085) | ~2× | Errors are isolated characters in mostly-correct words, best case |
| Florence-2-large (0.061 / 0.170) | ~3× | Errors cluster in content words (names: `Khrushchov`→`Khrushdov`, `Austria`→`husbia`) |
| SmolDocling full-form (1.47 / 1.50) | ~1× | WER ≈ CER → **hallucination**: entire chunks fabricated, characters and words equally wrong |

When WER ≫ CER, the model is mostly fluent but botches specific words. When WER ≈ CER, the output is structurally broken.

### Bounding Box IoU

Intersection-over-Union for axis-aligned `[x1, y1, x2, y2]` boxes.

```
inter_area = max(0, min(x2_a, x2_b) - max(x1_a, x1_b))
           × max(0, min(y2_a, y2_b) - max(y1_a, y1_b))

area_a = (x2_a - x1_a) × (y2_a - y1_a)
area_b = (x2_b - x1_b) × (y2_b - y1_b)

IoU = inter_area / (area_a + area_b - inter_area)
```

**Block-level aggregation** (`compute_block_iou`):
```
for each predicted block:
    find the unmatched GT block with highest IoU
    if IoU ≥ 0.1 (permissive noise-filter threshold):
        record as a match

mean_iou   = average IoU over all matched pairs
recall     = matched / total_gt_blocks
precision  = matched / total_pred_blocks
```

**Bbox format normalization** before IoU:
| Source format | Conversion |
|---|---|
| `[x, y, w, h]` (xywh) | → `[x, y, x+w, y+h]` |
| `[x1,y1,x2,y2,x3,y3,x4,y4]` (quad) | → `[min(xs), min(ys), max(xs), max(ys)]` |
| Auto-detect | Treat as xywh if width < x1 or height < y1 (negative dims) |

**Aggregation:** per-image mean IoU → arithmetic mean. Images with **0 matched blocks are excluded** from the aggregate (they contribute nothing to the numerator).

### Reading Order: Kendall's τ-b

Measures how well the model's top-to-bottom, left-to-right ordering of text blocks matches the ground-truth reading order.

**Step 1, Extract predicted order** (`extract_reading_order`):
```python
# Sort predicted blocks by y (top-to-bottom), then x (left-to-right)
# Returns list of block indices in reading order
```
This is a **geometric heuristic**: reading order = sort by (y, x). It works perfectly for single-column IAM forms but would need a line-grouping step (blocks within 20px y-difference → same line) for multi-column documents in Phase 5.

**Step 2, Spatial alignment:**
```
for each predicted block in sorted order:
    find the unmatched GT block with highest IoU > 0.05
    record (predicted_position, gt_index) pair
```

**Step 3, Kendall's τ-b:**

$$\tau_b = \frac{C - D}{\sqrt{(T - T_{\text{pred}})(T - T_{\text{gt}})}}$$

Where for the two parallel rank lists (predicted and GT positions of matched blocks):
- $C$ = number of **concordant** pairs, both lists agree on which element comes first
- $D$ = number of **discordant** pairs, the lists disagree
- $T$ = total pairs = $n(n-1)/2$
- $T_{\text{pred}}$, $T_{\text{gt}}$ = tied pairs in each list

A pair $(i, j)$ is concordant if $(a_i - a_j) \cdot (b_i - b_j) > 0$, discordant if $< 0$, and tied if either difference is zero.

| τ value | Meaning |
|---|---|
| +1.0 | Perfect agreement (all pairs concordant) |
| 0.0 | No correlation (random ordering) |
| −1.0 | Perfect inverse (model reverses the order) |

**Aggregation:** per-image τ → arithmetic mean. Images with **fewer than 2 matched blocks are excluded** (Kendall's τ is undefined for n < 2).

**Why τ = 1.00 for single-column models:** IAM forms are single-column. Any top-to-bottom sort of predicted blocks exactly matches the ground-truth `[0, 1, 2, ...]` ordering. This confirms correct vertical ordering but does **not** test multi-column or unruled reading order, that is the focus of Phase 5. For region-level detectors (DocLayoutYOLO, PP-DocLayout-L), τ drops below 1.0 because region-level blocks don't map 1:1 to line-level GT blocks, producing detectable ordering disagreements.

---

## Results (Live)

> Updated as each candidate is evaluated. CER/WER are whitespace-normalized. Full JSON in `benchmark/results/`.
>
> **The authoritative accuracy comparison is the [handwriting-only results table](#handwriting-only-results-authoritative-sorted-by-cer).** The tables below cover the full-form evaluation (subject to the printed-text confound) and operational characteristics.

### Full-Form Results (printed-text confound, see caveat above)

Sorted by CER. Bbox IoU and reading-order τ are excluded here because they were scored in the cropped coordinate space (see handwriting-only table).

| Candidate | Latency (s) | CER | WER | Text Source | Note |
|---|---|---|---|---|---|
| **Florence-2-large** | 1.05 | **0.061** | 0.170 | Handwriting (no header) | Appears to read the handwritten section even on full forms |
| **MonkeyOCR** (GGUF, CPU) | 5.96 | 0.58 | 0.65 | Printed (headers in output) | CER driven by generation repetition, not misrecognition |
| **Nemotron OCR v2** | 0.09 | 1.17 | 1.24 | Printed (headers in output) | Recognizer trained on printed documents |
| *(baseline)* **Doc AI + Gemini** | 16.7 | 1.22 | 1.22 | Full form | Scope mismatch vs handwritten-only GT inflates CER (handwriting-only CER: 0.108 word-level, 0.095 line-level) |
| **SmolDocling-256M** | 12.2 | 1.47 | 1.50 | Ambiguous (skips header) | Repetition/hallucination on handwriting |
| **GOT-OCR2.0** | 35.9 | 2.93 | 3.92 | Ambiguous (skips header) | Transcribes printed + handwritten; chat-template tokens need cleanup |
| **PaddleOCR-VL-1.6** | 31.94 (cropped) | **0.045** | 0.085 | Handwriting (cropped) | **#1 CER.** Bimodal latency (3.7-61.9s), 12 GB VRAM causes WDDM spill on long-text images. Standalone benchmark. |

### Operational Characteristics

| Candidate | Latency: full form / cropped (s) | VRAM Peak | Throughput | Setup (1 = easiest) | Flexibility (5 = best) |
|---|---|---|---|---|---|
| **Florence-2-large** | 1.05 / 1.05 (1.49 with bboxes) | 2.0 GB | - | 4 | 5 |
| **PaddleOCR-VL-1.6** | 22.48 / 3.97 | not recorded (~2-3 GB est.) | - | - | - |
| **Nemotron OCR v2** | 0.09 / 0.17 | 0.6 GB | 664 ppm | 4 | 4 |
| **SmolDocling-256M** | 12.2 / 6.23 | 0.8 GB | 4.9 ppm | 3 | 3 |
| **MonkeyOCR** (GGUF) | 5.96 / 3.79 | 0 (CPU) | 10.1 ppm | 3 | 2 |
| **GOT-OCR2.0** | 35.9 / 38.95 | 3.4 GB | - | 2 | 3 |
| **DocLayoutYOLO** (layout only) | - / 0.10 | not recorded | - | - | - |
| *(baseline)* **Doc AI + Gemini** | 16.7 / 2.8 (OCR only) | N/A (cloud) | N/A | N/A | N/A |

### Florence-2-large Detailed Findings (incl. Bbox & Reading Order)

| Metric | Value | Notes |
|---|---|---|
| Model | `microsoft/Florence-2-large` (770M) | Runs in `florencetf` conda env (transformers 4.40.0) |
| Avg latency | 1.05 s (plain `<OCR>`) / 1.49 s (`<OCR_WITH_REGION>`) | Fastest GPU model; 0.44s overhead for boxes |
| VRAM peak | 2.0 GB | Fits 12 GB with large headroom |
| Handwriting CER | **0.061** | #1 line-level of all models (Doc AI word-level: 0.108, Qwen3-VL-8B word-level: 0.035) |
| **Mean Bbox IoU** | **0.76** | vs IAM XML line-level GT. All GT blocks matched on all 25 images (100% recall). |
| **Mean Kendall's τ** | **1.00** | Top-to-bottom y-sort of model bboxes exactly matches GT order (single-column). |
| Bbox format | 4-corner quad → xyxy | `[x1,y1,x2,y2,x3,y3,x4,y4]` converted to `[x_min,y_min,x_max,y_max]` |
| Script | `scripts/eval_bbox_reading_order.py florence2` | Requires `conda activate florencetf` |
| Speed-optimized variant | Florence-2-base (230M): CER 0.187, 1.2 s | Fallback if latency budget tightens |
| Key finding | - | The 28% IoU gap vs GT is primarily because Florence-2 outputs phrase-level regions while GT is line-level, plus some vertical offset in predicted boxes, not mislocalization. |

### PaddleOCR-VL-1.6 Detailed Findings

| Metric | Value | Notes |
|---|---|---|
| Model | PaddleOCR-VL-1.6 via native `PaddleOCRVL` pipeline | NOT HuggingFace transformers, PaddlePaddle inference engine |
| Avg latency | 31.94 s (cropped handwriting) | Bimodal: 3.7-4.9s for short text, 30-62s for long text. WDDM sysmem fallback on 12 GB VRAM causes bimodality. On >=16 GB expect sub-5s uniform. |
| Handwriting CER | **0.045** | First model to beat the <5% CER target (now #4 overall) |
| Handwriting WER | **0.085** | Beats <10% WER target |
| Bounding boxes | Document-level blocks | 1 block per cropped image covering ~entire handwritten region. Not comparable to line-level GT (IoU/tau not scored). |
| VRAM peak | ~8 GB | Dual-model load: PP-DocLayoutV3 + PaddleOCR-VL-1.6. Leaves ~4 GB headroom on 12 GB cards, insufficient for long-text KV cache. |
| Environment | Docker (`sm120-offline` image) | `.venv_paddleocr` native path confirmed broken on Blackwell. Docker is the only working path, see [PaddleOCR-VL Docker Workflow](#paddleocr-vl-docker-workflow). |
| Blocker history | PaddlePaddle 3.2.1 (PyPI default) lacks sm_120 | Resolved via official `sm120-offline` Docker image. Also discovered: import conflict with project modules, WSL2 VRAM leak between `--rm` containers, and JIT compilation overhead. |
| Benchmark claim | 96.3% OmniDocBench (SOTA doc VLM) | Vendor-reported |
| Script | `scripts/bench_paddleocr_handwritten.py` | Standalone, zero project imports, data-only Docker mount. Also: `scripts/paddle_hw.py` (HW-only), `scripts/paddle_ff.py` (full-form), `scripts/viz_paddleocr.py` (visualizations). |

#### PaddleOCR-VL Layout Output Format

`PaddleOCRVL.predict()` returns a list of result objects, each with three accessors:

| Accessor | Path | Content | Format |
|---|---|---|---|
| **JSON** | `res.json["res"]` | Structured document parse with bboxes + layout | `{"parsing_res_list": [...], "layout_det_res": [...]}` |
| **Markdown** | `res.markdown["markdown_texts"]` | Full document as markdown | String (preserves structure: headings, tables, lists) |
| **Plain text** | `res.text` | Plain text transcription | String (no structure) |

**`parsing_res_list` entries** (one per document-level block):
```json
{
  "block_bbox": "[x1, y1, x2, y2]",   // string or list — xyxy coordinates
  "block_content": "text content..."     // text within this block
}
```

**`layout_det_res`** (layout detection):
```json
{
  "bbox": [x1, y1, x2, y2],   // region bounding box
  "label": "text"             // region type: text, table, figure, formula, etc.
}
```

**Key limitation for this benchmark:** PaddleOCR-VL is a *document structure* model, it outputs paragraph/block-level bboxes, not line-level. On the cropped handwritten images, it typically outputs 1 block covering the entire handwritten region (bbox spans 0-1857px). This is correct document structure but cannot be scored against line-level GT bboxes via IoU or reading-order τ. For line-level bboxes, Florence-2-large (IoU 0.76) is the correct tool.

**Structured output example** (from `a04-039.png`):
- 1 block: `bbox=[0.0, 1.0, 1857.0, 1762.0]` containing all 470 characters of handwritten text
- Markdown: preserves line breaks and paragraph structure from the original
- Plain text: continuous transcription (same content, no structure)

#### PaddleOCR PP-DocLayout-L Layout Detection

PaddleOCR provides a standalone `LayoutDetection` module (separate from the PaddleOCR-VL VLM) that uses an RT-DETR-L detection model to identify document regions with class labels and bounding boxes. It runs in the same Docker image, is fast (~0.07s), and produces per-region bboxes suitable for IoU and reading-order evaluation.

**API:**
```python
from paddleocr import LayoutDetection
model = LayoutDetection(model_name="PP-DocLayout-L")
output = model.predict("image.png", batch_size=1, layout_nms=True)
# → [{"boxes": [{"cls_id": 2, "label": "text", "score": 0.931,
#                "coordinate": [xmin, ymin, xmax, ymax]}, ...]}]
```

**Available models** (in ascending size):

| Model | mAP | Params | Size | Notes |
|---|---|---|---|---|
| PP-DocLayout-S | 70.9% | 18M | 4.8 MB | Fastest, PicoDet-S. 0 detected regions on cropped handwriting. |
| PP-DocLayout-M | 75.2% | 43M | 22 MB | Balanced. |
| **PP-DocLayout-L** | **90.4%** | 503M | 123 MB | **Used in this benchmark.** RT-DETR-L. 3-9 regions per page. |
| PP-DocLayout_plus-L | 83.2% | 634M | 126 MB | 20-category, higher precision on complex layouts. |
| PP-DocBlockLayout | 95.9% | 506M | 123 MB | 1-category (block), highest mAP. |

**Results on IAM forms (25 images):**

| Dataset | Avg Latency | Regions/img | Bbox IoU | Reading τ | Notes |
|---|---|---|---|---|---|
| Full-form (curated) | 0.071s | 3-9 (mostly 4) | 0.08 | **0.92** | Detects text, headers, paragraph titles. g05-098.png: 9 regions, IoU 0.78. |
| Handwritten (cropped) | 0.096s | 0-1 (mostly 1) | 0.02 | 0.80 | Detects 1 region but misclassifies handwriting as "formula"/"image". Single-region τ = 0.80 is trivial (1-item order). |

**Key findings:**
- PP-DocLayout-L is the best layout detector tested on IAM forms, 0.07s, τ = 0.92, vs DocLayoutYOLO's τ = -0.17.
- On full-form images, it consistently detects the text region, header, and additional structural elements.
- On cropped handwritten images, it detects 1 region (correct bbox, wrong label). Layout detection isn't designed for single-text-region crops.
- IoU of 0.08 is region-level vs line-level GT, same granularity mismatch as all layout detectors. Per-image IoU can be high (0.78-0.83) when region count matches.
- The model can be combined with PaddleOCR-VL in a pipeline: LayoutDetection for region bboxes + reading order, PaddleOCR-VL for text transcription.

**Visualizations:** `benchmark/visualizations/paddleocr_layout/` (5 representative full-form images).

**Result files:** `benchmark/results/paddleocr_layoutdet_handwritten.json`, `benchmark/results/paddleocr_layoutdet_curated.json`.

### GOT-OCR2.0 Detailed Findings

| Metric | Value | Notes |
|---|---|---|
| Model | `stepfun-ai/GOT-OCR-2.0-hf` | HuggingFace transformers, BF16 |
| Avg latency | 35.9 s | 70% slower than cloud baseline (21.1s) |
| VRAM peak | 3.4 GB | Fits 12 GB comfortably |
| CER (normalized) | 2.93 (full form) / 0.088 (cropped) | Cropped score dramatically better than previously reported 4.32 (single-image bug). Full-form score inflated by printed/handwritten scope mismatch. |
| WER (normalized) | 3.92 / 0.263 (cropped) | Same scope mismatch on full form; cropped score competitive |
| Reading order | - | Not evaluated (plain OCR mode, no structured output) |
| Bounding boxes | None | No bbox output in any mode. Format mode = text formatting only. Fine-grained mode takes a bbox as *input*, not output. Verified against HF transformers and original model source. |
| Setup complexity | 2/5 | Straightforward `AutoModelForImageTextToText` + `AutoProcessor`. One-line install. |
| Flexibility | 3/5 | Handles varied handwriting on full forms. Output includes chat-template tokens requiring cleanup. |
| Key issue | - | Output includes system/user/assistant role markers and IAM metadata headers; cleanup regex needed. Requires full-page context, unsuitable for the cropped-handwriting pipeline. |

### SmolDocling-256M Detailed Findings

| Metric | Value | Notes |
|---|---|---|
| Model | `docling-project/SmolDocling-256M-preview` | `AutoModelForMultimodalLM` (requires transformers 5.x) |
| Avg latency | 12.2 s (full form) / 6.23 s (cropped) | First candidate to beat the 21.1s baseline (by 42% on full forms) |
| VRAM peak | 0.8 GB | Fits 12 GB with 15× headroom |
| CER (normalized) | 1.47 (full form) / 0.107 (cropped) | Full-form score inflated by hallucination/repetition; cropped score is competitive |
| WER (normalized) | 1.50 (full form) / 0.232 (cropped) | Same hallucination pattern as CER |
| Throughput | 4.9 ppm | |
| Reading order | τ = 1.00 | DocTags includes positional coordinates; order inferable from bbox sorting |
| Bounding boxes | Partial (IoU 0.24) | DocTags parser extracts 2-5 coarse blocks per page with pixel coordinates. Full DocTags→DoclingDocument parsing would need the `docling` library. |
| Setup complexity | 3/5 | Requires transformers 5.x, chat template, and a custom regex parser for DocTags. |
| Flexibility | 3/5 | Handles printed documents well; hallucinates/repeats on handwriting (not trained on it). |
| Key caveat | - | Full-form CER/WER are deceptively low because whitespace normalization masks repetition. |
| DocTags format | - | Proprietary tokens (`<loc_N>` for coords, `<text>`, `<table>`, etc.). Coordinates in 0-999 normalized bin space. Parsable via regex. |

### Nemotron OCR v2 Detailed Findings

| Metric | Value | Notes |
|---|---|---|
| Model | `nvidia/nemotron-ocr-v2` (v2_english) | 54M params: detector (RegNetX-8GF) + recognizer (Transformer) + relational model |
| Avg latency | 0.09 s (full form) / 0.07 s (cropped) | **239× faster than cloud baseline.** Confirmed across all 25 curated images. |
| VRAM peak | 0.6 GB | Smallest footprint; 20× headroom |
| CER (normalized) | 1.17 (full form, printed text) / 0.214 (cropped handwriting) | Recognizer trained on printed documents, handwriting CER improved from previously reported 0.74 (single-image bug) |
| WER (normalized) | 1.24 / 0.523 (cropped) | Same printed-document bias |
| Throughput | 664 ppm | 100×+ any other candidate. Production-ready. |
| Reading order | Built-in (τ = 0.89) | Relational model predicts reading order; not uniformly 1.0, only ~40% of GT blocks matched across images. |
| Bounding boxes | Partial (IoU 0.28) | 4-corner quads, denormalized 0-1 → pixels. ~40% of line-level blocks matched. Bbox normalizer bug fixed (was double-converting xyxy as xywh). |
| Setup complexity | 4/5 | git-lfs clone, CUDA toolkit, C++ build; CUDA version must match PyTorch. Needs the separate `aiml` conda env (CUDA 13.0 + PyTorch 2.12). |
| Flexibility | 4/5 | Detection/localization works on everything; recognition only reliable on printed text. |
| Text source | Printed | Outputs form headers ("Sentence Database", "A04-039"), confirmed reading the machine-printed region on full forms. |

### MonkeyOCR Detailed Findings

| Metric | Value | Notes |
|---|---|---|
| Model | `dinhquangson/MonkeyOCR-pro-1.2B-Vision-GGUF` | Qwen2-VL based, GGUF Q4_K_M quantization |
| Avg latency | 5.96 s (full form) / 3.79 s (cropped) | CPU-only, llama.cpp Vulkan backend not detected on RTX 5070 Ti. Requires `ctx-size=8192` + `image-min-tokens=1024`. |
| VRAM peak | 0 MB | CPU inference |
| CER (normalized) | 0.58 (full form) / 0.11 (cropped) | Full-form score improved from 0.81 after fixing context starvation; remaining error driven by generation repetition, not misrecognition. Cropped score from single image, needs rerun on full 25-image set. |
| WER (normalized) | 0.65 (full form) | Improved from 0.78 after context fix |
| Throughput | 10.1 ppm | Down from 56.2 ppm due to longer output (676 vs 87 chars) |
| Reading order / bboxes (this eval) | None | GGUF/llama-server path is text-only, recognition component without structure detection |
| Bounding boxes (official pipeline) | Native | DocLayoutYOLO → layout bboxes → VLM recognition → layoutreader for reading order; structured markdown output. Requires LMDeploy/vLLM backend + separate layout weights, see official repo. |
| Setup complexity | 3/5 | Pre-built llama.cpp b9596 binaries; start llama-server with specific flags. No Python dependency issues. |
| Flexibility | 2/5 | Accurate on typed text; repeats itself on longer passages. Officially does not support handwritten content. |
| Key issue | - | **Generation-control problem, not OCR problem.** Recognizes text correctly but cannot stop, repeats the same paragraph 2-3× with variations. At `ctx-size=4096`, image tokens don't fit and output truncates to 87 chars. ("Sentence Database" in output is NOT a hallucination, it's printed on the form.) |
| Root cause | - | Qwen2-VL is a generative LMM, not a dedicated OCR engine; the LLM "completes" text beyond the visible image. Known issue with small VLMs used for transcription. |
| Text source | Printed | Outputs form headers on full forms, confirmed reading the machine-printed region. |

### Phase 2 Environment Changes

| Change | Detail |
|---|---|
| **transformers 4.57.6 → 5.8.1** | Required for SmolDocling `AutoModelForMultimodalLM`. Backward compat verified for GOT-OCR2.0. |
| **PaddlePaddle 3.2.1 installed (then superseded)** | Initial install alongside PyTorch 2.11.0+cu130 failed on Blackwell (no sm_120). Superseded by 3.4.0+ cu129 in `.venv_paddleocr`. |
| **PaddleOCR 3.6.0 installed** | Native PaddleOCR-VL API, working once the Blackwell-compatible PaddlePaddle build was in place. |
| **Florence-2 unblocked via `florencetf` env** | Works with transformers 4.40.0. Base (230M): CER 0.187. Large (770M): CER 0.061. |
| **`.venv_paddleocr` created** | Isolates PaddlePaddle 3.4.0+cu129 (bundled CUDA 12.9 NCCL/cuBLAS) from PyTorch's CUDA 13.0 stack. |

### Blocker Log

**Resolved**

| Candidate | Blocker | Root Cause | Resolution |
|---|---|---|---|
| **PaddleOCR-VL-1.6** | `RuntimeError: Unsupported GPU architecture` in `paddle_inference.create_predictor()` | PaddlePaddle 3.2.1 (PyPI default) does not support Blackwell sm_120 | Install PaddlePaddle 3.4.0+ from the cu129 index in the isolated `.venv_paddleocr`, or use the official `sm120` Docker image. (A vLLM backend that bypasses the PaddlePaddle engine remains an untested alternative.) |

**Open**

| Item | Impact | Notes |
|---|---|---|
| **bitsandbytes lacks Blackwell support** | INT4 quantization unavailable → blocks Qwen3-VL-8B-Instruct (Tier 2, Phase 3) on the 12 GB research GPU | Revisit when bitsandbytes ships sm_120 kernels, or evaluate in BF16 on a larger GPU / cloud API |

### Decisions

* **Research is empirical:** every finding is implemented and measured locally; no theoretical-only evaluations.
* **12 GB VRAM is a research-phase limit only:** models requiring >11 GB are evaluated via quantization; larger GPUs offer more headroom.
* **Two-stage architecture assumed**, with investigation into end-to-end alternatives.
* **English only:** multilingual is out of scope.
* **Auditability TBD:** Phase 7 will determine the best approach.
* **Cloud APIs allowed for Stage 2 comparison:** the research goal is fully local; cloud APIs are an acceptable fallback.
* **transformers 5.x adopted:** SmolDocling required `AutoModelForMultimodalLM`; the 4.57.6 constraint is lifted. Backward compat verified for GOT-OCR2.0.

## Further Considerations

1. **Fine-tuning:** If no off-the-shelf model meets accuracy targets (note: best local CER is 6.1% vs the <5% target), a Phase 9 could explore fine-tuning SmolDocling-256M or GOT-OCR2.0 on a custom handwriting dataset.
2. **vLLM / SGLang acceleration:** For VLM candidates, optimized inference engines could significantly improve throughput. SmolDocling reports 0.35s/page on A100 via vLLM, test on RTX 5070 Ti where applicable.
3. **ONNX / TensorRT:** Converting the best model to ONNX or TensorRT could further reduce latency (out of scope for this research phase).
4. **PaddleOCR-VL vLLM backend:** No longer needed to unblock Blackwell (resolved via cu129 wheels) but may still be worth testing for serving throughput.
5. **Florence-2 on old transformers:** Resolved by pinning `transformers==4.40.0` in the separate `florencetf` conda env. See `scripts/bench_florence2.py`.
6. **MonkeyOCR, llama.cpp for text, DocLayoutYOLO for bboxes:** Pre-built llama.cpp binaries with Qwen2-VL native support made text recognition trivial to serve. DocLayoutYOLO (40.7 MB, PyTorch, no PaddlePaddle) provides native layout-detection bboxes at 0.10s/image and works on Blackwell. See `scripts/eval_bbox_reading_order.py doclayout_yolo` and `scripts/generate_model_visualizations.py monkeyocr`.
7. **Handwriting evaluation completed:** All 6 Tier-1 candidates have handwriting-only CER via XML-guided cropping, see [Handwriting-Only Re-Evaluation](#handwriting-only-re-evaluation-xml-guided-cropping). Florence-2-large leads at CER 0.061.
8. **Cloud baseline bbox evaluation:** `scripts/eval_baseline_bbox.py` evaluates Google Document AI block-level IoU and reading-order τ on the cropped handwritten dataset (25 images). Results: IoU 0.58, τ 0.91, 2.19s/page. See `benchmark/results/baseline_google_docai_layout.json`.
9. **GOT-OCR2.0 has no native bbox output (verified 2025-06-11):** Text-only OCR. Format mode = LaTeX/markdown; fine-grained mode takes a bbox as *input*. For bboxes use Florence-2 (IoU 0.76), Nemotron (0.28), SmolDocling (0.24), or DocLayoutYOLO (0.12, region-level).
10. **MonkeyOCR for handwriting:** The GGUF/llama-server path is text-only; the official pipeline adds DocLayoutYOLO + layoutreader. But MonkeyOCR **does not support handwritten content** per its official limitations, **for handwriting bboxes, use Florence-2 (IoU 0.76).**

---

## License

This project is licensed under the Apache 2.0 License. See [LICENSE](./LICENSE) for details.
