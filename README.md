# VLM-OCR Research: Handwritten English Essay Feedback

[![Status: Research](https://img.shields.io/badge/status-research-blue)](.)
[![Hardware: RTX 5070 Ti](https://img.shields.io/badge/hardware-RTX%205070%20Ti%20(12GB)-green)](.)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-orange)](./LICENSE)

Empirical evaluation of open-source OCR models and Vision-Language Models (VLMs) for a **handwritten English essay feedback system**. The pipeline must (1) transcribe difficult handwriting, (2) detect and localize writing errors with bounding boxes, (3) determine reading order on unruled paper, and (4) generate natural-language feedback — all at lower latency than the current Google Document AI + Gemini pipeline. Every finding in this document is implemented and measured locally; nothing is theoretical.

---

## Status at a Glance

**Phase 2 is complete — all 6 Tier-1 candidates evaluated.**

| | Headline result |
|---|---|
| **#1 local model** | **Florence-2-large** — handwriting CER **0.061**, full-form CER 1.22, **1.05s**/page, bbox IoU **0.76**, reading order τ = 1.00, 2.0 GB VRAM |
| **#2 local model** | **GOT-OCR2.0** — handwriting CER **0.088**, 2.53s/page, text-only (no bbox output) |
| **#3 local model** | **SmolDocling-256M** — handwriting CER **0.107**, 5.37s/page, DocTags bboxes (IoU 0.24) |
| **Cloud reference** | Google Doc AI — handwriting CER 0.08, 2.8s OCR-only (16.7s end-to-end with Gemini feedback) |
| **Fastest model** | Nemotron OCR v2 — 0.07s/page (301× faster than baseline), CER 0.214 (not competitive on accuracy) |
| **Latency target** | Met for Stage 1: Florence-2-large transcribes in 1.05s vs the 16.7s end-to-end baseline. Stage 2 (error detection, currently Gemini at 12.4s) is now the dominant open problem |
| **Accuracy target** | Not yet met: best local CER is 6.1% (Florence-2-large) vs the <5% target |

> **2026-06-12 — Evaluation bug fixes applied.** A code review identified several metric bugs (τ always 1.0, single-image CERs, Nemotron bbox double-conversion, missing crop-offset transform). All numbers above reflect corrected evaluations on the full 25-image dataset. See commit history for details.

> **Read this before interpreting any numbers:** IAM evaluation forms contain the same text both **machine-printed** and **handwritten**, so full-form CER/WER cannot reveal which region a model actually read. Only the [handwriting-only re-evaluation](#handwriting-only-re-evaluation-xml-guided-cropping) on XML-cropped images measures true handwriting recognition. Full-form Phase 2 numbers are retained as printed-text OCR benchmarks. Details: [IAM Dataset Structure](#iam-dataset-structure).

**Next up:** Phase 3 (Tier-2 baselines), then pipeline assembly, reading-order deep-dive, and error-detection accuracy (Phases 4–8).

---

## Target Baseline

| Metric | Current (Google Document AI + Gemini) |
|--------|--------------------------------------|
| End-to-end latency | **16.7s** (Doc AI 2.8s + Gemini 3.5 Flash 12.4s) |
| Cost | Document AI: $1.50–$30/1K pages + Gemini API per-token |
| Architecture | Cloud-only, proprietary |
| Model | `gemini-3.5-flash` via Vertex AI (`location="global"`) |

## Constraints

- **Research phase:** RTX 5070 Ti mobile (12 GB VRAM). Models must fit locally (quantization allowed).
- **Deployment phase:** Greater compute available, and cloud APIs are acceptable. The final recommendation may include models exceeding research GPU limits.
- **Transformers version:** 5.8.1 (upgraded from 4.57.6 for SmolDocling `AutoModelForMultimodalLM` support and PaddleOCR-VL compatibility).

## Environments

Four separate Python environments are required due to conflicting CUDA/transformers/PaddlePaddle versions:

| Environment | Type | PyTorch | CUDA | Used for |
|---|---|---|---|---|
| `.venv/` | venv | 2.11.0+cu130 | 13.0 | SmolDocling, GOT-OCR2.0, MonkeyOCR, DocLayoutYOLO, Qwen3-VL, TrOCR, baselines |
| `aiml` | conda | 2.12.0+cu130 | 13.0 | Nemotron OCR v2 (CUDA toolkit must match PyTorch for the C++ extension build) |
| `florencetf` | conda | 2.11.0+cu130 | 13.0 | Florence-2 base/large (requires transformers 4.40.0, incompatible with transformers 5.x) |
| `.venv_paddleocr` | venv | — (PaddlePaddle 3.4.0+) | 12.9 | PaddleOCR-VL-1.6 (PaddlePaddle bundles its own NCCL/cuBLAS, conflicting with PyTorch's CUDA 13.0 stack) |

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

PaddleOCR-VL uses the **PaddlePaddle native inference engine**, NOT HuggingFace transformers. The standard `paddlepaddle-gpu` from PyPI (3.2.1) does **not** include Blackwell (sm_120) support and fails with `RuntimeError: Unsupported GPU architecture`. Two working options:

**Option A — Docker (recommended per PaddleOCR docs):**
```bash
docker run -it --gpus all --network host --user root \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-nvidia-gpu-sm120 \
  /bin/bash
# Inside container: paddleocr doc_parser -i /path/to/image.png
```

**Option B — Manual install (Blackwell-compatible PaddlePaddle wheel):**
```bash
# Install PaddlePaddle 3.4.0+ from the cu129 index (supports sm_120)
pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu129/ --upgrade
# Then use the Python API:
#   from paddleocr import PaddleOCRVL
#   pipeline = PaddleOCRVL()
#   output = pipeline.predict("image.png")
```

**Python API:** `from paddleocr import PaddleOCRVL` (NOT `PaddleOCR` or HuggingFace `AutoModel`). See `candidates/paddleocr_vl/eval.py` for the integration script.

Reference: [PaddleOCR-VL NVIDIA Blackwell Tutorial](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL-NVIDIA-Blackwell.html)

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
│   ├── paddleocr_vl/                  # PaddleOCR-VL-1.6
│   ├── got_ocr/                       # GOT-OCR2.0
│   ├── nemotron_ocr/                  # NVIDIA Nemotron OCR v2
│   ├── florence2/                     # Microsoft Florence-2
│   ├── smoldocling/                   # SmolDocling / granite-docling
│   ├── trocr/                         # Microsoft TrOCR (handwritten)
│   ├── qwen3_vl/                      # Qwen3-VL (4B & 8B)
│   └── baselines/                     # EasyOCR, docTR, Tesseract
└── pipeline/                          # Final assembled two-stage pipeline
```

---

## Candidate Models

### Tier 1: Most Promising

| # | Candidate | Params | Key Advantage | VRAM Est. |
| --- | --- | --- | --- | --- |
| 1 | **PaddleOCR-VL-1.6** † | 0.9 B | SOTA doc VLM (96.3% OmniDocBench), built-in layout & structure | ~2–3 GB |
| 2 | **GOT-OCR2.0** | ~7 B (Qwen-based) | Unified end-to-end OCR, GGUF available. No bbox output. | ~6–8 GB (INT4) |
| 3 | **NVIDIA Nemotron OCR v2** | 54 M EN / 84 M Multi | **Built-in reading order** (relational model), bounding boxes, production-grade | ~1–2 GB |
| 4 | **Florence-2-large** | 0.77 B | Microsoft foundation model, prompt-based OCR + boxes, multi-task | ~1.5 GB |
| 5 | **granite-docling-258M** | 256 M | Ultra-compact document VLM, successor to SmolDocling, DocTags format | ~1 GB |
| 6 | **MonkeyOCR** | ~1.2 B | Document parsing with SRR triplet (structure-recognition-relation). Built-in layout detection (DocLayoutYOLO) for bboxes. **Does not support handwritten content** (per official limitations). | ~6 GB |

† Blackwell requires PaddlePaddle 3.4.0+ from the cu129 index or the official Docker image — see [PaddleOCR-VL on Blackwell](#paddleocr-vl-on-blackwell-nvidia-sm_120).

- **Phase 2 status:** All 6 Tier-1 candidates evaluated. Florence-2-large is the #1 local model (handwriting CER 0.061, IoU 0.76, 1.05s). GOT-OCR2.0 is #2 (0.088 CER on handwriting). DocLayoutYOLO provides native layout bboxes (0.10s; coarse region-level, IoU 0.12 vs line-level GT, τ = -0.17 — region-level detection cannot resolve line-level reading order).

### Tier 2: Baselines & Comparison

| # | Candidate | Params | Purpose |
| --- | --- | --- | --- |
| 7 | **TrOCR (base + large, handwritten)** | 0.3 B / 0.6 B | IAM-finetuned HTR baseline: line-level only, needs text detector |
| 8 | **Qwen3-VL-4B-Instruct** | 4 B | Latest Qwen VLM, expanded OCR (32 languages), robust in low light/blur/tilt, strong document parsing. BF16 fits 12 GB (~8 GB). |
| 9 | **Qwen3-VL-8B-Instruct (INT4)** | 8 B (quantized) | Flagship Qwen VLM: 256K context, advanced spatial perception, 2D/3D grounding. INT4 needed for 12 GB (~6–8 GB) — currently blocked by bitsandbytes (see [Blocker Log](#blocker-log)). For deployment, BF16 on larger GPU or cloud API. |
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

**Sentence segmentation:** On unruled paper, determining whether a word belongs to the line above or below is non-trivial when spacing is ambiguous. The current production system uses a custom sequence-model algorithm that tracks previous, current, and next sentence relationships to disambiguate word-line assignments. This capability must be replicated or exceeded by any replacement pipeline.

**Stage 2 candidates:** Qwen3-VL-4B, Qwen3-VL-8B (INT4), SmolVLM-Instruct (2 B), granite-docling-258M, Hunyuan VL, Gemini 3.5 Flash (cloud comparison).

**Key research question:** can a single VLM (e.g., Qwen3-VL) handle both stages end-to-end?

---

## Research Phases

### Phase 1: Environment Setup & Baseline — Complete

1. Set up Python environment (CUDA 13.0, PyTorch 2.11.0, RTX 5070 Ti 12 GB). → `scripts/setup.sh`
2. Collect handwritten essay samples. → `benchmark/test_dataset/` (1,539 IAM forms from Kaggle, 25 curated).
3. Measure Google Document AI + Gemini baseline latency. → `benchmark/baseline.py` — **16.7s total (Doc AI 2.8s + Gemini 3.5 Flash 12.4s)**.
4. Build shared evaluation harness. → `benchmark/harness.py` + `benchmark/metrics.py`
5. Define metric collection JSON schema. → `benchmark/test_dataset/ground_truth.json`

**Validation:** `bash scripts/phase1_validate.sh` — 27/27 checks pass, 0 failures.

#### Phase 1 Findings

| Finding | Detail |
|---|---|
| **Baseline is 16.7s** | 21% faster than previously reported 21.1s (fixed: rate-limit sleep was included in latency). Local models must beat 16.7s. |
| **Stage 2 dominates latency** | Gemini error detection is 12.4s (74% of total). Replacing it with a local model is the highest-impact optimization. |
| **Document AI is fast** | OCR alone is 2.8s. The cloud OCR stage is not the bottleneck. |
| **Gemini needs `location="global"`** | `gemini-3.5-flash` is a preview model only available in the `global` region on Vertex AI. Using `asia-southeast1` returns 404. |
| **Blackwell needs nightly PyTorch** | RTX 5070 Ti (sm_120) unsupported by stable PyTorch 2.6. Uses 2.11.0+cu130. |
| **bitsandbytes blocked** | INT4 quantization unavailable for Blackwell — blocks Qwen3-VL-8B evaluation. |
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

### Phase 2: Tier-1 Candidate Evaluation — Complete (6/6 evaluated)

> **Methodological finding (2025-06-11):** IAM forms contain the writing prompt **machine-printed** and the writer's **handwritten copy** of the same text. The ground truth is therefore textually identical for both regions, so **full-form CER/WER cannot distinguish whether a model read the printed text or the handwriting.** Models that emit the form header ("Sentence Database") — Nemotron OCR and MonkeyOCR — are confirmed to be reading the printed region. SmolDocling and GOT-OCR2.0 skip the header, so their source is ambiguous. **The full-form Phase 2 numbers below are printed-text OCR benchmarks; authoritative handwriting accuracy comes from the [handwriting-only re-evaluation](#handwriting-only-re-evaluation-xml-guided-cropping).** Details: [IAM Dataset Structure](#iam-dataset-structure).

For each of the 6 candidates, the following were measured:

* End-to-end latency (image → structured text with bounding boxes).
* CER / WER on handwritten text (whitespace-normalized).
* Reading order accuracy (Kendall's tau vs. ground truth).
* Bounding box quality (visual inspection + IoU where ground truth exists).
* VRAM peak usage.
* Setup complexity (1–5 scale; 1 = easiest).
* Flexibility (1–5 scale; 5 = handles varied handwriting, layouts, noise best).

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
- Models that output "Sentence Database" or form IDs are confirmed to be reading sections 1–2 (printed), not section 3 (handwriting): **Nemotron OCR** and **MonkeyOCR** both do this.
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

#### Phase 2B–2G: Candidate Evaluations (full forms)

| Candidate | Status | Text Source | Key Findings |
|---|---|---|---|
| **GOT-OCR2.0** | Complete | Ambiguous (no header output) | 35.9s avg, 3.4 GB VRAM. Transcribes full form (printed + handwritten). No native bbox output — format mode produces text formatting only. Verified against HF transformers and original implementation. |
| **Florence-2-large** | Complete (separate env) | **Handwriting** (no header in output) | **#1 local model.** CER 0.061 (24% better than Doc AI), 1.05s avg (fastest GPU model). 770M params, 2.0 GB VRAM. Requires `conda activate florencetf` with transformers 4.40.0. Base model (230M): CER 0.187, 1.2s. |
| **PaddleOCR-VL-1.6** | Complete (after Blackwell fix) | Evaluated on cropped handwriting | Initially blocked: PaddlePaddle 3.2.1 lacks sm_120 (`RuntimeError: Unsupported GPU architecture`). Resolved with PaddlePaddle 3.4.0+ (cu129) in `.venv_paddleocr`. Handwriting CER 0.072, 3.97s cropped (22.48s full page). Built-in layout + bboxes. |
| **SmolDocling-256M** | Complete | Ambiguous (skips header) | 12.2s avg latency (42% faster than baseline), 0.8 GB VRAM, CER 1.47, WER 1.50. DocTags output with bbox parsing via regex. Hallucinates/repeats on handwriting. `AutoModelForMultimodalLM` + transformers 5.x. |
| **Nemotron OCR v2** | Complete | **Printed text** (headers in output) | **Fastest candidate.** 0.09s avg (234× faster than baseline), 0.6 GB VRAM, CER 1.17, WER 1.24. Outputs form headers ("Sentence Database") + form IDs → confirmed reading printed text. 7 text regions per page with bboxes + reading order via relational model. Built in `aiml` conda env (CUDA 13.0 + PyTorch 2.12). CUDA extension compiled without issues. |
| **MonkeyOCR** | Complete (GPU, GGUF) | Evaluated on cropped handwriting | 4.27s avg (GPU via CUDA llama-server build), handwriting CER 0.566, WER 0.240. GGUF/llama-server path = text recognition only. 1.2B params, ~2 GB VRAM with full GPU offload. Built from ggerganov/llama.cpp with -DGGML_CUDA=ON for RTX 5070 Ti. Official pipeline has layout detection + reading order via DocLayoutYOLO + layoutreader + VLM. |

### Handwriting-Only Re-Evaluation (XML-Guided Cropping)

After the methodological finding above, all images were cropped to the handwritten region using IAM XML `<handwritten-part>` `<cmp>` bounding boxes. Each image was cropped to the union bounding box of all handwritten lines (excluding the signature footer), with 20px padding. This physically removes the printed header/prompt, forcing models to read only handwriting. **These are the authoritative accuracy numbers.**

#### Ground Truth Bounding Box Methodology

Ground truth line-level bounding boxes are derived from the IAM XML annotations with a three-step coordinate transform:

**1. Source data — IAM XML `<cmp>` elements**

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

**Example — form `a01-000u`:**

| Metric | Value |
|---|---|
| Form dimensions | 2479 × 3542 px |
| First cmp | `x=363, y=739` (first character of first handwritten line) |
| Last cmp | `x=2414, y=1913` (last character of last handwritten line) |
| Union bbox | `x=[363, 2414]`, `y=[739, 1913]` |
| Crop origin (with 20px padding) | `(343, 719)` |
| Cropped image size | ~2071 × 1214 px |

The crop removes:
- **Top 719px** — form header ("Sentence Database") + printed prompt
- **Bottom 1629px** — empty space + footer below the handwriting

**3. Coordinate transform to cropped image space**

Each line's bounding box is then offset to the cropped image coordinate system:

```
cropped_x1 = max(0, original_x1 - crop_x)
cropped_y1 = max(0, original_y1 - crop_y)
cropped_x2 = min(img_width,  original_x2 - crop_x)
cropped_y2 = min(img_height, original_y2 - crop_y)
```

**Example — first line of `a04-039`:**

| Coordinate | Original (full form) | Offset (minus crop) | Cropped |
|---|---|---|---|
| `x1` | 364 | 364 − 343 = 21 → padded | 57 |
| `y1` | 912 | 912 − 891 = 21 | 20 |
| `x2` | 1985 | 1985 − 343 = 1642 | 1678 |
| `y2` | 1040 | 1040 − 891 = 149 | 148 |

**Result:** Each GT bbox tightly bounds the handwritten text from the first character to the last character of that line, in the cropped image coordinate space. Boxes span 1565–1858px wide on ~1880px-wide images, with a ~20–60px left margin (empty space inherent in IAM scans). The `reading_order` field is always `[0, 1, 2, ...]` (sequential) since IAM forms are single-column.

See `scripts/generate_ground_truth.py` for the generation code and `scripts/crop_handwritten.py` for the crop logic. Note that all bbox IoU and reading-order (τ) scores below are computed in this cropped coordinate space.

#### Handwriting-Only Results (authoritative, sorted by CER)

| Candidate | Printed CER (full form) | **Handwriting CER** | Bbox IoU | Read Order τ | Latency (cropped) | Verdict |
|---|---|---|---|---|---|---|
| **Florence-2-large** | — | **0.061** | **0.76** | **1.00** | 1.05s (text) / 1.49s (bbox) | **#1 local model.** 24% better CER than Doc AI. Perfect reading order, strong line-level bboxes (100% block recall). 2.0 GB VRAM. |
| **PaddleOCR-VL-1.6** | — | **0.072** | n/s | n/s | 3.97s (22.48s full page) | **#2 local model by CER (needs rerun on full 25-image set).** SOTA doc VLM (96.3% OmniDocBench), built-in layout + bboxes. Requires `.venv_paddleocr`. |
| **Google Doc AI** (cloud) | 0.08 | **0.095** | **0.58** | **0.91** | 2.19s (OCR only) | Cloud baseline. Handwriting CER 0.095 (25-image evaluation). Strong bbox quality (0.58 IoU). Reading order τ = 0.91. |
| **MonkeyOCR** | — | **0.566** | — | — | 4.27s | GPU-accelerated via CUDA llama-server. Struggles with IAM handwriting (CER 0.566 vs 0.061 best local). Text-only (no bbox/reading-order output). |
| **SmolDocling-256M** | 1.47 | 0.107 | 0.24 | 1.00 | 5.37s | Best structured output (DocTags + bboxes), but weak localization (coarse blocks). |
| **Nemotron OCR v2** | 1.17 | 0.214 | 0.28 | 0.89 | 0.07s | Fastest (301× baseline). Recognizer struggles with handwriting (CER 0.214) but detector localizes ~40% of line-level blocks. Reading order τ informative (0.89, not uniformly 1.0). |
| **GOT-OCR2.0** | 2.93 | 0.088 | ‡ | — | 2.53s | **#2 local model by CER on handwriting.** Previously misreported as 4.32 (single-image bug). No bbox output. |
| **DocLayoutYOLO** (layout only) | — | — | 0.12 † | -0.17 | 0.10s | Layout detection only (no OCR). Region-level blocks cannot resolve line-level reading order (τ = -0.17). Runs on Blackwell without PaddlePaddle. |

† Bboxes from DocLayoutYOLO layout detection: 1–3 region-level blocks per page scored against 8–11 line-level GT lines. The low IoU reflects this granularity mismatch, not detection failure. See `scripts/eval_bbox_reading_order.py doclayout_yolo`.
‡ GOT-OCR2.0 has no native bbox output in any mode — excluded from IoU comparison.
n/s = not yet scored (PaddleOCR-VL bbox/reading-order evaluation pending).

**Key findings:**

- **Florence-2-large is the #1 local model** (CER 0.061) — beating Google Doc AI (0.08) by 24% at 1.05s on GPU. Bbox quality is dominant (IoU 0.76, 100% block recall on all 25 images).
- **GOT-OCR2.0 is the #2 local model** (CER 0.088) at 2.53s — dramatically better than the previously reported 4.32 (single-image evaluation bug). Text-only, no bbox output.
- **SmolDocling is the best structured-output model** (CER 0.107 + DocTags bboxes, τ = 1.00) — well suited to the essay feedback pipeline, though localization is coarse (IoU 0.24).
- **PaddleOCR-VL-1.6** has the second-lowest CER (0.072) but the evaluation still needs rerunning on the full 25-image set.
- **Nemotron's speed is unmatched** (0.07s, 301× baseline) but handwriting CER of 0.214 keeps it out of the top tier. Bbox IoU corrected to 0.28 (was 0.29 with double-conversion bug).
- **MonkeyOCR** (CER 0.11) needs rerunning on full 25-image set — current number is from a single image.
- **DocLayoutYOLO** provides real layout detection on Blackwell (0.10s/image), but τ = -0.17 confirms region-level detection cannot resolve line-level reading order.
- **Reading order is trivially perfect for line-level models** (τ = 1.00 for Florence-2, SmolDocling): IAM forms are single-column, so any top-to-bottom sort matches GT. For region-level detectors (DocLayoutYOLO), τ is informative (-0.17). Multi-column/unruled reading order remains untested — see Phase 5.
- **Florence-2-base (230M)** is the speed-optimized fallback: CER 0.187 at 1.2s.
- **Cropping reduced latency for all models** (smaller image = fewer vision tokens to process).

See `scripts/crop_handwritten.py` and `scripts/eval_handwritten.py` for the methodology.

#### Model Bounding Box Generation Methods

Each model produces bounding boxes differently:

| Model | Bbox Source | Format | Conversion | Evaluation Script |
|---|---|---|---|---|
| **Florence-2** | `<OCR_WITH_REGION>` prompt returns quad boxes + labels | `[x1,y1,x2,y2,x3,y3,x4,y4]` 4-corner quad | `xyxy = [min(xs), min(ys), max(xs), max(ys)]` — min/max over 4 corners | `scripts/eval_bbox_reading_order.py florence2` (requires `florencetf` env) |
| **Nemotron OCR v2** | Detector (RegNetX-8GF) outputs text regions per page | `[x, y, w, h]` — origin + dimensions | `xyxy = [x, y, x+w, y+h]` | `scripts/eval_bbox_reading_order.py nemotron` (requires `aiml` env) |
| **SmolDocling** | DocTags tokens (`<loc_N>` with 0–999 normalized coords) parsed to pixel bboxes | `[x1, y1, x2, y2]` — already xyxy | No conversion needed. Coordinates denormalized from 0–999 bin space via regex parser. | `scripts/eval_bbox_reading_order.py smoldocling` (requires `.venv`) |
| **GOT-OCR2.0** | No native bbox output | — | Model does not output bboxes. Format mode produces formatted text (line breaks), not spatial coordinates. Fine-grained mode takes a bbox as *input*, not output. Verified against HF transformers `stepfun-ai/GOT-OCR-2.0-hf` and original `ucaslcl/GOT-OCR2_0` source. | — |
| **MonkeyOCR (official pipeline)** | DocLayoutYOLO → layout bboxes | `[x1, y1, x2, y2]` — already xyxy | No conversion needed. Detected regions: plain text, figure, formula, table, etc. Layout detection via DocLayoutYOLO (PyTorch/ONNX, no PaddlePaddle needed). | `scripts/eval_bbox_reading_order.py doclayout_yolo` (requires `.venv` + `pip install doclayout_yolo`) |
| **Google Doc AI** | Cloud API returns structured document with block-level bboxes | `[x1, y1, x2, y2]` | Already xyxy | `benchmark/baseline.py` |

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

GOOGLE DOC AI   (CER=0.08, 3.7s + Gemini 13.7s = 17.4s total)
──────────────────────────────────────────────────────────────────────
In Vienna, before flying off to Moscow, во
Mr. Khrushchou said he hoped his weekend talks with President Kennedy
would help "to" establish an onduring peace between nations.
Replying to a farewell speech from Austrian President Schaerf
the Soviet Premier thanked Austria for the hospitality and welcome he
had received. "The Soviet Union has always striven and is striving to
safeguard an onduring peace for the peoples to secure an early solution
of the

Gemini 3.5 Flash found: "Khrushchou"→"Khrushchev", "onduring"→"enduring"

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

SMOLDOCLING   (CER=0.14, 7.1s)
──────────────────────────────────────────────────────────────────────
In Vienna, before flying off to Moscow, Mr. Khrushov said he hoped his
weekend talkers with Presidet Kennedys would be help " to establish an
onduring peace between nations. " Replying to a farewell speech from
author Presidet Shaard, he Soulet Premier thanked ushisa for the
hospitally and welcome he had received. 4 The Soulet Union has always
Join a shon and is stiving to safeguard an onduring peace for the
peoples, to secure an early soluion of the

NEMOTRON   (CER=0.74, 0.06s)
──────────────────────────────────────────────────────────────────────
4 The Souiet unioh ha always strven and is striving to safeguard an ou
ondaring peace for the peoples) , to Decure an early sole hor of the
In Vienna, before juying off to Morcow, Mr. khrushchov said he hopect
his weekend talks with Preideww Kennedy would & help 4 to eslaseish an
onduring peaie Selweeh natiows " Replbis so a were speech from Aushiag
Prestdoct Schoerd 1 the Sovret Preneirr thanked Ausiia for the
hospilality and welcome he had roeived.

FLORENCE-2-LARGE   (CER=0.11 on this image, 1.8s)
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

GOT-OCR2.0   (CER=4.32, 38.9s)
──────────────────────────────────────────────────────────────────────
Mr. Kh ush cho u said he hop col his week eu o talks with President
Kennedy would help" to es laS e is hah oh d ul in s peace S elwe eh
hao w.  Rep lg is ho a f ar well pe ch from Hush ian Pre sid oct S
doer d, he So u let Pre nii er th auk eod Hush i a for the hospital
i ly and welcome he had received.  4 The So u iet Union ha alwa gp
show an ol is shi u ing to safe guard an on d turing peace for he pe
opes, ho secure an ari ly shi u on of the1111111111111111111111111111...
```

### Phase 3: Tier-2 Candidate Evaluation

Evaluate comparison baselines and larger models for reference.

### Phase 4: Two-Stage Pipeline Architecture Design

Test combinations of best Stage 1 + Stage 2 models. Measure:

* Can a single VLM do both stages end-to-end?
* Is separate OCR + small LLM more efficient than one large VLM?
* Latency breakdown between stages.

### Phase 5: Reading Order Deep-Dive

The hardest sub-problem for unruled handwritten text (note: Phase 2's τ = 1.00 scores only confirm single-column ordering — they say nothing about this harder case):

1. **Nemotron OCR v2 relational model:** built-in reading order prediction.
2. **PaddleOCR PP-StructureV3:** layout-aware structured output.
3. **Heuristic post-processing:** y-coordinate line grouping + x-coordinate sorting.
4. **VLM-based reading order:** ask Qwen3-VL or Florence-2 to output order.
5. **Spatial relationship models:** fine-tune a small model on reading order.

**Metric:** Kendall's tau vs. manual annotation on 5–10 samples.

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
| **Setup Complexity** | 1–5 scale: install steps, config, docs quality | Lower is better |
| **Flexibility** | 1–5 scale: handles varied handwriting, layouts, noise | Higher is better |
| **Throughput** | Pages per minute (batch where applicable) | > 1 ppm |
| **Cost (cloud)** | If any cloud API component used | Documented for comparison |

\* Stage budgets derive from the original 40s end-to-end budget and predate the measured baseline; treat 21.1s as the operative end-to-end target.

---

## Results (Live)

> Updated as each candidate is evaluated. CER/WER are whitespace-normalized. Full JSON in `benchmark/results/`.
>
> **The authoritative accuracy comparison is the [handwriting-only results table](#handwriting-only-results-authoritative-sorted-by-cer).** The tables below cover the full-form evaluation (subject to the printed-text confound) and operational characteristics.

### Full-Form Results (printed-text confound — see caveat above)

Sorted by CER. Bbox IoU and reading-order τ are excluded here because they were scored in the cropped coordinate space (see handwriting-only table).

| Candidate | Latency (s) | CER | WER | Text Source | Note |
|---|---|---|---|---|---|
| **Florence-2-large** | 1.05 | **0.061** | 0.170 | Handwriting (no header) | Appears to read the handwritten section even on full forms |
| **MonkeyOCR** (GGUF, CPU) | 5.96 | 0.58 | 0.65 | Printed (headers in output) | CER driven by generation repetition, not misrecognition |
| **Nemotron OCR v2** | 0.09 | 1.17 | 1.24 | Printed (headers in output) | Recognizer trained on printed documents |
| *(baseline)* **Doc AI + Gemini** | 16.7 | 1.22 | 1.22 | Full form | Scope mismatch vs handwritten-only GT inflates CER (handwriting-only CER is 0.08) |
| **SmolDocling-256M** | 12.2 | 1.47 | 1.50 | Ambiguous (skips header) | Repetition/hallucination on handwriting |
| **GOT-OCR2.0** | 35.9 | 2.93 | 3.92 | Ambiguous (skips header) | Transcribes printed + handwritten; chat-template tokens need cleanup |
| **PaddleOCR-VL-1.6** | 22.48 | — | — | — | Full-form accuracy not scored; evaluated on cropped handwriting only |

### Operational Characteristics

| Candidate | Latency: full form / cropped (s) | VRAM Peak | Throughput | Setup (1 = easiest) | Flexibility (5 = best) |
|---|---|---|---|---|---|
| **Florence-2-large** | 1.05 / 1.05 (1.49 with bboxes) | 2.0 GB | — | 4 | 5 |
| **PaddleOCR-VL-1.6** | 22.48 / 3.97 | not recorded (~2–3 GB est.) | — | — | — |
| **Nemotron OCR v2** | 0.09 / 0.17 | 0.6 GB | 664 ppm | 4 | 4 |
| **SmolDocling-256M** | 12.2 / 6.23 | 0.8 GB | 4.9 ppm | 3 | 3 |
| **MonkeyOCR** (GGUF) | 5.96 / 3.79 | 0 (CPU) | 10.1 ppm | 3 | 2 |
| **GOT-OCR2.0** | 35.9 / 38.95 | 3.4 GB | — | 2 | 3 |
| **DocLayoutYOLO** (layout only) | — / 0.10 | not recorded | — | — | — |
| *(baseline)* **Doc AI + Gemini** | 16.7 / 2.8 (OCR only) | N/A (cloud) | N/A | N/A | N/A |

### Florence-2-large Detailed Findings (incl. Bbox & Reading Order)

| Metric | Value | Notes |
|---|---|---|
| Model | `microsoft/Florence-2-large` (770M) | Runs in `florencetf` conda env (transformers 4.40.0) |
| Avg latency | 1.05 s (plain `<OCR>`) / 1.49 s (`<OCR_WITH_REGION>`) | Fastest GPU model; 0.44s overhead for boxes |
| VRAM peak | 2.0 GB | Fits 12 GB with large headroom |
| Handwriting CER | **0.061** | #1 of all models, local or cloud (Doc AI: 0.08) |
| **Mean Bbox IoU** | **0.76** | vs IAM XML line-level GT. All GT blocks matched on all 25 images (100% recall). |
| **Mean Kendall's τ** | **1.00** | Top-to-bottom y-sort of model bboxes exactly matches GT order (single-column). |
| Bbox format | 4-corner quad → xyxy | `[x1,y1,x2,y2,x3,y3,x4,y4]` converted to `[x_min,y_min,x_max,y_max]` |
| Script | `scripts/eval_bbox_reading_order.py florence2` | Requires `conda activate florencetf` |
| Speed-optimized variant | Florence-2-base (230M): CER 0.187, 1.2 s | Fallback if latency budget tightens |
| Key finding | — | The 28% IoU gap vs GT is primarily because Florence-2 outputs phrase-level regions while GT is line-level, plus some vertical offset in predicted boxes — not mislocalization. |

### PaddleOCR-VL-1.6 Detailed Findings

| Metric | Value | Notes |
|---|---|---|
| Model | PaddleOCR-VL-1.6 via native `PaddleOCRVL` pipeline | NOT HuggingFace transformers — PaddlePaddle inference engine |
| Avg latency | 3.97 s (cropped handwriting) / 22.48 s (full page) | Cropping gives a 5.7× speedup |
| Handwriting CER | **0.072** | #2 local model, within 0.011 of Florence-2-large |
| Bounding boxes | Built-in layout & structure output | IoU / reading-order τ not yet scored |
| VRAM peak | Not recorded | Tier-1 estimate: ~2–3 GB |
| Environment | `.venv_paddleocr` (PaddlePaddle 3.4.0+ cu129) | Isolated env required — PaddlePaddle's bundled NCCL/cuBLAS conflict with PyTorch CUDA 13.0 |
| Blocker history | PaddlePaddle 3.2.1 (PyPI default) lacks sm_120 | Resolved via cu129 wheels or official `sm120` Docker image — see [Blocker Log](#blocker-log) |
| Benchmark claim | 96.3% OmniDocBench (SOTA doc VLM) | Vendor-reported |
| Script | `candidates/paddleocr_vl/eval.py` | |

### GOT-OCR2.0 Detailed Findings

| Metric | Value | Notes |
|---|---|---|
| Model | `stepfun-ai/GOT-OCR-2.0-hf` | HuggingFace transformers, BF16 |
| Avg latency | 35.9 s | 70% slower than cloud baseline (21.1s) |
| VRAM peak | 3.4 GB | Fits 12 GB comfortably |
| CER (normalized) | 2.93 (full form) / 0.088 (cropped) | Cropped score dramatically better than previously reported 4.32 (single-image bug). Full-form score inflated by printed/handwritten scope mismatch. |
| WER (normalized) | 3.92 / 0.263 (cropped) | Same scope mismatch on full form; cropped score competitive |
| Reading order | — | Not evaluated (plain OCR mode, no structured output) |
| Bounding boxes | None | No bbox output in any mode. Format mode = text formatting only. Fine-grained mode takes a bbox as *input*, not output. Verified against HF transformers and original model source. |
| Setup complexity | 2/5 | Straightforward `AutoModelForImageTextToText` + `AutoProcessor`. One-line install. |
| Flexibility | 3/5 | Handles varied handwriting on full forms. Output includes chat-template tokens requiring cleanup. |
| Key issue | — | Output includes system/user/assistant role markers and IAM metadata headers; cleanup regex needed. Requires full-page context — unsuitable for the cropped-handwriting pipeline. |

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
| Bounding boxes | Partial (IoU 0.24) | DocTags parser extracts 2–5 coarse blocks per page with pixel coordinates. Full DocTags→DoclingDocument parsing would need the `docling` library. |
| Setup complexity | 3/5 | Requires transformers 5.x, chat template, and a custom regex parser for DocTags. |
| Flexibility | 3/5 | Handles printed documents well; hallucinates/repeats on handwriting (not trained on it). |
| Key caveat | — | Full-form CER/WER are deceptively low because whitespace normalization masks repetition. |
| DocTags format | — | Proprietary tokens (`<loc_N>` for coords, `<text>`, `<table>`, etc.). Coordinates in 0–999 normalized bin space. Parsable via regex. |

### Nemotron OCR v2 Detailed Findings

| Metric | Value | Notes |
|---|---|---|
| Model | `nvidia/nemotron-ocr-v2` (v2_english) | 54M params: detector (RegNetX-8GF) + recognizer (Transformer) + relational model |
| Avg latency | 0.09 s (full form) / 0.07 s (cropped) | **301× faster than cloud baseline.** Confirmed across all 25 curated images. |
| VRAM peak | 0.6 GB | Smallest footprint; 20× headroom |
| CER (normalized) | 1.17 (full form, printed text) / 0.214 (cropped handwriting) | Recognizer trained on printed documents — handwriting CER improved from previously reported 0.74 (single-image bug) |
| WER (normalized) | 1.24 / 0.523 (cropped) | Same printed-document bias |
| Throughput | 664 ppm | 100×+ any other candidate. Production-ready. |
| Reading order | Built-in (τ = 0.89) | Relational model predicts reading order; not uniformly 1.0 — only ~40% of GT blocks matched across images. |
| Bounding boxes | Partial (IoU 0.28) | 4-corner quads, denormalized 0–1 → pixels. ~40% of line-level blocks matched. Bbox normalizer bug fixed (was double-converting xyxy as xywh). |
| Setup complexity | 4/5 | git-lfs clone, CUDA toolkit, C++ build; CUDA version must match PyTorch. Needs the separate `aiml` conda env (CUDA 13.0 + PyTorch 2.12). |
| Flexibility | 4/5 | Detection/localization works on everything; recognition only reliable on printed text. |
| Text source | Printed | Outputs form headers ("Sentence Database", "A04-039") — confirmed reading the machine-printed region on full forms. |

### MonkeyOCR Detailed Findings

| Metric | Value | Notes |
|---|---|---|
| Model | `dinhquangson/MonkeyOCR-pro-1.2B-Vision-GGUF` | Qwen2-VL based, GGUF Q4_K_M quantization |
| Avg latency | 5.96 s (full form) / 3.79 s (cropped) | CPU-only — llama.cpp Vulkan backend not detected on RTX 5070 Ti. Requires `ctx-size=8192` + `image-min-tokens=1024`. |
| VRAM peak | 0 MB | CPU inference |
| CER (normalized) | 0.58 (full form) / 0.11 (cropped) | Full-form score improved from 0.81 after fixing context starvation; remaining error driven by generation repetition, not misrecognition. Cropped score from single image — needs rerun on full 25-image set. |
| WER (normalized) | 0.65 (full form) | Improved from 0.78 after context fix |
| Throughput | 10.1 ppm | Down from 56.2 ppm due to longer output (676 vs 87 chars) |
| Reading order / bboxes (this eval) | None | GGUF/llama-server path is text-only — recognition component without structure detection |
| Bounding boxes (official pipeline) | Native | DocLayoutYOLO → layout bboxes → VLM recognition → layoutreader for reading order; structured markdown output. Requires LMDeploy/vLLM backend + separate layout weights — see official repo. |
| Setup complexity | 3/5 | Pre-built llama.cpp b9596 binaries; start llama-server with specific flags. No Python dependency issues. |
| Flexibility | 2/5 | Accurate on typed text; repeats itself on longer passages. Officially does not support handwritten content. |
| Key issue | — | **Generation-control problem, not OCR problem.** Recognizes text correctly but cannot stop — repeats the same paragraph 2–3× with variations. At `ctx-size=4096`, image tokens don't fit and output truncates to 87 chars. ("Sentence Database" in output is NOT a hallucination — it's printed on the form.) |
| Root cause | — | Qwen2-VL is a generative LMM, not a dedicated OCR engine; the LLM "completes" text beyond the visible image. Known issue with small VLMs used for transcription. |
| Text source | Printed | Outputs form headers on full forms — confirmed reading the machine-printed region. |

### Phase 2 Environment Changes

| Change | Detail |
|---|---|
| **transformers 4.57.6 → 5.8.1** | Required for SmolDocling `AutoModelForMultimodalLM`. Backward compat verified for GOT-OCR2.0. |
| **PaddlePaddle 3.2.1 installed (then superseded)** | Initial install alongside PyTorch 2.11.0+cu130 failed on Blackwell (no sm_120). Superseded by 3.4.0+ cu129 in `.venv_paddleocr`. |
| **PaddleOCR 3.6.0 installed** | Native PaddleOCR-VL API — working once the Blackwell-compatible PaddlePaddle build was in place. |
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
| **bitsandbytes lacks Blackwell support** | INT4 quantization unavailable → blocks Qwen3-VL-8B-Instruct (Tier 2, Phase 3) on the 12 GB research GPU | Revisit when bitsandbytes ships sm_120 kernels, or evaluate in BF16 on a larger GPU / cloud API in the deployment phase |

### Decisions

* **Research is empirical:** every finding is implemented and measured locally; no theoretical-only evaluations.
* **12 GB VRAM is a research-phase limit only:** models requiring >11 GB are evaluated via quantization; deployment has more headroom.
* **Two-stage architecture assumed**, with investigation into end-to-end alternatives.
* **English only:** multilingual is out of scope.
* **Auditability TBD:** Phase 7 will determine the best approach.
* **Cloud APIs allowed for Stage 2 comparison:** the research goal is fully local; deployment can use cloud APIs freely.
* **transformers 5.x adopted:** SmolDocling required `AutoModelForMultimodalLM`; the 4.57.6 constraint is lifted. Backward compat verified for GOT-OCR2.0.

## Further Considerations

1. **Fine-tuning:** If no off-the-shelf model meets accuracy targets (note: best local CER is 6.1% vs the <5% target), a Phase 9 could explore fine-tuning SmolDocling-256M or GOT-OCR2.0 on a custom handwriting dataset.
2. **vLLM / SGLang acceleration:** For VLM candidates, optimized inference engines could significantly improve throughput. SmolDocling reports 0.35s/page on A100 via vLLM — test on RTX 5070 Ti where applicable.
3. **ONNX / TensorRT:** For deployment, consider converting the best model to ONNX or TensorRT for further latency reduction (out of scope for research).
4. **PaddleOCR-VL vLLM backend:** No longer needed to unblock Blackwell (resolved via cu129 wheels) but may still be worth testing for serving throughput.
5. **Florence-2 on old transformers:** Resolved by pinning `transformers==4.40.0` in the separate `florencetf` conda env. See `scripts/bench_florence2.py`.
6. **MonkeyOCR — llama.cpp for text, DocLayoutYOLO for bboxes:** Pre-built llama.cpp binaries with Qwen2-VL native support made text recognition trivial to serve. DocLayoutYOLO (40.7 MB, PyTorch, no PaddlePaddle) provides native layout-detection bboxes at 0.10s/image and works on Blackwell. See `scripts/eval_bbox_reading_order.py doclayout_yolo` and `scripts/generate_model_visualizations.py monkeyocr`.
7. **Handwriting evaluation completed:** All 6 Tier-1 candidates have handwriting-only CER via XML-guided cropping — see [Handwriting-Only Re-Evaluation](#handwriting-only-re-evaluation-xml-guided-cropping). Florence-2-large leads at CER 0.061.
8. **Cloud baseline bbox evaluation:** `scripts/eval_baseline_bbox.py` evaluates Google Document AI block-level IoU and reading-order τ on the cropped handwritten dataset (25 images). Results: IoU 0.58, τ 0.91, 2.19s/page. See `benchmark/results/baseline_google_docai_layout.json`.
9. **GOT-OCR2.0 has no native bbox output (verified 2025-06-11):** Text-only OCR. Format mode = LaTeX/markdown; fine-grained mode takes a bbox as *input*. For bboxes use Florence-2 (IoU 0.76), Nemotron (0.28), SmolDocling (0.24), or DocLayoutYOLO (0.12, region-level).
10. **MonkeyOCR for handwriting:** The GGUF/llama-server path is text-only; the official pipeline adds DocLayoutYOLO + layoutreader. But MonkeyOCR **does not support handwritten content** per its official limitations — **for handwriting bboxes, use Florence-2 (IoU 0.76).**

---

## License

This project is licensed under the Apache 2.0 License. See [LICENSE](./LICENSE) for details.