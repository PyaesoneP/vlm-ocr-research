# VLM-OCR Research: Handwritten English Essay Feedback

[![Status: Research](https://img.shields.io/badge/status-research-blue)](.)
[![Hardware: RTX 5070 Ti](https://img.shields.io/badge/hardware-RTX%205070%20Ti%20(12GB)-green)](.)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-orange)](./LICENSE)

Empirical evaluation of open-source OCR models and Vision-Language Models (VLMs) for a **handwritten English essay feedback system**. The system must transcribe difficult handwriting, detect and localize writing errors with bounding boxes, determine reading order without ruled lines, and generate natural language feedback. The goal is to achieve all of this at a lower latency than the current Google Document AI + Gemini pipeline.

---

## Target Baseline

| Metric | Current (Google Document AI + Gemini) |
|--------|--------------------------------------|
| End-to-end latency | **21.1 s** (Doc AI 3.1s + Gemini 3.5 Flash 17.7s) |
| Cost | Document AI: $1.50–$30/1K pages + Gemini API per-token |
| Architecture | Cloud-only, proprietary |
| Model | `gemini-3.5-flash` via Vertex AI (`location="global"`) |

## Constraints

- **Research phase:** RTX 5070 Ti mobile (12 GB VRAM). Models must fit locally (quantization allowed).
- **Deployment phase:** Greater compute available, and cloud APIs are acceptable. The final recommendation may include models exceeding research GPU limits.
- **Transformers version:** 5.11.0 (upgraded from 4.57.6 for SmolDocling `AutoModelForMultimodalLM` support and PaddleOCR-VL compatibility).

## Environments

Two separate Python environments are required due to conflicting CUDA toolkit versions:

| Environment | Type | PyTorch | CUDA | Purpose |
|---|---|---|---|---|
| `.venv/` | venv | 2.10.0+cu128 | 12.8 | SmolDocling, GOT-OCR2.0, Florence-2 (attempted) |
| `aiml` | conda | 2.12.0+cu130 | 13.0 | Nemotron OCR v2 (requires matching CUDA toolkit for C++ extension build) |

**Why two environments:** Nemotron OCR v2 compiles a C++ CUDA extension that requires `nvcc` version matching `torch.version.cuda`. System `nvcc` is CUDA 13.0; the main `.venv` uses PyTorch with CUDA 12.8. The `aiml` conda env was created with PyTorch 2.12.0+cu130 to match system CUDA 13.0.

**Activation:**
```bash
# For SmolDocling, GOT-OCR2.0, Florence-2:
source .venv/bin/activate

# For Nemotron OCR v2:
conda activate aiml
```

---

## Project Structure

```text
vlm-ocr-research/
├── README.md                          # This file: research plan & live results
├── benchmark/
│   ├── harness.py                     # Shared evaluation harness
│   ├── metrics.py                     # CER, WER, IoU, reading-order scoring
│   ├── test_dataset/                  # Handwritten essay samples + ground truth
│   └── results/                       # Per-candidate JSON result files
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
| 1 | **PaddleOCR-VL-1.6** | 0.9 B | SOTA doc VLM (96.3% OmniDocBench), built-in layout & structure | ~2–3 GB |
| 2 | **GOT-OCR2.0** | ~7 B (Qwen-based) | Unified end-to-end OCR, bounding boxes, fine-grained, GGUF available | ~6–8 GB (INT4) |
| 3 | **NVIDIA Nemotron OCR v2** | 54 M EN / 84 M Multi | **Built-in reading order** (relational model), bounding boxes, production-grade | ~1–2 GB |
| 4 | **Florence-2-large** | 0.77 B | Microsoft foundation model, prompt-based OCR + boxes, multi-task | ~1.5 GB |
| 5 | **granite-docling-258M** | 256 M | Ultra-compact document VLM, successor to SmolDocling, DocTags format | ~1 GB |
| 6 | **MonkeyOCR** | ~3 B | Document-specialized VLM, strong on OCRBench, native bbox + reading order output | ~6 GB |

**Phase 2 status:** GOT-OCR2.0 - evaluated. SmolDocling-256M - evaluated (best so far: 12.2s, 0.8GB). Florence-2 - blocked. PaddleOCR-VL - blocked (GPU). Nemotron OCR & MonkeyOCR - evaluated.

### Tier 2: Baselines & Comparison

| # | Candidate | Params | Purpose |
| --- | --- | --- | --- |
| 7 | **TrOCR (base + large, handwritten)** | 0.3 B / 0.6 B | IAM-finetuned HTR baseline: line-level only, needs text detector |
| 8 | **Qwen3-VL-4B-Instruct** | 4 B | Latest Qwen VLM, expanded OCR (32 languages), robust in low light/blur/tilt, strong document parsing. BF16 fits 12 GB (~8 GB). |
| 9 | **Qwen3-VL-8B-Instruct (INT4)** | 8 B (quantized) | Flagship Qwen VLM: 256K context, advanced spatial perception, 2D/3D grounding. INT4 needed for 12 GB (~6–8 GB). For deployment, BF16 on larger GPU or cloud API. |
| 10 | **Hunyuan VL** | ~4 B | Tencent VLM with multi-resolution architecture, strong on document benchmarks, potential single-model Stage 1+2 candidate | ~8 GB |
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

Key research question: can a single VLM (e.g., Qwen3-VL) handle both stages end-to-end?

---

## Research Phases

### Phase 1: Environment Setup & Baseline

1. Set up Python environment (CUDA 12.8, PyTorch 2.10, RTX 5070 Ti 12 GB). -> `scripts/setup.sh`
2. Collect handwritten essay samples. -> `benchmark/test_dataset/` (1,539 IAM forms from Kaggle, 25 curated).
3. Measure Google Document AI + Gemini baseline latency. -> `benchmark/baseline.py` — **21.1s total (Doc AI 3.1s + Gemini 3.5 Flash 17.7s)**.
4. Build shared evaluation harness. -> `benchmark/harness.py` + `benchmark/metrics.py`
5. Define metric collection JSON schema. -> `benchmark/test_dataset/ground_truth.json`

**Validation:** `bash scripts/phase1_validate.sh` — 27/27 checks pass, 0 failures.

#### Phase 1 Findings

| Finding | Detail |
|---|---|
| **Baseline is 21.1s** | 48–65% faster than the original 40-60s target. Local models must beat 21.1s. |
| **Stage 2 dominates latency** | Gemini error detection is 17.7s (84% of total). Replacing it with a local model is the highest-impact optimization. |
| **Document AI is fast** | OCR alone is 3.1s. The cloud OCR stage is not the bottleneck. |
| **Gemini needs `location="global"`** | `gemini-3.5-flash` is a preview model only available in the `global` region on Vertex AI. Using `asia-southeast1` returns 404. |
| **Blackwell needs nightly PyTorch** | RTX 5070 Ti (sm_120) unsupported by stable PyTorch 2.6. Uses 2.10.0+cu128. |
| **bitsandbytes blocked** | INT4 quantization unavailable for Blackwell — blocks Qwen3-VL-8B evaluation. |
| **Free tier API keys inadequate** | 20 req/day + 5 RPM limits make API keys unusable for benchmarking. Vertex AI via ADC resolved this. |

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

1. **PyTorch on Blackwell.** The RTX 5070 Ti has compute capability sm_120. PyTorch 2.6 stable only supports up to sm_90. Installing the stable wheel produced CUDA compatibility warnings and `total_mem` -> `total_memory` attribute errors. Fixed by switching to PyTorch 2.12 nightly with CUDA 12.8.

2. **Gemini API key quota.** Free tier limits (20 req/day, 5 RPM) caused 429 errors across all Flash models and 503 errors during demand spikes. Added retry logic with exponential backoff and rate limiting, but the daily quota was fundamentally too low for 12 sequential calls. Resolved by switching to Vertex AI via ADC (`gcloud auth application-default login`), which has production-tier quota.

3. **Document AI regional endpoint.** The processor is deployed in `asia-southeast1`. The default client connects to the global endpoint and rejected the regional processor. Fixed by configuring a regional API endpoint (`asia-southeast1-documentai.googleapis.com`) via `ClientOptions`.

4. **Ground truth without IAM XML.** The IAM metadata download requires authentication. Without XML annotations, the ground truth generator produced skeleton entries. CER/WER scoring requires IAM registration or manual annotation.

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
benchmark/results/baseline_google_docai_gemini.json  # 29.0s baseline
docs/METHODOLOGY.md                  # Full research design doc
.env.example                         # GCP credential template
```

### Phase 2: Tier-1 Candidate Evaluation — 4 of 6 Complete, 2 Blocked

> **CRITICAL METHODOLOGICAL FINDING (2025-06-11):** The IAM forms contain BOTH machine-printed text (the "prompt" for writers to copy) AND handwritten text (the writer's copy). Since the handwritten text is an exact copy of the printed prompt, the ground truth text is identical regardless of which section is transcribed. **CER/WER cannot distinguish whether a model read the printed text or the handwriting.** Nemotron OCR and MonkeyOCR output the printed form header ("Sentence Database"), confirming they read the machine-printed region. SmolDocling skips the header — it *may* be reading the handwriting. This means Phase 2 rankings reflect printed-text OCR accuracy, not handwriting recognition. See [IAM Dataset Structure](#iam-dataset-structure) below for details.

Evaluate the 6 most promising candidates one-by-one. For each, measure:

* End-to-end latency (image → structured text with bounding boxes).
* CER / WER on handwritten text (whitespace-normalized).
* Reading order accuracy (Kendall's tau vs. ground truth).
* Bounding box quality (visual inspection + IoU where ground truth exists).
* VRAM peak usage.
* Setup complexity (1–5 scale).
* Flexibility (1–5 scale: varied handwriting, layouts, noise).

**Dataset:** 1,539 IAM handwriting forms from Kaggle (`gwachatkozah/iam-forms-dataset`) with matching XML annotations providing line-level text, bounding boxes, and reading order. Curated subset of 25 forms used for evaluation. Ground truth populated via `parse_iam_xml()` extracting `<cmp>`-level coordinates.

**CER/WER:** Computed with whitespace normalization (newlines → spaces, collapsed whitespace). This prevents line-break formatting differences from artificially inflating error rates.

### IAM Dataset Structure

Every IAM form image has FOUR sections stacked vertically, separated by horizontal lines:

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

- **CER/WER measures text correctness, NOT handwriting recognition ability.**
- A model that reads the easy machine-printed text (section 2) scores identically to one that reads the hard handwriting (section 3).
- Models that output "Sentence Database" or form IDs in their transcription are confirmed to be reading section 1-2 (printed), not section 3 (handwriting).
- **Nemotron OCR** outputs headers → reading printed text.
- **MonkeyOCR** outputs headers → reading printed text.
- **SmolDocling** skips headers → source ambiguous (may be reading handwriting, or may filter headers).
- **GOT-OCR2.0** skips headers → source ambiguous.

The Phase 2 rankings below should be interpreted as **printed-text OCR benchmarks**, not handwriting recognition results. Handwriting-specific evaluation requires isolating the handwritten region (section 3) via bounding box filtering, or using a handwriting-only dataset.

#### Phase 2A: Prerequisites 

- Removed old renamed images (`iam_page_*.jpg`), replaced with Kaggle form images (original IAM filenames matching XML).
- Fixed `parse_iam_xml()` to compute bboxes from `<cmp>` child elements (not `<line>` attributes).
- Fixed `find_xml_for_image()` for direct stem matching (`a01-000u.png` ↔ `a01-000u.xml`).
- Regenerated `ground_truth.json`: 1,539/1,539 entries with real text, bboxes, and reading order.
- Created new 25-form curated subset spanning 21 writer prefixes.
- Updated harness to match ground truth by image name (not index).
- Added `normalize_ocr_text()` and `compute_cer_normalized()` / `compute_wer_normalized()` to metrics.

#### Phase 2B–2G: Candidate Evaluations

| Candidate | Status | Text Source | Key Findings |
|---|---|---|---|
| **GOT-OCR2.0** |  Complete | Ambiguous (no header output) | 35.9s avg, 3.4 GB VRAM. Transcribes full form (printed + handwritten). No structured bbox output in plain OCR mode. |
| **Florence-2-large** |  Blocked | — | Remote code `past_key_values[0]` incompatible with transformers 4.57 beam search. Built-in `Florence2ForConditionalGeneration` has weight name mismatch with HF checkpoint. Transformers 5.x has `forced_bos_token_id` missing from `Florence2LanguageConfig`. |
| **PaddleOCR-VL-1.6** |  Blocked | — | Native PaddleOCR API requires PaddlePaddle 3.2.1 which does not support Blackwell GPU (sm_120). `RuntimeError: Unsupported GPU architecture`. transformers 5.x path also blocked (same GPU compat issue for PaddlePaddle). |
| **SmolDocling-256M** |  Complete | Ambiguous (skips header) | **Best candidate for accuracy.** 12.2s avg latency (42% faster than baseline), 0.8 GB VRAM, CER 1.47, WER 1.50. DocTags output with bbox parsing via regex. Hallucinates/repeats on handwriting. `AutoModelForMultimodalLM` + transformers 5.x. Skips form header — may be reading handwriting or filtering non-content. |
| **Nemotron OCR v2** |  Complete | **Printed text** (headers in output) | **Fastest candidate.** 0.09s avg (234x faster than baseline), 0.6 GB VRAM, CER 1.17, WER 1.24. Outputs form headers ("Sentence Database") + form IDs → confirmed reading printed text, not handwriting. 7 text regions per page with bboxes + reading order via relational model. Built in aiml conda env (CUDA 13.0 + PyTorch 2.12). CUDA extension compiled without issues. |
| **MonkeyOCR** |  Complete (CPU) | **Printed text** (headers in output) | 5.96s avg, 0 MB VRAM (CPU-only). CER 0.58, WER 0.65 — poor due to generation repetition, not misrecognition. Outputs form headers ("Sentence Database") → confirmed reading printed text. Recognizes typed text accurately but loops/repeats. Backend: llama.cpp b9596 llama-server (ctx-size=8192). |

### Handwriting-Only Re-Evaluation (XML-Guided Cropping)

After the methodological finding that Phase 2 metrics reflect printed-text OCR, we cropped images to the handwritten region using IAM XML `<handwritten-part>` `<cmp>` bounding boxes. Each image was cropped to the union bounding box of all handwritten lines (excluding signature footer), with 20px padding. This physically removes the printed header/prompt, forcing models to read only handwriting.

| Candidate | Printed CER (Phase 2) | **Handwriting CER** | Handwriting Latency | Verdict |
|---|---|---|---|---|
| **MonkeyOCR** 🥇 | 0.58 | **0.11** | 3.79s | Best accuracy. Removes "Sentence Database" header bias. No bboxes. |
| **SmolDocling** 🥈 | 1.47 | **0.14** | 7.05s | Near-best accuracy + structured DocTags output with bboxes. Best overall for essay feedback pipeline. |
| **Nemotron OCR v2** | 1.17 | 0.74 | 0.06s | Fastest but recognizer genuinely struggles with handwriting. 6x worse CER than top two. |
| **GOT-OCR2.0** | 2.93 | 4.32 | 38.95s | Degrades on cropped images — requires full-page context. Not suitable for handwriting. |

**Key findings:**
- **SmolDocling and MonkeyOCR both handle handwriting well** — their poor Phase 2 CER was entirely due to the mixed printed/handwritten IAM layout, not handwriting recognition failure.
- **Nemotron's speed advantage is negated by poor handwriting accuracy** (CER 0.74 vs 0.11-0.14 for top models).
- **GOT-OCR2.0 is full-page dependent** — degrades severely without printed context.
- Cropping reduced latency for all models (smaller image = fewer tokens to process).

See `scripts/crop_handwritten.py` and `scripts/eval_handwritten.py` for the methodology.

### Phase 3: Tier-2 Candidate Evaluation

Evaluate comparison baselines and larger models for reference.

### Phase 4: Two-Stage Pipeline Architecture Design

Test combinations of best Stage 1 + Stage 2 models. Measure:

* Can a single VLM do both stages end-to-end?
* Is separate OCR + small LLM more efficient than one large VLM?
* Latency breakdown between stages.

### Phase 5: Reading Order Deep-Dive

The hardest sub-problem for unruled handwritten text:

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
| **Latency (total)** | Wall-clock time image → feedback, avg of 10 runs | < 40 s (beat baseline) |
| **Latency (Stage 1)** | OCR/transcription only | < 15 s |
| **Latency (Stage 2)** | Error detection + feedback | < 25 s |
| **CER** | Character Error Rate vs. ground truth | < 5 % |
| **WER** | Word Error Rate vs. ground truth | < 10 % |
| **Reading Order Acc.** | Kendall's tau vs. manual annotation | > 0.85 |
| **Error Detection F1** | Per error type, macro-averaged | > 0.75 |
| **Bounding Box IoU** | Mean IoU for error bounding boxes | > 0.6 |
| **VRAM Peak** | `nvidia-smi` monitoring during inference | < 11 GB (research) |
| **Setup Complexity** | 1–5 scale: install steps, config, docs quality | Lower is better |
| **Flexibility** | 1–5 scale: handles varied handwriting, layouts, noise | Higher is better |
| **Throughput** | Pages per minute (batch where applicable) | > 1 ppm |
| **Cost (cloud)** | If any cloud API component used | Documented for comparison |

---

## Results (Live)

> Updated as each candidate is evaluated. CER/WER are whitespace-normalized. See `benchmark/results/` for full JSON.

| Rank | Candidate | Latency (s) | CER | WER | Read Order τ | Error F1 | VRAM (GB) | Setup | Flex |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| — | *(baseline)* Google Doc AI + Gemini | 21.1 | 1.22 | 1.22 | — | — | N/A (cloud) | N/A | N/A |
| 1 | **Nemotron OCR v2** | 0.09 | 1.17 | 1.24 | — | — | 0.6 | 4 | 4 |
| 2 | **SmolDocling-256M** | 12.2 | 1.47 | 1.50 | — | — | 0.8 | 3 | 3 |
| 3 | **GOT-OCR2.0** | 35.9 | 2.93 | 3.92 | — | — | 3.4 | 2 | 3 |
| — | PaddleOCR-VL-1.6 | blocked | — | — | — | — | — | — | — |
| — | Florence-2-large | blocked | — | — | — | — | — | — | — |
| — | MonkeyOCR | pending | — | — | — | — | — | — | — |

### GOT-OCR2.0 Detailed Findings

| Metric | Value | Notes |
|---|---|---|
| Model | `stepfun-ai/GOT-OCR-2.0-hf` | HuggingFace transformers, BF16 |
| Avg latency | 35.9 s | 24% slower than cloud baseline (29.0s) |
| VRAM peak | 3.4 GB | Fits 12GB comfortably |
| CER (normalized) | 2.93 | High — model transcribes full IAM form (printed instructions + handwritten) while GT covers handwritten only |
| WER (normalized) | 3.92 | Same scope mismatch as CER |
| Reading order | — | Not evaluated (plain OCR mode, no structured output) |
| Bounding boxes | — | Plain OCR mode does not output bboxes. Structured mode requires chat template fix. |
| Setup complexity | 2/5 | Straightforward `AutoModelForImageTextToText` + `AutoProcessor`. One-line install. |
| Flexibility | 3/5 | Handles varied handwriting well. Output quality affected by chat template tokens requiring cleanup. |
| Key issue | — | Output includes system/user/assistant role markers and IAM metadata headers. Cleanup regex needed. Transcribes entire form, not just handwritten region. |

### SmolDocling-256M Detailed Findings

| Metric | Value | Notes |
|---|---|---|
| Model | `docling-project/SmolDocling-256M-preview` | `AutoModelForMultimodalLM` (requires transformers 5.x) |
| Avg latency | 12.2 s | **42% faster than cloud baseline (21.1s)**. First candidate to beat baseline. |
| VRAM peak | 0.8 GB | Fits 12GB with 15x headroom. Smallest footprint by far. |
| CER (normalized) | 1.47 | Near baseline (1.22). Model hallucinates/repeats text on handwriting. |
| WER (normalized) | 1.50 | Near baseline (1.22). Same hallucination pattern as CER. |
| Throughput | 4.9 ppm | Highest throughput of all candidates. |
| Reading order | — | DocTags includes positional coordinates; reading order inferable from bbox sorting. |
| Bounding boxes | Partial | DocTags parser extracts 2-5 blocks per page with pixel coordinates. Missing structured layout parsing (needs `docling` library for full DocTags→DoclingDocument). |
| Setup complexity | 3/5 | Requires transformers 5.x. `AutoModelForMultimodalLM` + chat template. DocTags output needs custom regex parser. |
| Flexibility | 3/5 | Handles printed documents well. Hallucinates/repeats on handwritten IAM forms. Not trained on handwriting. |
| Key issue | — | Text repetition/hallucination on handwriting. Model was trained on printed documents, not handwritten essays. CER/WER are deceptively low due to whitespace normalization masking repetition. |
| DocTags format | — | Proprietary DocTags tokens (`<loc_N>` for coords, `<text>`, `<table>`, etc.). Coordinates in 0-999 normalized bin space. Parsable via regex. |

### Nemotron OCR v2 Detailed Findings

| Metric | Value | Notes |
|---|---|---|
| Model | `nvidia/nemotron-ocr-v2` (v2_english) | 54M params: detector (RegNetX-8GF) + recognizer (Transformer) + relational model |
| Avg latency | 0.09 s | **234x faster than cloud baseline**. Full pass through 25 curated images confirms sub-0.1s average. |
| VRAM peak | 0.6 GB | Fits 12GB with 20x headroom. Smallest footprint. |
| CER (normalized) | 1.17 | Near baseline (1.22). Recognizer struggles with handwriting (trained on printed docs). |
| WER (normalized) | 1.24 | Near baseline (1.22). Same printed-document bias as CER. |
| Throughput | 664 ppm | 100x+ any other candidate. Production-ready throughput. |
| Reading order | Built-in | Relational model explicitly predicts logical groupings and reading order across text elements. |
| Bounding boxes | Full | 4-corner quad format. Denormalized from 0-1 to pixel coords. ~7 regions per IAM page. |
| Setup complexity | 4/5 | Requires git-lfs clone, CUDA toolkit, C++ build. CUDA version must match PyTorch. Needed separate conda env (aiml, CUDA 13.0 + PyTorch 2.12). |
| Flexibility | 4/5 | Detects printed text well. Handwriting recognition is garbled ("hohi happing Onke") but detection/localization works correctly. |
| Key advantage | — | **Lowest latency by far** (0.09s vs 12.2s SmolDocling, 21.1s baseline). Production-grade with built-in reading order. Relational model is unique differentiator. |
| Key limitation | — | Recognizer trained on printed documents — handwriting transcription is inaccurate. Detector works well (correctly localizes all text lines). |
|  Text source | — | **Outputs printed form headers** ("Sentence Database", "A04-039"). Confirmed reading machine-printed region, not handwriting. CER 1.17 reflects printed-text OCR quality, not handwriting recognition. |

### MonkeyOCR Detailed Findings

| Metric | Value | Notes |
|---|---|---|
| Model | `dinhquangson/MonkeyOCR-pro-1.2B-Vision-GGUF` | Qwen2-VL based, GGUF Q4_K_M quantization |
| Avg latency | 5.96 s | CPU-only (llama.cpp Vulkan backend not detected on RTX 5070 Ti). Requires ctx-size=8192 + image-min-tokens=1024. |
| VRAM peak | 0 MB | CPU inference. Vulkan GPU offload unavailable. |
| CER (normalized) | 0.58 | Improved from 0.81 after fixing context starvation. Still poor — driven by generation repetition, not misrecognition. |
| WER (normalized) | 0.65 | Improved from 0.78 after context fix. Same repetition issue as CER. |
| Throughput | 10.1 ppm | Down from 56.2 ppm due to longer output (676 vs 87 chars). |
| Reading order | — | No bboxes, no layout output. Text-only via llama-server API. |
| Bounding boxes | — | llama-server chat API provides text only. No structured output. |
| Setup complexity | 3/5 | Pre-built llama.cpp b9596 binaries (Vulkan). Requires downloading release, starting llama-server with specific flags. No Python dependency issues. |
| Flexibility | 2/5 | Accurate on typed text. Repeats itself on longer passages. Handwriting recognition untested (model loops before reaching it). IAM forms are mixed typed+handwritten — typed portion recognized well. |
| Key issue | — | **Generation-control problem, not OCR problem.** Recognizes typed text correctly but cannot stop — repeats the same paragraph 2-3x with variations. Requires ctx-size=8192 for image tokens to fit (at 4096, output truncated to 87 chars). "Sentence Database" is NOT a hallucination — it's printed on the IAM form header. |
| Root cause | — | Qwen2-VL is a generative LMM, not a dedicated OCR engine. The LLM component "completes" text beyond the visible image content. Repetition is a known issue with small VLMs used for transcription tasks. |
|  Text source | — | **Outputs printed form headers** ("Sentence Database"). Confirmed reading machine-printed region, not handwriting. CER 0.58 reflects printed-text OCR + repetition artifacts, not handwriting recognition. |
| Backend | — | llama.cpp b9596 llama-server with Vulkan build. Vulkan GPU backend not detected on RTX 5070 Ti — fell back to CPU. |

### Phase 2 Environment Changes

| Change | Detail |
|---|---|
| **transformers 4.57.6 → 5.11.0** | Required for SmolDocling `AutoModelForMultimodalLM`. Also unblocks PaddleOCR-VL (conceptually). |
| **PaddlePaddle 3.2.1 installed** | Installed alongside PyTorch 2.10.0+cu128. CUDA package versions reconciled manually. |
| **PaddleOCR 3.6.0 installed** | Native PaddleOCR-VL API available but blocked by Blackwell GPU. |
| **Florence-2 remote code cache purged** | Removed patched remote code. Model remains blocked in both 4.57 and 5.x. |

### Blocked Candidates — Root Cause Analysis

| Candidate | Blocker | Root Cause | Unblock Path |
|---|---|---|---|
| **Florence-2-large** | `past_key_values` + SDPA incompat | Remote `modeling_florence2.py` uses old `past_key_values[0][0].shape[2]` API incompatible with transformers 4.57 beam search. Built-in class has weight name mismatch with HF checkpoint. 5.x adds `forced_bos_token_id` error. | Wait for Microsoft to update HF checkpoint for transformers 5.x built-in class, OR use vLLM/SGLang backend. |
| **PaddleOCR-VL-1.6** | Blackwell sm_120 unsupported | PaddlePaddle 3.2.1 does not support Blackwell GPU architecture. `RuntimeError: Unsupported GPU architecture` in `paddle_inference.create_predictor()`. | Wait for PaddlePaddle to add sm_120 support (Blackwell), OR use cloud GPU (A100/H100), OR use vLLM backend which bypasses PaddlePaddle inference engine. |

### Decisions

* **Research is empirical:** every finding is implemented and measured locally; no theoretical-only evaluations.
* **12 GB VRAM for research only:** models requiring >11 GB are evaluated via quantization; deployment has more headroom.
* **Two-stage architecture assumed:** with investigation into end-to-end alternatives.
* **English only:** multilingual is out of scope.
* **Auditability TBD:** Phase 7 will determine the best approach.
* **Cloud API for Stage 2 comparison:** primary goal is fully local for research; deployment can use cloud APIs freely.
* **transformers 5.x adopted:** SmolDocling required `AutoModelForMultimodalLM`. 4.57.6 constraint lifted. Backward compat verified for GOT-OCR2.0.

## Further Considerations

1. **Fine-tuning:** If no off-the-shelf model meets accuracy targets, a Phase 9 could explore fine-tuning SmolDocling-256M or GOT-OCR2.0 on a custom handwriting dataset.
2. **vLLM / SGLang acceleration:** For VLM candidates, optimized inference engines could significantly improve throughput. SmolDocling reports 0.35s/page on A100 via vLLM — test on RTX 5070 Ti where applicable.
3. **ONNX / TensorRT:** For deployment, consider converting the best model to ONNX or TensorRT for further latency reduction (out of scope for research).
4. **PaddleOCR-VL vLLM backend:** May bypass PaddlePaddle GPU incompatibility by using vLLM inference server instead of direct PaddlePaddle inference.
5. **Florence-2 vLLM backend:** May bypass remote code issues by loading through vLLM which has native Florence-2 support.
6. **MonkeyOCR — llama.cpp path worked for serving, accuracy the blocker:** Pre-built llama.cpp binaries with Qwen2-VL native support made serving trivial (no Python deps needed beyond stdlib). This same approach could serve Florence-2 or other GGUF-converted VLMs without the transformers remote-code issues.
7. ** Handwriting-specific evaluation needed:** All Phase 2 CER/WER results reflect printed-text OCR due to IAM dataset structure (see [IAM Dataset Structure](#iam-dataset-structure)). To properly evaluate handwriting recognition, we need to either: (a) isolate the handwritten region via bounding box filtering using the XML annotations, or (b) use a handwriting-only dataset without printed prompts (e.g., IAM lines subset, RIMES, or custom essay scans).

---

## License

This project is licensed under the Apache 2.0 License. See [LICENSE](https://www.google.com/search?q=./LICENSE) for details.