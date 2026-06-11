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
- **Transformers version:** 4.57.6 (pinned). Models requiring 5.x (PaddleOCR-VL) or with incompatible remote code (Florence-2) are blocked pending version resolution.

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

**Phase 2 status:** GOT-OCR2.0 - evaluated. Florence-2 - blocked (transformers compat). PaddleOCR-VL - blocked (needs transformers 5.x).

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
#   GCP_PROJECT=vlm-ocr-research
#   GCP_LOCATION=asia-southeast1          # Document AI region
#   DOCAI_PROCESSOR_ID=50e3c2a5ddb25e78
#   GEMINI_MODEL=gemini-3.5-flash         # Must use location="global" in code

# 3. Run the baseline (5 curated images)
.venv/bin/python -u benchmark/baseline.py
```

**Critical:** Two auth paths supported:
```python
# API Key (AI Studio) — simplest, requires GOOGLE_API_KEY or GEMINI_API_KEY env var
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

### Phase 2: Tier-1 Candidate Evaluation 🔄 IN PROGRESS

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

#### Phase 2A: Prerequisites 

- Removed old renamed images (`iam_page_*.jpg`), replaced with Kaggle form images (original IAM filenames matching XML).
- Fixed `parse_iam_xml()` to compute bboxes from `<cmp>` child elements (not `<line>` attributes).
- Fixed `find_xml_for_image()` for direct stem matching (`a01-000u.png` ↔ `a01-000u.xml`).
- Regenerated `ground_truth.json`: 1,539/1,539 entries with real text, bboxes, and reading order.
- Created new 25-form curated subset spanning 21 writer prefixes.
- Updated harness to match ground truth by image name (not index).
- Added `normalize_ocr_text()` and `compute_cer_normalized()` / `compute_wer_normalized()` to metrics.

#### Phase 2B–2G: Candidate Evaluations

| Candidate | Status | Key Findings |
|---|---|---|
| **GOT-OCR2.0** |  Complete | 35.9s avg, 3.4 GB VRAM. Transcribes full form (printed + handwritten). No structured bbox output in plain OCR mode. |
| **Florence-2-large** |  Blocked | `Florence2VisionConfig` missing `embed_dim` in transformers 4.57 built-in. Remote code has `_supports_sdpa` incompatibility with PyTorch 2.10. |
| **PaddleOCR-VL-1.6** |  Blocked | Not in transformers 4.57 `AutoModelForVision2Seq`/`AutoModelForImageTextToText` supported configs. Requires transformers 5.x. |
| **granite-docling-258M** |  Next | 256M params, HuggingFace-native, DocTags format. |
| **Nemotron OCR v2** |  Pending | Built-in reading order. Requires Python 3.12 + CUDA toolkit + custom install. |
| **MonkeyOCR** |  Pending | No eval script yet. Needs full implementation. |

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
| 1 | **GOT-OCR2.0** | 35.9 | 2.93 | 3.92 | — | — | 3.4 | 2 | 3 |
| — | PaddleOCR-VL-1.6 |  blocked | — | — | — | — | — | — | — |
| — | Florence-2-large |  blocked | — | — | — | — | — | — | — |
| — | Nemotron OCR v2 |  | — | — | — | — | — | — | — |
| — | granite-docling-258M |  | — | — | — | — | — | — | — |
| — | MonkeyOCR |  | — | — | — | — | — | — | — |
| — | TrOCR (handwritten) |  | — | — | — | — | — | — | — |
| — | Qwen3-VL-4B |  | — | — | — | — | — | — | — |
| — | Qwen3-VL-8B (INT4) |  | — | — | — | — | — | — | — |

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
| — | Hunyuan VL | — | — | — | — | — | — | — | — |
| — | EasyOCR | — | — | — | — | — | — | — | — |
| — | docTR | — | — | — | — | — | — | — | — |
| — | Tesseract 5 | — | — | — | — | — | — | — | — |

---

## Decisions

* **Research is empirical:** every finding is implemented and measured locally; no theoretical-only evaluations.
* **12 GB VRAM for research only:** models requiring >11 GB are evaluated via quantization; deployment has more headroom.
* **Two-stage architecture assumed:** with investigation into end-to-end alternatives.
* **English only:** multilingual is out of scope.
* **Auditability TBD:** Phase 7 will determine the best approach.
* **Cloud API for Stage 2 comparison:** primary goal is fully local for research; deployment can use cloud APIs freely.

## Further Considerations

1. **Fine-tuning:** If no off-the-shelf model meets accuracy targets, a Phase 9 could explore fine-tuning GOT-OCR2.0 or TrOCR on a custom handwriting dataset.
2. **vLLM / SGLang acceleration:** For VLM candidates, optimized inference engines could significantly improve throughput. Test where applicable.
3. **ONNX / TensorRT:** For deployment, consider converting the best model to ONNX or TensorRT for further latency reduction (out of scope for research).

---

## License

This project is licensed under the Apache 2.0 License. See [LICENSE](https://www.google.com/search?q=./LICENSE) for details.