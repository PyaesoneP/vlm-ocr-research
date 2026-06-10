# VLM-OCR Research: Handwritten English Essay Feedback

[![Status: Research](https://img.shields.io/badge/status-research-blue)](.)
[![Hardware: RTX 5070 Ti](https://img.shields.io/badge/hardware-RTX%205070%20Ti%20(12GB)-green)](.)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-orange)](./LICENSE)

Empirical evaluation of open-source OCR models and Vision-Language Models (VLMs)
for a **handwritten English essay feedback system**. The system must transcribe
difficult handwriting, detect and localize writing errors with bounding boxes,
determine reading order without ruled lines, and generate natural language
feedback — all at lower latency than the current Google Document AI + Gemini
pipeline.

---

## Target Baseline

| Metric | Current (Google Document AI + Gemini) |
|--------|--------------------------------------|
| End-to-end latency | ~40–60 seconds per handwritten page |
| Cost | Document AI: $1.50–$30/1K pages + Gemini API per-token |
| Architecture | Cloud-only, proprietary |

## Constraints

- **Research phase**: RTX 5070 Ti mobile — 12 GB VRAM. Models must fit locally (quantization allowed).
- **Deployment phase**: Greater compute available + cloud APIs acceptable. Final recommendation may include models exceeding research GPU limits.

---

## Project Structure

```
vlm-ocr-research/
├── README.md                          # This file — research plan & live results
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

### Tier 1 — Most Promising

| # | Candidate | Params | Key Advantage | VRAM Est. |
|---|-----------|--------|---------------|-----------|
| 1 | **PaddleOCR-VL-1.6** | 0.9 B | SOTA doc VLM (96.3 % OmniDocBench), built-in layout & structure | ~2–3 GB |
| 2 | **GOT-OCR2.0** | ~7 B (Qwen-based) | Unified end-to-end OCR, bounding boxes, fine-grained, GGUF available | ~6–8 GB (INT4) |
| 3 | **NVIDIA Nemotron OCR v2** | 54 M EN / 84 M Multi | **Built-in reading order** (relational model), bounding boxes, production-grade | ~1–2 GB |
| 4 | **Florence-2-large** | 0.77 B | Microsoft foundation model, prompt-based OCR + boxes, multi-task | ~1.5 GB |
| 5 | **granite-docling-258M** | 256 M | Ultra-compact document VLM, successor to SmolDocling, DocTags format | ~1 GB |

### Tier 2 — Baselines & Comparison

| # | Candidate | Params | Purpose |
|---|-----------|--------|---------|
| 6 | **TrOCR (base + large, handwritten)** | 0.3 B / 0.6 B | IAM-finetuned HTR baseline — line-level only, needs text detector |
| 7 | **Qwen3-VL-4B-Instruct** | 4 B | Latest Qwen VLM, expanded OCR (32 languages), robust in low light/blur/tilt, strong document parsing. BF16 fits 12 GB (~8 GB). |
| 8 | **Qwen3-VL-8B-Instruct (INT4)** | 8 B (quantized) | Flagship Qwen VLM — 256K context, advanced spatial perception, 2D/3D grounding. INT4 needed for 12 GB (~6–8 GB). For deployment: BF16 on larger GPU or cloud API. |
| 9 | **EasyOCR** | ~50 M | Popular easy-to-use OCR; handwriting support on roadmap |
| 10 | **docTR** | varies | Modular PyTorch detection + recognition, good for documents |
| 11 | **Tesseract 5** | N/A | Traditional baseline for comparison |

---

## Architecture Insight

No single model handles everything. A **two-stage pipeline** is needed:

```
┌──────────────────────────────────────────────────────┐
│ Stage 1 — Transcription + Localization (OCR/VLM)      │
│  Image → { text, bounding boxes, reading order }      │
└──────────────────────────┬───────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────┐
│ Stage 2 — Error Detection & Feedback (LLM/VLM)        │
│  { text + bboxes + image } → { error bboxes + NL fb } │
└──────────────────────────────────────────────────────┘
```

**Stage 2 candidates**: Qwen3-VL-4B, Qwen3-VL-8B (INT4), SmolVLM-Instruct (2 B),
granite-docling-258M, Gemini 2.5 Flash (cloud comparison).

Key research question: can a single VLM (e.g., Qwen3-VL) handle both stages
end-to-end?

---

## Research Phases

### Phase 1 — Environment Setup & Baseline

1. Set up Python environment (CUDA 12.x, PyTorch, model dependencies)
2. Collect 10–20 diverse handwritten English essay samples (varying handwriting quality, layouts, error types)
3. Measure precise Google Document AI + Gemini baseline latency
4. Build shared evaluation harness
5. Define metric collection JSON schema

### Phase 2 — Tier-1 Candidate Evaluation

Evaluate the 5 most promising candidates. For each, measure:
- End-to-end latency (image → structured text with bounding boxes)
- CER / WER on handwritten text
- Reading order accuracy (manual evaluation)
- Bounding box quality (visual inspection + IoU where ground truth exists)
- VRAM peak usage
- Setup complexity (1–5 scale)
- Flexibility (1–5 scale: varied handwriting, layouts, noise)

### Phase 3 — Tier-2 Candidate Evaluation

Evaluate comparison baselines and larger models for reference.

### Phase 4 — Two-Stage Pipeline Architecture Design

Test combinations of best Stage 1 + Stage 2 models. Measure:
- Can a single VLM do both stages end-to-end?
- Is separate OCR + small LLM more efficient than one large VLM?
- Latency breakdown between stages

### Phase 5 — Reading Order Deep-Dive

The hardest sub-problem for unruled handwritten text:

1. **Nemotron OCR v2 relational model** — built-in reading order prediction
2. **PaddleOCR PP-StructureV3** — layout-aware structured output
3. **Heuristic post-processing** — y-coordinate line grouping + x-coordinate sorting
4. **VLM-based reading order** — ask Qwen3-VL or Florence-2 to output order
5. **Spatial relationship models** — fine-tune a small model on reading order

**Metric**: Kendall's tau vs. manual annotation on 5–10 samples.

### Phase 6 — Error Detection Accuracy

Per error-type evaluation:

| Error Type | Detection Method | Metrics |
|-----------|-----------------|---------|
| Capitalization | Rule-based + VLM image verification | Precision, Recall, F1 |
| Spelling | Dictionary + context (VLM/LLM) | Precision, Recall, F1 |
| Grammar | LLM analysis of transcription | Precision, Recall, F1 |
| Punctuation | Rule-based + VLM verification | Precision, Recall, F1 |
| Structural | Layout analysis (indentation, margins) | Precision, Recall, F1 |

**Bounding box accuracy**: IoU between predicted and actual error locations.

### Phase 7 — Auditability Strategy

Evaluate three approaches:
1. **Per-word image crops** — store cropped image of each word with transcript
2. **Annotated overlay** — draw recognized text over original with confidence highlighting
3. **Side-by-side storage** — original image + structured JSON output

Compare storage overhead, visual verifiability, and implementation complexity.

### Phase 8 — Final Pipeline Assembly & Benchmark

1. Assemble best Stage 1 + Stage 2 combination
2. Run full end-to-end benchmark against baseline
3. Document final architecture, latency breakdown, and accuracy scores
4. Compare total cost of ownership (local GPU amortization vs. cloud API)

---

## Metrics Framework

| Metric | How Measured | Target |
|--------|-------------|--------|
| **Latency (total)** | Wall-clock time image → feedback, avg of 10 runs | < 40 s (beat baseline) |
| **Latency (Stage 1)** | OCR/transcription only | < 15 s |
| **Latency (Stage 2)** | Error detection + feedback | < 25 s |
| **CER** | Character Error Rate vs. ground truth | < 5 % |
| **WER** | Word Error Rate vs. ground truth | < 10 % |
| **Reading Order Acc.** | Kendall's tau vs. manual annotation | > 0.85 |
| **Error Detection F1** | Per error type, macro-averaged | > 0.75 |
| **Bounding Box IoU** | Mean IoU for error bounding boxes | > 0.6 |
| **VRAM Peak** | `nvidia-smi` monitoring during inference | < 11 GB (research) |
| **Setup Complexity** | 1–5: install steps, config, docs quality | Lower is better |
| **Flexibility** | 1–5: handles varied handwriting, layouts, noise | Higher is better |
| **Throughput** | Pages per minute (batch where applicable) | > 1 ppm |
| **Cost (cloud)** | If any cloud API component used | Documented for comparison |

---

## Results (Live)

> Results will be populated as each candidate is evaluated.

| Rank | Candidate | Latency (s) | CER (%) | WER (%) | Read Order τ | Error F1 | VRAM (GB) | Setup | Flex |
|------|-----------|-------------|---------|---------|-------------|----------|-----------|-------|------|
| — | *(baseline)* Google Doc AI + Gemini | ~50 | TBD | TBD | TBD | TBD | N/A (cloud) | N/A | N/A |
| — | PaddleOCR-VL-1.6 | — | — | — | — | — | — | — | — |
| — | GOT-OCR2.0 | — | — | — | — | — | — | — | — |
| — | Nemotron OCR v2 | — | — | — | — | — | — | — | — |
| — | Florence-2-large | — | — | — | — | — | — | — | — |
| — | granite-docling-258M | — | — | — | — | — | — | — | — |
| — | TrOCR (handwritten) | — | — | — | — | — | — | — | — |
| — | Qwen3-VL-4B | — | — | — | — | — | — | — | — |
| — | Qwen3-VL-8B (INT4) | — | — | — | — | — | — | — | — |
| — | EasyOCR | — | — | — | — | — | — | — | — |
| — | docTR | — | — | — | — | — | — | — | — |
| — | Tesseract 5 | — | — | — | — | — | — | — | — |

---

## Decisions

- **Research is empirical** — every finding is implemented and measured locally; no theoretical-only evaluations
- **12 GB VRAM for research only** — models requiring >11 GB are evaluated via quantization; deployment has more headroom
- **Two-stage architecture assumed** with investigation into end-to-end alternatives
- **English only** — multilingual is out of scope
- **Auditability TBD** — Phase 7 will determine the best approach
- **Cloud API for Stage 2 comparison** — primary goal is fully local for research; deployment can use cloud APIs freely

## Further Considerations

1. **Fine-tuning**: If no off-the-shelf model meets accuracy targets, a Phase 9 could explore fine-tuning GOT-OCR2.0 or TrOCR on a custom handwriting dataset.
2. **vLLM / SGLang acceleration**: For VLM candidates, optimized inference engines could significantly improve throughput. Test where applicable.
3. **ONNX / TensorRT**: For deployment, consider converting the best model to ONNX or TensorRT for further latency reduction (out of scope for research).

---

## License

This project is licensed under the Apache 2.0 License — see [LICENSE](./LICENSE) for details.
