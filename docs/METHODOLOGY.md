# Research Methodology & Model Selection Rationale

## Problem Statement

The task is to build a system that takes a handwritten English essay page as input and produces: (1) a full transcription of the text, (2) bounding-box localization of each text region, (3) correct reading order without ruled lines, (4) detection and localization of writing errors (capitalization, spelling, grammar, punctuation, structural), and (5) natural language feedback for the student.

The current production pipeline uses Google Document AI for OCR followed by Gemini for error analysis, with end-to-end latency of 40-60 seconds per page and cloud-only architecture. The research goal is to match or exceed this accuracy at lower latency, ideally with a fully local deployment option.

A critical sub-problem is **sentence segmentation** on unruled paper: determining whether a given word belongs to the line above or the line below when spacing is ambiguous. The current production system uses a custom sequence-model algorithm that tracks relationships between previous, current, and next sentences to disambiguate word-line assignments. Any replacement pipeline must match or exceed this capability.

## Architecture: Why Two Stages

No single open-source model handles the full pipeline. OCR models excel at transcription and layout but lack essay-grading reasoning. Vision-language models can reason about content but are inconsistent at fine-grained text localization. The two-stage design separates these concerns:

```
Stage 1 (Perception): Image -> { text, bounding boxes, reading order }
                                     Includes sentence segmentation on unruled paper
Stage 2 (Reasoning):  { text + bboxes + image } -> { error bboxes + feedback }
```

This separation has several advantages:

- **Independent optimization**: Each stage can use the best model for its specific task. Stage 1 benefits from document-specialized architectures; Stage 2 benefits from strong language reasoning.
- **Partial fallback**: If Stage 2 is unavailable (e.g., cloud quota), Stage 1 still produces a usable transcription with bounding boxes.
- **Cost efficiency**: A smaller, cheaper model can handle Stage 1 at high throughput, while a more capable model is reserved for the reasoning-intensive Stage 2.
- **Auditability**: Storing intermediate outputs (transcription + bboxes) enables human-in-the-loop verification before feedback generation.

A secondary research question is whether a single VLM (e.g., Qwen3-VL) can handle both stages end-to-end. This will be evaluated in Phase 4.

## Model Selection Criteria

Candidates were selected across two tiers based on four criteria:

| Criterion | Description | Measurement |
|---|---|---|
| **VRAM fit** | Must run on 12 GB RTX 5070 Ti (quantization allowed) | `nvidia-smi` peak monitoring |
| **Bounding box output** | Must localize text regions, not just transcribe | IoU vs. ground truth |
| **Reading order** | Must order text blocks correctly without ruled lines | Kendall's tau vs. manual annotation |
| **Handwriting robustness** | Must handle varied legibility, slant, stroke width | CER/WER on IAM handwriting samples |

Secondary criteria: setup complexity, inference speed, license, and community support.

## Tier 1 Candidates (Most Promising)

### 1. PaddleOCR-VL-1.6 (0.9B params)

**Why**: Achieves 96.3% on OmniDocBench v1.6, the leading document understanding benchmark. Built specifically for document parsing with native layout analysis, structured output (Markdown/JSON with coordinates), and a compact architecture that fits comfortably in 12 GB (~2-3 GB VRAM).

**Key capability**: End-to-end document VLM — image in, structured JSON with text blocks, bboxes, and reading order out. No separate detection/recognition pipeline needed.

**Risk**: Trained primarily on printed and typewritten documents. Handwriting performance on IAM-style pages is unverified. May need prompt engineering for handwriting-specific output.

### 2. GOT-OCR2.0 (~7B params, Qwen-based)

**Why**: Unified end-to-end OCR supporting plain text, formatted text, and fine-grained OCR with bounding boxes. The "General OCR Theory" architecture handles diverse inputs without task-specific heads. GGUF quantization available for 12 GB fit (~6-8 GB at INT4).

**Key capability**: Single model handles detection, recognition, and formatting. Supports both plain text output and structured formats with coordinate annotations.

**Risk**: 7B parameters is large for 12 GB even at INT4. Inference speed may be slower than specialized smaller models. Community GGUF support is early-stage.

### 3. NVIDIA Nemotron OCR v2 (54M EN / 84M Multi)

**Why**: Production-grade OCR with three explicit components: text detector (RegNetX-8GF backbone), text recognizer (Transformer), and relational model for reading order prediction. The relational model is unique among open-source OCR systems — it explicitly predicts reading order, which is the hardest sub-problem for unruled handwritten text.

**Key capability**: Built-in reading order prediction via the relational model. Extremely VRAM-efficient (~1-2 GB). Per-phase timing via `verbose_post` mode enables precise latency breakdown.

**Risk**: Optimized for printed/scanned documents. Handwriting accuracy on IAM data is unknown. Requires Python 3.12 and specific CUDA toolkit version. Installation complexity is higher than HuggingFace-based models.

### 4. Florence-2-large (0.77B params)

**Why**: Microsoft's lightweight vision foundation model. Prompt-based multitask architecture supports `<OCR>` and `<OCR_WITH_REGION>` tasks out of the box. Well-established HuggingFace integration with extensive documentation.

**Key capability**: Prompt-based interface means no fine-tuning needed for different OCR tasks. Returns quad boxes with text labels. Tiny footprint (~1.5 GB VRAM).

**Risk**: Designed as a general vision model, not OCR-specialized. Handwriting accuracy may be lower than document-specific models. No built-in reading order — relies on positional heuristics.

### 5. granite-docling-258M (256M params)

**Why**: Ultra-compact document VLM from IBM, successor to SmolDocling. Uses DocTags format for structured document output including layout, bounding boxes, and reading order. At 256M parameters, it's the smallest candidate by a wide margin.

**Key capability**: Extreme efficiency — fits in ~1 GB VRAM. DocTags format preserves document structure in a compact token representation. Fast inference due to small size.

**Risk**: Smallest model means potential accuracy ceiling. DocTags format requires post-processing for standard bbox/text extraction. Handwriting performance on non-printed text is untested.

### 6. MonkeyOCR (~3B params)

**Why**: Document-specialized VLM with strong performance on OCRBench, a comprehensive benchmark covering text recognition, scene text, document parsing, and handwriting. Native support for bounding box output and reading order prediction. At ~3B parameters, fits comfortably in 12 GB at BF16 (~6 GB VRAM).

**Key capability**: Purpose-built for OCR tasks rather than general visual understanding. Stronger document-specific inductive biases compared to general VLMs of similar size. Competitive with larger models on structured OCR benchmarks.

**Risk**: Newer model with smaller community than established options. Handwriting-specific performance on IAM-style data is not yet independently verified in published benchmarks. May require custom inference code outside the HuggingFace ecosystem.

## Tier 2 Candidates (Baselines & Comparison)

### 6. TrOCR (0.3B base / 0.6B large, handwritten)

**Why**: Microsoft's Transformer-based OCR, fine-tuned specifically on the IAM handwriting dataset. Represents the traditional approach: dedicated HTR model trained on the target domain.

**Key capability**: Strong performance on IAM-style handwriting since it was trained on that exact distribution. Simple HuggingFace API.

**Risk**: Line-level only — requires a separate text detector (CRAFT, DBNet) for full-page input. This adds complexity and latency. No reading order or layout understanding.

### 7. Qwen3-VL-4B-Instruct (4B params)

**Why**: Latest Qwen vision-language model with expanded OCR support (32 languages), robust to low light, blur, and tilt. Strong document parsing capabilities with 256K context window for long documents.

**Key capability**: Can potentially handle both Stage 1 (OCR + bboxes) and Stage 2 (error detection) in a single model. Spatial grounding enables 2D/3D coordinate output. BF16 fits in 12 GB (~8 GB VRAM).

**Risk**: General VLM, not OCR-specialized. Bounding box accuracy for fine-grained text regions may be lower than dedicated OCR models. Prompt sensitivity — output format depends heavily on prompt engineering.

### 8. Qwen3-VL-8B-Instruct (INT4, 8B quantized)

**Why**: Flagship Qwen VLM with advanced spatial perception. INT4 quantization needed for 12 GB fit (~6-8 GB VRAM). Represents the "single-model ceiling" — if any model can do both stages end-to-end, it's this one.

**Key capability**: 256K context, 2D/3D grounding, DeepStack for fine-grained visual details. Strongest reasoning capabilities among open-source VLMs in this size class.

**Risk**: Requires bitsandbytes INT4 quantization, which is currently incompatible with Blackwell (sm_120) GPUs. This candidate is blocked until bitsandbytes adds Blackwell support. For deployment (larger GPU or cloud), BF16 is preferred.

### 10. Hunyuan VL (~4B params)

**Why**: Tencent's vision-language model with strong performance on document understanding benchmarks. Multi-resolution architecture handles varied image sizes well -- particularly relevant for handwritten essays with inconsistent text scaling. Available in multiple sizes with the ~4B variant fitting within 12 GB.

**Key capability**: Multi-resolution processing adapts to varied text sizes without lossy resizing. Strong benchmark scores on document parsing tasks. Potential single-model candidate for both Stage 1 (OCR + bboxes) and Stage 2 (error detection).

**Risk**: Newer ecosystem with less community tooling for structured OCR output extraction compared to HuggingFace-native models. Document-specific benchmarks show promise but handwriting use case is untested in published work. May require vLLM or custom inference pipeline.

### 11-13. Traditional Baselines

**EasyOCR** (~50M params): Popular easy-to-use OCR with 80+ language support. CRNN-based architecture. Provides a "minimum viable" baseline — if VLMs can't beat EasyOCR on handwriting, they're not viable.

**docTR** (varies): Modular PyTorch library with separate detection (DBNet, LinkNet) and recognition (CRNN, SAR, ViTSTR) components. Represents the modular approach where components can be swapped independently.

**Tesseract 5** (traditional): Long-standing OCR engine using LSTM-based recognition. Provides a historical baseline to measure how far VLM-based approaches have advanced.

## Models Considered But Not Included (Yet)

The initial candidate list prioritizes models with **explicit OCR or document-parsing capabilities** — architectures that output bounding boxes and structured text natively, rather than relying on prompt engineering to extract coordinates from free-text VLM responses. This is a practical scoping decision for Phase 2-3, not a judgment on model quality.

Several strong general-purpose VLMs were excluded from the initial round but could be added as the research progresses:

### Gemma 4 (Google)

Google's latest open VLM family. Strong multimodal reasoning with competitive benchmark scores. Available in multiple sizes including variants that fit within 12 GB.

**Why not in Tier 1**: Gemma 4 is a general VLM, not document-specialized. Like Qwen3-VL, it would require prompt engineering for structured bbox output. Its OCR-specific performance on handwritten text is not well-documented in published benchmarks. The model's strength is reasoning, not fine-grained text localization.

**When to add**: If Qwen3-VL performs well as a single-model solution in Phase 4, Gemma 4 becomes a strong comparison point for the "general VLM vs. specialized pipeline" question. It would also be a natural Stage 2 candidate given its reasoning capabilities.

### Other Notable Exclusions

| Model | Reason for exclusion | When to reconsider |
|---|---|---|
| **InternVL 3** | General VLM, strong on benchmarks but no OCR specialization | Phase 4 — single-model comparison if Qwen3-VL shows promise |
| **Phi-4-vision** | Compact (good for 12 GB) but designed for chart/diagram reasoning, not OCR | If small model efficiency becomes the priority |
| **Pixtral (Mistral)** | Strong VLM but 12B parameters — too large for research GPU even quantized | Deployment phase with larger hardware |
| **LLaVA-NeXT** | Academic VLM with strong community, but OCR is not a design focus | Phase 4 comparison if needed |
| **MiniCPM-V** | Very compact (2-3B), fits easily in 12 GB | Worth testing if granite-docling-258M underperforms — similar size class |
| **OlmOCR** | Purpose-built for document OCR with structured output | Should be in Tier 1 — oversight during initial selection |
| **ColPali/ColQwen** | Retrieval-focused VLMs, not transcription/bbox models | Not applicable to this task |

### Selection Bias Acknowledgment

The initial candidate list over-represents models from the HuggingFace `transformers` ecosystem (Florence-2, TrOCR, granite-docling, Qwen3-VL, GOT-OCR2.0) and under-represents models requiring custom inference pipelines (Gemma via JAX/Flax, MonkeyOCR via its native toolkit). This is a pragmatic bias: the shared harness expects a `transformers`-compatible `inference_fn`, and adding non-HuggingFace backends increases per-candidate setup time.

This bias is actively being corrected. Hunyuan VL and MonkeyOCR have been promoted from the exclusion list to Tier 2 and Tier 1 respectively. Models that show strong benchmark performance will be added even if they require custom inference code -- the harness is designed to accept any callable, not just HuggingFace pipelines.

## Evaluation Metrics

| Metric | What it measures | Why it matters |
|---|---|---|
| **CER** (Character Error Rate) | Character-level transcription accuracy | Fine-grained measure of OCR quality, sensitive to handwriting legibility |
| **WER** (Word Error Rate) | Word-level transcription accuracy | More meaningful for downstream tasks (spell checking, grammar analysis) |
| **Reading Order (Kendall's tau)** | Correctness of text block ordering | Critical for unruled handwriting where spatial position does not imply reading order |
| **Error Detection F1** | Precision/recall of writing error identification | Measures Stage 2 quality — can the system find and classify student mistakes? |
| **Bounding Box IoU** | Spatial accuracy of error localization | Determines whether feedback annotations point to the correct location on the page |
| **End-to-end Latency** | Wall-clock time from image to feedback | The primary success criterion — must beat 40-60s baseline |
| **VRAM Peak** | Maximum GPU memory during inference | Gate for research-phase feasibility on 12 GB hardware |
| **Throughput (PPM)** | Pages per minute | Deployment-readiness metric |

## Hardware Constraints & Their Implications

The research phase uses an RTX 5070 Ti mobile GPU with 12 GB VRAM. This constraint drives several methodological decisions:

- **Quantization is allowed**: Models requiring >11 GB at full precision are evaluated at INT4/INT8. This simulates the cost-performance tradeoff that deployment engineers would face.
- **Deployment headroom is documented**: Models that exceed 12 GB during research are not rejected — they are noted as "deployment-only" candidates for when larger GPUs or cloud instances are available.
- **bitsandbytes gap**: The RTX 5070 Ti uses the Blackwell architecture (sm_120), which is not yet supported by bitsandbytes (as of June 2026). This temporarily blocks INT4 quantization for Qwen3-VL-8B. The workaround is to evaluate the 4B variant at BF16 and benchmark the 8B variant when bitsandbytes adds Blackwell support or when running on deployment hardware.
- **VRAM monitoring**: Every benchmark run records peak VRAM via `torch.cuda.max_memory_allocated()` and `nvidia-smi` to provide accurate per-candidate resource profiles.

## Research Phases

The methodology follows an eight-phase empirical approach:

1. **Environment & Baseline**: Establish the measurement infrastructure and capture current production pipeline performance.
2. **Tier 1 Evaluation**: Test the five most promising candidates. Measure latency, accuracy, VRAM, and qualitative scores.
3. **Tier 2 Evaluation**: Test baselines and larger models for comparison context.
4. **Pipeline Architecture**: Test Stage 1 + Stage 2 combinations. Answer the key question: one VLM or two specialized models?
5. **Reading Order Deep-Dive**: The hardest sub-problem. Compare Nemotron's relational model, heuristic post-processing, and VLM-based ordering. Includes sentence segmentation: disambiguating word-line assignments on unruled paper. The current production system uses a custom sequence-model algorithm tracking previous/current/next sentence relationships — any replacement must match or exceed this.
6. **Error Detection Accuracy**: Per-error-type evaluation (capitalization, spelling, grammar, punctuation, structural).
7. **Auditability Strategy**: Evaluate three approaches to make the system's outputs verifiable by humans.
8. **Final Assembly & Benchmark**: Combine the best components and run the full end-to-end benchmark against the baseline.

Each phase produces quantitative JSON results in `benchmark/results/` and updates the live results table in the README. No theoretical-only evaluations — every finding is backed by measured data on real hardware with real handwritten samples.
