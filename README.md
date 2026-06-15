# VLM-OCR for Handwritten Essay Feedback

[![Status: Research](https://img.shields.io/badge/status-research-blue)](.)
[![Hardware: RTX 5070 Ti](https://img.shields.io/badge/hardware-RTX%205070%20Ti%20(12GB)-green)](.)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-orange)](./LICENSE)

Empirical evaluation of open-source OCR and vision-language models for a handwritten English essay feedback pipeline. The system must (1) transcribe difficult handwriting, (2) localize writing errors with bounding boxes, (3) recover reading order on unruled paper, and (4) generate natural-language feedback, matching or beating a commercial cloud pipeline (Google Document AI + Gemini) on accuracy while running fully local at lower latency and zero marginal cost. Every number below is measured locally on an RTX 5070 Ti (12 GB). Nothing is theoretical.

---

## Contents

- [TL;DR](#tldr)
- [The Confound (read before any number)](#the-confound-read-before-any-number)
- [Results](#results)
  - [Handwriting CER leaderboard](#handwriting-cer-leaderboard)
  - [Word-level bounding boxes](#word-level-bounding-boxes)
  - [Qualitative comparison](#qualitative-comparison)
- [Architecture](#architecture)
- [Reproducibility](#reproducibility)
  - [Environments](#environments)
  - [PaddleOCR-VL on Blackwell (Docker)](#paddleocr-vl-on-blackwell-docker)
  - [MonkeyOCR GPU setup](#monkeyocr-gpu-setup)
  - [Cloud baseline](#cloud-baseline)
- [How the metrics are computed](#how-the-metrics-are-computed)
- [Roadmap](#roadmap)
- [Open issues & decisions](#open-issues--decisions)
- [Project structure](#project-structure)
- [License](#license)

---

## TL;DR

**13/13 candidates evaluated. The cloud baseline is beaten on every metric.** The authoritative metric is handwriting CER on XML-cropped images (see [the confound](#the-confound-read-before-any-number) for why full-form numbers don't count).

| Verdict | Model | CER | Notes |
|---|---|---|---|
| **Best automatable** | Qwen3-VL-8B | **0.035** (word) | Word IoU 0.722, via API. 4B runs locally: CER 0.022 line / 0.049 word, IoU 0.718. |
| **Best speed/accuracy** | Florence-2-large | 0.061 | 1.05s, 2.0 GB VRAM, but **line-level bboxes only**. |
| **Highest CER overall** | Hunyuan VL | 0.015 | Manual-only (lmarena, 5 images). No API/HF access, no bbox. Not automatable. |
| **Cloud baseline** | Google Doc AI | 0.108 (word) | Beaten on both CER (→0.035) and word IoU (0.611 → 0.722). |

**The bottleneck has moved.** Stage 1 (OCR) is solved. The open problem is now Stage 2, Gemini error detection at 12.4s, which is 74% of the cloud pipeline's 16.7s.

---

## The Confound (read before any number)

Every IAM evaluation form contains the same text **twice**: once machine-printed (the prompt) and once in the writer's hand (the copy). Both regions therefore have *identical ground truth*, so **full-form CER/WER cannot tell whether a model read the easy printed text or the hard handwriting.**

```
┌──────────────────────────────────────┐
│ 1. HEADER (printed)                  │  "Sentence Database" + form ID
├──────────────────────────────────────┤
│ 2. PRINTED PROMPT (machine-printed)  │  serif paragraph — the "answer key"
├──────────────────────────────────────┤
│ 3. HANDWRITTEN COPY                  │  writer's copy of the prompt (~6 lines)
├──────────────────────────────────────┤
│ 4. FOOTER                            │  "Name:" + handwritten signature
└──────────────────────────────────────┘
```

Nemotron and MonkeyOCR emit the form header ("Sentence Database"), confirming they read the printed region; SmolDocling and GOT-OCR2.0 skip the header, so their source is ambiguous.

**Fix:** all authoritative numbers are measured on images cropped to section 3 using the IAM XML `<handwritten-part>` `<cmp>` bounding boxes (union of all handwritten lines, excluding the signature, 20px padding). This physically removes sections 1-2, forcing models to read only handwriting. Full-form numbers are retained elsewhere purely as printed-text OCR benchmarks. Crop logic: `scripts/crop_handwritten.py`; GT generation: `scripts/generate_ground_truth.py`.

> A 2026-06-12 code review fixed several metric bugs (τ stuck at 1.0, single-image CERs, Nemotron bbox double-conversion, a missing crop-offset transform). All numbers below reflect the corrected evaluation on the full 25-image dataset.

---

## Results

All accuracy numbers are on the **cropped-handwriting** dataset: 25 IAM forms, `ground_truth_handwritten.json`. CER/WER are whitespace-normalized. Full JSON per candidate is in `benchmark/results/`.

### Handwriting CER leaderboard

| # | Model | CER | WER | Latency | VRAM | Bbox | Notes |
|---|---|---|---|---|---|---|---|
| 1 | **Hunyuan VL** | **0.015** | - | ~5-10s (chat) | cloud | none | Manual via lmarena, 5 images. Text-only, not automatable. |
| 2 | **Qwen3-VL-4B** | **0.022** | 0.054 | 14.0s | 9.6 GB | line | Local. Best automatable by line CER. |
| 3 | **Qwen3-VL-8B** | **0.035** (word) | 0.223 | 26.8s (API) | - | word | Via novita API; can't run local (see [open issues](#open-issues--decisions)). Best word-level accuracy of any model. |
| 4 | **PaddleOCR-VL-1.6** | **0.045** | 0.085 | 31.94s | ~8 GB | block | Docker-only on Blackwell. Bimodal 3.7-61.9s. SOTA doc VLM (96.3% OmniDocBench, vendor). |
| 5 | **Florence-2-large** | **0.061** | 0.170 | 1.05s | 2.0 GB | line | Best speed/accuracy. Line IoU 0.76 / word IoU 0.176. |
| 6 | GOT-OCR2.0 | 0.088 | 0.263 | 2.53s | 3.4 GB | none | Text-only; output needs chat-token cleanup. |
| 7 | Google Doc AI | 0.095 line / 0.108 word | 0.315 | 2.19s line / 3.6s word | cloud | word | Cloud baseline. |
| 8 | SmolDocling-256M | 0.107 | 0.232 | 5.37s | 0.8 GB | coarse | DocTags + bboxes (IoU 0.24). Hallucinates/repeats on handwriting. |
| 9 | Nemotron OCR v2 | 0.214 | 0.523 | 0.07s | 0.6 GB | partial | **Fastest (239× baseline).** Recognizer trained on printed text. IoU 0.28, reading-order τ 0.89. |
| 10 | TrOCR-large | 0.244 | 0.506 | 1.35s | 2.2 GB | heuristic | IAM-finetuned, line-level. |
| 11 | docTR | 0.272 | 0.757 | 1.09s | ~1 GB | word | Best traditional OCR for handwriting. |
| 12 | TrOCR-base | 0.380 | 0.681 | 1.22s | ~1.5 GB | heuristic | ~50% worse CER than large. |
| 13 | Tesseract 5 | 0.439 | 0.862 | 0.50s | CPU | word | Best bbox precision (word IoU 0.812), unusable CER. |
| 14 | EasyOCR | 0.628 | 1.086 | 0.88s | ~0.5 GB | word | Not viable for handwriting. |

What the table doesn't say in numbers:

- **No traditional OCR engine breaks CER 0.27.** VLMs are 5-8× more accurate at transcription on handwriting.
- **PaddleOCR-VL's bimodal latency** is a 12 GB artifact: long-text images overflow VRAM and trigger WDDM sysmem fallback (~100× slowdown). Expect sub-5s uniform on ≥16 GB.
- **GOT-OCR2.0 has no bbox output in any mode**: format mode is text formatting, fine-grained mode takes a bbox as *input*. Verified against HF `stepfun-ai/GOT-OCR-2.0-hf` and original source.
- **MonkeyOCR's error is generation control, not OCR.** It reads correctly but can't stop, repeating paragraphs 2-3×. Officially does not support handwriting.
- **Qwen3-VL needed a class fix** (`Qwen3VLForConditionalGeneration` → `AutoModelForImageTextToText`) and bbox-token stripping; the initial run reported a misleading CER of 0.597.

### Word-level bounding boxes

Per-word IoU against 1891 IAM XML words, greedy spatial match at IoU ≥ 0.05.

| Model | Word IoU | CER | τ | Latency | Bbox source |
|---|---|---|---|---|---|
| **Tesseract 5** | **0.812** | 0.443 | 0.993 | 0.5s | native word engine |
| **Qwen3-VL-8B** | **0.722** | 0.035 | - | 26.8s (API) | prompted |
| **Qwen3-VL-4B** | **0.718** | 0.049 | 1.000 | ~80s | prompted |
| Google Doc AI | 0.611 | 0.108 | - | 3.6s | native (cloud) |
| EasyOCR | 0.597 | 0.625 | 0.760 | 1.1s | native |
| docTR | 0.581 | 0.275 | 0.999 | 1.2s | native |
| Florence-2-large | 0.176\* | 0.091 | 1.000 | 1.7s | line-level only |

\* Florence-2 emits ~10 line boxes per page, so word-granularity IoU collapses; its line-vs-line IoU is 0.76. PaddleOCR-VL outputs block-level boxes, so word IoU isn't meaningful for it.

Takeaways: **Tesseract wins on box precision but is unusable for reading. Qwen3-VL is the only approach delivering both accurate transcription and usable word localization.** Reading order is effectively solved for single-column forms (τ > 0.99 for all but EasyOCR), multi-column/unruled ordering is untested and is the focus of [Phase 5](#roadmap). For Stage 2 error boxes, Tesseract or Qwen are the only viable candidates.

### Qualitative comparison

Same line (`a04-039.png`), abbreviated. Full outputs for all models are in `benchmark/results/`.

```
GROUND TRUTH
  In Vienna, before flying off to Moscow, Mr. Khrushchov said he hoped his
  weekend talks with President Kennedy would help "to establish an enduring
  peace between nations." ...

GOOGLE DOC AI (CER 0.095 line / 0.108 word)   — fragments badly, drops words
  In Vienna, before flying off to Moscow, ... Mr. Khrushchou said ... would
  help " to " establish an onduring peace between nations. ...

FLORENCE-2-LARGE (CER 0.061)                   — fluent, errs on names
  ... Mr. Khrushdov ... talks with President Kennedy would be help 4 to
  establish an enduring peace ... an ordinary peace for he people ...

NEMOTRON (CER 0.214)                            — readable but reading order scrambled
  4 The Souiet unioh ha always strven ... In Vienna, before juying off to
  Morcow, Mr. khrushchov said he hopect his weekend talks ...
```

The pattern is diagnostic: fluent models (Florence-2, GOT) miss specific content words (`Khrushchov`, `Austria`), while broken models (SmolDocling, Nemotron on handwriting) garble structure. See [WER vs CER as a diagnostic](#wer-vs-cer-as-a-diagnostic).

---

## Architecture

No single model does everything. The pipeline is two-stage:

```
Stage 1 — Transcription + localization (OCR/VLM)
  image → { text, bounding boxes, reading order }   (+ sentence segmentation)
                          │
Stage 2 — Error detection + feedback (LLM/VLM)
  { text + bboxes + image } → { error bboxes + NL feedback }
```

- **Sentence segmentation** on unruled paper is hard: assigning a word to the line above or below is ambiguous when spacing is. It's a first-class evaluation criterion (Phase 5).
- **Stage 2 candidates:** Qwen3-VL-4B/8B, granite-docling-258M, Hunyuan VL, Gemini 3.5 Flash (cloud comparison).
- **Open question:** can one VLM (e.g. Qwen3-VL) handle both stages end-to-end?

---

## Reproducibility

### Environments

Four Python environments are required because of conflicting CUDA / transformers / PaddlePaddle versions.

| Env | Type | PyTorch / CUDA | Used for |
|---|---|---|---|
| `.venv` | venv | 2.11.0+cu130 / 13.0 | SmolDocling, GOT-OCR2.0, MonkeyOCR, DocLayoutYOLO, Qwen3-VL, TrOCR, baselines |
| `aiml` | conda | 2.12.0+cu130 / 13.0 | Nemotron OCR v2 (CUDA toolkit must match PyTorch for the C++ extension build) |
| `florencetf` | conda | 2.11.0+cu130 / 13.0 | Florence-2 (needs transformers 4.40.0, incompatible with 5.x) |
| `.venv_paddleocr` | venv | PaddlePaddle 3.4.0+ / 12.9 | PaddleOCR-VL (bundles its own NCCL/cuBLAS, conflicts with PyTorch's CUDA 13.0) |

```bash
source .venv/bin/activate          # most models
conda activate aiml                # Nemotron OCR v2
conda activate florencetf          # Florence-2
source .venv_paddleocr/bin/activate # PaddleOCR-VL (native path — broken on Blackwell, see below)
```

Blackwell (sm_120) is unsupported by stable PyTorch; this project uses 2.11.0+cu130. transformers is pinned at 5.8.1 (needed for SmolDocling's `AutoModelForMultimodalLM`).

### PaddleOCR-VL on Blackwell (Docker)

PaddleOCR-VL uses the **PaddlePaddle native engine**, not HuggingFace transformers (`from paddleocr import PaddleOCRVL`). PyPI `paddlepaddle-gpu` lacks sm_120 support. **Docker is the only reliable path on WSL2 + Blackwell**, the native `.venv_paddleocr` (PaddlePaddle 3.3.1) hangs at `paddle.to_tensor()`.

Five distinct failure modes; the last line printed before a freeze identifies which:

| Last line printed | Cause | Fix |
|---|---|---|
| `Checking connectivity to the model hosters...` | pings HF/BOS/ModelScope, stalls on bad routes | `-e PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` |
| `Fetching N files...` | ~2 GB BOS download stalls outside China; `--rm` re-downloads | named volume `-v paddlex_models:/home/paddleocr/.paddlex` |
| after `generation_config.json`, before `Latency:` | JIT kernel compile or `/dev/shm` exhaustion (Docker default 64 MB) | `--shm-size=8g` + persistent container |
| worked before, now hangs immediately | stuck GPU / zombie VRAM (WSL2 leaks between `--rm` runs) | `wsl --shutdown`; set NVIDIA Control Panel → CUDA Sysmem Fallback → *Prefer No Sysmem Fallback* |
| prints `Latency:` and results, never exits | known Paddle teardown hang | `import os; os._exit(0)` at script end |

**WSL2 gotchas:** VRAM is not freed between `docker run --rm` containers (WDDM leak), so subsequent runs spill weights into shared memory over PCIe, a ~100× slowdown that looks like a freeze. `wsl --shutdown` between runs clears it; the sysmem-fallback policy converts the silent slowdown into a fast, visible OOM. **Never `docker kill`** during active CUDA work (SIGKILL leaks VRAM via dxgkrnl), always `docker stop`.

> Import conflict (2026-06-12): importing the project's `candidates` package before `PaddleOCRVL.predict()` triggers sysmem fallback even on a clean GPU. Use a standalone script with **zero project imports** and **data-only mounts**.

```bash
# Pull once (Chinese registry, be patient):
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-nvidia-gpu-sm120-offline

# Benchmark (standalone script, data-only mounts):
docker run --rm --gpus all --network host --shm-size=8g \
  -e PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True -e PYTHONUNBUFFERED=1 \
  -v paddlex_models:/home/paddleocr/.paddlex \
  -v $PWD/benchmark/test_dataset:/data:ro \
  -v $PWD/benchmark/results:/results \
  -v $PWD/scripts/bench_paddleocr_handwritten.py:/scripts/bench.py:ro \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-nvidia-gpu-sm120-offline \
  python3 -u /scripts/bench.py
```

For rapid iteration, run a persistent container (`-d ... sleep infinity`) and `docker exec` into it to keep the JIT cache warm; `docker stop` to tear down.

Reference: [PaddleOCR-VL NVIDIA Blackwell tutorial](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL-NVIDIA-Blackwell.html).

### MonkeyOCR GPU setup

Pre-built llama.cpp binaries are CPU-only (54s/image). Building from source with CUDA gives 4.27s/image (10×).

```bash
git clone https://github.com/ggerganov/llama.cpp.git /tmp/llama.cpp && cd /tmp/llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target llama-server

cd build/bin && LD_LIBRARY_PATH=. ./llama-server \
  -hf dinhquangson/MonkeyOCR-pro-1.2B-Vision-GGUF \
  --host 0.0.0.0 --port 8080 -ngl 99 -c 8192 \
  --mmproj-offload --image-min-tokens 1024
curl -s http://localhost:8080/health   # → {"status":"ok"}
```

`-ngl 99` offloads all layers; `--mmproj-offload` puts the vision projector on GPU (critical for encoding speed). At `-c 4096` the image tokens don't fit and output truncates.

### Cloud baseline

**16.7s end-to-end** (Doc AI 2.8s + Gemini 3.5 Flash 12.4s). `gemini-3.5-flash` is a preview model available only in the `global` region on Vertex AI (`asia-southeast1` returns 404). Document AI runs in `asia-southeast1` via a regional endpoint.

```bash
gcloud auth application-default login
set -a && source .env && set +a   # GCP_PROJECT, DOCAI_PROCESSOR_ID, GEMINI_MODEL=gemini-3.5-flash
.venv/bin/python -u benchmark/baseline.py
```

`baseline.py` prefers an API key, falls back to Vertex AI via ADC (production-tier quota; free-tier keys at 20 req/day are unusable for benchmarking).

---

## How the metrics are computed

All bbox/reading-order metrics first run **greedy spatial matching**: each predicted block matches the best-IoU unmatched GT block; only matched pairs score.

**CER**: Levenshtein character distance on whitespace-normalized text (newlines→spaces, collapsed, stripped), divided by GT length. Both empty = 0.0; GT empty + pred non-empty = 1.0. Per-image, then arithmetic mean across all 25 (no exclusions).

**WER**: identical algorithm on `.split()` word tokens. WER ≥ CER always, since a wrong word costs ≥1 character error.

*Worked example:* `Khrushchov` → `Khrushdov`: 1 substitution + 1 deletion = distance 2, CER = 2/10 = **0.20**; the whole word is wrong, so WER = **1.0**.

**Bbox IoU**: standard intersection-over-union on `[x1,y1,x2,y2]`. Formats normalized first: `[x,y,w,h]` → `[x,y,x+w,y+h]`; 4-corner quad → `[min,min,max,max]`. Match threshold IoU ≥ 0.1; mean over matched pairs. Images with 0 matches are excluded.

**Reading order: Kendall's τ-b.** Predicted order = sort blocks by (y, x); align to GT by IoU > 0.05; τ-b over the parallel rank lists (+1 perfect, 0 random, −1 reversed). Per-image mean, excluding images with <2 matches.

> τ = 1.00 for line-level models is *expected*, not impressive: IAM forms are single-column, so any top-to-bottom sort matches `[0,1,2,...]`. Region-level detectors (DocLayoutYOLO τ = −0.17, PP-DocLayout-L τ = 0.92) drop below 1.0 because their blocks don't map 1:1 to line-level GT. Genuine reading-order difficulty (multi-column/unruled) is Phase 5.

### WER vs CER as a diagnostic

The ratio reveals the *type* of error:

| WER/CER | Meaning | Example |
|---|---|---|
| ~2× | isolated character errors in mostly-correct words (best case) | PaddleOCR-VL |
| ~3× | errors cluster in content words (names) | Florence-2 (`Khrushchov`→`Khrushdov`) |
| ~1× | WER ≈ CER → **hallucination**, whole chunks fabricated | SmolDocling full-form |

---

## Roadmap

| Phase | Status | Summary |
|---|---|---|
| 1: Setup & baseline | Completed | Environments, IAM dataset (1,539 forms, 25 curated), 16.7s baseline, harness + metrics. 27/27 validation checks pass. |
| 2: Tier-1 evaluation | 6/6 Completed| Surfaced the [printed-text confound](#the-confound-read-before-any-number); established cropped-handwriting as the authoritative protocol. |
| 3: Tier-2 evaluation | 13/13 Completed | Full leaderboard above. Hunyuan #1 CER (manual); Qwen3-VL-8B best automatable. |
| 4: Pipeline assembly | Pending | Best Stage 1 + Stage 2 combos. One end-to-end VLM vs. OCR + small LLM? Latency breakdown. |
| 5: Reading-order deep-dive | Pending | The hard case: unruled/multi-column. Nemotron relational model, PP-StructureV3, heuristics, VLM prompting. τ vs. manual annotation. |
| 6: Error-detection accuracy | Pending | Per error type (capitalization, spelling, grammar, punctuation, structural): P/R/F1 + error-box IoU. |
| 7: Auditability | Pending | Per-word crops vs. annotated overlay vs. side-by-side JSON: storage, verifiability, complexity. |
| 8: Final assembly & benchmark | Pending | Full end-to-end vs. baseline; document architecture, latency, and total cost of ownership. |

**Targets:** latency < 16.7s · CER < 5% · WER < 10% · reading-order τ > 0.85 · error-detection F1 > 0.75 · error-box IoU > 0.6 · VRAM < 11 GB (research phase).

---

## Open issues & decisions

**Open**

- **bitsandbytes lacks Blackwell (sm_120) kernels** → no local INT4, which blocks Qwen3-VL-8B on the 12 GB research GPU. Worked around via the novita HF API; revisit when sm_120 kernels ship, or run BF16 on a larger GPU.

**Resolved**

- PaddleOCR-VL `Unsupported GPU architecture` (PaddlePaddle 3.2.1 lacks sm_120) → official `sm120-offline` Docker image.

**Decisions**

- **Direct handwritten evaluation only**, full-form is confounded by printed text (Phase 2).
- **English only**; multilingual out of scope.
- **Cloud APIs allowed for Stage 2 comparison**, the goal is fully open-source, local or API.
- **transformers 5.x adopted**, required by SmolDocling; backward-compat verified for GOT-OCR2.0.
- **12 GB is a research-phase limit only**, production assumes larger compute and permits cloud APIs.

**Possible extensions:** fine-tuning a compact model (SmolDocling-258M / GOT-OCR2.0) if a target deployment misses accuracy targets; vLLM/SGLang for VLM throughput (SmolDocling claims 0.35s/page on A100 via vLLM); ONNX/TensorRT for latency.

---

## Project structure

```text
vlm-ocr-research/
├── README.md                # this file
├── benchmark/
│   ├── harness.py           # shared evaluation harness
│   ├── metrics.py           # CER, WER, IoU, reading-order scoring
│   ├── test_dataset/        # handwritten essay samples + ground truth
│   ├── results/             # per-candidate JSON
│   └── visualizations/      # bbox overlays + comparisons
├── candidates/              # one dir per model (qwen3_vl, florence2, got_ocr, …)
├── scripts/
│   ├── bench_paddleocr_handwritten.py  # standalone PaddleOCR-VL Docker benchmark
│   ├── crop_handwritten.py             # XML-guided crop to the handwritten region
│   ├── eval_handwritten.py             # authoritative cropped-handwriting eval
│   └── …
└── pipeline/                # final two-stage pipeline
```

---

## License

Apache 2.0. See [LICENSE](./LICENSE).