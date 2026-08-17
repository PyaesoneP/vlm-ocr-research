# VLM-OCR for Handwritten Essay Feedback

[![Status: Research](https://img.shields.io/badge/status-research-blue)](.)
[![Hardware: RTX 5070 Ti](https://img.shields.io/badge/hardware-RTX%205070%20Ti%20(12GB)-green)](.)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-orange)](./LICENSE)

Empirical evaluation of open-source OCR and vision-language models for a handwritten English essay feedback pipeline. The system must (1) transcribe difficult handwriting exactly as written, (2) localize writing errors with bounding boxes, (3) recover reading order on unruled paper, and (4) generate natural-language feedback. The research target is to match or beat a commercial cloud pipeline (Google Document AI + Gemini) while running locally where possible at lower latency and zero marginal cost. Results are marked local/API/cloud/manual where relevant; nothing below is a hypothetical model-card claim.

---

## Contents

- [TL;DR](#tldr)
- [The Confound (read before any number)](#the-confound-read-before-any-number)
- [Results](#results)
  - [Handwriting CER leaderboard](#handwriting-cer-leaderboard)
  - [Word-level bounding boxes](#word-level-bounding-boxes)
  - [Qualitative comparison](#qualitative-comparison)
- [Architecture](#architecture)
  - [Phase 4 pipeline assembly](#phase-4-pipeline-assembly)
- [Reproducibility](#reproducibility)
- [How the metrics are computed](#how-the-metrics-are-computed)
- [Roadmap](#roadmap)
- [Open issues & decisions](#open-issues--decisions)
- [Project structure](#project-structure)
- [License](#license)

### Deep dives

- [Phase 4 experiment log](docs/phase4.md) - every result matrix, verifier, and the next-steps plan
- [Reproducibility & environment setup](docs/reproducibility.md) - environments, Docker/Blackwell failure modes, per-model setup
- [Metrics definitions](docs/metrics.md) - CER/WER/IoU/tau definitions, worked example, WER/CER diagnostic

---

## TL;DR

**14/14 Stage 1 OCR/localization candidates evaluated.** On clean IAM handwriting crops, the best OCR/VLMs beat the Google Document AI baseline on transcription and word localization. The full feedback pipeline is not solved yet: the real-world error set shows that fluent OCR can "helpfully" correct student mistakes before the grader sees them.

| Verdict | Model | CER | Notes |
|---|---|---|---|
| **Best automatable** | Qwen3-VL-8B | **0.035** (word) | Word IoU 0.722, via API. 4B runs locally: CER 0.022 line / 0.049 word, IoU 0.718. |
| **Best speed/accuracy** | Florence-2-large | 0.061 | 1.05s, 2.0 GB VRAM, but **line-level bboxes only**. |
| **Lowest observed CER** | Hunyuan VL | 0.015 | Manual-only (lmarena, 5 images). No API/HF access, no bbox. Not automatable. |
| **Cloud OCR baseline** | Google Doc AI | 0.108 (word) | Beaten on Stage 1 CER (→0.035) and word IoU (0.611 → 0.722). |

**Current bottleneck:** clean OCR/localization is strong, and the reviewed source-text Stage 2 gate is now recovered with `contract_v3` plus adjudication. The remaining blocker is Stage 1 truthfulness and text/box alignment: OCR still must preserve what the student actually wrote (`bred`, `minuts`, `forgoten`) and provide boxes anchored to the same word list the grader sees.

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
| LocateAnything-3B | 0.592 | 0.722 | 0.176 | 12.6s | prompted word boxes |
| docTR | 0.581 | 0.275 | 0.999 | 1.2s | native |
| Florence-2-large | 0.176\* | 0.091 | 1.000 | 1.7s | line-level only |

\* Florence-2 emits ~10 line boxes per page, so word-granularity IoU collapses; its line-vs-line IoU is 0.76. PaddleOCR-VL outputs block-level boxes, so word IoU isn't meaningful for it.

Takeaways: **Tesseract wins on box precision but is unusable for reading. Qwen3-VL is the only approach delivering both accurate transcription and usable word localization.** LocateAnything-3B is now measured: its adaptive word-box prompt reaches EasyOCR/docTR-tier localization (IoU 0.592) but the text labels are not transcription-quality, so it remains a localization candidate rather than an OCR replacement. Reading order is effectively solved for single-column forms (τ > 0.99 for all but EasyOCR), multi-column/unruled ordering is untested and is the focus of [Phase 5](#roadmap). For Stage 2 error boxes, Tesseract, Qwen, and LocateAnything-as-localizer are the candidates worth testing.

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

The pattern is diagnostic: fluent models (Florence-2, GOT) miss specific content words (`Khrushchov`, `Austria`), while broken models (SmolDocling, Nemotron on handwriting) garble structure. See [WER vs CER as a diagnostic](docs/metrics.md#wer-vs-cer-as-a-diagnostic).

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

### Phase 4 pipeline assembly

Phase 4 is a **pipeline assembly benchmark**: can the Stage 1 components preserve real student mistakes well enough for a grader to catch them? The full experiment log - all result matrices, verifiers, the evidence-graph/CTC work, and the next-steps plan - is in [docs/phase4.md](docs/phase4.md).

**Current state** (reviewed 20-page real-world set, local-only runs):

- **Stage 2 is solved on reviewed source text.** `contract_v3` + rule-based adjudication: valid JSON 1.00, error-detection F1 1.000, error-box IoU 1.000, 0 clean-page false positives - the task's source-text upper bound.
- **Best live local full pipeline:** Qwen3-VL-4B verbatim text + Qwen word boxes + `contract_v3` + adjudication -> error-detection F1 0.585, error-box IoU 0.711, 5 false positives. Below the 0.75 target, so the two-stage architecture does not advance yet.
- **Blocker: Stage 1 truthfulness.** Qwen verbatim preserves 12/21 intended error spans, crop-verified 13/21, normal 10/21; the promotion gate is 17/21 with <=2 correction leaks. Tesseract is the best box source (word IoU 0.827) but its transcript is too noisy for Stage 2.
- **Optical CTC scoring stays metadata-only.** The GFCN line-context scorer top-ranks 17/21 visible error forms; a guarded margin-0.4 policy auto-supports only `forgoten`, `usualy`, and `Thursday` with 0 clean-page corruptions. Development-set calibration, not a production selector.
- **No missing Stage 1 candidate was promoted** (GOT-OCR2.0, Florence-2, PaddleOCR-VL, Nemotron, MonkeyOCR, SmolDocling, TrOCR all fail the gates).
- **Next:** freeze the rules, reach >=17/21 evidence recovery (or a plausible path to F1 >= 0.75), then rerun the focused full-pipeline matrix.

---

## Reproducibility

Five Python environments are required (conflicting CUDA / transformers / PaddlePaddle versions); Blackwell sm_120 needs PyTorch 2.11.0+cu130, and PaddleOCR-VL runs reliably only via Docker on this machine. Full setup details, the Docker failure-mode table, and per-model commands are in [docs/reproducibility.md](docs/reproducibility.md).

```bash
source .venv/bin/activate                # most models
conda activate aiml                      # Nemotron OCR v2
conda activate florencetf                # Florence-2
source .venv_paddleocr/bin/activate      # PaddleOCR-VL (native broken on Blackwell; use Docker)
source .venv_locateanything/bin/activate # LocateAnything-3B
```

---

## How the metrics are computed

CER/WER are Levenshtein distance on whitespace-normalized text or word tokens; bbox and reading-order metrics use greedy IoU spatial matching, with reading order scored by Kendall's tau-b over parallel rank lists. Full definitions, edge cases, a worked example, and the WER/CER diagnostic are in [docs/metrics.md](docs/metrics.md).

---

## Roadmap

| Phase | Status | Summary |
|---|---|---|
| 1: Setup & baseline | Completed | Environments, IAM dataset (1,539 forms, 25 curated), 16.7s baseline, harness + metrics. 27/27 validation checks pass. |
| 2: Tier-1 evaluation | 6/6 Completed| Surfaced the [printed-text confound](#the-confound-read-before-any-number); established cropped-handwriting as the authoritative protocol. |
| 3: Tier-2 evaluation | 14/14 Completed | Full leaderboard above. Hunyuan #1 CER (manual); Qwen3-VL-8B best automatable. |
| 4: Pipeline assembly | In progress | Reviewed annotations, Stage 2 source-text recovery, adjudicated two-stage matrix, and single-pass diagnostic complete. Current focus: evidence-preserving OCR plus shared text/box alignment. |
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
│   ├── benchmark_phase4.py             # single-VLM vs two-stage architecture benchmark
│   ├── crop_handwritten.py             # XML-guided crop to the handwritten region
│   ├── eval_handwritten.py             # authoritative cropped-handwriting eval
│   └── …
└── pipeline/                # Phase 4 contracts, prompts, parsing, metrics, runners
```

---

## License

Apache 2.0. See [LICENSE](./LICENSE).
