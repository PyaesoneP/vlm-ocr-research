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
  - [Environments](#environments)
  - [PaddleOCR-VL on Blackwell (Docker)](#paddleocr-vl-on-blackwell-docker)
  - [MonkeyOCR GPU setup](#monkeyocr-gpu-setup)
  - [LocateAnything-3B setup](#locateanything-3b-setup)
  - [Cloud baseline](#cloud-baseline)
- [How the metrics are computed](#how-the-metrics-are-computed)
- [Roadmap](#roadmap)
- [Open issues & decisions](#open-issues--decisions)
- [Project structure](#project-structure)
- [License](#license)

---

## TL;DR

**14/14 Stage 1 OCR/localization candidates evaluated.** On clean IAM handwriting crops, the best OCR/VLMs beat the Google Document AI baseline on transcription and word localization. The full feedback pipeline is not solved yet: the real-world error set shows that fluent OCR can "helpfully" correct student mistakes before the grader sees them.

| Verdict | Model | CER | Notes |
|---|---|---|---|
| **Best automatable** | Qwen3-VL-8B | **0.035** (word) | Word IoU 0.722, via API. 4B runs locally: CER 0.022 line / 0.049 word, IoU 0.718. |
| **Best speed/accuracy** | Florence-2-large | 0.061 | 1.05s, 2.0 GB VRAM, but **line-level bboxes only**. |
| **Lowest observed CER** | Hunyuan VL | 0.015 | Manual-only (lmarena, 5 images). No API/HF access, no bbox. Not automatable. |
| **Cloud OCR baseline** | Google Doc AI | 0.108 (word) | Beaten on Stage 1 CER (→0.035) and word IoU (0.611 → 0.722). |

**Current bottleneck:** clean OCR/localization is strong, but Phase 4 found that truthful OCR is the gate for error detection. The Stage 1 transcript must preserve what the student actually wrote (`bred`, `minuts`, `forgoten`) instead of normalizing it to the intended correction.

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

### Phase 4 pipeline assembly

Phase 4 is implemented as a **pipeline assembly benchmark**, not a Qwen-only test and not a positive-error leaderboard. Stage 1 transcription and word localization were evaluated in Phases 2-3 on clean IAM handwriting, and Phase 4 now tests whether those components preserve real student mistakes well enough for a grader to catch them. The code lives in `pipeline/` and `scripts/benchmark_phase4.py`, with generated Stage 1 cache files under `pipeline_output/phase4_cache/`.

| Axis | Current options | Notes |
|---|---|---|
| Text source | Saved Phase 2/3 artifacts for IAM; live real-world sources for Qwen3-VL-4B normal/verbatim, Tesseract, docTR, EasyOCR, Florence-2, GOT-OCR2.0, SmolDocling, Nemotron OCR v2, PaddleOCR-VL, MonkeyOCR, and TrOCR | Real-world matrix rows must come from live runs on the raw real-world images; IAM artifacts are not substituted. Cloud/manual/API sources remain gated. |
| Box source | Qwen3-VL-4B word boxes, Tesseract word boxes, `same_stage1_boxes`, `realworld_aligned_words`, `no_boxes` diagnostic | `same_stage1_boxes` uses the boxes emitted by a live OCR/localization source. Historical result JSONs mostly store metrics, not reusable coordinates. |
| Stage 2 grader | Local Qwen3-VL-4B JSON grader | Gemini / Qwen API graders stay gated until explicit approval and cost estimate. |
| End-to-end arm | `single_qwen3vl_4b_e2e` | Tests whether one VLM prompt can replace the pipeline. |
| Stage 1 only | `--stage1-only` | Scores truthful transcription, evidence preservation, correction leaks, and localization without paying the Qwen grader latency. |

IAM handwritten crops remain the clean negative/control set: Phase 4 localization is measured as **word IoU** against `ground_truth_wordlevel.json`, plus latency, valid JSON rate, and false positives. `benchmark/test_dataset/phase4_positive_controls.json` contains tiny schema/behavior probes so the Stage 2 contract can be tested with known error types, but those probes are not treated as handwriting accuracy evidence. Error F1 and error-box IoU are reported as `not_applicable` whenever a dataset has no ground-truth errors.

Strategy names encode the composition:

```
two_stage__TEXT_SOURCE__BOX_SOURCE__GRADER
```

IAM one-image mixed local sanity run (`benchmark/results/phase4_pipeline.json`, `a04-039.png`):

The single-pass arm used `--max-new-tokens 1024`; the Stage 2 grader used `--grader-max-new-tokens 768`.

| Text + boxes -> grader | CER | WER | Word IoU | Valid JSON | False positives | Latency |
|---|---:|---:|---:|---:|---:|---:|
| Single Qwen3-VL-4B end-to-end | 0.052 | 0.241 | 0.000 | 1.00 | 1 | 29.6s |
| Qwen3-VL-4B + Qwen boxes -> Qwen grader | 0.056 | 0.217 | 0.765 | 1.00 | 0 | 121.9s |
| Qwen3-VL-4B + Tesseract boxes -> Qwen grader | 0.056 | 0.217 | 0.863 | 1.00 | 0 | 70.4s |
| Qwen3-VL-8B API artifact + Tesseract boxes -> Qwen grader | 0.024 | 0.108 | 0.863 | 1.00 | 0 | 30.0s |
| Florence-2 + Tesseract boxes -> Qwen grader | 0.097 | 0.337 | 0.863 | 1.00 | 0 | 38.0s |
| Tesseract + Tesseract boxes -> Qwen grader | 0.440 | 0.904 | 0.863 | 1.00 | 0 | 5.0s |

```bash
# No-model smoke test.
python scripts/benchmark_phase4.py --smoke-fake --max-images 1 --positive-controls

# Small mixed local matrix.
python scripts/benchmark_phase4.py \
  --text-sources qwen3vl_4b_wordlevel florence2_large_wordlevel tesseract_wordlevel \
  --box-sources tesseract_word_boxes \
  --max-images 5
```

**Carry-forward decision:** the single-VLM path advances only if it is faster and simpler than the best two-stage local path while preserving word-level localization quality, producing valid JSON reliably, and keeping false positives low on clean IAM pages. Positive error-recall scoring remains Phase 6.

#### Real-world handwritten error set

IAM is useful as a clean negative/control set, but it cannot validate whether Stage 2 catches real writing errors because the copied text has no intentional mistakes. The real-world probe set fills that gap: handwritten pages on paper, photographed/scanned, then annotated against the final image pixels.

Dataset protocol:

- Include both clean pages and pages with intentional errors.
- Use dark pen, one page per image, good lighting, flat page, and no shadows when possible.
- Save images without resizing/cropping after capture.
- Drop files under `benchmark/test_dataset/realworld_raw/`.
- Current files are named `rw_1.jpg` through `rw_20.jpg`.
- Provide either the corrected version for each page or short notes describing the intended errors.

Codex task:

- Generate rough word boxes from the frozen image using candidate localizers.
- Build the dataset JSON under `benchmark/test_dataset/realworld_writing_errors.json`.
- Manually inspect/fix word text and coordinates as needed.
- Anchor errors to `word_indices`; compute error bboxes from the union of those word boxes.
- Run the Phase 4 mixed matrix and the later Phase 6 error-detection metrics.

Target schema:

```json
{
  "image": "rw_1.jpg",
  "text": "i recieved the letter yesterday.",
  "corrected_text": "I received the letter yesterday.",
  "words": [
    {"index": 0, "text": "i", "bbox": [40, 50, 55, 78]},
    {"index": 1, "text": "recieved", "bbox": [70, 50, 165, 78]}
  ],
  "errors": [
    {
      "type": "capitalization",
      "word_indices": [0],
      "evidence_text": "i",
      "correction": "I"
    },
    {
      "type": "spelling",
      "word_indices": [1],
      "evidence_text": "recieved",
      "correction": "received"
    }
  ]
}
```

The current probe has 10 clean pages and 10 positive pages. A larger 20 positive / 20 clean set would be a good next step before treating error-detection numbers as a proper leaderboard.

Bootstrap commands once raw images and `source_texts.md` exist:

```bash
.venv/bin/python scripts/bootstrap_realworld_dataset.py
.venv/bin/python scripts/seed_realworld_draft_boxes.py
.venv/bin/python scripts/align_realworld_words.py
.venv/bin/python scripts/visualize_realworld_draft_boxes.py
.venv/bin/python scripts/visualize_realworld_draft_boxes.py --field words
python scripts/report_realworld_alignment.py
```

Current real-world status: `realworld_writing_errors.json` has 20 pages, 10 clean / 10 positive, 21 intended errors, Tesseract draft boxes under `draft_words`, and auto-aligned source-word boxes under `words`. All entries are still marked `auto_aligned_needs_review`; review the overlays in `benchmark/visualizations/realworld_aligned_words/` before treating word-IoU or error-box IoU as authoritative.

First real-world Qwen Stage 2 probe (`rw_11.jpg`, source text + auto-aligned words): valid JSON, `error_text_f1=1.00`, `error_detection_f1=0.667`, `error_box_iou=0.942`. Qwen found all three spelling errors, but one error bbox landed on the wrong line, so detection and localization should be reported separately.

Five-page real-world full-pipeline smoke (`rw_1`, `rw_2`, `rw_11`, `rw_12`, `rw_13`; two clean, three positive) is saved in `benchmark/results/phase4_realworld_full_pipeline_5image.json`.

| Full pipeline strategy | CER | WER | Word IoU | Valid JSON | Error text F1 | Error-box IoU | False positives | Avg latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Single Qwen3-VL-4B end-to-end | 0.371 | 0.387 | 0.100 | 0.60 | 0.000 | 0.000 | 0 | 63.3s |
| Tesseract live OCR/boxes -> Qwen grader | 0.324 | 0.902 | 0.900 | 1.00 | 0.000 | 0.000 | 0 | 39.4s |
| Qwen live OCR/boxes -> Qwen grader | 0.029 | 0.107 | 0.720 | 1.00 | 0.000 | 0.000 | 0 | 57.5s |

Interpretation: Qwen live OCR is much better than Tesseract for transcription, and Tesseract gives strong box overlap on these provisional auto-aligned boxes. However, the full two-stage pipelines missed the intentional errors because the OCR transcription often normalized the handwritten mistakes into corrected words before the grader saw them. The source-text probe shows the grader can catch errors when the evidence text is preserved; the full-pipeline result shows Phase 4 still needs either an OCR mode that preserves misspellings, a grader that compares image evidence, or an explicit observed-vs-corrected text contract. The single-pass arm is not reliable yet: two of five responses were invalid/truncated and word localization was poor.

Useful real-world full-pipeline commands:

```bash
.venv/bin/python scripts/benchmark_phase4.py --dataset realworld \
  --strategies two_stage__qwen3vl_4b_live_word_ocr__same_stage1_boxes__qwen3vl_4b_grader \
  --image rw_11.jpg --output benchmark/results/phase4_realworld_full_pipeline_qwen.json --include-raw

.venv/bin/python scripts/benchmark_phase4.py --dataset realworld --max-images 0 \
  --strategies single_qwen3vl_4b_e2e \
  --image rw_1.jpg --image rw_2.jpg --image rw_11.jpg --image rw_12.jpg --image rw_13.jpg \
  --output benchmark/results/phase4_realworld_full_pipeline_5image_single_qwen.json --include-raw

.venv/bin/python scripts/visualize_phase4_results.py \
  --result benchmark/results/phase4_realworld_full_pipeline_5image.json
```

Five-page real-world **live Stage 1 matrix** (`benchmark/results/phase4_realworld_live_matrix_5image.json`) reruns available local OCR engines directly on the real-world images, not on previous IAM artifacts. It also adds truthfulness metrics: `evidence_preserved_rate` measures whether intended erroneous spans survive Stage 1, and `correction_leak_total` counts cases where OCR outputs the corrected word instead of the written word.

| Live Stage 1 -> Qwen grader | Verbatim CER | Evidence preserved | Correction leaks | Word IoU | Error text F1 | Error detection F1 | Error-box IoU | False positives | Avg latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| docTR live word OCR | 0.205 | 2/7 | 0 | 0.601 | 0.074 | 0.000 | 0.041 | 0 | 35.1s |
| EasyOCR live word OCR | 0.598 | 0/7 | 0 | 0.637 | 0.000 | 0.000 | 0.000 | 0 | 36.8s |
| Qwen3-VL-4B normal word OCR | 0.024 | 2/7 | 4 | 0.720 | 0.133 | 0.000 | 0.000 | 4 | 52.8s |
| Qwen3-VL-4B verbatim word OCR | 0.008 | 5/7 | 1 | 0.719 | 0.434 | 0.246 | 0.455 | 1 | 60.7s |
| Tesseract live word OCR | 0.324 | 1/7 | 1 | 0.900 | 0.000 | 0.000 | 0.000 | 0 | 28.7s |

Current interpretation: the verbatim Qwen prompt is the best local evidence-preserving Stage 1 so far. It keeps Qwen-level word localization while reducing correction leaks and letting Stage 2 catch more real errors. It is still not perfect: `rw_11` preserved `bred` and `minuts` but corrected `forgoten`, and `rw_13` read `umbrela` as `umbrele`. Traditional OCR engines preserve fewer exact intended errors because their handwriting transcripts are too noisy; Tesseract still has the best spatial boxes.

The Phase 4 Stage 1 registry now records all Phase 2/3 candidates, including Hunyuan, Qwen3-VL-8B API, PaddleOCR-VL, Florence-2, GOT-OCR2.0, SmolDocling, Nemotron, MonkeyOCR, TrOCR, Doc AI, LocateAnything, and the local OCR baselines. Only live-local sources in the active environment were run in the table above. Cloud/manual/Docker/alternate-env sources are present in the registry but require separate approved live runs; they are not substituted from IAM artifacts for the real-world matrix.

One-page **all-live Stage 1 availability/truthfulness probe** (`benchmark/results/phase4_realworld_stage1_all_live_attempt_1image.json`, `rw_11.jpg`) now attempts every local/alt-env/server Stage 1 source directly on the real image and records failures instead of silently dropping them. This probe is Stage 1 only, so it isolates truthful OCR/localization from Qwen grader latency.

| Live Stage 1 source | Status in active `.venv` | Verbatim CER | Evidence preserved | Correction leaks | Word IoU | Stage 1 latency |
|---|---|---:|---:|---:|---:|---:|
| docTR live word OCR | ran | 0.190 | 2/3 | 0 | 0.614 | 0.9s |
| EasyOCR live word OCR | ran | 0.626 | 0/3 | 0 | 0.642 | 11.1s |
| Qwen3-VL-4B normal word OCR | ran | 0.074 | 0/3 | 3 | 0.707 | 29.5s |
| Qwen3-VL-4B verbatim word OCR | ran | 0.006 | 2/3 | 1 | 0.710 | 40.0s |
| Tesseract live word OCR | ran | 0.282 | 0/3 | 1 | 0.946 | 0.9s |
| Florence-2 live region OCR | failed here | not_applicable | 0/3 | 0 | 0.000 | 0.0s |
| GOT-OCR2.0 live OCR | failed here | not_applicable | 0/3 | 0 | 0.000 | 0.0s |
| SmolDocling live OCR | failed here | not_applicable | 0/3 | 0 | 0.000 | 0.0s |
| Nemotron OCR v2 live OCR | requires `aiml` env | not_applicable | 0/3 | 0 | 0.000 | 0.0s |
| PaddleOCR-VL live OCR | requires Paddle/Docker env | not_applicable | 0/3 | 0 | 0.000 | 0.0s |
| MonkeyOCR live OCR | requires local server | not_applicable | 0/3 | 0 | 0.000 | 0.0s |
| TrOCR base/large live line OCR | missing local HF processor cache | not_applicable | 0/3 | 0 | 0.000 | 0.0s |

The failed rows are useful: they mean the active `.venv` did not run those models live, and the benchmark did not backfill with IAM artifacts. Florence/GOT/SmolDocling failed on offline Hugging Face metadata/client checks in this sandbox; Nemotron, PaddleOCR-VL, MonkeyOCR, and TrOCR need their documented environment/server/cache path before they can be counted as real-world live rows.

```bash
.venv/bin/python scripts/benchmark_phase4.py --dataset realworld --max-images 0 \
  --strategies \
    two_stage__doctr_live_word_ocr__same_stage1_boxes__qwen3vl_4b_grader \
    two_stage__easyocr_live_word_ocr__same_stage1_boxes__qwen3vl_4b_grader \
    two_stage__qwen3vl_4b_live_word_ocr__same_stage1_boxes__qwen3vl_4b_grader \
    two_stage__qwen3vl_4b_verbatim_word_ocr__same_stage1_boxes__qwen3vl_4b_grader \
    two_stage__tesseract_live_word_ocr__same_stage1_boxes__qwen3vl_4b_grader \
  --image rw_1.jpg --image rw_2.jpg --image rw_11.jpg --image rw_12.jpg --image rw_13.jpg \
  --output benchmark/results/phase4_realworld_live_matrix_5image.json \
  --include-raw --continue-on-error

.venv/bin/python scripts/visualize_phase4_results.py \
  --result benchmark/results/phase4_realworld_live_matrix_5image.json \
  --output-dir benchmark/visualizations/phase4_realworld_live_matrix_5image

.venv/bin/python scripts/benchmark_phase4.py --dataset realworld --image rw_11.jpg \
  --text-sources all_live_local --box-sources same_stage1_boxes \
  --stage1-only --continue-on-error \
  --output benchmark/results/phase4_realworld_stage1_all_live_attempt_1image.json
```

#### Phase 4 resume plan

Next work should turn the current smoke results into an architecture decision. The highest-priority issue is still Stage 1 truthfulness: the OCR transcript must preserve the student's actual written evidence, because Stage 2 cannot catch errors that Stage 1 has already corrected away.

Current stopping point:

- Best current local Stage 1 candidate: `qwen3vl_4b_verbatim_word_ocr`.
- Best current box-only source: Tesseract word boxes, but its transcription is too noisy for grading.
- Most useful artifact to continue from: `benchmark/results/phase4_realworld_live_matrix_5image.json`.
- Most useful diagnostic artifact: `benchmark/results/phase4_realworld_stage1_all_live_attempt_1image.json`.
- Do not treat word IoU or error-box IoU as final until `auto_aligned_needs_review` boxes are reviewed.
- Do not use IAM Phase 2/3 artifacts as substitutes for real-world Stage 1 truthfulness.

Recommended order:

1. Review/fix the real-world word annotations.
   - Current `words` boxes are still `auto_aligned_needs_review`.
   - Inspect `benchmark/visualizations/realworld_aligned_words/`.
   - Fix bad word boxes and `word_indices` for the 20 real-world pages before trusting word IoU or error-box IoU as authoritative.

2. Run the full 20-page Stage 1 truthfulness matrix.
   - Use `--stage1-only` first so OCR truthfulness is isolated from Qwen grader latency.
   - Primary metrics: `verbatim_cer`, `verbatim_wer`, `evidence_preserved_rate`, `correction_leak_total`, and `word_iou`.

```bash
.venv/bin/python scripts/benchmark_phase4.py --dataset realworld --max-images 0 \
  --text-sources all_live_local --box-sources same_stage1_boxes \
  --stage1-only --continue-on-error \
  --output benchmark/results/phase4_realworld_stage1_all_live_20image.json
```

3. Get missing live Stage 1 candidates running one environment at a time.
   - Florence-2: run from `florencetf` / ensure local HF cache is available.
   - GOT-OCR2.0, SmolDocling, TrOCR: ensure local HF processor/model cache or allow a deliberate download setup step.
   - Nemotron OCR v2: run from `aiml`.
   - PaddleOCR-VL: use the Docker path, not the broken native `.venv_paddleocr` path.
   - MonkeyOCR: start the local llama.cpp server before the probe.
   - Cloud/manual/API sources stay gated: estimate cost and get explicit approval before live Doc AI, Gemini, Qwen3-VL-8B API, or Hunyuan runs.

4. Mix the best text source with the best box source.
   - Likely candidates: Qwen verbatim text + Qwen boxes, Qwen verbatim text + Tesseract boxes, docTR text + Tesseract boxes, and Qwen normal + Tesseract boxes as a control.
   - This answers whether a slightly noisy but truthful OCR source beats a fluent normalizing VLM for downstream error detection.

5. Tighten Stage 2 after Stage 1 evidence is preserved.
   - The source-text probe proves Qwen can catch errors when evidence survives.
   - Next prompt work should reduce false punctuation/capitalization calls, require exact `evidence_text`, and avoid invented errors.

6. Retest single-VLM end-to-end only after the two-stage evidence-preserving path is stable.
   - Current single Qwen has invalid/truncated responses and poor word localization.
   - A future single-pass arm should use the same verbatim contract as Stage 1 and advance only if it is valid, faster, and comparable on evidence preservation plus localization.

---

## Reproducibility

### Environments

Five Python environments are required because of conflicting CUDA / transformers / PaddlePaddle versions.

| Env | Type | PyTorch / CUDA | Used for |
|---|---|---|---|
| `.venv` | venv | 2.11.0+cu130 / 13.0 | Qwen3-VL, baselines, MonkeyOCR client, and HF model code when the required local cache/network is available |
| `aiml` | conda | 2.12.0+cu130 / 13.0 | Nemotron OCR v2 (CUDA toolkit must match PyTorch for the C++ extension build) |
| `florencetf` | conda | 2.11.0+cu130 / 13.0 | Florence-2 (needs transformers 4.40.0, incompatible with 5.x) |
| `.venv_paddleocr` | venv | PaddlePaddle 3.3.1 native path / Docker sm120 offline preferred | PaddleOCR-VL (native path is broken on this Blackwell setup; Docker is the measured path) |
| `.venv_locateanything` | venv | CUDA-matched PyTorch / transformers 4.57.1 | LocateAnything-3B / NVLabs Eagle Embodied text localization |

```bash
source .venv/bin/activate          # most models
conda activate aiml                # Nemotron OCR v2
conda activate florencetf          # Florence-2
source .venv_paddleocr/bin/activate # PaddleOCR-VL (native path — broken on Blackwell, see below)
source .venv_locateanything/bin/activate # LocateAnything-3B
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

### LocateAnything-3B setup

LocateAnything-3B (NVLabs Eagle Embodied) is evaluated as a **Stage 1 text-localization candidate**, not as a direct handwriting transcription replacement. The public task is scene text detection / grounding, so CER/WER are reported only if the model emits actual text labels in `<ref>...</ref>` spans. Primary metrics are word IoU, recall/precision, reading-order τ, latency, and VRAM.

It needs an isolated environment because the released stack requires `transformers==4.57.1`, `numpy==1.25.0`, and `Pillow==11.1.0`.

```bash
# Python 3.11 is required for the pinned numpy==1.25.0 wheel.
/home/pyaes/.pyenv/versions/3.11.14/bin/python -m venv .venv_locateanything
source .venv_locateanything/bin/activate
pip install --upgrade pip

# Install CUDA-matched PyTorch first. On this Blackwell setup, use the
# same torch/CUDA family as the other working environments.
pip install torch==2.11.0 torchvision==0.26.0

# Then install the LocateAnything stack:
pip install -r requirements-locateanything.txt

# Smoke test one cropped handwriting image:
python candidates/locateanything/eval.py benchmark/test_dataset/handwritten/a04-039.png

# CPU fallback is intentionally disabled for benchmarks. For a slow parser-only
# smoke test on CPU, opt in explicitly:
LOCATEANYTHING_ALLOW_CPU=1 python candidates/locateanything/eval.py benchmark/test_dataset/handwritten/a04-039.png

# Full word-level localization benchmark:
LOCATEANYTHING_MAX_IMAGE_SIDE=1024 LOCATEANYTHING_MAX_NEW_TOKENS=2048 \
  python scripts/eval_wordlevel_iou.py locateanything
```

Outputs:

- `benchmark/results/locateanything_wordlevel_handwritten.json`
- `benchmark/visualizations/locateanything_wordlevel/`

The model is under NVIDIA's non-commercial research license. Do not add it to the CER leaderboard unless the benchmark confirms transcription-quality labels; otherwise compare it only in the word-level localization table.

On 12 GB VRAM, full-resolution pages OOM in the vision encoder unless optimized attention is available. The candidate resizes inference images to `LOCATEANYTHING_MAX_IMAGE_SIDE` (default 1024) and maps predicted boxes back to original image coordinates before scoring. It also uses an explicit word-level prompt, then retries the broader scene-text prompt when the first pass emits fewer than `LOCATEANYTHING_MIN_WORD_BOXES` boxes (default 20). Current measured result: word IoU 0.592, CER 0.722, τ 0.176, 12.6s/image, 11.9 GB peak VRAM.

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
| 3: Tier-2 evaluation | 14/14 Completed | Full leaderboard above. Hunyuan #1 CER (manual); Qwen3-VL-8B best automatable. |
| 4: Pipeline assembly | In progress | Mixed text-source / box-source / grader matrix implemented; 5-image real-world matrix and all-live Stage 1 availability probe complete. Current focus: truthful OCR that preserves student mistakes. |
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
