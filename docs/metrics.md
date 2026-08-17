# How the Metrics Are Computed

Part of [vlm-ocr-research](../README.md). Full definitions and edge cases for every metric in the benchmark, plus the WER/CER diagnostic ratio.

---

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

