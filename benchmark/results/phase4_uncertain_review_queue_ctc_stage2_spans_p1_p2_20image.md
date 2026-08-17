# Phase 4 CTC Uncertain Review Queue

This queue is metadata only. It does not rewrite OCR text or create automatic error claims.

## Summary

- Queue records: 7
- Priority counts: `{'P1': 4, 'P2': 3}`
- Decision counts: `{'REJECT_DISAGREEMENT': 3, 'REVIEW_ALTERNATIVE_READING': 4}`
- Filtered review kept: 4

## Records

### P1 rw_13_w011 (rw_13.jpg)

- Canonical: `umbrele`
- Word top: `umbrek`
- Line top: `umbrek`
- Review text: `umbrek`
- Decision: `REVIEW_ALTERNATIVE_READING` -> `REVIEW_ALTERNATIVE_READING`
- Word indices: `[11]`
- BBox: `[668, 427, 1040, 533]`
- Word crop: `pipeline_output/phase4_inference_evidence_graph/crops/rw_13/w011.png`
- Line crop: `pipeline_output/phase4_inference_evidence_graph/line_crops/rw_13/rw_13_w011_line.png`
- Reasons: `['long_token', 'qwen_normal_verbatim_disagreement', 'geometry_text_disagreement']`

### P1 rw_14_w026 (rw_14.jpg)

- Canonical: `fo`
- Word top: `to`
- Line top: `to`
- Review text: `to`
- Decision: `REVIEW_ALTERNATIVE_READING` -> `REVIEW_ALTERNATIVE_READING`
- Word indices: `[26]`
- BBox: `[410, 1012, 490, 1094]`
- Word crop: `pipeline_output/phase4_inference_evidence_graph/crops/rw_14/w026.png`
- Line crop: `pipeline_output/phase4_inference_evidence_graph/line_crops/rw_14/rw_14_w026_line.png`
- Reasons: `['qwen_normal_verbatim_disagreement', 'geometry_text_disagreement']`

### P1 rw_18_w013 (rw_18.jpg)

- Canonical: `intresting`
- Word top: `intesting`
- Line top: `intesting`
- Review text: `intesting`
- Decision: `REVIEW_ALTERNATIVE_READING` -> `REVIEW_ALTERNATIVE_READING`
- Word indices: `[13]`
- BBox: `[818, 419, 1196, 570]`
- Word crop: `pipeline_output/phase4_inference_evidence_graph/crops/rw_18/w013.png`
- Line crop: `pipeline_output/phase4_inference_evidence_graph/line_crops/rw_18/rw_18_w013_line.png`
- Reasons: `['long_token', 'qwen_normal_verbatim_disagreement', 'geometry_text_disagreement']`

### P1 rw_5_w020 (rw_5.jpg)

- Canonical: `woodon`
- Word top: `wooden`
- Line top: `wooden`
- Review text: `wooden`
- Decision: `REVIEW_ALTERNATIVE_READING` -> `REVIEW_ALTERNATIVE_READING`
- Word indices: `[20]`
- BBox: `[23, 760, 351, 849]`
- Word crop: `pipeline_output/phase4_inference_evidence_graph/crops/rw_5/w020.png`
- Line crop: `pipeline_output/phase4_inference_evidence_graph/line_crops/rw_5/rw_5_w020_line.png`
- Reasons: `['repeated_letter_candidate', 'qwen_normal_verbatim_disagreement', 'geometry_text_disagreement', 'alignment_instability']`

### P2 rw_20_w005 (rw_20.jpg)

- Canonical: `thursdav`
- Word top: `Thursday`
- Line top: `thrsdav`
- Review text: `Thursday`
- Decision: `REJECT_DISAGREEMENT` -> `REJECT_DISAGREEMENT`
- Word indices: `[5]`
- BBox: `[187, 277, 598, 407]`
- Word crop: `pipeline_output/phase4_inference_evidence_graph/crops/rw_20/w005.png`
- Line crop: `pipeline_output/phase4_inference_evidence_graph/line_crops/rw_20/rw_20_w005_line.png`
- Reasons: `['long_token', 'qwen_normal_verbatim_disagreement', 'geometry_text_disagreement']`

### P2 rw_20_w024 (rw_20.jpg)

- Canonical: `floor.`
- Word top: `Roc.`
- Line top: `flor`
- Review text: `Roc.`
- Decision: `REJECT_DISAGREEMENT` -> `REJECT_DISAGREEMENT`
- Word indices: `[24]`
- BBox: `[859, 1023, 1081, 1127]`
- Word crop: `pipeline_output/phase4_inference_evidence_graph/crops/rw_20/w024.png`
- Line crop: `pipeline_output/phase4_inference_evidence_graph/line_crops/rw_20/rw_20_w024_line.png`
- Reasons: `['punctuation_sensitive_token', 'repeated_letter_candidate', 'geometry_text_disagreement']`

### P2 rw_8_w009 (rw_8.jpg)

- Canonical: `Talips.`
- Word top: `Tulips.`
- Line top: `tlips`
- Review text: `Tulips.`
- Decision: `REJECT_DISAGREEMENT` -> `REJECT_DISAGREEMENT`
- Word indices: `[9]`
- BBox: `[1085, 244, 1333, 385]`
- Word crop: `pipeline_output/phase4_inference_evidence_graph/crops/rw_8/w009.png`
- Line crop: `pipeline_output/phase4_inference_evidence_graph/line_crops/rw_8/rw_8_w009_line.png`
- Reasons: `['punctuation_sensitive_token', 'qwen_normal_verbatim_disagreement', 'geometry_text_disagreement']`

