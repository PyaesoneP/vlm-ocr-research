# Held-out capture protocol

Use this folder for the frozen held-out set. Do not tune prompts, thresholds, guards, or candidate generation after scoring it.

## What to write

Copy the pages in `source_texts.md` by hand.

- Use one physical page per section.
- Save the images as `rw_101.jpg` through `rw_140.jpg`.
- Copy the **Write exactly** paragraph verbatim.
- For positive pages, keep the mistakes exactly as written.
- Do not write the "Correct version" or "Intended errors" sections on the page.

## Capture rules

- Use dark pen on plain white or lightly ruled paper.
- Keep the page flat.
- Use good lighting with minimal shadow.
- Include the whole written page in the image.
- Do not crop, resize, sharpen, or rotate after capture.
- If a photo is blurry or partly cut off, retake it before adding it here.

## After images are added

First rebuild the source-truth skeleton and confirm the expected counts:

```bash
.venv/bin/python scripts/bootstrap_realworld_dataset.py \
  --source benchmark/test_dataset/realworld_heldout_raw/source_texts.md \
  --image-dir benchmark/test_dataset/realworld_heldout_raw \
  --output benchmark/test_dataset/realworld_heldout_writing_errors.json

.venv/bin/python scripts/validate_realworld_dataset.py \
  --dataset benchmark/test_dataset/realworld_heldout_writing_errors.json \
  --image-dir benchmark/test_dataset/realworld_heldout_raw \
  --expected-pages 40 --expected-clean 20 --expected-positive 20 --expected-errors 40 \
  --allow-skeleton
```

Then seed draft boxes, align them to the source text, and render overlays:

```bash
.venv/bin/python scripts/seed_realworld_draft_boxes.py \
  --input benchmark/test_dataset/realworld_heldout_writing_errors.json \
  --output benchmark/test_dataset/realworld_heldout_writing_errors.json \
  --image-dir benchmark/test_dataset/realworld_heldout_raw

.venv/bin/python scripts/align_realworld_words.py \
  --input benchmark/test_dataset/realworld_heldout_writing_errors.json \
  --output benchmark/test_dataset/realworld_heldout_writing_errors.json

.venv/bin/python scripts/visualize_realworld_draft_boxes.py \
  --dataset benchmark/test_dataset/realworld_heldout_writing_errors.json \
  --image-dir benchmark/test_dataset/realworld_heldout_raw \
  --field words \
  --output-dir benchmark/visualizations/realworld_heldout_aligned_words

.venv/bin/python scripts/report_realworld_alignment.py \
  --dataset benchmark/test_dataset/realworld_heldout_writing_errors.json \
  --limit 160
```

After manual word-box review is complete, validate without skeleton mode:

```bash
.venv/bin/python scripts/validate_realworld_dataset.py \
  --dataset benchmark/test_dataset/realworld_heldout_writing_errors.json \
  --image-dir benchmark/test_dataset/realworld_heldout_raw \
  --expected-pages 40 --expected-clean 20 --expected-positive 20 --expected-errors 40
```

Only treat the held-out set as scoreable after that final validation passes.
