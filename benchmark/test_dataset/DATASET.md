# Test Dataset

## Source

Images in this directory are from the **IAM Handwriting Database** (https://fki.tic.heia-fr.ch/databases/iam-handwriting-database), a widely-used benchmark for handwritten text recognition.

## Contents

| Resource | Description |
|----------|-------------|
| `iam_page_*.jpg` | Full-page IAM forms with multiple lines of handwritten copy-text (~100 pages) |
| `curated/` | Diverse 15–20 page subset selected for benchmarking (symlinks to originals) |
| `curated_manifest.json` | Metadata for the curated subset (text density, line count, contrast) |
| `ground_truth_template.json` | Schema reference for ground truth annotations |
| `ground_truth.json` | Auto-generated ground truth (transcriptions, bounding boxes, reading order) |
| `iam_xml/` | IAM XML metadata (requires IAM account — see below) |

## IAM XML Metadata

The IAM dataset includes per-form XML annotations with:
- Writer ID
- Line-level transcriptions
- Bounding box coordinates
- Segmentation information

To obtain the XML metadata:

1. Register at https://fki.tic.heia-fr.ch/databases/iam-handwriting-database
2. Download `xml.tgz` (metadata only, ~8 MB)
3. Extract to `benchmark/test_dataset/iam_xml/`
4. Run `python scripts/generate_ground_truth.py` to produce `ground_truth.json`

Without XML metadata, `ground_truth.json` will contain only image references
with empty text/block fields — manual annotation required.

## Limitations

- **IAM pages are copy-text, not essays**: Writers copied printed text. Pages do not contain the types of writing errors (spelling, grammar, capitalization) that the full pipeline is designed to detect.
- **No error annotations**: The `errors` array in `ground_truth.json` is empty. Error detection evaluation (Phase 6) will require purpose-built samples with known writing errors.
- **Limited handwriting diversity**: While IAM spans ~500 writers, our subset captures only ~100 pages. Additional diversity may be needed for robust evaluation.
- **Form layout**: IAM forms have printed guide lines, unlike real essay paper. Models evaluated here may perform differently on unlined paper.

## Future Work

- Supplement with real student essay samples (anonymized, with consent)
- Add synthetic error-annotated pages for error detection metrics
- Include unlined paper samples for reading-order evaluation (Phase 5)

## License

IAM dataset is available for non-commercial research use. See:
https://fki.tic.heia-fr.ch/databases/iam-handwriting-database
