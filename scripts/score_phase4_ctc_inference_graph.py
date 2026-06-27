"""Attach CTC candidate scores to a label-free Phase 4 inference graph."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.optical_candidate_scorer import EasyOCRCTCCandidateScorer


DEFAULT_GRAPH = Path("benchmark/results/phase4_inference_evidence_graph_unscored_20image.json")
DEFAULT_OUTPUT = Path("benchmark/results/phase4_inference_evidence_graph_ctc_rw1_rw2_rw11.json")
DEFAULT_DATASET = Path("benchmark/test_dataset/realworld_writing_errors.json")
DEFAULT_IMAGE_DIR = Path("benchmark/test_dataset/realworld_raw")
DEFAULT_LINE_CROP_DIR = Path("pipeline_output/phase4_inference_evidence_graph/line_crops")

SUPPORTED_ALTERNATIVE = "SUPPORTED_ALTERNATIVE"
SUPPORTED_CANONICAL = "SUPPORTED_CANONICAL"
UNCERTAIN_REVIEW = "UNCERTAIN_REVIEW"


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def load_dataset(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        for key in ("samples", "pages", "items"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"Unsupported dataset schema: {path}")
    return {str(sample.get("image", "")): sample for sample in data}


def crop_box(image_path: Path, bbox: list[int], out_path: Path, *, padding: int) -> str:
    from PIL import Image

    with Image.open(image_path) as image:
        width, height = image.size
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid line crop bbox for {image_path}: {bbox}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.crop((x1, y1, x2, y2)).save(out_path)
    return str(out_path)


def union_bbox(words: list[dict[str, Any]]) -> list[int]:
    boxes = [word.get("bbox", [0, 0, 0, 0]) for word in words]
    return [
        min(int(box[0]) for box in boxes),
        min(int(box[1]) for box in boxes),
        max(int(box[2]) for box in boxes),
        max(int(box[3]) for box in boxes),
    ]


def line_context(
    record: dict[str, Any],
    *,
    dataset_by_image: dict[str, dict[str, Any]],
    image_dir: Path,
    line_crop_dir: Path,
    padding: int,
) -> dict[str, Any]:
    sample = dataset_by_image.get(str(record.get("image", "")))
    if not sample:
        return {}
    words = [word for word in sample.get("words", []) if len(word.get("bbox", [])) == 4]
    if not words:
        return {}
    target_box = [int(v) for v in record.get("bbox", [0, 0, 0, 0])[:4]]
    target_center = (target_box[1] + target_box[3]) / 2.0
    target_height = max(1, target_box[3] - target_box[1])
    threshold = max(80.0, target_height * 0.9)
    line_words = [
        word for word in words
        if abs(((int(word["bbox"][1]) + int(word["bbox"][3])) / 2.0) - target_center) <= threshold
    ]
    line_words.sort(key=lambda word: (int(word["bbox"][0]), int(word.get("index", 0))))
    if not line_words:
        return {}

    line_indices = [int(word.get("index", -1)) for word in line_words]
    target_indices = [int(index) for index in record.get("word_indices", [])]
    target_set = set(target_indices)
    try:
        first_pos = next(i for i, word in enumerate(line_words) if int(word.get("index", -1)) in target_set)
    except StopIteration:
        return {}

    line_bbox = union_bbox(line_words)
    line_crop_path = crop_box(
        image_dir / str(record["image"]),
        line_bbox,
        line_crop_dir / Path(str(record["image"])).stem / f"{record['pair_id']}_line.png",
        padding=padding,
    )
    canonical_parts = [str(word.get("text", "")) for word in line_words]
    target_len = max(1, len(target_indices))
    return {
        "line_crop_path": line_crop_path,
        "line_bbox": line_bbox,
        "line_word_indices": line_indices,
        "line_text": " ".join(canonical_parts),
        "target_line_offset": first_pos,
        "target_line_length": target_len,
        "canonical_parts": canonical_parts,
    }


def line_candidate_text(context: dict[str, Any], candidate_text: str) -> str:
    parts = list(context["canonical_parts"])
    offset = int(context["target_line_offset"])
    length = int(context["target_line_length"])
    parts[offset:offset + length] = [str(candidate_text)]
    return " ".join(parts)


def score_record(
    record: dict[str, Any],
    scorer: EasyOCRCTCCandidateScorer,
    *,
    context: str,
    dataset_by_image: dict[str, dict[str, Any]],
    image_dir: Path,
    line_crop_dir: Path,
    line_padding: int,
) -> dict[str, Any]:
    line_info = {}
    if context == "line":
        line_info = line_context(
            record,
            dataset_by_image=dataset_by_image,
            image_dir=image_dir,
            line_crop_dir=line_crop_dir,
            padding=line_padding,
        )
    crop_path = Path(line_info.get("line_crop_path") or record["crop_path"])
    scored = []
    for candidate in record.get("candidates", []):
        candidate_text = str(candidate.get("text", ""))
        scorer_input = line_candidate_text(line_info, candidate_text) if line_info else candidate_text
        score = scorer.score_candidate(crop_path, scorer_input)
        row = {
            **candidate,
            "ctc_context": "line" if line_info else "word",
            "ctc_crop_path": str(crop_path),
            "ctc_input_text": scorer_input,
            "ctc_raw_logprob": score.raw_logprob,
            "ctc_normalized_logprob": score.normalized_logprob,
            "ctc_char_count": score.char_count,
            "ctc_target_length": score.target_length,
            "ctc_timesteps": score.timesteps,
            "ctc_scorer_text": score.scorer_text,
            "ctc_unsupported_chars": score.unsupported_chars,
            "ctc_supported_chars": score.supported,
            "ctc_latency_ms": score.latency_ms,
            # Compatibility with the existing selector/audit vocabulary.
            "visual_gain": score.normalized_logprob,
        }
        scored.append(row)

    scored.sort(key=lambda row: row["ctc_normalized_logprob"], reverse=True)
    for rank, row in enumerate(scored, start=1):
        row["rank"] = rank

    top = scored[0] if scored else None
    runner_up = scored[1] if len(scored) > 1 else None
    canonical = next((row for row in scored if row.get("is_canonical")), None)
    best_alternative = next((row for row in scored if not row.get("is_canonical")), None)
    margin = (
        float(top["ctc_normalized_logprob"] - runner_up["ctc_normalized_logprob"])
        if top
        and runner_up
        and finite(top.get("ctc_normalized_logprob"))
        and finite(runner_up.get("ctc_normalized_logprob"))
        else None
    )
    alt_margin_over_canonical = (
        float(best_alternative["ctc_normalized_logprob"] - canonical["ctc_normalized_logprob"])
        if best_alternative
        and canonical
        and finite(best_alternative.get("ctc_normalized_logprob"))
        and finite(canonical.get("ctc_normalized_logprob"))
        else None
    )
    return {
        **record,
        **{
            key: value for key, value in line_info.items()
            if key != "canonical_parts"
        },
        "top_text": top.get("text", "") if top else "",
        "top_is_canonical": bool(top and top.get("is_canonical")),
        "top_supported_by_ocr": bool(top and top.get("supported_by_ocr")),
        "top_ctc_supported_chars": bool(top and top.get("ctc_supported_chars")),
        "top_ctc_unsupported_chars": top.get("ctc_unsupported_chars", []) if top else [],
        "margin": margin,
        "alternative_margin_over_canonical": alt_margin_over_canonical,
        "hypotheses": scored,
    }


def decide(record: dict[str, Any], *, threshold: float, require_ocr_support: bool) -> str:
    top_is_canonical = bool(record.get("top_is_canonical"))
    if top_is_canonical:
        return SUPPORTED_CANONICAL
    margin = record.get("alternative_margin_over_canonical")
    if margin is None or float(margin) < threshold:
        return UNCERTAIN_REVIEW
    if not record.get("top_ctc_supported_chars"):
        return UNCERTAIN_REVIEW
    if require_ocr_support and not record.get("top_supported_by_ocr"):
        return UNCERTAIN_REVIEW
    return SUPPORTED_ALTERNATIVE


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records_total": len(records),
        "supported_alternative": sum(1 for row in records if row.get("decision") == SUPPORTED_ALTERNATIVE),
        "supported_canonical": sum(1 for row in records if row.get("decision") == SUPPORTED_CANONICAL),
        "uncertain_review": sum(1 for row in records if row.get("decision") == UNCERTAIN_REVIEW),
        "top_alternative": sum(1 for row in records if not row.get("top_is_canonical")),
        "top_alternative_ocr_supported": sum(
            1 for row in records
            if not row.get("top_is_canonical") and row.get("top_supported_by_ocr")
        ),
        "top_alternative_ctc_supported_chars": sum(
            1 for row in records
            if not row.get("top_is_canonical") and row.get("top_ctc_supported_chars")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--line-crop-dir", type=Path, default=DEFAULT_LINE_CROP_DIR)
    parser.add_argument("--image", action="append", dest="images", default=[])
    parser.add_argument("--context", choices=["word", "line"], default="word")
    parser.add_argument("--line-padding", type=int, default=24)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--allow-unsupported-alternatives", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--img-h", type=int, default=64)
    parser.add_argument("--img-w", type=int, default=256)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    data = json.loads(args.graph.read_text())
    image_filter = set(args.images)
    records = data.get("records", [])
    if image_filter:
        records = [record for record in records if record.get("image") in image_filter]

    scorer = EasyOCRCTCCandidateScorer(device=args.device, img_h=args.img_h, img_w=args.img_w)
    dataset_by_image = load_dataset(args.dataset) if args.context == "line" else {}
    scored = []
    for offset, record in enumerate(records, start=1):
        scored_record = score_record(
            record,
            scorer,
            context=args.context,
            dataset_by_image=dataset_by_image,
            image_dir=args.image_dir,
            line_crop_dir=args.line_crop_dir,
            line_padding=args.line_padding,
        )
        scored_record["decision"] = decide(
            scored_record,
            threshold=args.threshold,
            require_ocr_support=not args.allow_unsupported_alternatives,
        )
        scored.append(scored_record)
        if args.progress_every > 0 and offset % args.progress_every == 0:
            print(f"[ctc-graph] scored {offset}/{len(records)} records", flush=True)

    result = {
        **{key: value for key, value in data.items() if key != "records"},
        "source_graph": str(args.graph),
        "label_free": True,
        "scored": True,
        "score_backend": "easyocr_ctc",
        "ctc_context": args.context,
        "scorer": scorer.metadata(),
        "threshold": args.threshold,
        "require_ocr_support": not args.allow_unsupported_alternatives,
        "images": sorted(image_filter),
        "summary": summarize(scored),
        "records": scored,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
