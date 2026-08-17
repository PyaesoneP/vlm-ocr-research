"""Build a label-free Phase 4 inference evidence graph from lattice candidates.

Unlike the minimal-pair diagnostic, this script does not use ground-truth
visible/corrected labels. It starts from the Stage 1 alternative lattice, crops
the canonical word boxes, scores canonical text against candidate alternatives,
and records whether the crop visually supports the canonical OCR word or an
alternative reading.

The output is intended for the next Stage 2/adjudication experiment: only
SUPPORTED_ALTERNATIVE records should be passed as possible evidence hints, and
Stage 2 still has to decide whether the supported text is actually a writing
error.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.score_phase4_qwen_visual_gain import FakeScorer, QwenVisualGainScorer


DEFAULT_LATTICE = Path("benchmark/results/phase4_realworld_stage1_qwen_alternative_lattice_20image.json")
DEFAULT_IMAGE_DIR = Path("benchmark/test_dataset/realworld_raw")
DEFAULT_CROP_DIR = Path("pipeline_output/phase4_inference_evidence_graph/crops")
DEFAULT_OUTPUT = Path("benchmark/results/phase4_inference_evidence_graph_qwen_visual_gain.json")

SUPPORTED_ALTERNATIVE = "SUPPORTED_ALTERNATIVE"
SUPPORTED_CANONICAL = "SUPPORTED_CANONICAL"
UNCERTAIN_REVIEW = "UNCERTAIN_REVIEW"


def compact(text: str) -> str:
    return "".join(ch.lower() for ch in str(text) if ch.isalnum())


def surface_key(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]+", "", str(text)).strip())


def plain(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def load_lattice_outputs(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    outputs = []
    for strategy in data.get("strategies", []):
        for image in strategy.get("images", []):
            output = image.get("output", {})
            if output:
                outputs.append(output)
    return outputs


def alternatives_for_output(output: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = output.get("metadata", {})
    return list(metadata.get("word_alternatives") or metadata.get("_word_alternatives") or [])


def crop_box(image_path: Path, bbox: list[int], out_path: Path, *, padding: int) -> str:
    with Image.open(image_path) as image:
        width, height = image.size
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid crop bbox for {image_path}: {bbox}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.crop((x1, y1, x2, y2)).save(out_path)
    return str(out_path)


def candidate_records(item: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(text: str, source: str, payload: dict[str, Any] | None = None) -> None:
        key = surface_key(text)
        compact_key = compact(text)
        if not key or key in seen:
            return
        seen.add(key)
        candidates.append({
            "text": text,
            "norm": compact_key,
            "surface_key": key,
            "source": source,
            "sources": payload.get("sources", [source]) if payload else [source],
            "supported_by_ocr": bool(payload.get("supported_by_ocr", False)) if payload else True,
            "edit_distance": payload.get("edit_distance") if payload else None,
            "ious": payload.get("ious", []) if payload else [],
            "is_canonical": source == "canonical",
        })

    add(str(item.get("canonical_text", "")), "canonical")
    for alternative in item.get("alternatives", []) or []:
        if isinstance(alternative, dict):
            add(str(alternative.get("observed_text", "")), "alternative", alternative)
    return candidates


def one_edit_variants(text: str, *, limit: int = 16) -> list[str]:
    """Generate bounded OCR-like variants without using a dictionary or labels."""

    value = plain(text)
    if len(value) < 3 or len(value) > 14:
        return []
    variants: list[str] = []
    seen = {value}

    def add(candidate: str) -> None:
        candidate = plain(candidate)
        if len(candidate) >= 3 and candidate not in seen:
            seen.add(candidate)
            variants.append(candidate)

    # Vowel substitutions catch near-miss endings such as umbrele -> umbrela.
    vowels = "aeiou"
    if value[-1] in vowels:
        for vowel in vowels:
            if vowel != value[-1]:
                add(value[:-1] + vowel)

    # Deletions catch extra-letter normalizations: forgotten -> forgoten,
    # interesting -> intresting, minutes -> minuts. Repeated-letter deletions are
    # most plausible, so add them before all other deletions.
    for i in range(len(value) - 1):
        if value[i] == value[i + 1]:
            add(value[:i] + value[i + 1:])
    for i in range(len(value)):
        add(value[:i] + value[i + 1:])

    for i, char in enumerate(value):
        if char not in vowels:
            continue
        for vowel in vowels:
            if vowel != char:
                add(value[:i] + vowel + value[i + 1:])

    # Adjacent transpositions catch common handwriting/OCR order ambiguity.
    for i in range(len(value) - 1):
        if value[i] != value[i + 1]:
            add(value[:i] + value[i + 1] + value[i] + value[i + 2:])

    return variants[:limit]


def add_candidate(
    candidates: list[dict[str, Any]],
    seen: set[str],
    *,
    text: str,
    source: str,
    supported_by_ocr: bool,
    payload: dict[str, Any] | None = None,
) -> None:
    key = surface_key(text)
    compact_key = compact(text)
    if not key or key in seen:
        return
    seen.add(key)
    candidates.append({
        "text": text,
        "norm": compact_key,
        "surface_key": key,
        "source": source,
        "sources": payload.get("sources", [source]) if payload else [source],
        "supported_by_ocr": supported_by_ocr,
        "edit_distance": payload.get("edit_distance") if payload else None,
        "ious": payload.get("ious", []) if payload else [],
        "is_canonical": source == "canonical",
    })


def expanded_candidate_records(item: dict[str, Any], *, lexical_limit: int) -> list[dict[str, Any]]:
    candidates = candidate_records(item)
    seen = {str(candidate.get("surface_key", "")) for candidate in candidates}
    for variant in one_edit_variants(str(item.get("canonical_text", "")), limit=lexical_limit):
        add_candidate(
            candidates,
            seen,
            text=variant,
            source="generic_one_edit_variant",
            supported_by_ocr=False,
            payload={"sources": ["generic_one_edit_variant"]},
        )
    return candidates


def bbox_union(boxes: list[dict[str, Any]], indices: list[int]) -> list[int]:
    selected = [boxes[index].get("bbox", [0, 0, 0, 0]) for index in indices if 0 <= index < len(boxes)]
    if not selected:
        return [0, 0, 0, 0]
    return [
        min(int(box[0]) for box in selected),
        min(int(box[1]) for box in selected),
        max(int(box[2]) for box in selected),
        max(int(box[3]) for box in selected),
    ]


def box_text(boxes: list[dict[str, Any]], indices: list[int]) -> str:
    return " ".join(str(boxes[index].get("text", "")) for index in indices if 0 <= index < len(boxes))


def alternatives_by_index(items: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    for item in items:
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError, AttributeError):
            continue
        out[index] = [alt for alt in item.get("alternatives", []) or [] if isinstance(alt, dict)]
    return out


def phrase_candidate_records(
    *,
    canonical_text: str,
    alternatives: list[str],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    add_candidate(candidates, seen, text=canonical_text, source="canonical", supported_by_ocr=True)
    for text in alternatives:
        add_candidate(
            candidates,
            seen,
            text=text,
            source="phrase_alternative",
            supported_by_ocr=True,
            payload={"sources": ["phrase_alternative"]},
        )
    return candidates


def phrase_records(output: dict[str, Any], alt_by_index: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Add label-free multiword records for grammar-like canonical spans."""

    boxes = output.get("boxes", [])
    image = str(output.get("image", ""))
    records: list[dict[str, Any]] = []

    def add(indices: list[int], reason: str, alternatives: list[str] | None = None) -> None:
        canonical = box_text(boxes, indices)
        records.append({
            "pair_id": f"{Path(image).stem}_w{'_'.join(f'{i:03d}' for i in indices)}",
            "image": image,
            "word_index": indices[0],
            "word_indices": indices,
            "bbox": bbox_union(boxes, indices),
            "canonical_text": canonical,
            "suspicion_reasons": [reason],
            "candidates": phrase_candidate_records(
                canonical_text=canonical,
                alternatives=alternatives or [],
            ),
        })

    words = [plain(box.get("text", "")) for box in boxes]
    for i in range(len(words) - 1):
        if words[i] in {"they", "he", "she", "it"} and words[i + 1] in {"lives", "come", "know"}:
            add([i, i + 1], "subject_verb_candidate")
        if words[i] == "should" and words[i + 1] == "of":
            add([i, i + 1], "modal_of_candidate")
        if words[i] and words[i] == words[i + 1]:
            add([i, i + 1], "repeated_word_candidate")

        # Combine a supported single-word alternative with its neighbor to catch
        # repeated-word errors such as canonical "tho the" with alternative "the".
        alt_texts = [plain(alt.get("observed_text", "")) for alt in alt_by_index.get(i, [])]
        if words[i + 1] and words[i + 1] in alt_texts:
            add([i, i + 1], "alternative_repeated_word_candidate", [f"{boxes[i + 1].get('text', '')} {boxes[i + 1].get('text', '')}"])

    for i in range(len(words) - 2):
        if words[i:i + 3] == ["took", "us", "hour"]:
            add([i, i + 1, i + 2], "missing_article_candidate")

    return records


def build_records(
    outputs: list[dict[str, Any]],
    *,
    image_dir: Path,
    crop_dir: Path,
    padding: int,
    max_records: int,
    image_filter: set[str],
    lexical_limit: int,
) -> list[dict[str, Any]]:
    records = []
    for output in outputs:
        image = str(output.get("image", ""))
        if image_filter and image not in image_filter:
            continue
        alternative_items = alternatives_for_output(output)
        alt_by_index = alternatives_by_index(alternative_items)
        for item in alternative_items:
            try:
                index = int(item.get("index"))
            except (TypeError, ValueError, AttributeError):
                continue
            bbox = [int(v) for v in item.get("bbox", [0, 0, 0, 0])[:4]]
            pair_id = f"{Path(image).stem}_w{index:03d}"
            crop_path = crop_box(
                image_dir / image,
                bbox,
                crop_dir / Path(image).stem / f"w{index:03d}.png",
                padding=padding,
            )
            candidates = expanded_candidate_records(item, lexical_limit=lexical_limit)
            if len(candidates) < 2:
                continue
            records.append({
                "pair_id": pair_id,
                "image": image,
                "word_index": index,
                "word_indices": [index],
                "bbox": bbox,
                "crop_path": crop_path,
                "canonical_text": item.get("canonical_text", ""),
                "suspicion_reasons": item.get("reasons", []),
                "candidates": candidates,
            })
            if max_records > 0 and len(records) >= max_records:
                return records
        for record in phrase_records(output, alt_by_index):
            crop_path = crop_box(
                image_dir / image,
                record["bbox"],
                crop_dir / Path(image).stem / f"{record['pair_id']}_phrase.png",
                padding=padding,
            )
            record["crop_path"] = crop_path
            records.append(record)
            if max_records > 0 and len(records) >= max_records:
                return records
    return records


def score_record(record: dict[str, Any], scorer: Any) -> dict[str, Any]:
    crop_path = Path(record["crop_path"])
    scored = []
    for candidate in record.get("candidates", []):
        scores = scorer.score_candidate(crop_path, str(candidate.get("text", "")))
        scored.append({**candidate, **scores})
    scored.sort(key=lambda row: row["visual_gain"], reverse=True)
    top = scored[0] if scored else None
    runner_up = scored[1] if len(scored) > 1 else None
    canonical = next((row for row in scored if row.get("is_canonical")), None)
    best_alternative = next((row for row in scored if not row.get("is_canonical")), None)
    margin = float(top["visual_gain"] - runner_up["visual_gain"]) if top and runner_up else None
    alt_margin_over_canonical = (
        float(best_alternative["visual_gain"] - canonical["visual_gain"])
        if best_alternative and canonical else None
    )
    return {
        **record,
        "top_text": top.get("text", "") if top else "",
        "top_is_canonical": bool(top and top.get("is_canonical")),
        "top_supported_by_ocr": bool(top and top.get("supported_by_ocr")),
        "margin": margin,
        "alternative_margin_over_canonical": alt_margin_over_canonical,
        "hypotheses": scored,
    }


def decide(record: dict[str, Any], *, threshold: float, require_ocr_support: bool) -> str:
    margin = record.get("alternative_margin_over_canonical")
    top_is_canonical = bool(record.get("top_is_canonical"))
    if top_is_canonical:
        return SUPPORTED_CANONICAL
    if margin is None or float(margin) < threshold:
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
        "ocr_supported_alternative": sum(
            1 for row in records
            if row.get("decision") == SUPPORTED_ALTERNATIVE and row.get("top_supported_by_ocr")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lattice-result", type=Path, default=DEFAULT_LATTICE)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--crop-dir", type=Path, default=DEFAULT_CROP_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--padding", type=int, default=16)
    parser.add_argument("--lexical-limit", type=int, default=4)
    parser.add_argument("--max-records", type=int, default=0, help="Limit records; 0 means all.")
    parser.add_argument("--image", dest="images", action="append", default=[])
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--allow-unsupported-alternatives", action="store_true")
    parser.add_argument("--score", action="store_true", help="Run Qwen visual-gain scoring.")
    parser.add_argument("--smoke-fake", action="store_true", help="Use deterministic fake scores.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--normalize-by", choices=["char", "token"], default="char")
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    records = build_records(
        load_lattice_outputs(args.lattice_result),
        image_dir=args.image_dir,
        crop_dir=args.crop_dir,
        padding=args.padding,
        max_records=args.max_records,
        image_filter=set(args.images),
        lexical_limit=args.lexical_limit,
    )
    if args.score:
        scorer = FakeScorer() if args.smoke_fake else QwenVisualGainScorer(
            model_id=args.model_id,
            alpha=args.alpha,
            normalize_by=args.normalize_by,
        )
        scored = []
        for offset, record in enumerate(records, start=1):
            scored_record = score_record(record, scorer)
            scored_record["decision"] = decide(
                scored_record,
                threshold=args.threshold,
                require_ocr_support=not args.allow_unsupported_alternatives,
            )
            scored.append(scored_record)
            if args.progress_every > 0 and offset % args.progress_every == 0:
                print(f"[inference-evidence] scored {offset}/{len(records)} records", flush=True)
        records = scored
    else:
        for record in records:
            record["decision"] = UNCERTAIN_REVIEW

    result = {
        "lattice_result": str(args.lattice_result),
        "image_dir": str(args.image_dir),
        "crop_dir": str(args.crop_dir),
        "label_free": True,
        "scored": bool(args.score),
        "threshold": args.threshold,
        "require_ocr_support": not args.allow_unsupported_alternatives,
        "summary": summarize(records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
