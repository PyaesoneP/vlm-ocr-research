"""Build a minimal-pair crop dataset for Phase 4 candidate scoring.

The output is a small development/evaluation artifact for the forensic OCR
path. It pairs the visible annotated form with its normalized form for positive
errors, and creates matched clean controls where the clean word appears on a
clean page.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_DATASET = Path("benchmark/test_dataset/realworld_writing_errors.json")
DEFAULT_LATTICE = Path("benchmark/results/phase4_realworld_stage1_qwen_alternative_lattice_20image.json")
DEFAULT_IMAGE_DIR = Path("benchmark/test_dataset/realworld_raw")
DEFAULT_OUTPUT = Path("benchmark/results/phase4_minimal_pair_dataset.json")
DEFAULT_CROP_DIR = Path("pipeline_output/phase4_minimal_pairs/crops")


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def load_dataset(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("samples", "pages", "items"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"Unsupported dataset schema: {path}")


def load_outputs(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text())
    outputs: dict[str, dict[str, Any]] = {}
    for strategy in data.get("strategies", []):
        for image in strategy.get("images", []):
            output = image.get("output", {})
            image_name = output.get("image") or image.get("image")
            if image_name:
                outputs[str(image_name)] = output
    return outputs


def crop_box(
    image_path: Path,
    bbox: list[int],
    out_path: Path,
    *,
    padding: int,
) -> str:
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


def text_for_indices(words: list[dict[str, Any]], indices: list[int]) -> str:
    parts = []
    for index in indices:
        if 0 <= index < len(words):
            parts.append(str(words[index].get("text", "")))
    return " ".join(parts)


def bbox_for_indices(words: list[dict[str, Any]], indices: list[int]) -> list[int]:
    boxes = [words[index].get("bbox", [0, 0, 0, 0]) for index in indices if 0 <= index < len(words)]
    if not boxes:
        return [0, 0, 0, 0]
    return [
        min(int(box[0]) for box in boxes),
        min(int(box[1]) for box in boxes),
        max(int(box[2]) for box in boxes),
        max(int(box[3]) for box in boxes),
    ]


def alternatives_by_index(output: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    metadata = output.get("metadata", {})
    raw_items = metadata.get("word_alternatives", metadata.get("_word_alternatives", []))
    items: dict[int, list[dict[str, Any]]] = {}
    for item in raw_items or []:
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError, AttributeError):
            continue
        alternatives = item.get("alternatives", [])
        if isinstance(alternatives, list):
            items[index] = [alt for alt in alternatives if isinstance(alt, dict)]
    return items


def candidate_records(
    *,
    visible: str,
    normalized: str,
    canonical: str,
    alternatives: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(text: str, source: str, rank: int | None = None, payload: dict[str, Any] | None = None) -> None:
        compact = norm(text)
        if not compact or compact in seen:
            return
        seen.add(compact)
        candidates.append({
            "text": text,
            "norm": compact,
            "source": source,
            "rank": rank,
            "is_visible_label": compact == norm(visible),
            "is_normalized_label": compact == norm(normalized),
            "supported_by_ocr": bool(payload.get("supported_by_ocr")) if payload else source == "canonical",
            "sources": payload.get("sources", []) if payload else [source],
            "edit_distance": payload.get("edit_distance") if payload else None,
            "ious": payload.get("ious", []) if payload else [],
        })

    add(canonical, "canonical")
    add(visible, "label_visible")
    add(normalized, "label_normalized")
    for rank, alternative in enumerate(alternatives, start=1):
        add(str(alternative.get("observed_text", "")), "lattice", rank=rank, payload=alternative)
    return candidates


def positive_pairs(
    dataset: list[dict[str, Any]],
    outputs: dict[str, dict[str, Any]],
    *,
    image_dir: Path,
    crop_dir: Path,
    padding: int,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for sample in dataset:
        image = str(sample.get("image", ""))
        output = outputs.get(image, {})
        output_boxes = output.get("boxes", [])
        output_alternatives = alternatives_by_index(output)
        for error_idx, error in enumerate(sample.get("errors", []) or []):
            indices = [int(index) for index in error.get("word_indices", []) if isinstance(index, int)]
            visible = str(error.get("evidence_text", ""))
            normalized = str(error.get("correction", ""))
            bbox = bbox_for_indices(sample.get("words", []), indices)
            canonical = text_for_indices(output_boxes, indices)
            alternatives = output_alternatives.get(indices[0], []) if len(indices) == 1 else []
            crop_path = crop_box(
                image_dir / image,
                bbox,
                crop_dir / "positive" / f"{Path(image).stem}_e{error_idx:02d}.png",
                padding=padding,
            )
            candidates = candidate_records(
                visible=visible,
                normalized=normalized,
                canonical=canonical,
                alternatives=alternatives,
            )
            pairs.append({
                "pair_id": f"{Path(image).stem}_e{error_idx:02d}",
                "split": "positive",
                "image": image,
                "word_indices": indices,
                "bbox": bbox,
                "crop_path": crop_path,
                "visible_form": visible,
                "normalized_form": normalized,
                "canonical_text": canonical,
                "error_type": error.get("type", ""),
                "candidate_visible_present": any(c["is_visible_label"] for c in candidates),
                "candidate_visible_ocr_supported": any(
                    c["is_visible_label"] and c["supported_by_ocr"] for c in candidates
                ),
                "candidates": candidates,
            })
    return pairs


def clean_control_pairs(
    dataset: list[dict[str, Any]],
    outputs: dict[str, dict[str, Any]],
    positive: list[dict[str, Any]],
    *,
    image_dir: Path,
    crop_dir: Path,
    padding: int,
) -> list[dict[str, Any]]:
    target_pairs = {(norm(pair["normalized_form"]), pair["visible_form"]) for pair in positive}
    controls: list[dict[str, Any]] = []
    for sample in dataset:
        if sample.get("split") != "clean":
            continue
        image = str(sample.get("image", ""))
        output = outputs.get(image, {})
        output_boxes = output.get("boxes", [])
        output_alternatives = alternatives_by_index(output)
        for box in output_boxes:
            try:
                index = int(box.get("index"))
            except (TypeError, ValueError, AttributeError):
                continue
            clean_text = str(box.get("text", ""))
            for clean_norm, plausible_error in target_pairs:
                if norm(clean_text) != clean_norm:
                    continue
                bbox = list(box.get("bbox", [0, 0, 0, 0]))
                crop_path = crop_box(
                    image_dir / image,
                    bbox,
                    crop_dir / "clean" / f"{Path(image).stem}_w{index:03d}_{norm(plausible_error)}.png",
                    padding=padding,
                )
                alternatives = output_alternatives.get(index, [])
                candidates = candidate_records(
                    visible=clean_text,
                    normalized=plausible_error,
                    canonical=clean_text,
                    alternatives=alternatives,
                )
                controls.append({
                    "pair_id": f"{Path(image).stem}_w{index:03d}_{norm(plausible_error)}",
                    "split": "clean_control",
                    "image": image,
                    "word_indices": [index],
                    "bbox": bbox,
                    "crop_path": crop_path,
                    "visible_form": clean_text,
                    "normalized_form": plausible_error,
                    "canonical_text": clean_text,
                    "error_type": "clean_control",
                    "candidate_visible_present": True,
                    "candidate_visible_ocr_supported": True,
                    "candidates": candidates,
                })
    return controls


def summarize(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    positive = [pair for pair in pairs if pair["split"] == "positive"]
    clean = [pair for pair in pairs if pair["split"] == "clean_control"]
    return {
        "pairs_total": len(pairs),
        "positive_pairs": len(positive),
        "clean_control_pairs": len(clean),
        "positive_visible_candidate_recall": (
            sum(1 for pair in positive if pair["candidate_visible_present"]) / len(positive)
            if positive else 0.0
        ),
        "positive_visible_ocr_supported_recall": (
            sum(1 for pair in positive if pair["candidate_visible_ocr_supported"]) / len(positive)
            if positive else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--lattice-result", type=Path, default=DEFAULT_LATTICE)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--crop-dir", type=Path, default=DEFAULT_CROP_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--padding", type=int, default=16)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    outputs = load_outputs(args.lattice_result)
    positive = positive_pairs(
        dataset,
        outputs,
        image_dir=args.image_dir,
        crop_dir=args.crop_dir,
        padding=args.padding,
    )
    clean = clean_control_pairs(
        dataset,
        outputs,
        positive,
        image_dir=args.image_dir,
        crop_dir=args.crop_dir,
        padding=args.padding,
    )
    pairs = positive + clean
    result = {
        "dataset": str(args.dataset),
        "lattice_result": str(args.lattice_result),
        "image_dir": str(args.image_dir),
        "crop_dir": str(args.crop_dir),
        "summary": summarize(pairs),
        "pairs": pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
