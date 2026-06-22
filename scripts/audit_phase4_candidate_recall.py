"""Audit candidate recall for Phase 4 evidence-lattice OCR outputs.

This is an offline decomposition tool. It does not call any model. It checks
whether each annotated erroneous span appears in the canonical OCR text or in
the attached alternative lattice, and whether that candidate is supported by an
OCR source rather than only by a generic lexical neighbor.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_DATASET = Path("benchmark/test_dataset/realworld_writing_errors.json")
DEFAULT_LATTICE = Path("benchmark/results/phase4_realworld_stage1_qwen_alternative_lattice_20image.json")
DEFAULT_OUTPUT = Path("benchmark/results/phase4_candidate_recall_audit.json")


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def spaced_norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


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


def text_for_indices(words: list[dict[str, Any]], indices: list[int]) -> str:
    parts = []
    for index in indices:
        if 0 <= index < len(words):
            parts.append(str(words[index].get("text", "")))
    return " ".join(parts)


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


def candidate_texts_for_indices(
    alternatives: dict[int, list[dict[str, Any]]],
    indices: list[int],
) -> list[dict[str, Any]]:
    if len(indices) != 1:
        return []
    index = indices[0]
    candidates = []
    for rank, alternative in enumerate(alternatives.get(index, []), start=1):
        text = str(alternative.get("observed_text", ""))
        sources = [str(source) for source in alternative.get("sources", [])]
        candidates.append({
            "rank": rank,
            "text": text,
            "norm": norm(text),
            "supported_by_ocr": bool(alternative.get("supported_by_ocr")),
            "sources": sources,
            "edit_distance": alternative.get("edit_distance"),
            "ious": alternative.get("ious", []),
        })
    return candidates


def audit_positive_errors(
    dataset: list[dict[str, Any]],
    outputs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in dataset:
        image = str(sample.get("image", ""))
        output = outputs.get(image, {})
        boxes = output.get("boxes", [])
        alternatives = alternatives_by_index(output)
        for error in sample.get("errors", []) or []:
            indices = [int(index) for index in error.get("word_indices", []) if isinstance(index, int)]
            evidence = str(error.get("evidence_text", ""))
            correction = str(error.get("correction", ""))
            canonical_text = text_for_indices(boxes, indices)
            candidates = candidate_texts_for_indices(alternatives, indices)
            evidence_norm = norm(evidence)
            candidate_hits = [c for c in candidates if c["norm"] == evidence_norm]
            ocr_hits = [c for c in candidate_hits if c["supported_by_ocr"]]
            rows.append({
                "image": image,
                "type": error.get("type", ""),
                "word_indices": indices,
                "evidence_text": evidence,
                "correction": correction,
                "canonical_text": canonical_text,
                "canonical_matches_evidence": norm(canonical_text) == evidence_norm,
                "canonical_matches_correction": norm(canonical_text) == norm(correction),
                "candidate_recall": bool(candidate_hits),
                "candidate_recall_ocr_supported": bool(ocr_hits),
                "candidate_rank": candidate_hits[0]["rank"] if candidate_hits else None,
                "candidate_sources": candidate_hits[0]["sources"] if candidate_hits else [],
                "candidate_count": len(candidates),
                "candidates": candidates,
            })
    return rows


def audit_clean_controls(
    dataset: list[dict[str, Any]],
    outputs: dict[str, dict[str, Any]],
    positive_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pairs = {(norm(row["correction"]), norm(row["evidence_text"])) for row in positive_rows}
    rows: list[dict[str, Any]] = []
    for sample in dataset:
        if sample.get("split") != "clean":
            continue
        image = str(sample.get("image", ""))
        output = outputs.get(image, {})
        alternatives = alternatives_by_index(output)
        for box in output.get("boxes", []):
            try:
                index = int(box.get("index"))
            except (TypeError, ValueError, AttributeError):
                continue
            canonical = str(box.get("text", ""))
            canonical_norm = norm(canonical)
            for clean_norm, erroneous_norm in pairs:
                if canonical_norm != clean_norm:
                    continue
                candidates = candidate_texts_for_indices(alternatives, [index])
                erroneous_hits = [c for c in candidates if c["norm"] == erroneous_norm]
                rows.append({
                    "image": image,
                    "word_index": index,
                    "clean_text": canonical,
                    "erroneous_control_norm": erroneous_norm,
                    "erroneous_candidate_present": bool(erroneous_hits),
                    "erroneous_candidate_ocr_supported": any(c["supported_by_ocr"] for c in erroneous_hits),
                    "candidate_rank": erroneous_hits[0]["rank"] if erroneous_hits else None,
                    "candidate_sources": erroneous_hits[0]["sources"] if erroneous_hits else [],
                    "candidate_count": len(candidates),
                })
    return rows


def summarize(positive_rows: list[dict[str, Any]], clean_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(positive_rows)
    canonical_hits = sum(1 for row in positive_rows if row["canonical_matches_evidence"])
    candidate_hits = sum(1 for row in positive_rows if row["candidate_recall"])
    ocr_hits = sum(1 for row in positive_rows if row["candidate_recall_ocr_supported"])
    evidence_available = sum(
        1 for row in positive_rows if row["canonical_matches_evidence"] or row["candidate_recall"]
    )
    missed_by_canonical = [row for row in positive_rows if not row["canonical_matches_evidence"]]
    missed_total = len(missed_by_canonical)
    missed_candidate_hits = sum(1 for row in missed_by_canonical if row["candidate_recall"])
    missed_ocr_hits = sum(1 for row in missed_by_canonical if row["candidate_recall_ocr_supported"])
    normalized_leaks = sum(1 for row in positive_rows if row["canonical_matches_correction"])
    clean_total = len(clean_rows)
    clean_bad = sum(1 for row in clean_rows if row["erroneous_candidate_present"])
    clean_bad_ocr = sum(1 for row in clean_rows if row["erroneous_candidate_ocr_supported"])
    return {
        "positive_errors": total,
        "canonical_evidence_hits": canonical_hits,
        "canonical_evidence_recall": canonical_hits / total if total else 0.0,
        "candidate_recall_at_k": candidate_hits / total if total else 0.0,
        "candidate_recall_ocr_supported_at_k": ocr_hits / total if total else 0.0,
        "canonical_or_candidate_evidence_available": evidence_available,
        "canonical_or_candidate_evidence_recall": evidence_available / total if total else 0.0,
        "canonical_miss_count": missed_total,
        "candidate_recall_on_canonical_misses": (
            missed_candidate_hits / missed_total if missed_total else 0.0
        ),
        "candidate_recall_ocr_supported_on_canonical_misses": (
            missed_ocr_hits / missed_total if missed_total else 0.0
        ),
        "normalized_canonical_leaks": normalized_leaks,
        "clean_controls": clean_total,
        "clean_erroneous_candidate_rate": clean_bad / clean_total if clean_total else 0.0,
        "clean_erroneous_ocr_supported_rate": clean_bad_ocr / clean_total if clean_total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--lattice-result", type=Path, default=DEFAULT_LATTICE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    outputs = load_outputs(args.lattice_result)
    positive_rows = audit_positive_errors(dataset, outputs)
    clean_rows = audit_clean_controls(dataset, outputs, positive_rows)
    result = {
        "dataset": str(args.dataset),
        "lattice_result": str(args.lattice_result),
        "summary": summarize(positive_rows, clean_rows),
        "positive_errors": positive_rows,
        "clean_controls": clean_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    summary = result["summary"]
    print(
        "Candidate recall: "
        f"{summary['candidate_recall_at_k']:.3f} "
        f"({summary['candidate_recall_ocr_supported_at_k']:.3f} OCR-supported), "
        f"canonical evidence recall {summary['canonical_evidence_recall']:.3f}, "
        f"canonical-or-candidate recall {summary['canonical_or_candidate_evidence_recall']:.3f}"
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
