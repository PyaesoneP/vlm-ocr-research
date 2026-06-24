#!/usr/bin/env python3
"""Audit general Stage 1 uncertainty signals for real-world truthfulness misses.

This report is intentionally diagnostic. It asks whether current OCR misses
would have been flagged by dataset-independent uncertainty signals, without
adding word-specific fixes for the current 20-page development set.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.localization import union_bboxes


DEFAULT_DATASET = PROJECT_ROOT / "benchmark" / "test_dataset" / "realworld_writing_errors.json"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "pipeline_output" / "phase4_cache"
DEFAULT_OUTPUT = PROJECT_ROOT / "benchmark" / "results" / "phase4_stage1_uncertainty_signal_audit.json"

DEFAULT_PRIMARY_SOURCE = "qwen3vl_4b_crop_verified_word_v2"
DEFAULT_SOURCES = [
    "qwen3vl_4b_verbatim_word",
    "qwen3vl_4b_word",
    "qwen3vl_4b_crop_verified_word_v2",
    "tesseract_word_boxes",
    "doctr_live_word_ocr",
    "easyocr_live_word_ocr",
    "got_ocr2_live_ocr",
    "florence2_live_region_ocr",
    "paddleocr_vl_live_ocr",
    "nemotron_ocr_v2_live_ocr",
]

QWEN_VERBATIM = "qwen3vl_4b_verbatim_word"
QWEN_NORMAL = "qwen3vl_4b_word"
QWEN_CROP_VERIFIED = "qwen3vl_4b_crop_verified_word_v2"

BROAD_CONTEXT_SIGNALS = {
    "primary_not_preserved",
    "ocr_source_disagreement",
    "geometry_text_disagreement",
}


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def tokens(text: str) -> list[str]:
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def contains_span(text: str, span: str) -> bool:
    normalized_text = f" {normalize_text(text)} "
    normalized_span = normalize_text(span)
    return bool(normalized_span and f" {normalized_span} " in normalized_text)


def raw_contains_punct_sensitive(text: str, span: str) -> bool:
    return contains_span(text, span) and str(span).strip() not in str(text)


def bbox_iou(left: list[int] | list[float], right: list[int] | list[float]) -> float:
    if len(left) < 4 or len(right) < 4:
        return 0.0
    lx1, ly1, lx2, ly2 = [float(v) for v in left[:4]]
    rx1, ry1, rx2, ry2 = [float(v) for v in right[:4]]
    ix1 = max(lx1, rx1)
    iy1 = max(ly1, ry1)
    ix2 = min(lx2, rx2)
    iy2 = min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def edit_distance(left: str, right: str) -> int:
    left = normalize_text(left).replace(" ", "")
    right = normalize_text(right).replace(" ", "")
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, 1):
        current = [i]
        for j, right_char in enumerate(right, 1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_cache_entry(cache_dir: Path, source: str, image_name: str) -> dict[str, Any] | None:
    path = cache_dir / source / f"{Path(image_name).stem}.json"
    if not path.exists():
        return None
    return load_json(path)


def source_text(entry: dict[str, Any] | None) -> str:
    return str(entry.get("text", "")) if entry else ""


def source_boxes(entry: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not entry:
        return []
    boxes = entry.get("blocks", entry.get("boxes", []))
    return boxes if isinstance(boxes, list) else []


def error_bbox_from_words(page: dict[str, Any], error: dict[str, Any]) -> list[int]:
    words = page.get("words", [])
    bboxes = []
    for index in error.get("word_indices", []):
        if isinstance(index, int) and 0 <= index < len(words):
            bbox = words[index].get("bbox", [])
            if isinstance(bbox, list) and len(bbox) >= 4:
                bboxes.append([int(v) for v in bbox[:4]])
    return union_bboxes(bboxes) if bboxes else [0, 0, 0, 0]


def nearby_boxes(
    boxes: list[dict[str, Any]],
    gt_bbox: list[int],
    *,
    min_iou: float,
) -> list[dict[str, Any]]:
    nearby = []
    for box in boxes:
        bbox = box.get("bbox", [])
        if not isinstance(bbox, list) or len(bbox) < 4:
            continue
        iou = bbox_iou(bbox[:4], gt_bbox)
        if iou >= min_iou:
            nearby.append({
                "text": str(box.get("text", "")),
                "bbox": [int(v) for v in bbox[:4]],
                "iou": iou,
                "confidence": box.get("confidence"),
            })
    return sorted(nearby, key=lambda item: item["iou"], reverse=True)


def matching_evidence_iou(boxes: list[dict[str, Any]], evidence: str, gt_bbox: list[int]) -> float:
    evidence_norm = normalize_text(evidence)
    best = 0.0
    for box in boxes:
        if evidence_norm and evidence_norm not in normalize_text(str(box.get("text", ""))):
            continue
        bbox = box.get("bbox", [])
        if isinstance(bbox, list) and len(bbox) >= 4:
            best = max(best, bbox_iou(bbox[:4], gt_bbox))
    return best


def source_status(
    source: str,
    entry: dict[str, Any] | None,
    evidence: str,
    correction: str,
    gt_bbox: list[int],
    *,
    min_iou: float,
) -> dict[str, Any]:
    text = source_text(entry)
    boxes = source_boxes(entry)
    evidence_present = contains_span(text, evidence)
    correction_present = contains_span(text, correction)
    near = nearby_boxes(boxes, gt_bbox, min_iou=min_iou)
    nearby_texts = [item["text"] for item in near]
    nearby_norms = sorted({normalize_text(text) for text in nearby_texts if normalize_text(text)})

    if entry is None:
        classification = "missing_cache"
    elif evidence_present and correction_present:
        classification = "ambiguous"
    elif evidence_present:
        classification = "evidence"
    elif correction_present:
        classification = "correction"
    elif text.strip():
        classification = "other"
    else:
        classification = "empty"

    return {
        "source": source,
        "has_cache": entry is not None,
        "classification": classification,
        "evidence_present": evidence_present,
        "correction_present": correction_present,
        "punctuation_sensitive_match": raw_contains_punct_sensitive(text, evidence),
        "matching_evidence_iou": matching_evidence_iou(boxes, evidence, gt_bbox),
        "nearby_texts": nearby_texts[:5],
        "nearby_normalized_texts": nearby_norms[:5],
        "best_nearby_iou": near[0]["iou"] if near else 0.0,
        "best_nearby_text": near[0]["text"] if near else "",
    }


def consecutive_repeat(values: list[str]) -> bool:
    return any(left == right for left, right in zip(values, values[1:]))


def tokenization_risk(evidence: str, statuses: dict[str, dict[str, Any]]) -> bool:
    evidence_tokens = tokens(evidence)
    if not evidence_tokens:
        return False
    nearby_token_counts = []
    for status in statuses.values():
        for text in status.get("nearby_normalized_texts", []):
            nearby_token_counts.append(len(tokens(text)))
    if not nearby_token_counts:
        return False
    if len(evidence_tokens) == 1 and any(count > 1 for count in nearby_token_counts):
        return True
    if len(evidence_tokens) > 1 and any(count == 1 for count in nearby_token_counts):
        return True
    return False


def crop_verifier_touched(entry: dict[str, Any] | None, gt_bbox: list[int]) -> bool:
    if not entry:
        return False
    records = []
    records.extend(entry.get("_crop_verified_replacements", []))
    records.extend(entry.get("_crop_verified_rejected", []))
    for record in records:
        bbox = record.get("bbox", [])
        if isinstance(bbox, list) and len(bbox) >= 4 and bbox_iou(bbox[:4], gt_bbox) >= 0.05:
            return True
    return False


def build_signals(
    *,
    primary_status: dict[str, Any],
    statuses: dict[str, dict[str, Any]],
    evidence: str,
    correction: str,
    crop_entry: dict[str, Any] | None,
    gt_bbox: list[int],
    alignment_threshold: float,
) -> tuple[dict[str, bool], dict[str, bool]]:
    qwen_normal = statuses.get(QWEN_NORMAL, {})
    qwen_verbatim = statuses.get(QWEN_VERBATIM, {})
    status_classes = {
        status["classification"]
        for status in statuses.values()
        if status.get("has_cache")
    }
    nearby_text_sets = {
        tuple(status.get("nearby_normalized_texts", []))
        for status in statuses.values()
        if status.get("nearby_normalized_texts")
    }

    general = {
        "primary_not_preserved": primary_status["classification"] not in {"evidence", "ambiguous"},
        "qwen_normal_verbatim_disagreement": (
            bool(qwen_normal)
            and bool(qwen_verbatim)
            and (
                qwen_normal.get("classification") != qwen_verbatim.get("classification")
                or qwen_normal.get("nearby_normalized_texts") != qwen_verbatim.get("nearby_normalized_texts")
            )
        ),
        "ocr_source_disagreement": len(status_classes) > 1 or len(nearby_text_sets) > 1,
        "low_primary_alignment_confidence": (
            primary_status.get("evidence_present")
            and primary_status.get("matching_evidence_iou", 0.0) < alignment_threshold
        ) or (
            not primary_status.get("nearby_texts")
            and primary_status.get("has_cache")
        ),
        "geometry_text_disagreement": len(nearby_text_sets) > 1,
        "punctuation_sensitive_token": any(
            status.get("punctuation_sensitive_match") for status in statuses.values()
        ),
        "repeated_token_span": consecutive_repeat(tokens(evidence)),
        "multiword_span": len(tokens(evidence)) > 1,
        "fused_or_split_candidate": tokenization_risk(evidence, statuses),
        "crop_verifier_already_flagged": crop_verifier_touched(crop_entry, gt_bbox),
    }
    diagnostic_only = {
        "evidence_correction_edit_close": (
            bool(normalize_text(evidence))
            and bool(normalize_text(correction))
            and edit_distance(evidence, correction) <= 2
        ),
        "correction_seen_in_primary": primary_status.get("correction_present", False),
    }
    return general, diagnostic_only


def audit_page(
    page: dict[str, Any],
    *,
    cache_dir: Path,
    primary_source: str,
    sources: list[str],
    nearby_iou: float,
    alignment_threshold: float,
) -> list[dict[str, Any]]:
    entries = {
        source: load_cache_entry(cache_dir, source, page["image"])
        for source in sources
    }
    rows = []
    for error_index, error in enumerate(page.get("errors", [])):
        evidence = str(error.get("evidence_text", ""))
        correction = str(error.get("correction", ""))
        gt_bbox = error.get("bbox") or error_bbox_from_words(page, error)
        statuses = {
            source: source_status(
                source,
                entry,
                evidence,
                correction,
                gt_bbox,
                min_iou=nearby_iou,
            )
            for source, entry in entries.items()
        }
        primary_status = statuses.get(primary_source)
        if not primary_status:
            primary_status = source_status(
                primary_source,
                load_cache_entry(cache_dir, primary_source, page["image"]),
                evidence,
                correction,
                gt_bbox,
                min_iou=nearby_iou,
            )
            statuses[primary_source] = primary_status

        general, diagnostic_only = build_signals(
            primary_status=primary_status,
            statuses=statuses,
            evidence=evidence,
            correction=correction,
            crop_entry=entries.get(QWEN_CROP_VERIFIED),
            gt_bbox=gt_bbox,
            alignment_threshold=alignment_threshold,
        )
        rows.append({
            "image": page["image"],
            "error_index": error_index,
            "type": error.get("type", ""),
            "evidence_text": evidence,
            "correction": correction,
            "word_indices": error.get("word_indices", []),
            "gt_bbox": gt_bbox,
            "primary_source": primary_source,
            "primary_classification": primary_status["classification"],
            "primary_evidence_iou": primary_status["matching_evidence_iou"],
            "caught_by_any_general_signal": any(general.values()),
            "general_signals": general,
            "diagnostic_only_signals": diagnostic_only,
            "source_statuses": statuses,
        })
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    signal_counts: Counter[str] = Counter()
    diagnostic_counts: Counter[str] = Counter()
    primary_counts: Counter[str] = Counter()
    missed_rows = []
    for row in rows:
        primary_counts[row["primary_classification"]] += 1
        for signal, enabled in row["general_signals"].items():
            if enabled:
                signal_counts[signal] += 1
        for signal, enabled in row["diagnostic_only_signals"].items():
            if enabled:
                diagnostic_counts[signal] += 1
        if row["primary_classification"] not in {"evidence", "ambiguous"}:
            missed_rows.append(row)

    caught_misses = [
        row for row in missed_rows
        if any(row["general_signals"].values())
    ]
    actionable_caught_misses = [
        row for row in missed_rows
        if any(
            enabled and signal not in BROAD_CONTEXT_SIGNALS
            for signal, enabled in row["general_signals"].items()
        )
    ]
    uncaught_misses = [
        {
            "image": row["image"],
            "evidence_text": row["evidence_text"],
            "correction": row["correction"],
            "primary_classification": row["primary_classification"],
            "diagnostic_only_signals": row["diagnostic_only_signals"],
        }
        for row in missed_rows
        if not any(
            enabled and signal not in BROAD_CONTEXT_SIGNALS
            for signal, enabled in row["general_signals"].items()
        )
    ]
    return {
        "total_errors": len(rows),
        "primary_classification_counts": dict(primary_counts),
        "primary_miss_count": len(missed_rows),
        "primary_misses_caught_by_general_signals": len(caught_misses),
        "primary_miss_general_signal_recall": (
            len(caught_misses) / len(missed_rows) if missed_rows else 1.0
        ),
        "broad_context_signals": sorted(BROAD_CONTEXT_SIGNALS),
        "primary_misses_caught_by_actionable_signals": len(actionable_caught_misses),
        "primary_miss_actionable_signal_recall": (
            len(actionable_caught_misses) / len(missed_rows) if missed_rows else 1.0
        ),
        "general_signal_counts": dict(signal_counts),
        "diagnostic_only_signal_counts": dict(diagnostic_counts),
        "uncaught_primary_misses_by_actionable_signals": uncaught_misses,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--primary-source", default=DEFAULT_PRIMARY_SOURCE)
    parser.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES)
    parser.add_argument("--nearby-iou", type=float, default=0.05)
    parser.add_argument("--alignment-threshold", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_json(args.dataset)
    sources = list(dict.fromkeys([args.primary_source, *args.sources]))
    rows = []
    for page in dataset:
        rows.extend(audit_page(
            page,
            cache_dir=args.cache_dir,
            primary_source=args.primary_source,
            sources=sources,
            nearby_iou=args.nearby_iou,
            alignment_threshold=args.alignment_threshold,
        ))

    result = {
        "dataset": str(args.dataset.relative_to(PROJECT_ROOT)),
        "cache_dir": str(args.cache_dir.relative_to(PROJECT_ROOT)),
        "primary_source": args.primary_source,
        "sources": sources,
        "nearby_iou": args.nearby_iou,
        "alignment_threshold": args.alignment_threshold,
        "generalization_policy": {
            "uses_ground_truth_for_evaluation_only": True,
            "word_specific_fixes_allowed": False,
            "production_trigger_rule": "Prefer actionable general signals. Treat broad context signals as audit context, not sufficient verifier triggers by themselves.",
        },
        "summary": summarize(rows),
        "errors": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    summary = result["summary"]
    print(
        f"Primary misses caught by general signals: "
        f"{summary['primary_misses_caught_by_general_signals']}/{summary['primary_miss_count']}"
    )
    print(
        f"Primary misses caught by actionable signals: "
        f"{summary['primary_misses_caught_by_actionable_signals']}/{summary['primary_miss_count']}"
    )
    for signal, count in sorted(summary["general_signal_counts"].items()):
        print(f"{signal}: {count}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
