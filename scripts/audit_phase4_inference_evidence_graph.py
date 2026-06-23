"""Audit a label-free Phase 4 inference evidence graph against dev labels.

The evidence graph itself is label-free. This audit uses the reviewed
development annotations only after the fact to decompose failures into:

- no graph record for the error span
- visible evidence absent from candidates
- visible evidence present but not top-ranked
- visible evidence top-ranked but blocked by policy
- visible evidence auto-supported
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_GRAPH = Path("benchmark/results/phase4_inference_evidence_graph_qwen_visual_gain_rw11.json")
DEFAULT_DATASET = Path("benchmark/test_dataset/realworld_writing_errors.json")
DEFAULT_OUTPUT = Path("benchmark/results/phase4_inference_evidence_graph_audit.json")


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def norm_words(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def candidate_contains(candidate_text: str, evidence_text: str) -> bool:
    candidate_norm = norm(candidate_text)
    evidence_norm = norm(evidence_text)
    if candidate_norm == evidence_norm:
        return True
    candidate_words = f" {norm_words(candidate_text)} "
    evidence_words = norm_words(evidence_text)
    return bool(evidence_words and f" {evidence_words} " in candidate_words)


def load_dataset(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("samples", "pages", "items"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"Unsupported dataset schema: {path}")


def bbox_iou(left: list[float], right: list[float]) -> float:
    if len(left) != 4 or len(right) != 4:
        return 0.0
    xa = max(left[0], right[0])
    ya = max(left[1], right[1])
    xb = min(left[2], right[2])
    yb = min(left[3], right[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_l = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    area_r = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    union = area_l + area_r - inter
    return inter / union if union else 0.0


def error_bbox(sample: dict[str, Any], indices: list[int]) -> list[int]:
    words = sample.get("words", [])
    boxes = [words[index].get("bbox", [0, 0, 0, 0]) for index in indices if 0 <= index < len(words)]
    if not boxes:
        return [0, 0, 0, 0]
    return [
        min(int(box[0]) for box in boxes),
        min(int(box[1]) for box in boxes),
        max(int(box[2]) for box in boxes),
        max(int(box[3]) for box in boxes),
    ]


def records_by_image(path: Path) -> dict[str, list[dict[str, Any]]]:
    data = json.loads(path.read_text())
    records: dict[str, list[dict[str, Any]]] = {}
    for record in data.get("records", []) or []:
        records.setdefault(str(record.get("image", "")), []).append(record)
    return records


def best_record_for_error(
    records: list[dict[str, Any]],
    bbox: list[int],
    evidence_text: str,
    *,
    min_iou: float,
) -> dict[str, Any] | None:
    best: tuple[float, dict[str, Any]] | None = None
    evidence_norm = norm(evidence_text)
    for record in records:
        iou = bbox_iou(record.get("bbox", [0, 0, 0, 0]), bbox)
        if iou < min_iou:
            continue
        candidates = record.get("hypotheses") or record.get("candidates") or []
        has_visible = any(candidate_contains(candidate.get("text", ""), evidence_text) for candidate in candidates)
        score = iou + (1.0 if has_visible else 0.0)
        if best is None or score > best[0]:
            best = (score, record)
    return best[1] if best else None


def visible_status(error: dict[str, Any], record: dict[str, Any] | None) -> dict[str, Any]:
    evidence = str(error.get("evidence_text", ""))
    evidence_norm = norm(evidence)
    if record is None:
        return {
            "status": "missing_record",
            "record_id": "",
            "visible_candidate_present": False,
            "visible_top_rank": None,
            "visible_supported_by_ocr": False,
            "top_supported_by_ocr": False,
            "top_source": "",
        }

    hypotheses = record.get("hypotheses") or record.get("candidates") or []
    top = hypotheses[0] if hypotheses else {}
    visible_rank = None
    visible_supported_by_ocr = False
    visible_source = ""
    for rank, candidate in enumerate(hypotheses, start=1):
        if candidate_contains(candidate.get("text", ""), evidence):
            visible_rank = rank
            visible_supported_by_ocr = bool(candidate.get("supported_by_ocr", False))
            visible_source = str(candidate.get("source", ""))
            break

    if visible_rank is None:
        status = "visible_absent"
    elif visible_rank == 1 and record.get("decision") == "SUPPORTED_ALTERNATIVE":
        status = "auto_supported"
    elif visible_rank == 1:
        status = "top_but_blocked"
    else:
        status = "present_not_top"

    return {
        "status": status,
        "record_id": record.get("pair_id", ""),
        "visible_candidate_present": visible_rank is not None,
        "visible_top_rank": visible_rank,
        "visible_supported_by_ocr": visible_supported_by_ocr,
        "visible_source": visible_source,
        "decision": record.get("decision"),
        "canonical_text": record.get("canonical_text"),
        "top_text": record.get("top_text"),
        "top_supported_by_ocr": bool(top.get("supported_by_ocr", False)),
        "top_source": str(top.get("source", "")),
        "alternative_margin_over_canonical": record.get("alternative_margin_over_canonical"),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    return {
        "errors_total": total,
        "record_present": sum(1 for row in rows if row["status"] != "missing_record"),
        "visible_candidate_present": sum(1 for row in rows if row["visible_candidate_present"]),
        "visible_candidate_recall": (
            sum(1 for row in rows if row["visible_candidate_present"]) / total if total else 0.0
        ),
        "visible_top_rank_1": sum(1 for row in rows if row.get("visible_top_rank") == 1),
        "visible_top_rank_1_rate": (
            sum(1 for row in rows if row.get("visible_top_rank") == 1) / total if total else 0.0
        ),
        "visible_ocr_supported": sum(1 for row in rows if row["visible_supported_by_ocr"]),
        "auto_supported": sum(1 for row in rows if row["status"] == "auto_supported"),
        "top_but_blocked": sum(1 for row in rows if row["status"] == "top_but_blocked"),
        "present_not_top": sum(1 for row in rows if row["status"] == "present_not_top"),
        "visible_absent": sum(1 for row in rows if row["status"] == "visible_absent"),
        "missing_record": sum(1 for row in rows if row["status"] == "missing_record"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image", action="append", dest="images", default=[])
    parser.add_argument("--min-iou", type=float, default=0.05)
    args = parser.parse_args()

    records = records_by_image(args.graph)
    image_filter = set(args.images)
    rows = []
    for sample in load_dataset(args.dataset):
        image = str(sample.get("image", ""))
        if image_filter and image not in image_filter:
            continue
        for error_idx, error in enumerate(sample.get("errors", []) or []):
            indices = [int(i) for i in error.get("word_indices", []) if isinstance(i, int)]
            record = best_record_for_error(
                records.get(image, []),
                error_bbox(sample, indices),
                str(error.get("evidence_text", "")),
                min_iou=args.min_iou,
            )
            rows.append({
                "image": image,
                "error_index": error_idx,
                "word_indices": indices,
                "evidence_text": error.get("evidence_text", ""),
                "correction": error.get("correction", ""),
                "type": error.get("type", ""),
                **visible_status(error, record),
            })

    result = {
        "graph": str(args.graph),
        "dataset": str(args.dataset),
        "images": sorted(image_filter),
        "summary": summarize(rows),
        "errors": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
