"""Apply scored evidence-graph decisions to an existing Phase 4 result.

This is a diagnostic bridge, not a production inference path. The current
minimal-pair evidence graph is built from development-set labels, so this script
answers a bounded question: what would Phase 4 metrics look like if the visual
selector's SUPPORTED_ERROR records were available to the Stage 2/adjudication
layer?
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.contracts import ErrorFinding, Feedback, PipelineOutput, TextBox
from pipeline.metrics import aggregate_phase4_metrics, evaluate_phase4_output


DEFAULT_RESULT = Path("benchmark/results/phase4_realworld_mixed_best_20image_adjudicated.json")
DEFAULT_EVIDENCE_GRAPH = Path("benchmark/results/phase4_evidence_graph_qwen_visual_gain.json")
DEFAULT_GROUND_TRUTH = Path("benchmark/test_dataset/realworld_writing_errors.json")
DEFAULT_OUTPUT = Path("benchmark/results/phase4_realworld_evidence_graph_adjudication_diagnostic.json")
DEFAULT_STRATEGY = "two_stage__qwen3vl_4b_verbatim_word_ocr__same_stage1_boxes__qwen3vl_4b_grader"


def load_ground_truth(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        for key in ("samples", "pages", "items"):
            if isinstance(data.get(key), list):
                return {str(item.get("image")): item for item in data[key]}
    if isinstance(data, list):
        return {str(item.get("image")): item for item in data}
    raise ValueError(f"Unsupported ground-truth schema: {path}")


def load_evidence_records(path: Path) -> dict[str, list[dict[str, Any]]]:
    data = json.loads(path.read_text())
    by_image: dict[str, list[dict[str, Any]]] = {}
    for record in data.get("records", []):
        if record.get("decision") != "SUPPORTED_ERROR":
            continue
        by_image.setdefault(str(record.get("image")), []).append(record)
    return by_image


def output_from_dict(data: dict[str, Any]) -> PipelineOutput:
    return PipelineOutput(
        strategy_name=str(data.get("strategy_name", "")),
        image=str(data.get("image", "")),
        text=str(data.get("text", "")),
        boxes=[TextBox.from_dict(item, index=i) for i, item in enumerate(data.get("boxes", []))],
        errors=[ErrorFinding.from_dict(item) for item in data.get("errors", [])],
        feedback=Feedback.from_any(data.get("feedback", {})),
        stage1_latency=float(data.get("stage1_latency", 0.0) or 0.0),
        stage2_latency=float(data.get("stage2_latency", 0.0) or 0.0),
        total_latency=float(data.get("total_latency", 0.0) or 0.0),
        parse_valid=bool(data.get("parse_valid", True)),
        repair_attempted=bool(data.get("repair_attempted", False)),
        repair_succeeded=bool(data.get("repair_succeeded", False)),
        raw_response=str(data.get("raw_response", "")),
        notes=[str(item) for item in data.get("notes", [])],
        metadata=dict(data.get("metadata", {})),
    )


def evidence_error(record: dict[str, Any]) -> ErrorFinding:
    return ErrorFinding(
        type=str(record.get("error_type", "spelling") or "spelling"),
        bbox=[int(v) for v in record.get("bbox", [0, 0, 0, 0])[:4]],
        description="Supported by scored visual evidence graph.",
        correction=str(record.get("normalized_form", "")),
        evidence_text=str(record.get("visible_form", "")),
        word_indices=[int(i) for i in record.get("word_indices", []) if isinstance(i, int)],
    )


def error_key(error: ErrorFinding) -> tuple[tuple[int, ...], str, str, str]:
    return (
        tuple(error.word_indices),
        error.type,
        _norm(error.evidence_text),
        _norm(error.correction),
    )


def _norm(text: str) -> str:
    return "".join(ch.lower() for ch in str(text) if ch.isalnum())


def add_evidence_errors(
    output: PipelineOutput,
    evidence_records: list[dict[str, Any]],
    *,
    mode: str,
) -> list[dict[str, Any]]:
    additions = [evidence_error(record) for record in evidence_records]
    if mode == "replace":
        output.errors = additions
        return [record_summary(record) for record in evidence_records]

    existing = {error_key(error) for error in output.errors}
    applied = []
    for record, candidate in zip(evidence_records, additions):
        key = error_key(candidate)
        if key in existing:
            continue
        output.errors.append(candidate)
        existing.add(key)
        applied.append(record_summary(record))
    return applied


def record_summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "pair_id": record.get("pair_id"),
        "word_indices": record.get("word_indices", []),
        "evidence_text": record.get("visible_form", ""),
        "correction": record.get("normalized_form", ""),
        "type": record.get("error_type", ""),
        "margin": record.get("margin"),
        "top_text": record.get("top_text", ""),
    }


def find_strategy(result: dict[str, Any], name: str) -> dict[str, Any]:
    for strategy in result.get("strategies", []):
        if strategy.get("name") == name:
            return strategy
    names = [strategy.get("name") for strategy in result.get("strategies", [])]
    raise ValueError(f"Strategy not found: {name}. Available: {names}")


def evaluate_images(
    strategy: dict[str, Any],
    evidence_by_image: dict[str, list[dict[str, Any]]],
    gt_by_image: dict[str, dict[str, Any]],
    *,
    mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    image_rows = []
    metric_rows = []
    for image_record in strategy.get("images", []):
        image = str(image_record.get("image", ""))
        output = output_from_dict(copy.deepcopy(image_record.get("output", {})))
        applied = add_evidence_errors(output, evidence_by_image.get(image, []), mode=mode)
        metrics = evaluate_phase4_output(output, gt_by_image.get(image), gt_by_image.get(image))
        metric_rows.append({"image": image, **metrics})
        image_rows.append({
            "image": image,
            "run": image_record.get("run", 1),
            "evidence_graph_records_applied": applied,
            "metrics": metrics,
            "output": output.to_dict(),
        })
    return image_rows, aggregate_phase4_metrics(metric_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--evidence-graph", type=Path, default=DEFAULT_EVIDENCE_GRAPH)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument("--mode", choices=["augment", "replace"], default="augment")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    result = json.loads(args.result.read_text())
    strategy = find_strategy(result, args.strategy)
    evidence_by_image = load_evidence_records(args.evidence_graph)
    gt_by_image = load_ground_truth(args.ground_truth)
    images, aggregate = evaluate_images(
        strategy,
        evidence_by_image,
        gt_by_image,
        mode=args.mode,
    )
    output = {
        "diagnostic": True,
        "warning": (
            "Uses a development-set minimal-pair evidence graph. Do not treat as "
            "held-out or production inference."
        ),
        "source_result": str(args.result),
        "source_strategy": args.strategy,
        "evidence_graph": str(args.evidence_graph),
        "ground_truth": str(args.ground_truth),
        "mode": args.mode,
        "aggregate": aggregate,
        "images": images,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"mode": args.mode, "aggregate": aggregate}, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
