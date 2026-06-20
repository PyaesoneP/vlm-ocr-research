#!/usr/bin/env python3
"""Stage 2-only regression runner for reviewed real-world Phase 4 annotations."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.contracts import PipelineOutput, TextBox
from pipeline.localization import union_bboxes
from pipeline.metrics import aggregate_phase4_metrics, evaluate_phase4_output
from pipeline.strategies import TwoStageStrategy
from scripts.benchmark_phase4 import (
    QWEN_MODEL_ID,
    Qwen3VLAdapter,
    fake_model_call,
    get_peak_vram_mb,
)

REALWORLD_RAW_DIR = PROJECT_ROOT / "benchmark" / "test_dataset" / "realworld_raw"
REALWORLD_DATASET = PROJECT_ROOT / "benchmark" / "test_dataset" / "realworld_writing_errors.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "benchmark" / "results" / "phase4_stage2_source_text_qwen_contract_v2.json"


def source_output(entry: dict[str, Any]) -> PipelineOutput:
    boxes = [
        TextBox.from_dict(
            word,
            index=int(word.get("index", i)),
            source="reviewed_realworld_words",
        )
        for i, word in enumerate(entry.get("words", []))
    ]
    return PipelineOutput(
        strategy_name="realworld_source_text_reviewed_words",
        image=entry["image"],
        text=str(entry.get("text", "")),
        boxes=boxes,
        stage1_latency=0.0,
        total_latency=0.0,
        metadata={
            "annotation_status": entry.get("annotation_status", ""),
            "source": "realworld_writing_errors.json",
        },
    )


def fake_grader_call(prompt: str, image_path: Path | None) -> str:
    return fake_model_call(prompt, image_path)


def qwen_grader_call(adapter: Qwen3VLAdapter, max_new_tokens: int):
    def call(prompt: str, image_path: Path | None) -> str:
        return adapter.generate(prompt, image_path, max_new_tokens=max_new_tokens)

    return call


def select_entries(
    entries: list[dict[str, Any]],
    *,
    max_images: int,
    images: list[str] | None,
) -> list[dict[str, Any]]:
    by_image = {entry["image"]: entry for entry in entries}
    if images:
        missing = [image for image in images if image not in by_image]
        if missing:
            raise SystemExit(f"Requested images not found: {missing}")
        return [by_image[image] for image in images]
    selected = sorted(entries, key=lambda entry: entry["image"])
    if max_images > 0:
        selected = selected[:max_images]
    return selected


def contract_metrics(output: PipelineOutput) -> dict[str, Any]:
    indexed_boxes = {
        box.index if box.index is not None else i: box
        for i, box in enumerate(output.boxes)
    }
    predicted_total = len(output.errors)
    anchored = 0
    contiguous = 0
    evidence_exact = 0
    evidence_normalized = 0
    bbox_union_exact = 0
    bbox_union_iou_ok = 0

    for error in output.errors:
        indices = error.word_indices
        valid_indices = [index for index in indices if index in indexed_boxes]
        if indices and len(valid_indices) == len(indices):
            anchored += 1
        sorted_indices = sorted(valid_indices)
        if sorted_indices and sorted_indices == list(range(sorted_indices[0], sorted_indices[-1] + 1)):
            contiguous += 1

        referenced_text = " ".join(indexed_boxes[index].text for index in sorted_indices)
        if referenced_text == error.evidence_text:
            evidence_exact += 1
        if _normalize(referenced_text) == _normalize(error.evidence_text):
            evidence_normalized += 1

        union = union_bboxes([indexed_boxes[index].bbox for index in sorted_indices])
        if union == error.bbox:
            bbox_union_exact += 1
        if _bbox_iou(union, error.bbox) >= 0.98:
            bbox_union_iou_ok += 1

    return {
        "contract_predicted_errors": predicted_total,
        "contract_anchored_errors": anchored,
        "contract_contiguous_errors": contiguous,
        "contract_evidence_exact": evidence_exact,
        "contract_evidence_normalized": evidence_normalized,
        "contract_bbox_union_exact": bbox_union_exact,
        "contract_bbox_union_iou_ok": bbox_union_iou_ok,
    }


def _normalize(text: Any) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _bbox_iou(a: list[int], b: list[int]) -> float:
    if len(a) != 4 or len(b) != 4:
        return 0.0
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def aggregate_contract(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "contract_predicted_errors",
        "contract_anchored_errors",
        "contract_contiguous_errors",
        "contract_evidence_exact",
        "contract_evidence_normalized",
        "contract_bbox_union_exact",
        "contract_bbox_union_iou_ok",
    ]
    totals = {key: sum(int(row.get(key, 0)) for row in rows) for key in keys}
    predicted = totals["contract_predicted_errors"]
    totals.update({
        "contract_anchor_rate": totals["contract_anchored_errors"] / predicted if predicted else 1.0,
        "contract_evidence_exact_rate": totals["contract_evidence_exact"] / predicted if predicted else 1.0,
        "contract_bbox_union_exact_rate": totals["contract_bbox_union_exact"] / predicted if predicted else 1.0,
    })
    return totals


def run_mode(
    *,
    mode: str,
    entries: list[dict[str, Any]],
    dataset_by_image: dict[str, dict[str, Any]],
    grader_call,
    include_raw: bool,
    continue_on_error: bool,
) -> dict[str, Any]:
    strategy = TwoStageStrategy(
        name=f"stage2_source_text__{mode}__qwen3vl_4b_grader",
        ocr_call=lambda image_path: source_output(dataset_by_image[image_path.name]),
        grader_call=grader_call,
        stage2_prompt_mode=mode,
    )

    print(f"[stage2] Running {strategy.name} on {len(entries)} image(s)")
    image_records = []
    metric_rows = []
    latencies = []
    vram_peaks = []

    for entry in entries:
        image_path = REALWORLD_RAW_DIR / entry["image"]
        try:
            output = strategy.run_from_ocr(source_output(entry), image_path=image_path)
            failed = False
            failure = ""
        except Exception as exc:
            if not continue_on_error:
                raise
            output = PipelineOutput(
                strategy_name=strategy.name,
                image=entry["image"],
                parse_valid=False,
                notes=[f"{type(exc).__name__}: {exc}"],
                metadata={"failed": True, "failure": f"{type(exc).__name__}: {exc}"},
            )
            failed = True
            failure = output.notes[0]

        metrics = evaluate_phase4_output(output, entry, entry)
        metrics.update(contract_metrics(output))
        metrics["stage_failed"] = failed
        if failure:
            metrics["failure"] = failure
        metric_rows.append({"image": entry["image"], **metrics})
        latencies.append(output.total_latency)
        vram_peaks.append(get_peak_vram_mb())

        output_data = output.to_dict()
        if not include_raw:
            output_data.pop("raw_response", None)
        image_records.append({
            "image": entry["image"],
            "output": output_data,
            "metrics": metrics,
        })
        print(
            f"  {entry['image']}: total={output.total_latency:.2f}s "
            f"valid_json={output.parse_valid} errors={len(output.errors)} "
            f"det_f1={metrics['error_detection_f1']} fp={metrics['false_positive_count']}"
            + (f" FAILED={failure}" if failed else "")
        )

    aggregate = aggregate_phase4_metrics(metric_rows)
    aggregate.update(aggregate_contract(metric_rows))
    aggregate.update({
        "latency_total_avg": sum(latencies) / len(latencies) if latencies else 0.0,
        "latency_stage2_avg": sum(latencies) / len(latencies) if latencies else 0.0,
        "vram_peak_mb": max(vram_peaks) if vram_peaks else 0,
    })
    return {
        "name": strategy.name,
        "aggregate": aggregate,
        "images": image_records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=REALWORLD_DATASET)
    parser.add_argument(
        "--prompt-modes",
        nargs="+",
        default=["contract_v2"],
        choices=[
            "baseline",
            "contract_v2",
            "contract_v3",
            "image_verify",
            "image_verify_v3",
        ],
    )
    parser.add_argument("--max-images", type=int, default=0, help="Number of images to run. Use 0 for all.")
    parser.add_argument("--image", dest="images", action="append", help="Specific real-world image filename.")
    parser.add_argument("--model-id", default=QWEN_MODEL_ID)
    parser.add_argument("--grader-max-new-tokens", type=int, default=768)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-raw", action="store_true")
    parser.add_argument("--smoke-fake", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = json.loads(args.dataset.read_text())
    dataset_by_image = {entry["image"]: entry for entry in entries}
    selected = select_entries(entries, max_images=args.max_images, images=args.images)
    if not selected:
        raise SystemExit("No real-world entries selected.")

    if args.smoke_fake:
        grader_call = fake_grader_call
        model_id = "fake"
    else:
        adapter = Qwen3VLAdapter(model_id=args.model_id, max_new_tokens=args.grader_max_new_tokens)
        grader_call = qwen_grader_call(adapter, args.grader_max_new_tokens)
        model_id = args.model_id

    result = {
        "phase": 4,
        "task": "stage2_source_text_regression",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": {
            "name": "realworld",
            "ground_truth": str(args.dataset.relative_to(PROJECT_ROOT)),
            "image_dir": str(REALWORLD_RAW_DIR.relative_to(PROJECT_ROOT)),
            "images": [entry["image"] for entry in selected],
            "model_id": model_id,
            "grader_max_new_tokens": args.grader_max_new_tokens,
            "prompt_modes": args.prompt_modes,
        },
        "notes": [
            "Stage 2-only regression: reviewed source text and reviewed word boxes are fixed inputs.",
            "No OCR, Stage 1 cache, or box-source mixing is used in this benchmark.",
            "Cloud/API graders are not used by this script.",
        ],
        "strategies": [],
    }

    for mode in args.prompt_modes:
        result["strategies"].append(
            run_mode(
                mode=mode,
                entries=selected,
                dataset_by_image=dataset_by_image,
                grader_call=grader_call,
                include_raw=args.include_raw,
                continue_on_error=args.continue_on_error,
            )
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    print(f"[stage2] Wrote {args.output}")


if __name__ == "__main__":
    main()
