#!/usr/bin/env python3
"""Standalone PaddleOCR-VL runner for Phase 4 real-world Stage 1 cache files.

This script intentionally has zero project imports. It is meant to run inside
the PaddleOCR-VL Docker image and write cache JSON files that
`scripts/benchmark_phase4.py` can consume as `paddleocr_vl_live_ocr`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DATA_DIR = Path(os.environ.get("DATA_DIR", "/data/realworld_raw"))
DEFAULT_GT_PATH = Path(os.environ.get("GT_PATH", "/data/realworld_writing_errors.json"))
DEFAULT_CACHE_DIR = Path(os.environ.get("CACHE_DIR", "/phase4_cache/paddleocr_vl_live_ocr"))
DEFAULT_RESULT_PATH = Path(os.environ.get("RESULT_PATH", "/results/phase4_realworld_stage1_paddleocr_vl_docker_raw.json"))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def discover_images(data_dir: Path, names: list[str] | None, max_images: int) -> list[Path]:
    if names:
        images = [data_dir / name for name in names]
    else:
        images = sorted(path for path in data_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if max_images > 0:
        images = images[:max_images]
    missing = [str(path) for path in images if not path.exists()]
    if missing:
        raise SystemExit(f"Missing image(s): {', '.join(missing)}")
    return images


def load_ground_truth(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {
        str(entry.get("image", "")): entry
        for entry in data
        if entry.get("image")
    }


def normalize_bbox(raw: Any) -> list[int]:
    if isinstance(raw, str):
        raw = raw.strip().strip("[]")
        try:
            values = [float(part.strip()) for part in raw.split(",")]
        except ValueError:
            return [0, 0, 0, 0]
    elif isinstance(raw, (list, tuple)):
        try:
            values = [float(value) for value in list(raw)[:4]]
        except (TypeError, ValueError):
            return [0, 0, 0, 0]
    else:
        return [0, 0, 0, 0]
    if len(values) < 4:
        return [0, 0, 0, 0]
    x1, y1, x2, y2 = [int(round(value)) for value in values[:4]]
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return [max(0, x1), max(0, y1), max(0, x2), max(0, y2)]


def parse_paddle_output(output: Any) -> tuple[str, list[dict[str, Any]]]:
    blocks: list[dict[str, Any]] = []
    text_parts: list[str] = []
    if not output:
        return "", blocks

    for res in output:
        if hasattr(res, "json") and res.json:
            inner = res.json.get("res", res.json) if isinstance(res.json, dict) else {}
            for item in inner.get("parsing_res_list", []):
                text = str(item.get("block_content", "")).strip()
                if not text:
                    continue
                bbox = normalize_bbox(item.get("block_bbox", [0, 0, 0, 0]))
                blocks.append({
                    "bbox": bbox,
                    "text": text,
                    "confidence": 1.0,
                    "source": "paddleocr_vl_live_ocr",
                })
                text_parts.append(text)

        if not text_parts and hasattr(res, "markdown") and res.markdown:
            markdown = res.markdown
            if isinstance(markdown, dict):
                text = markdown.get("markdown_texts", "")
            else:
                text = str(markdown)
            if text:
                text_parts.append(str(text))
        if not text_parts and hasattr(res, "text") and res.text:
            text_parts.append(str(res.text))

    return " ".join(part.strip() for part in text_parts if part.strip()), blocks


def sync_paddle() -> None:
    try:
        import paddle

        paddle.device.synchronize()
    except Exception:
        pass


def run_image(pipeline: Any, image_path: Path) -> dict[str, Any]:
    sync_paddle()
    start = time.perf_counter()
    output = pipeline.predict(str(image_path))
    sync_paddle()
    latency = time.perf_counter() - start
    text, blocks = parse_paddle_output(output)
    return {
        "text": text,
        "blocks": blocks,
        "stage1_latency": latency,
        "_stage1_composition": "paddleocr_vl_docker_standalone",
        "_box_granularity": "block",
        "_raw_block_count": len(blocks),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--gt-path", type=Path, default=DEFAULT_GT_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--result-path", type=Path, default=DEFAULT_RESULT_PATH)
    parser.add_argument("--image", action="append", dest="images")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--no-os-exit", action="store_true", help="Return normally instead of os._exit(0) after saving.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    images = discover_images(args.data_dir, args.images, args.max_images)
    gt_by_name = load_ground_truth(args.gt_path)

    print("Loading PaddleOCR-VL pipeline ...", flush=True)
    from paddleocr import PaddleOCRVL

    pipeline = PaddleOCRVL(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )
    print("Pipeline loaded.", flush=True)

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    result_records: list[dict[str, Any]] = []
    for index, image_path in enumerate(images, start=1):
        print(f"[{index}/{len(images)}] {image_path.name}", flush=True)
        try:
            record = run_image(pipeline, image_path)
            failed = False
            failure = ""
        except Exception as exc:
            failed = True
            failure = f"{type(exc).__name__}: {exc}"
            record = {
                "text": "",
                "blocks": [],
                "stage1_latency": 0.0,
                "_stage1_composition": "paddleocr_vl_docker_standalone",
                "_failure": failure,
            }

        cache_path = args.cache_dir / f"{image_path.stem}.json"
        cache_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
        result_records.append({
            "image": image_path.name,
            "cache_path": str(cache_path),
            "failed": failed,
            "failure": failure,
            "text": record.get("text", ""),
            "blocks": record.get("blocks", []),
            "stage1_latency": record.get("stage1_latency", 0.0),
            "ground_truth_errors": len(gt_by_name.get(image_path.name, {}).get("errors", [])),
        })

    result = {
        "candidate": "paddleocr_vl_live_ocr",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "purpose": "Raw standalone Docker output for Phase 4 cache population",
        "data_dir": str(args.data_dir),
        "gt_path": str(args.gt_path),
        "cache_dir": str(args.cache_dir),
        "images": result_records,
    }
    args.result_path.parent.mkdir(parents=True, exist_ok=True)
    args.result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved raw result: {args.result_path}", flush=True)
    print(f"Saved cache files under: {args.cache_dir}", flush=True)

    if args.no_os_exit:
        return 0
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
