#!/usr/bin/env python3
"""Render Phase 4 result overlays against real-world ground truth."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = PROJECT_ROOT / "benchmark" / "results" / "phase4_realworld_full_pipeline_5image.json"
DEFAULT_GT = PROJECT_ROOT / "benchmark" / "test_dataset" / "realworld_writing_errors.json"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "benchmark" / "test_dataset" / "realworld_raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmark" / "visualizations" / "phase4_realworld_full_pipeline_5image"


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")


def load_gt(path: Path) -> dict[str, dict[str, Any]]:
    return {entry["image"]: entry for entry in json.loads(path.read_text())}


def font_pair():
    from PIL import ImageFont

    try:
        return (
            ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18),
            ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13),
        )
    except OSError:
        return ImageFont.load_default(), ImageFont.load_default()


def draw_label(draw, xy: tuple[int, int], text: str, *, fill: tuple[int, int, int], font) -> None:
    x, y = xy
    label = str(text)[:42]
    bbox = draw.textbbox((x, y), label, font=font)
    pad = 2
    draw.rectangle(
        [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
        fill=(255, 255, 255),
    )
    draw.text((x, y), label, fill=fill, font=font)


def draw_boxes(
    draw,
    boxes: list[dict[str, Any]],
    *,
    outline: tuple[int, int, int],
    width: int,
    font,
    label_prefix: str = "",
    limit_labels: int = 45,
) -> None:
    labels_drawn = 0
    for idx, item in enumerate(boxes):
        bbox = item.get("bbox", [])
        if len(bbox) != 4 or bbox == [0, 0, 0, 0]:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        draw.rectangle([x1, y1, x2, y2], outline=outline, width=width)
        if labels_drawn >= limit_labels:
            continue
        text = str(item.get("text", item.get("evidence_text", "")))
        label = f"{label_prefix}{idx}:{text}" if text else f"{label_prefix}{idx}"
        draw_label(draw, (x1 + 2, max(36, y1 - 16)), label, fill=outline, font=font)
        labels_drawn += 1


def draw_errors(
    draw,
    errors: list[dict[str, Any]],
    *,
    outline: tuple[int, int, int],
    font,
    label_prefix: str,
) -> None:
    for idx, item in enumerate(errors):
        bbox = item.get("bbox", [])
        if len(bbox) != 4 or bbox == [0, 0, 0, 0]:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        draw.rectangle([x1, y1, x2, y2], outline=outline, width=5)
        evidence = item.get("evidence_text", "")
        correction = item.get("correction", "")
        label = f"{label_prefix}{idx}:{item.get('type', '')} {evidence}->{correction}"
        draw_label(draw, (x1 + 3, min(y2 + 4, max(36, y1 + 4))), label, fill=outline, font=font)


def header(draw, image, title: str, subtitle: str, font, small_font) -> None:
    draw.rectangle([0, 0, image.width, 62], fill=(255, 255, 255))
    draw.text((10, 7), title, fill=(0, 0, 0), font=font)
    draw.text((10, 35), subtitle, fill=(40, 40, 40), font=small_font)


def render_panel(
    image,
    *,
    title: str,
    subtitle: str,
    word_boxes: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    word_color: tuple[int, int, int],
    error_color: tuple[int, int, int],
    word_label_prefix: str,
    error_label_prefix: str,
):
    from PIL import ImageDraw

    panel = image.copy().convert("RGB")
    draw = ImageDraw.Draw(panel)
    font, small_font = font_pair()
    draw_boxes(
        draw,
        word_boxes,
        outline=word_color,
        width=2,
        font=small_font,
        label_prefix=word_label_prefix,
    )
    draw_errors(
        draw,
        errors,
        outline=error_color,
        font=small_font,
        label_prefix=error_label_prefix,
    )
    header(draw, panel, title, subtitle, font, small_font)
    return panel


def format_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_record(
    *,
    image_path: Path,
    gt_entry: dict[str, Any],
    strategy_name: str,
    record: dict[str, Any],
    output_dir: Path,
) -> Path:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    output = record["output"]
    metrics = record["metrics"]
    gt_words = gt_entry.get("words", [])
    gt_errors = gt_entry.get("errors", [])
    pred_boxes = output.get("boxes", [])
    pred_errors = output.get("errors", [])

    gt_panel = render_panel(
        image,
        title=f"GT: {image_path.name}",
        subtitle=f"words={len(gt_words)} errors={len(gt_errors)}",
        word_boxes=gt_words,
        errors=gt_errors,
        word_color=(0, 150, 70),
        error_color=(235, 160, 0),
        word_label_prefix="G",
        error_label_prefix="GE",
    )
    pred_panel = render_panel(
        image,
        title=f"Prediction: {strategy_name}",
        subtitle=(
            f"valid={metrics.get('parse_valid')} cer={format_metric(metrics.get('cer'))} "
            f"wer={format_metric(metrics.get('wer'))} word_iou={format_metric(metrics.get('word_iou'))} "
            f"errors={metrics.get('error_count')}"
        ),
        word_boxes=pred_boxes,
        errors=pred_errors,
        word_color=(220, 50, 50),
        error_color=(165, 60, 210),
        word_label_prefix="P",
        error_label_prefix="PE",
    )

    gap = 18
    combined = Image.new("RGB", (gt_panel.width * 2 + gap, gt_panel.height), (245, 245, 245))
    combined.paste(gt_panel, (0, 0))
    combined.paste(pred_panel, (gt_panel.width + gap, 0))

    strategy_dir = output_dir / safe_name(strategy_name)
    strategy_dir.mkdir(parents=True, exist_ok=True)
    out_path = strategy_dir / f"{image_path.stem}.png"
    combined.save(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GT)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--strategy", action="append", help="Strategy name to render. Repeatable.")
    parser.add_argument("--image", action="append", help="Image filename to render. Repeatable.")
    args = parser.parse_args()

    data = json.loads(args.result.read_text())
    gt_by_name = load_gt(args.ground_truth)
    strategy_filter = set(args.strategy or [])
    image_filter = set(args.image or [])

    written = []
    for strategy in data.get("strategies", []):
        strategy_name = strategy["name"]
        if strategy_filter and strategy_name not in strategy_filter:
            continue
        for record in strategy.get("images", []):
            image_name = record["image"]
            if image_filter and image_name not in image_filter:
                continue
            image_path = args.image_dir / image_name
            if not image_path.exists() or image_name not in gt_by_name:
                continue
            written.append(render_record(
                image_path=image_path,
                gt_entry=gt_by_name[image_name],
                strategy_name=strategy_name,
                record=record,
                output_dir=args.output_dir,
            ))

    for path in written:
        print(path)
    print(f"Rendered {len(written)} overlay(s).")


if __name__ == "__main__":
    main()
