#!/usr/bin/env python3
"""
Render a captioned Qwen3-VL handwriting OCR demo video from existing artifacts.

This script only reads committed benchmark results and visualization images. It
does not run any model, call any API, or touch cloud services.

Usage:
    source .venv/bin/activate
    python scripts/render_qwen3vl_demo.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "benchmark" / "results"
VIZ_DIR = PROJECT_ROOT / "benchmark" / "visualizations"
HANDWRITTEN_DIR = PROJECT_ROOT / "benchmark" / "test_dataset" / "handwritten"

DEFAULT_OUTPUT = PROJECT_ROOT / "demo" / "qwen3vl_demo.mp4"
DEFAULT_SCRIPT_OUTPUT = PROJECT_ROOT / "demo" / "qwen3vl_demo_script.md"

BG = (15, 18, 23)
PANEL = (27, 32, 39)
PANEL_2 = (36, 42, 50)
TEXT = (240, 244, 248)
MUTED = (173, 184, 196)
SUBTLE = (91, 103, 116)
GREEN = (70, 216, 159)
RED = (255, 92, 92)
BLUE = (85, 166, 255)
AMBER = (245, 184, 86)
WHITE = (255, 255, 255)

RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS


@dataclass(frozen=True)
class Scene:
    key: str
    duration: float
    title: str
    caption: str
    narration: str


@dataclass
class Assets:
    width: int
    height: int
    fps: int
    qwen8: dict
    qwen4_line: dict
    qwen4_word: dict
    docai: dict
    tesseract: dict
    images: dict[str, Image.Image]
    fonts: dict[str, ImageFont.FreeTypeFont]


SCENES = [
    Scene(
        key="problem",
        duration=20.0,
        title="Qwen3-VL for handwriting OCR + localization",
        caption=(
            "The task is not just reading handwriting. The feedback pipeline also "
            "needs word locations, so every detected error can point back to the page."
        ),
        narration=(
            "This demo shows Qwen3-VL on the handwriting problem that matters for "
            "essay feedback: transcription plus word-level localization. A useful "
            "system needs both the text and the boxes."
        ),
    ),
    Scene(
        key="evaluation",
        duration=25.0,
        title="Evaluation setup: crop away the easy text",
        caption=(
            "Full IAM forms contain the same text twice. The benchmark uses only "
            "the handwritten crop, so the model cannot win by reading the printed prompt."
        ),
        narration=(
            "The benchmark avoids a common trap. IAM evaluation forms include a "
            "machine-printed prompt and a handwritten copy of the same text. These "
            "numbers come from cropped handwritten regions only."
        ),
    ),
    Scene(
        key="metrics",
        duration=40.0,
        title="Main result: lower CER and stronger word boxes",
        caption=(
            "Qwen3-VL-8B reaches CER 0.035 and word IoU 0.722 on 25 handwritten "
            "crops, beating the Google Document AI baseline on both metrics."
        ),
        narration=(
            "On the 25-image handwritten benchmark, Qwen3-VL-8B is the best "
            "automatable result: zero point zero three five CER and zero point "
            "seven two two word IoU. Google Document AI is at zero point one zero "
            "eight CER and zero point six one one word IoU."
        ),
    ),
    Scene(
        key="visual",
        duration=30.0,
        title="Visual proof: predicted boxes line up with words",
        caption=(
            "Green is ground truth. Red is the model prediction. Qwen's red boxes "
            "track the handwritten words tightly enough for downstream error feedback."
        ),
        narration=(
            "Here is the same handwritten page with word boxes overlaid. Green is "
            "ground truth and red is prediction. The point is not a pretty overlay; "
            "it is auditable localization for later feedback."
        ),
    ),
    Scene(
        key="range",
        duration=25.0,
        title="Honest range: best, median, and worst examples",
        caption=(
            "The result is strong, but not magic. The demo includes a best case, "
            "a median page, and the highest-CER Qwen3-VL-8B sample."
        ),
        narration=(
            "The benchmark is not cherry-picked. This section shows a clean page, "
            "a median page, and the highest-error Qwen3-VL-8B example so the range "
            "of behavior is visible."
        ),
    ),
    Scene(
        key="conclusion",
        duration=25.0,
        title="Practical conclusion",
        caption=(
            "Qwen3-VL is the only evaluated approach that combines strong "
            "handwriting transcription with usable word localization."
        ),
        narration=(
            "The practical conclusion is that Qwen3-VL is the only evaluated "
            "candidate that gives strong handwriting transcription and usable "
            "word boxes. The four billion parameter model runs locally; the eight "
            "billion result here was evaluated through an API because local INT4 "
            "is blocked on this Blackwell setup."
        ),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Qwen3-VL demo video from existing benchmark artifacts."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--script-output", type=Path, default=DEFAULT_SCRIPT_OUTPUT)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--resolution", default="1920x1080")
    parser.add_argument(
        "--no-debug-frames",
        action="store_true",
        help="Do not write representative PNG frames to demo/frames.",
    )
    return parser.parse_args()


def parse_resolution(value: str) -> tuple[int, int]:
    try:
        w_text, h_text = value.lower().split("x", 1)
        width, height = int(w_text), int(h_text)
    except ValueError as exc:
        raise SystemExit(f"Invalid --resolution '{value}'. Use WIDTHxHEIGHT.") from exc
    if width < 640 or height < 360:
        raise SystemExit("--resolution must be at least 640x360.")
    return width, height


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact is missing: {path}")
    return path


def load_json(path: Path) -> dict:
    require_file(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_image(path: Path) -> Image.Image:
    require_file(path)
    return Image.open(path).convert("RGB")


def load_font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    if mono:
        candidates = [
            "/usr/share/fonts/truetype/noto/NotoSansMono-Bold.ttf" if bold else
            "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf" if bold else
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/lato/Lato-Bold.ttf" if bold else
            "/usr/share/fonts/truetype/lato/Lato-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def load_assets(width: int, height: int, fps: int) -> Assets:
    images = {
        "original_a04": load_image(HANDWRITTEN_DIR / "a04-039.png"),
        "qwen_a04": load_image(VIZ_DIR / "qwen3vl_8b_wordlevel" / "a04-039_bboxes.png"),
        "qwen_f07": load_image(VIZ_DIR / "qwen3vl_8b_wordlevel" / "f07-069_bboxes.png"),
        "qwen_r02": load_image(VIZ_DIR / "qwen3vl_8b_wordlevel" / "r02-117_bboxes.png"),
        "docai_a04": load_image(VIZ_DIR / "docai_wordlevel" / "a04-039_bboxes.png"),
    }
    fonts = {
        "display": load_font(round(height * 0.065), bold=True),
        "h1": load_font(round(height * 0.047), bold=True),
        "h2": load_font(round(height * 0.034), bold=True),
        "body": load_font(round(height * 0.026)),
        "body_bold": load_font(round(height * 0.026), bold=True),
        "small": load_font(round(height * 0.020)),
        "small_bold": load_font(round(height * 0.020), bold=True),
        "mono": load_font(round(height * 0.024), mono=True),
        "metric": load_font(round(height * 0.052), bold=True),
        "caption": load_font(round(height * 0.026), bold=True),
    }
    return Assets(
        width=width,
        height=height,
        fps=fps,
        qwen8=load_json(RESULTS_DIR / "qwen3_vl_8b_wordlevel_handwritten.json"),
        qwen4_line=load_json(RESULTS_DIR / "qwen3_vl_4b_handwritten.json"),
        qwen4_word=load_json(RESULTS_DIR / "qwen3_vl_4b_wordlevel_handwritten.json"),
        docai=load_json(RESULTS_DIR / "google_docai_wordlevel_handwritten.json"),
        tesseract=load_json(RESULTS_DIR / "tesseract_wordlevel_handwritten.json"),
        images=images,
        fonts=fonts,
    )


def qwen8_image_stats(assets: Assets, image_name: str) -> dict:
    for item in assets.qwen8["images"]:
        if item["image"] == image_name:
            return item
    raise KeyError(f"Image metrics not found for {image_name}")


def metric_text(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def seconds_text(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}s"


def ease(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def lerp(a: float, b: float, value: float) -> float:
    return a + (b - a) * value


def text_bbox(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font) -> tuple[int, int, int, int]:
    return draw.textbbox(xy, text, font=font)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        trial = " ".join(current + [word])
        if text_bbox(draw, (0, 0), trial, font)[2] <= max_width or not current:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font,
    fill: tuple[int, int, int],
    max_width: int,
    line_gap: int = 8,
) -> int:
    x, y = xy
    for line in wrap_text(draw, text, font, max_width):
        draw.text((x, y), line, font=font, fill=fill)
        bbox = text_bbox(draw, (x, y), line, font)
        y += bbox[3] - bbox[1] + line_gap
    return y


def draw_title(canvas: Image.Image, title: str, subtitle: str | None = None) -> None:
    draw = ImageDraw.Draw(canvas, "RGBA")
    margin = round(canvas.width * 0.04)
    draw.text((margin, round(canvas.height * 0.045)), title, font=FONT["h1"], fill=TEXT)
    if subtitle:
        draw_wrapped(
            draw,
            (margin, round(canvas.height * 0.105)),
            subtitle,
            FONT["small"],
            MUTED,
            round(canvas.width * 0.78),
            line_gap=6,
        )


def draw_caption(canvas: Image.Image, text: str) -> None:
    draw = ImageDraw.Draw(canvas, "RGBA")
    margin = round(canvas.width * 0.04)
    box_h = round(canvas.height * 0.125)
    x1 = margin
    y1 = canvas.height - box_h - round(canvas.height * 0.035)
    x2 = canvas.width - margin
    y2 = canvas.height - round(canvas.height * 0.035)
    draw.rectangle((x1, y1, x2, y2), fill=(7, 9, 12, 224), outline=(64, 75, 88, 235), width=2)
    draw_wrapped(
        draw,
        (x1 + 28, y1 + 22),
        text,
        FONT["caption"],
        TEXT,
        x2 - x1 - 56,
        line_gap=8,
    )


def draw_progress(canvas: Image.Image, elapsed: float, total: float) -> None:
    draw = ImageDraw.Draw(canvas, "RGBA")
    margin = round(canvas.width * 0.04)
    y = canvas.height - 16
    draw.rectangle((margin, y, canvas.width - margin, y + 4), fill=(55, 64, 76, 180))
    progress = max(0.0, min(1.0, elapsed / total))
    draw.rectangle((margin, y, margin + int((canvas.width - 2 * margin) * progress), y + 4), fill=BLUE)


def base_canvas() -> Image.Image:
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(canvas, "RGBA")
    stripe_w = max(1, CANVAS_W // 24)
    for i in range(0, CANVAS_W, stripe_w * 2):
        draw.rectangle((i, 0, i + stripe_w, CANVAS_H), fill=(255, 255, 255, 5))
    return canvas


def panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill=PANEL, outline=(66, 78, 92)) -> None:
    draw.rectangle(box, fill=fill + (238,), outline=outline + (230,), width=2)


def crop_to_aspect(
    image: Image.Image,
    aspect: float,
    center: tuple[float, float] = (0.5, 0.5),
    scale: float = 1.0,
) -> Image.Image:
    iw, ih = image.size
    scale = max(0.08, min(1.0, scale))
    if iw / ih > aspect:
        crop_h = ih * scale
        crop_w = crop_h * aspect
        if crop_w > iw:
            crop_w = iw * scale
            crop_h = crop_w / aspect
    else:
        crop_w = iw * scale
        crop_h = crop_w / aspect
        if crop_h > ih:
            crop_h = ih * scale
            crop_w = crop_h * aspect
    cx = center[0] * iw
    cy = center[1] * ih
    left = max(0, min(iw - crop_w, cx - crop_w / 2))
    top = max(0, min(ih - crop_h, cy - crop_h / 2))
    return image.crop((int(left), int(top), int(left + crop_w), int(top + crop_h)))


def paste_image(
    canvas: Image.Image,
    image: Image.Image,
    box: tuple[int, int, int, int],
    mode: str = "contain",
    center: tuple[float, float] = (0.5, 0.5),
    scale: float = 1.0,
    border: tuple[int, int, int] | None = None,
) -> None:
    draw = ImageDraw.Draw(canvas, "RGBA")
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    draw.rectangle(box, fill=(245, 247, 250, 255))
    if mode == "cover":
        source = crop_to_aspect(image, bw / bh, center=center, scale=scale)
        resized = source.resize((bw, bh), RESAMPLE)
        canvas.paste(resized, (x1, y1))
    else:
        iw, ih = image.size
        factor = min(bw / iw, bh / ih) * scale
        rw, rh = max(1, int(iw * factor)), max(1, int(ih * factor))
        resized = image.resize((rw, rh), RESAMPLE)
        px = x1 + (bw - rw) // 2
        py = y1 + (bh - rh) // 2
        canvas.paste(resized, (px, py))
    if border:
        draw.rectangle(box, outline=border + (255,), width=4)


def draw_image_card(
    canvas: Image.Image,
    title: str,
    subtitle: str,
    image: Image.Image,
    box: tuple[int, int, int, int],
    accent: tuple[int, int, int],
    mode: str = "contain",
    center: tuple[float, float] = (0.5, 0.5),
    scale: float = 1.0,
) -> None:
    draw = ImageDraw.Draw(canvas, "RGBA")
    x1, y1, x2, y2 = box
    panel(draw, box, fill=PANEL_2, outline=accent)
    pad = 18
    draw.text((x1 + pad, y1 + pad), title, font=FONT["small_bold"], fill=TEXT)
    draw.text((x1 + pad, y1 + pad + 34), subtitle, font=FONT["small"], fill=MUTED)
    image_box = (x1 + pad, y1 + 76, x2 - pad, y2 - pad)
    paste_image(canvas, image, image_box, mode=mode, center=center, scale=scale, border=accent)


def draw_metric_card(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    label: str,
    value: str,
    detail: str,
    accent: tuple[int, int, int],
) -> None:
    draw = ImageDraw.Draw(canvas, "RGBA")
    panel(draw, box, fill=PANEL_2, outline=accent)
    x1, y1, x2, y2 = box
    draw.rectangle((x1, y1, x1 + 8, y2), fill=accent + (255,))
    draw.text((x1 + 26, y1 + 22), label, font=FONT["small_bold"], fill=TEXT)
    draw.text((x1 + 26, y1 + 74), value, font=FONT["metric"], fill=accent)
    draw_wrapped(draw, (x1 + 26, y1 + 150), detail, FONT["small"], MUTED, x2 - x1 - 52, line_gap=5)


def draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: tuple[int, int, int]) -> None:
    draw.line((start, end), fill=color + (255,), width=4)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    size = 13
    points = [
        end,
        (int(end[0] - size * math.cos(angle - 0.45)), int(end[1] - size * math.sin(angle - 0.45))),
        (int(end[0] - size * math.cos(angle + 0.45)), int(end[1] - size * math.sin(angle + 0.45))),
    ]
    draw.polygon(points, fill=color + (255,))


def draw_bullet(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, color: tuple[int, int, int], max_width: int) -> int:
    x, y = xy
    draw.ellipse((x, y + 10, x + 10, y + 20), fill=color + (255,))
    return draw_wrapped(draw, (x + 24, y), text, FONT["body"], TEXT, max_width - 24, line_gap=8) + 8


def render_problem(local_t: float, scene: Scene, assets: Assets) -> Image.Image:
    canvas = base_canvas()
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.text((76, 64), "Qwen3-VL", font=FONT["display"], fill=TEXT)
    draw.text((80, 145), "handwriting OCR + word localization", font=FONT["h1"], fill=MUTED)

    pipeline = [
        ("Handwritten image", BLUE, (86, 285, 396, 385)),
        ("Text transcript", GREEN, (520, 285, 830, 385)),
        ("Word boxes", AMBER, (86, 425, 396, 525)),
        ("Feedback target", RED, (520, 425, 830, 525)),
    ]
    for label, accent, box in pipeline:
        panel(draw, box, fill=PANEL_2, outline=accent)
        draw.text((box[0] + 22, box[1] + 33), label, font=FONT["small_bold"], fill=TEXT)
    draw_arrow(draw, (404, 335), (510, 335), SUBTLE)
    draw_arrow(draw, (675, 392), (675, 418), SUBTLE)
    draw_arrow(draw, (404, 475), (510, 475), SUBTLE)

    callout = (84, 590, 900, 835)
    panel(draw, callout, fill=PANEL)
    draw.text((112, 620), "What the benchmark measures", font=FONT["h2"], fill=TEXT)
    yy = 685
    yy = draw_bullet(draw, (118, yy), "Character error rate (CER) on handwritten text.", GREEN, 720)
    yy = draw_bullet(draw, (118, yy), "Word-level IoU against IAM XML word boxes.", AMBER, 720)
    draw_bullet(draw, (118, yy), "Reading order for downstream sentence feedback.", BLUE, 720)

    image_box = (1010, 230, 1810, 780)
    zoom = 1.0 - 0.12 * ease(local_t / scene.duration)
    paste_image(canvas, assets.images["original_a04"], image_box, mode="cover", center=(0.5, 0.36), scale=zoom, border=BLUE)
    draw.text((1010, 180), "Input crop: handwritten region only", font=FONT["small_bold"], fill=TEXT)
    draw.text((1010, 810), "No printed prompt is visible in the evaluation crop.", font=FONT["small"], fill=MUTED)
    draw_caption(canvas, scene.caption)
    return canvas


def render_evaluation(local_t: float, scene: Scene, assets: Assets) -> Image.Image:
    canvas = base_canvas()
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw_title(canvas, scene.title, "Same text appears in printed and handwritten regions; only the handwritten region counts.")

    form = (110, 210, 840, 760)
    panel(draw, form, fill=(248, 248, 245), outline=SUBTLE)
    bands = [
        ("Header", 0.13, (225, 231, 238)),
        ("Printed prompt (excluded)", 0.27, (255, 236, 217)),
        ("Handwritten copy (benchmark crop)", 0.43, (222, 247, 238)),
        ("Footer/signature", 0.17, (226, 231, 238)),
    ]
    y = form[1]
    for label, frac, color in bands:
        h = round((form[3] - form[1]) * frac)
        draw.rectangle((form[0], y, form[2], y + h), fill=color + (255,), outline=(125, 132, 140, 255), width=2)
        fill = (80, 72, 64) if "Printed" in label else (32, 46, 42)
        draw.text((form[0] + 24, y + max(18, h // 2 - 16)), label, font=FONT["small_bold"], fill=fill)
        y += h
    draw.line((form[0] + 10, 330, form[2] - 10, 330), fill=RED + (255,), width=5)
    draw.text((128, 785), "Printed text is physically removed before scoring.", font=FONT["small"], fill=MUTED)

    legend_x = 995
    draw.text((legend_x, 182), "Overlay legend", font=FONT["h2"], fill=TEXT)
    draw.rectangle((legend_x, 246, legend_x + 54, 290), outline=GREEN + (255,), width=5)
    draw.text((legend_x + 76, 250), "Green: ground truth word box", font=FONT["body"], fill=TEXT)
    draw.rectangle((legend_x, 318, legend_x + 54, 362), outline=RED + (255,), width=5)
    draw.text((legend_x + 76, 322), "Red: model-predicted word box", font=FONT["body"], fill=TEXT)

    crop_scale = 0.88 - 0.18 * ease(max(0.0, local_t - 8.0) / 12.0)
    draw_image_card(
        canvas,
        "Qwen3-VL-8B word-level overlay",
        "a04-039.png, cropped handwriting",
        assets.images["qwen_a04"],
        (970, 420, 1810, 820),
        GREEN,
        mode="cover",
        center=(0.53, 0.31),
        scale=crop_scale,
    )
    draw_caption(canvas, scene.caption)
    return canvas


def draw_bar(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    label: str,
    value: float,
    max_value: float,
    width: int,
    color: tuple[int, int, int],
    suffix: str,
    progress: float,
) -> None:
    draw.text((x, y - 4), label, font=FONT["small_bold"], fill=TEXT)
    draw.rectangle((x + 210, y, x + 210 + width, y + 32), fill=(52, 60, 70, 255))
    fill_w = int(width * min(1.0, value / max_value) * progress)
    draw.rectangle((x + 210, y, x + 210 + fill_w, y + 32), fill=color + (255,))
    draw.text((x + 230 + width, y - 2), f"{value:.3f}{suffix}", font=FONT["small"], fill=MUTED)


def render_metrics(local_t: float, scene: Scene, assets: Assets) -> Image.Image:
    canvas = base_canvas()
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw_title(canvas, scene.title, "All accuracy numbers below are on 25 handwritten-only IAM crops.")
    q8 = assets.qwen8["aggregate"]
    docai = assets.docai["aggregate"]
    tess = assets.tesseract["aggregate"]

    cards_y = 210
    draw_metric_card(
        canvas,
        (88, cards_y, 600, cards_y + 245),
        "Qwen3-VL-8B",
        f"CER {metric_text(q8['cer'])}",
        f"Word IoU {metric_text(q8['word_iou'])}; latency {seconds_text(q8['latency_avg_s'])}; API run.",
        GREEN,
    )
    draw_metric_card(
        canvas,
        (704, cards_y, 1216, cards_y + 245),
        "Google Doc AI",
        f"CER {metric_text(docai['cer'])}",
        f"Word IoU {metric_text(docai['word_iou'])}; latency {seconds_text(docai['latency_avg_s'])}; cloud baseline.",
        BLUE,
    )
    draw_metric_card(
        canvas,
        (1320, cards_y, 1832, cards_y + 245),
        "Tesseract 5",
        f"CER {metric_text(tess['cer'])}",
        f"Word IoU {metric_text(tess['word_iou'])}; strong boxes, unusable handwriting text.",
        AMBER,
    )

    chart = (110, 520, 1810, 835)
    panel(draw, chart, fill=PANEL)
    draw.text((140, 550), "Metric contrast", font=FONT["h2"], fill=TEXT)
    progress = ease(local_t / 10.0)
    draw.text((140, 612), "CER (lower is better)", font=FONT["body_bold"], fill=MUTED)
    draw_bar(draw, 150, 660, "Qwen3-VL-8B", q8["cer"], 0.45, 460, GREEN, "", progress)
    draw_bar(draw, 150, 710, "Google Doc AI", docai["cer"], 0.45, 460, BLUE, "", progress)
    draw_bar(draw, 150, 760, "Tesseract", tess["cer"], 0.45, 460, AMBER, "", progress)

    draw.text((980, 612), "Word IoU (higher is better)", font=FONT["body_bold"], fill=MUTED)
    draw_bar(draw, 990, 660, "Qwen3-VL-8B", q8["word_iou"], 1.0, 460, GREEN, "", progress)
    draw_bar(draw, 990, 710, "Google Doc AI", docai["word_iou"], 1.0, 460, BLUE, "", progress)
    draw_bar(draw, 990, 760, "Tesseract", tess["word_iou"], 1.0, 460, AMBER, "", progress)

    ratio = docai["cer"] / q8["cer"]
    draw.text((122, 858), f"Qwen CER is {ratio:.1f}x lower than Google Doc AI on this handwriting-only benchmark.", font=FONT["body_bold"], fill=TEXT)
    draw_caption(canvas, scene.caption)
    return canvas


def render_visual(local_t: float, scene: Scene, assets: Assets) -> Image.Image:
    canvas = base_canvas()
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw_title(canvas, scene.title, "Same page, same ground truth, different model overlays.")
    zoom_phase = ease(max(0.0, local_t - 10.0) / 16.0)
    scale = lerp(1.0, 0.48, zoom_phase)
    center = (0.52, lerp(0.50, 0.30, zoom_phase))
    mode = "contain" if scale > 0.94 else "cover"

    left = (80, 205, 930, 825)
    right = (990, 205, 1840, 825)
    draw_image_card(
        canvas,
        "Qwen3-VL-8B",
        "CER 0.0237 on this page; word IoU 0.6678",
        assets.images["qwen_a04"],
        left,
        GREEN,
        mode=mode,
        center=center,
        scale=scale,
    )
    draw_image_card(
        canvas,
        "Google Document AI",
        "Cloud baseline overlay for the same crop",
        assets.images["docai_a04"],
        right,
        BLUE,
        mode=mode,
        center=center,
        scale=scale,
    )
    draw.text((770, 852), "Green = ground truth    Red = prediction", font=FONT["small_bold"], fill=MUTED)
    draw_caption(canvas, scene.caption)
    return canvas


def render_range(local_t: float, scene: Scene, assets: Assets) -> Image.Image:
    canvas = base_canvas()
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw_title(canvas, scene.title, "Chosen from Qwen3-VL-8B per-image metrics.")

    samples = [
        ("Best CER", "f07-069.png", "qwen_f07", GREEN),
        ("Median-ish", "a04-039.png", "qwen_a04", BLUE),
        ("Highest CER", "r02-117.png", "qwen_r02", AMBER),
    ]
    x = 72
    card_w = 560
    card_h = 625
    reveal = ease(local_t / 8.0)
    for idx, (label, image_name, image_key, accent) in enumerate(samples):
        stats = qwen8_image_stats(assets, image_name)
        box = (x + idx * 610, 210, x + idx * 610 + card_w, 210 + card_h)
        draw_image_card(
            canvas,
            label,
            f"{image_name}  CER {stats['cer']:.4f}  IoU {stats['word_iou']:.4f}",
            assets.images[image_key],
            box,
            accent,
            mode="cover",
            center=(0.52, 0.33),
            scale=lerp(0.92, 0.62, reveal),
        )
    draw.text((110, 865), "The worst sample is still shown because technical demos should expose the edge of the envelope.", font=FONT["body_bold"], fill=TEXT)
    draw_caption(canvas, scene.caption)
    return canvas


def render_conclusion(local_t: float, scene: Scene, assets: Assets) -> Image.Image:
    canvas = base_canvas()
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw_title(canvas, scene.title, "What this means for the essay feedback pipeline.")
    q8 = assets.qwen8["aggregate"]
    q4_line = assets.qwen4_line
    q4_word = assets.qwen4_word["aggregate"]

    image_box = (1030, 205, 1818, 790)
    paste_image(
        canvas,
        assets.images["qwen_a04"],
        image_box,
        mode="cover",
        center=(0.52, 0.32),
        scale=0.62,
        border=GREEN,
    )
    draw.rectangle(image_box, fill=(15, 18, 23, 72))

    callout = (90, 205, 940, 790)
    panel(draw, callout, fill=PANEL)
    y = 250
    y = draw_bullet(draw, (132, y), "Qwen3-VL combines readable handwriting transcription with usable word-level localization.", GREEN, 740)
    y = draw_bullet(
        draw,
        (132, y),
        f"8B API result: CER {q8['cer']:.3f}, word IoU {q8['word_iou']:.3f}, latency {q8['latency_avg_s']:.1f}s.",
        BLUE,
        740,
    )
    y = draw_bullet(
        draw,
        (132, y),
        f"4B local result: line CER {q4_line['cer']:.3f}; word IoU {q4_word['word_iou']:.3f}.",
        AMBER,
        740,
    )
    y = draw_bullet(
        draw,
        (132, y),
        "8B local INT4 is blocked here by bitsandbytes support for Blackwell sm_120.",
        RED,
        740,
    )
    draw_bullet(
        draw,
        (132, y),
        "This video uses existing artifacts only: no live model run, no cloud call, no API cost.",
        GREEN,
        740,
    )
    draw_wrapped(
        draw,
        (1090, 820),
        "Next benchmark phase: end-to-end error detection and feedback.",
        FONT["body_bold"],
        TEXT,
        680,
        line_gap=8,
    )
    draw_caption(canvas, scene.caption)
    return canvas


RENDERERS: dict[str, Callable[[float, Scene, Assets], Image.Image]] = {
    "problem": render_problem,
    "evaluation": render_evaluation,
    "metrics": render_metrics,
    "visual": render_visual,
    "range": render_range,
    "conclusion": render_conclusion,
}


def make_writer(output: Path, fps: int, width: int, height: int) -> cv2.VideoWriter:
    output.parent.mkdir(parents=True, exist_ok=True)
    codecs = ["mp4v", "avc1", "H264"]
    for codec in codecs:
        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(
        "OpenCV could not open an MP4 writer. Tried codecs: mp4v, avc1, H264."
    )


def pil_to_bgr(frame: Image.Image) -> np.ndarray:
    rgb = np.asarray(frame.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def write_script(path: Path, scenes: list[Scene], assets: Assets) -> None:
    total = sum(scene.duration for scene in scenes)
    q8 = assets.qwen8["aggregate"]
    docai = assets.docai["aggregate"]
    lines = [
        "# Qwen3-VL Demo Narration Script",
        "",
        f"Total runtime: {format_time(total)}",
        "",
        "## Source metrics",
        "",
        f"- Qwen3-VL-8B word-level: CER {q8['cer']:.3f}, word IoU {q8['word_iou']:.3f}, latency {q8['latency_avg_s']:.1f}s.",
        f"- Google Document AI word-level: CER {docai['cer']:.3f}, word IoU {docai['word_iou']:.3f}, latency {docai['latency_avg_s']:.1f}s.",
        "- Evidence source: existing benchmark JSON and visualization PNG artifacts only.",
        "- No live model run, no cloud API call, no generated voice audio.",
        "",
        "## Timed narration",
        "",
    ]
    cursor = 0.0
    for scene in scenes:
        start = cursor
        end = cursor + scene.duration
        lines.extend(
            [
                f"### {format_time(start)}-{format_time(end)} - {scene.title}",
                "",
                scene.narration,
                "",
                f"On-screen caption: {scene.caption}",
                "",
            ]
        )
        cursor = end
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(round(seconds - minutes * 60))
    return f"{minutes:02d}:{secs:02d}"


def save_debug_frame(frame: Image.Image, frame_dir: Path, index: int, scene: Scene) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    safe = scene.key.replace("_", "-")
    frame.save(frame_dir / f"{index:02d}_{safe}.png")


def render_video(args: argparse.Namespace) -> None:
    global CANVAS_W, CANVAS_H, FONT
    CANVAS_W, CANVAS_H = parse_resolution(args.resolution)
    if args.fps <= 0:
        raise SystemExit("--fps must be positive.")
    assets = load_assets(CANVAS_W, CANVAS_H, args.fps)
    FONT = assets.fonts

    total_duration = sum(scene.duration for scene in SCENES)
    total_frames = int(round(total_duration * args.fps))
    writer = make_writer(args.output, args.fps, CANVAS_W, CANVAS_H)
    frame_dir = args.output.parent / "frames"

    print(f"Rendering {args.output}")
    print(f"Resolution: {CANVAS_W}x{CANVAS_H}, fps: {args.fps}, duration: {format_time(total_duration)}")

    elapsed = 0.0
    written = 0
    try:
        for scene_index, scene in enumerate(SCENES, start=1):
            renderer = RENDERERS[scene.key]
            scene_frames = int(round(scene.duration * args.fps))
            for frame_index in range(scene_frames):
                local_t = frame_index / args.fps
                frame = renderer(local_t, scene, assets)
                draw_progress(frame, elapsed + local_t, total_duration)
                if not args.no_debug_frames and frame_index == 0:
                    save_debug_frame(frame, frame_dir, scene_index, scene)
                writer.write(pil_to_bgr(frame))
                written += 1
            elapsed += scene.duration
            print(f"  {scene_index}/{len(SCENES)} {scene.key}: {scene.duration:.0f}s")
    finally:
        writer.release()

    if written != total_frames:
        print(f"Warning: expected {total_frames} frames, wrote {written} frames.", file=sys.stderr)
    write_script(args.script_output, SCENES, assets)
    print(f"Wrote video: {args.output}")
    print(f"Wrote script: {args.script_output}")
    if not args.no_debug_frames:
        print(f"Wrote debug frames: {frame_dir}")


def main() -> None:
    try:
        render_video(parse_args())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
