#!/usr/bin/env python3
"""Phase 4 architecture benchmark: single VLM vs two-stage pipeline.

Default use loads Qwen3-VL-4B and runs a mix-and-match local matrix on a small
subset: saved Stage 1 text artifacts crossed with explicit word-box sources,
then a local Qwen grader, plus the single-pass Qwen arm. Use `--smoke-fake`
for a fast no-model validation run.

Examples:
    python scripts/benchmark_phase4.py --smoke-fake --max-images 1 --positive-controls
    python scripts/benchmark_phase4.py --text-sources qwen3vl_4b_wordlevel florence2_large_wordlevel --box-sources tesseract_word_boxes --max-images 5
    python scripts/benchmark_phase4.py --strategies two_stage__qwen3vl_4b_wordlevel__tesseract_word_boxes__qwen3vl_4b_grader --max-images 25
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.contracts import PipelineOutput, TextBox
from pipeline.metrics import aggregate_phase4_metrics, evaluate_phase4_output
from pipeline.model_registry import (
    automated_artifact_text_sources,
    live_local_text_sources,
    registry_metadata,
)
from pipeline.prompts import VERBATIM_WORD_OCR_PROMPT
from pipeline.strategies import SinglePassStrategy, TwoStageStrategy


HANDWRITTEN_DIR = PROJECT_ROOT / "benchmark" / "test_dataset" / "handwritten"
GROUND_TRUTH_PATH = PROJECT_ROOT / "benchmark" / "test_dataset" / "ground_truth_handwritten.json"
WORD_GROUND_TRUTH_PATH = PROJECT_ROOT / "benchmark" / "test_dataset" / "ground_truth_wordlevel.json"
REALWORLD_RAW_DIR = PROJECT_ROOT / "benchmark" / "test_dataset" / "realworld_raw"
REALWORLD_GROUND_TRUTH_PATH = PROJECT_ROOT / "benchmark" / "test_dataset" / "realworld_writing_errors.json"
RESULTS_DIR = PROJECT_ROOT / "benchmark" / "results"
QWEN4B_WORDLEVEL_RESULT = PROJECT_ROOT / "benchmark" / "results" / "qwen3_vl_4b_wordlevel_handwritten.json"
POSITIVE_CONTROLS_PATH = PROJECT_ROOT / "benchmark" / "test_dataset" / "phase4_positive_controls.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "benchmark" / "results" / "phase4_pipeline.json"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "pipeline_output" / "phase4_cache"

QWEN_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
QWEN_BBOX_PATTERN = re.compile(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\s*")

STAGE1_TEXT_SOURCES = {
    "realworld_source_text": REALWORLD_GROUND_TRUTH_PATH,
    **automated_artifact_text_sources(PROJECT_ROOT),
}

LIVE_STAGE1_TEXT_SOURCES = set(live_local_text_sources())

TEXT_SOURCE_ALIASES = {"all_live_local", "all_automated_artifacts", "all_available"}
TEXT_SOURCE_CHOICES = set(STAGE1_TEXT_SOURCES) | LIVE_STAGE1_TEXT_SOURCES | TEXT_SOURCE_ALIASES

STAGE1_BOX_SOURCES = {
    "qwen3vl_4b_word_boxes",
    "realworld_aligned_words",
    "same_stage1_boxes",
    "tesseract_word_boxes",
    "no_boxes",
}

LOCAL_GRADERS = {
    "qwen3vl_4b_grader",
}

END_TO_END_STRATEGIES = {
    "single_qwen3vl_4b_e2e",
}

LEGACY_STRATEGY_ALIASES = {
    "two_stage_qwen3vl_4b": "two_stage__qwen3vl_4b_wordlevel__qwen3vl_4b_word_boxes__qwen3vl_4b_grader",
    "two_stage_qwen_text_tesseract_boxes": "two_stage__qwen3vl_4b_wordlevel__tesseract_word_boxes__qwen3vl_4b_grader",
}

CLOUD_STRATEGIES = {
    "single_qwen3vl_8b_api_e2e",
    "two_stage_docai_gemini",
}


QWEN_WORD_OCR_PROMPT = (
    "Transcribe every individual word in this handwritten text. "
    "For each word, output exactly: [x1, y1, x2, y2] the_word "
    "One word per output line. Preserve reading order. Do not group words together. "
    "Do not output boxes without their word labels."
)

QWEN_VERBATIM_WORD_CACHE_KEY = "qwen3vl_4b_verbatim_word"

CANDIDATE_LIVE_OCR_MODULES = {
    "florence2_live_region_ocr": "candidates.florence2.eval",
    "got_ocr2_live_ocr": "candidates.got_ocr.eval",
    "smoldocling_live_ocr": "candidates.smoldocling.eval",
    "nemotron_ocr_v2_live_ocr": "candidates.nemotron_ocr.eval",
    "paddleocr_vl_live_ocr": "candidates.paddleocr_vl.eval",
    "monkeyocr_live_ocr": "candidates.monkeyocr.eval",
}

TROCR_LIVE_SOURCES = {
    "trocr_base_live_line_ocr": ("base", "microsoft/trocr-base-handwritten"),
    "trocr_large_live_line_ocr": ("large", "microsoft/trocr-large-handwritten"),
}


@dataclass(frozen=True)
class TwoStageSpec:
    """One Phase 4 assembly point: Stage 1 text + boxes, then a grader."""

    text_source: str
    box_source: str
    grader: str

    @property
    def name(self) -> str:
        return f"two_stage__{self.text_source}__{self.box_source}__{self.grader}"


class Qwen3VLAdapter:
    """Small local adapter around transformers for Qwen3-VL-4B."""

    def __init__(self, model_id: str = QWEN_MODEL_ID, max_new_tokens: int = 4096):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.processor = None

    def _load(self) -> None:
        if self.model is not None and self.processor is not None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        print(f"[phase4] Loading {self.model_id} ...")
        model_kwargs = {
            "dtype": "auto",
            "device_map": "auto",
        }
        processor_kwargs = {"use_fast": True}
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                local_files_only=True,
                **model_kwargs,
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                local_files_only=True,
                **processor_kwargs,
            )
        except OSError:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                **model_kwargs,
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                **processor_kwargs,
            )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        print("[phase4] Model loaded.")

    def generate(
        self,
        prompt: str,
        image_path: Path | None,
        max_new_tokens: int | None = None,
    ) -> str:
        self._load()

        import torch
        from PIL import Image

        content: list[dict[str, Any]] = []
        if image_path is not None:
            image = Image.open(image_path).convert("RGB")
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]


class Stage1OnlyStrategy:
    """Run only Stage 1 OCR/localization for truthfulness and box diagnostics."""

    def __init__(self, name: str, ocr_call: Callable[[Path], PipelineOutput]):
        self.name = name
        self.ocr_call = ocr_call

    def run(self, image_path: str | Path) -> PipelineOutput:
        path = Path(image_path)
        output = self.ocr_call(path)
        output.strategy_name = self.name
        output.parse_valid = True
        output.stage2_latency = 0.0
        output.total_latency = output.stage1_latency
        output.metadata.update({"mode": "stage1_only"})
        return output


def load_ground_truth(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {entry["image"]: entry for entry in data}


def load_phase3_qwen_wordlevel() -> dict[str, dict[str, Any]]:
    """Load the Phase 3 Qwen word-level artifact as the solved Stage 1 source."""
    if not QWEN4B_WORDLEVEL_RESULT.exists():
        return {}
    data = json.loads(QWEN4B_WORDLEVEL_RESULT.read_text())
    return {entry["image"]: entry for entry in data.get("images", [])}


def load_stage1_artifact(source_name: str) -> dict[str, dict[str, Any]]:
    """Load a prior OCR/localization result artifact by image name."""
    path = STAGE1_TEXT_SOURCES.get(source_name)
    if path is None:
        raise KeyError(f"Unknown Stage 1 text source: {source_name}")
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    entries = data.get("images", []) if isinstance(data, dict) else data
    return {entry["image"]: entry for entry in entries}


def artifact_relpath(source_name: str) -> str:
    path = STAGE1_TEXT_SOURCES[source_name]
    return str(path.relative_to(PROJECT_ROOT))


def is_live_text_source(source_name: str) -> bool:
    return source_name in LIVE_STAGE1_TEXT_SOURCES


def artifact_latency(entry: dict[str, Any]) -> float:
    return float(entry.get("latency_s", entry.get("latency_avg_s", 0.0)) or 0.0)


def dataset_paths(dataset: str) -> tuple[Path, Path, Path]:
    if dataset == "realworld":
        return REALWORLD_RAW_DIR, REALWORLD_GROUND_TRUTH_PATH, REALWORLD_GROUND_TRUTH_PATH
    return HANDWRITTEN_DIR, GROUND_TRUTH_PATH, WORD_GROUND_TRUTH_PATH


def select_images(
    max_images: int,
    image_names: list[str] | None = None,
    image_dir: Path = HANDWRITTEN_DIR,
) -> list[Path]:
    if image_names:
        images = [image_dir / name for name in image_names]
        return [path for path in images if path.exists()]
    else:
        images = sorted([*image_dir.glob("*.png"), *image_dir.glob("*.jpg"), *image_dir.glob("*.jpeg")])
    images = [path for path in images if path.exists()]
    return images[:max_images] if max_images > 0 else images


def get_peak_vram_mb() -> int:
    try:
        import torch
    except ImportError:
        return 0
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.max_memory_allocated(0) // (1024 * 1024))


def parse_qwen_word_response(raw: str, image_path: Path) -> dict[str, Any]:
    from PIL import Image

    with Image.open(image_path) as image:
        img_w, img_h = image.size

    blocks = []
    clean_words = []

    matches = list(QWEN_BBOX_PATTERN.finditer(raw))
    for idx, match in enumerate(matches):
        bx1, by1, bx2, by2 = map(int, match.groups())
        x1 = int(bx1 / 999 * img_w)
        y1 = int(by1 / 999 * img_h)
        x2 = int(bx2 / 999 * img_w)
        y2 = int(by2 / 999 * img_h)
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw)
        text = raw[match.end():next_start].strip()
        if text and not re.search(r"[A-Za-z0-9]", text):
            text = ""
        if text:
            clean_words.append(text)
        blocks.append({"bbox": [x1, y1, x2, y2], "text": text, "confidence": 1.0})

    if not matches:
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            clean_words.append(line)
            blocks.append({"bbox": [0, 0, 0, 0], "text": line, "confidence": 1.0})

    return {
        "text": " ".join(clean_words),
        "blocks": blocks,
        "raw_response": raw,
        "_text_labels_present": bool(clean_words),
    }


def run_qwen_word_ocr(
    adapter: Qwen3VLAdapter,
    image_path: Path,
    prompt: str = QWEN_WORD_OCR_PROMPT,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    raw = adapter.generate(prompt, image_path)
    elapsed = time.perf_counter() - t0
    parsed = parse_qwen_word_response(raw, image_path)
    parsed["stage1_latency"] = elapsed
    return parsed


def run_tesseract_word_boxes(image_path: Path) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        import pytesseract
        from pytesseract import Output
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "pytesseract and Pillow are required for two_stage_qwen_text_tesseract_boxes"
        ) from exc

    with Image.open(image_path) as image:
        data = pytesseract.image_to_data(image, output_type=Output.DICT)

    blocks = []
    words = []
    for i, text in enumerate(data.get("text", [])):
        text = str(text).strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = 0.0
        if conf < 0:
            continue
        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])
        blocks.append({
            "bbox": [x, y, x + w, y + h],
            "text": text,
            "confidence": conf / 100.0,
        })
        words.append(text)

    return {
        "text": " ".join(words),
        "blocks": blocks,
        "stage1_latency": time.perf_counter() - t0,
    }


def cache_file(cache_dir: Path, cache_key: str, image_path: Path) -> Path:
    return cache_dir / cache_key / f"{image_path.stem}.json"


def cached_stage1(
    cache_dir: Path,
    cache_key: str,
    image_path: Path,
    refresh: bool,
    compute_fn,
) -> dict[str, Any]:
    path = cache_file(cache_dir, cache_key, image_path)
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    result = compute_fn(image_path)
    result["_cache_key"] = cache_key
    result["_cached_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def stage1_text_output(
    image_path: Path,
    source_name: str,
    entries: dict[str, dict[str, Any]],
) -> PipelineOutput:
    entry = entries.get(image_path.name, {})
    return PipelineOutput(
        strategy_name=f"{source_name}_text",
        image=image_path.name,
        text=str(entry.get("text", "")),
        stage1_latency=artifact_latency(entry),
        total_latency=artifact_latency(entry),
        metadata={
            "text_source": source_name,
            "text_artifact": artifact_relpath(source_name),
            "text_latency_s": artifact_latency(entry),
            "source_cer": entry.get("cer"),
            "source_wer": entry.get("wer"),
            "source_word_iou": entry.get("word_iou"),
            "source_iou_recall": entry.get("iou_recall"),
            "source_iou_precision": entry.get("iou_precision"),
        },
    )


def no_boxes_output(image_path: Path) -> PipelineOutput:
    return PipelineOutput(
        strategy_name="no_boxes",
        image=image_path.name,
        boxes=[],
        stage1_latency=0.0,
        total_latency=0.0,
        metadata={
            "box_source": "no_boxes",
            "box_latency_s": 0.0,
            "stage1_box_note": "Text-only diagnostic; localization metrics should be poor or not_applicable.",
        },
    )


def qwen_word_box_output(
    image_path: Path,
    adapter: Qwen3VLAdapter,
    args: argparse.Namespace,
    *,
    prompt: str = QWEN_WORD_OCR_PROMPT,
    cache_key: str = "qwen3vl_4b_word",
    strategy_name: str = "qwen3vl_4b_word_boxes",
) -> PipelineOutput:
    data = cached_stage1(
        args.cache_dir,
        cache_key,
        image_path,
        args.refresh_cache,
        lambda path: run_qwen_word_ocr(adapter, path, prompt=prompt),
    )
    if data.get("raw_response"):
        reparsed = parse_qwen_word_response(data["raw_response"], image_path)
        reparsed["stage1_latency"] = data.get("stage1_latency", 0.0)
        reparsed["_cache_key"] = data.get("_cache_key", cache_key)
        reparsed["_cached_at"] = data.get("_cached_at", "")
        data = reparsed
    output = PipelineOutput.from_ocr_dict(
        data,
        strategy_name=strategy_name,
        image=image_path,
    )
    output.metadata.update({
        "box_source": strategy_name,
        "box_latency_s": output.stage1_latency,
    })
    return output


def tesseract_word_box_output(
    image_path: Path,
    args: argparse.Namespace,
) -> PipelineOutput:
    data = cached_stage1(
        args.cache_dir,
        "tesseract_word_boxes",
        image_path,
        args.refresh_cache,
        run_tesseract_word_boxes,
    )
    output = PipelineOutput.from_ocr_dict(
        data,
        strategy_name="tesseract_word_boxes",
        image=image_path,
    )
    output.metadata.update({
        "box_source": "tesseract_word_boxes",
        "box_latency_s": output.stage1_latency,
    })
    return output


def baseline_live_word_output(
    image_path: Path,
    args: argparse.Namespace,
    *,
    source_name: str,
) -> PipelineOutput:
    if source_name == "easyocr_live_word_ocr":
        from candidates.baselines.eval import _easyocr_inference as inference_fn
    elif source_name == "doctr_live_word_ocr":
        from candidates.baselines.eval import _doctr_inference as inference_fn
    else:
        raise KeyError(f"Unknown baseline live OCR source: {source_name}")

    data = cached_stage1(
        args.cache_dir,
        source_name,
        image_path,
        args.refresh_cache,
        lambda path: inference_fn(str(path)),
    )
    output = PipelineOutput.from_ocr_dict(
        data,
        strategy_name=source_name,
        image=image_path,
    )
    output.metadata.update({
        "text_source": source_name,
        "box_source": source_name,
        "box_latency_s": output.stage1_latency,
        "stage1_composition": "live_ocr_text_and_boxes",
        "stage1_latency_policy": "single live OCR pass",
    })
    return output


def run_candidate_live_ocr(source_name: str, image_path: Path) -> dict[str, Any]:
    """Run one existing candidate inference function as a Phase 4 live Stage 1 source."""
    if source_name in TROCR_LIVE_SOURCES:
        variant, model_id = TROCR_LIVE_SOURCES[source_name]
        module = importlib.import_module("candidates.trocr.eval")
        module.MODEL_ID = model_id
        module.CANDIDATE_NAME = source_name
        loaded_source = getattr(module.inference_fn, "_phase4_source_name", None)
        if loaded_source and loaded_source != source_name:
            for attr in ("_model", "_processor"):
                if hasattr(module.inference_fn, attr):
                    delattr(module.inference_fn, attr)
        data = module.inference_fn(str(image_path))
        module.inference_fn._phase4_source_name = source_name
        data.setdefault("_trocr_variant", variant)
        return data

    module_name = CANDIDATE_LIVE_OCR_MODULES.get(source_name)
    if module_name is None:
        raise KeyError(f"Unknown candidate live OCR source: {source_name}")

    try:
        module = importlib.import_module(module_name)
    except SystemExit as exc:
        raise RuntimeError(f"{source_name} is not available in this environment") from exc

    return module.inference_fn(str(image_path))


def candidate_live_ocr_output(
    image_path: Path,
    args: argparse.Namespace,
    *,
    source_name: str,
) -> PipelineOutput:
    data = cached_stage1(
        args.cache_dir,
        source_name,
        image_path,
        args.refresh_cache,
        lambda path: run_candidate_live_ocr(source_name, path),
    )
    output = PipelineOutput.from_ocr_dict(
        data,
        strategy_name=source_name,
        image=image_path,
    )
    output.metadata.update({
        "text_source": source_name,
        "box_source": source_name,
        "box_latency_s": output.stage1_latency,
        "stage1_composition": "live_candidate_text_and_boxes",
        "stage1_latency_policy": "single live candidate OCR pass",
    })
    return output


def make_realworld_aligned_words_call() -> Callable[[Path], PipelineOutput]:
    entries = load_stage1_artifact("realworld_source_text")

    def box_call(image_path: Path) -> PipelineOutput:
        entry = entries.get(image_path.name, {})
        boxes = [
            TextBox.from_dict(word, index=i, source="realworld_aligned_words")
            for i, word in enumerate(entry.get("words", []))
        ]
        return PipelineOutput(
            strategy_name="realworld_aligned_words",
            image=image_path.name,
            boxes=boxes,
            stage1_latency=0.0,
            total_latency=0.0,
            metadata={
                "box_source": "realworld_aligned_words",
                "box_artifact": str(REALWORLD_GROUND_TRUTH_PATH.relative_to(PROJECT_ROOT)),
                "box_latency_s": 0.0,
                "box_annotation_status": entry.get("annotation_status", ""),
                "box_annotation_note": "Provisional auto-aligned boxes; review before authoritative IoU.",
            },
        )

    return box_call


def make_box_call(
    box_source: str,
    args: argparse.Namespace,
    adapter: Qwen3VLAdapter | None,
) -> Callable[[Path], PipelineOutput]:
    if box_source == "qwen3vl_4b_word_boxes":
        if adapter is None:
            raise RuntimeError("qwen3vl_4b_word_boxes requires the local Qwen adapter")
        return lambda image_path: qwen_word_box_output(image_path, adapter, args)
    if box_source == "tesseract_word_boxes":
        return lambda image_path: tesseract_word_box_output(image_path, args)
    if box_source == "realworld_aligned_words":
        return make_realworld_aligned_words_call()
    if box_source == "no_boxes":
        return no_boxes_output
    raise KeyError(f"Unknown Stage 1 box source: {box_source}")


def make_live_ocr_call(
    text_source: str,
    args: argparse.Namespace,
    adapter: Qwen3VLAdapter | None,
) -> Callable[[Path], PipelineOutput]:
    if text_source == "tesseract_live_word_ocr":
        def ocr_call(image_path: Path) -> PipelineOutput:
            output = tesseract_word_box_output(image_path, args)
            output.strategy_name = "tesseract_live_word_ocr"
            output.metadata.update({
                "text_source": text_source,
                "box_source": text_source,
                "stage1_composition": "live_ocr_text_and_boxes",
                "stage1_latency_policy": "single live OCR pass",
            })
            return output

        return ocr_call

    if text_source == "qwen3vl_4b_live_word_ocr":
        if adapter is None:
            raise RuntimeError("qwen3vl_4b_live_word_ocr requires the local Qwen adapter")

        def ocr_call(image_path: Path) -> PipelineOutput:
            output = qwen_word_box_output(image_path, adapter, args)
            output.strategy_name = "qwen3vl_4b_live_word_ocr"
            output.metadata.update({
                "text_source": text_source,
                "box_source": text_source,
                "stage1_composition": "live_ocr_text_and_boxes",
                "stage1_latency_policy": "single live OCR pass",
            })
            return output

        return ocr_call

    if text_source == "qwen3vl_4b_verbatim_word_ocr":
        if adapter is None:
            raise RuntimeError("qwen3vl_4b_verbatim_word_ocr requires the local Qwen adapter")

        def ocr_call(image_path: Path) -> PipelineOutput:
            output = qwen_word_box_output(
                image_path,
                adapter,
                args,
                prompt=VERBATIM_WORD_OCR_PROMPT,
                cache_key=QWEN_VERBATIM_WORD_CACHE_KEY,
                strategy_name="qwen3vl_4b_verbatim_word_ocr",
            )
            output.metadata.update({
                "text_source": text_source,
                "box_source": text_source,
                "stage1_composition": "live_ocr_text_and_boxes",
                "stage1_latency_policy": "single live OCR pass",
                "verbatim_prompt": True,
            })
            return output

        return ocr_call

    if text_source in {"easyocr_live_word_ocr", "doctr_live_word_ocr"}:
        return lambda image_path: baseline_live_word_output(
            image_path,
            args,
            source_name=text_source,
        )

    if text_source in CANDIDATE_LIVE_OCR_MODULES or text_source in TROCR_LIVE_SOURCES:
        return lambda image_path: candidate_live_ocr_output(
            image_path,
            args,
            source_name=text_source,
        )

    raise KeyError(f"Unknown live Stage 1 text source: {text_source}")


def make_composed_ocr_call(
    spec: TwoStageSpec,
    args: argparse.Namespace,
    adapter: Qwen3VLAdapter | None,
) -> Callable[[Path], PipelineOutput]:
    if is_live_text_source(spec.text_source):
        live_call = make_live_ocr_call(spec.text_source, args, adapter)
        if spec.box_source == "same_stage1_boxes":
            return live_call

        box_call = make_box_call(spec.box_source, args, adapter)

        def ocr_call(image_path: Path) -> PipelineOutput:
            text_output = live_call(image_path)
            box_output = box_call(image_path)
            stage1_latency = text_output.stage1_latency + box_output.stage1_latency
            metadata = {
                **text_output.metadata,
                **box_output.metadata,
                "text_source": spec.text_source,
                "stage1_composition": "live_text_plus_box_source",
                "stage1_latency_policy": "live_text_latency_s + box_latency_s",
            }
            return PipelineOutput(
                strategy_name=f"{spec.text_source}__{spec.box_source}",
                image=image_path.name,
                text=text_output.text,
                boxes=box_output.boxes,
                stage1_latency=stage1_latency,
                total_latency=stage1_latency,
                metadata=metadata,
            )

        return ocr_call

    if spec.box_source == "same_stage1_boxes":
        raise ValueError("same_stage1_boxes can only be used with a live Stage 1 text source")

    text_entries = load_stage1_artifact(spec.text_source)
    box_call = make_box_call(spec.box_source, args, adapter)

    def ocr_call(image_path: Path) -> PipelineOutput:
        text_output = stage1_text_output(image_path, spec.text_source, text_entries)
        box_output = box_call(image_path)
        stage1_latency = text_output.stage1_latency + box_output.stage1_latency
        metadata = {
            **text_output.metadata,
            **box_output.metadata,
            "stage1_composition": "text_artifact_plus_box_source",
            "stage1_latency_policy": "text_latency_s + box_latency_s",
        }
        return PipelineOutput(
            strategy_name=f"{spec.text_source}__{spec.box_source}",
            image=image_path.name,
            text=text_output.text,
            boxes=box_output.boxes,
            stage1_latency=stage1_latency,
            total_latency=stage1_latency,
            metadata=metadata,
        )

    return ocr_call


def phase3_qwen_text_output(image_path: Path, phase3_qwen: dict[str, dict[str, Any]]) -> PipelineOutput:
    entry = phase3_qwen.get(image_path.name, {})
    return PipelineOutput(
        strategy_name="qwen3vl_4b_phase3_text",
        image=image_path.name,
        text=str(entry.get("text", "")),
        stage1_latency=float(entry.get("latency_s", 0.0) or 0.0),
        total_latency=float(entry.get("latency_s", 0.0) or 0.0),
        metadata={
            "text_source": str(QWEN4B_WORDLEVEL_RESULT.relative_to(PROJECT_ROOT)),
            "phase3_cer": entry.get("cer"),
            "phase3_wer": entry.get("wer"),
            "phase3_word_iou": entry.get("word_iou"),
        },
    )


def make_qwen_ocr_call(
    adapter: Qwen3VLAdapter,
    args: argparse.Namespace,
    phase3_qwen: dict[str, dict[str, Any]],
):
    def ocr_call(image_path: Path) -> PipelineOutput:
        data = cached_stage1(
            args.cache_dir,
            "qwen3vl_4b_word",
            image_path,
            args.refresh_cache,
            lambda path: run_qwen_word_ocr(adapter, path),
        )
        if data.get("raw_response"):
            reparsed = parse_qwen_word_response(data["raw_response"], image_path)
            reparsed["stage1_latency"] = data.get("stage1_latency", 0.0)
            reparsed["_cache_key"] = data.get("_cache_key", "qwen3vl_4b_word")
            reparsed["_cached_at"] = data.get("_cached_at", "")
            data = reparsed
        phase3_entry = phase3_qwen.get(image_path.name, {})
        if phase3_entry.get("text"):
            data["text"] = phase3_entry["text"]
            data["_text_source"] = str(QWEN4B_WORDLEVEL_RESULT.relative_to(PROJECT_ROOT))
            data["_phase3_cer"] = phase3_entry.get("cer")
            data["_phase3_wer"] = phase3_entry.get("wer")
            data["_phase3_word_iou"] = phase3_entry.get("word_iou")
        return PipelineOutput.from_ocr_dict(
            data,
            strategy_name="qwen3vl_4b_word",
            image=image_path,
        )

    return ocr_call


def make_qwen_text_tesseract_boxes_call(
    args: argparse.Namespace,
    phase3_qwen: dict[str, dict[str, Any]],
):
    def ocr_call(image_path: Path) -> PipelineOutput:
        qwen_output = phase3_qwen_text_output(image_path, phase3_qwen)
        tess_data = cached_stage1(
            args.cache_dir,
            "tesseract_word_boxes",
            image_path,
            args.refresh_cache,
            run_tesseract_word_boxes,
        )
        tess_output = PipelineOutput.from_ocr_dict(
            tess_data,
            strategy_name="tesseract_word_boxes",
            image=image_path,
        )
        return PipelineOutput(
            strategy_name="qwen_text_tesseract_boxes",
            image=image_path.name,
            text=qwen_output.text,
            boxes=tess_output.boxes,
            stage1_latency=qwen_output.stage1_latency + tess_output.stage1_latency,
            total_latency=qwen_output.stage1_latency + tess_output.stage1_latency,
            metadata={
                "text_source": qwen_output.metadata.get("text_source"),
                "box_source": "tesseract_word_boxes",
                "phase3_cer": qwen_output.metadata.get("phase3_cer"),
                "phase3_wer": qwen_output.metadata.get("phase3_wer"),
            },
        )

    return ocr_call


def fake_model_call(prompt: str, image_path: Path | None) -> str:
    wants_error = "recieved" in prompt or "teh" in prompt
    if image_path is not None and image_path.name.startswith("pc_"):
        wants_error = True
    if "previous response" in prompt.lower():
        wants_error = True

    if "full transcription" in prompt:
        errors = []
        if wants_error:
            errors = [{
                "type": "spelling",
                "bbox": [72, 40, 142, 66],
                "description": "'recieved' should be 'received'",
                "correction": "received",
                "evidence_text": "recieved",
            }]
        return json.dumps({
            "text": "i recieved teh letter.",
            "blocks": [{"index": 0, "bbox": [40, 40, 260, 66], "text": "i recieved teh letter."}],
            "errors": errors,
            "feedback": {"summary": "Check spelling and sentence capitalization.", "strengths": [], "improvements": ["Revise spelling."]},
        })

    errors = []
    if wants_error:
        errors = [
            {
                "type": "capitalization",
                "bbox": [40, 40, 52, 66],
                "description": "Sentence should start with a capital letter.",
                "correction": "I",
                "evidence_text": "i",
            },
            {
                "type": "spelling",
                "bbox": [72, 40, 142, 66],
                "description": "'recieved' should be 'received'.",
                "correction": "received",
                "evidence_text": "recieved",
            },
        ]
    return json.dumps({
        "errors": errors,
        "feedback": {"summary": "Feedback generated by fake smoke model.", "strengths": [], "improvements": ["Review targeted edits."]},
    })


def fake_ocr_call(image_path: Path) -> PipelineOutput:
    return PipelineOutput(
        strategy_name="fake_ocr",
        image=image_path.name,
        text="Clean copied text with no writing errors.",
        boxes=[TextBox(index=0, bbox=[20, 20, 360, 52], text="Clean copied text with no writing errors.")],
        stage1_latency=0.01,
        total_latency=0.01,
    )


def strategy_name_from_spec(spec: TwoStageSpec) -> str:
    return spec.name


def parse_two_stage_strategy_name(name: str) -> TwoStageSpec:
    resolved = LEGACY_STRATEGY_ALIASES.get(name, name)
    parts = resolved.split("__")
    if len(parts) != 4 or parts[0] != "two_stage":
        raise ValueError(
            f"Unknown strategy {name!r}. Expected an end-to-end strategy or "
            "two_stage__TEXT_SOURCE__BOX_SOURCE__GRADER."
    )
    _, text_source, box_source, grader = parts
    if text_source not in TEXT_SOURCE_CHOICES:
        raise ValueError(f"Unknown Stage 1 text source: {text_source}")
    if box_source not in STAGE1_BOX_SOURCES:
        raise ValueError(f"Unknown Stage 1 box source: {box_source}")
    if box_source == "same_stage1_boxes" and not is_live_text_source(text_source):
        raise ValueError("same_stage1_boxes requires a live Stage 1 text source")
    if grader not in LOCAL_GRADERS:
        raise ValueError(f"Unknown grader: {grader}")
    return TwoStageSpec(text_source=text_source, box_source=box_source, grader=grader)


def requested_two_stage_specs(args: argparse.Namespace) -> list[TwoStageSpec]:
    if args.strategies:
        specs = []
        for name in args.strategies:
            resolved = LEGACY_STRATEGY_ALIASES.get(name, name)
            if resolved in END_TO_END_STRATEGIES or resolved in CLOUD_STRATEGIES:
                continue
            specs.append(parse_two_stage_strategy_name(name))
        return specs

    return [
        TwoStageSpec(text_source=text_source, box_source=box_source, grader=grader)
        for text_source in args.text_sources
        for box_source in args.box_sources
        for grader in args.graders
    ]


def expand_text_sources(source_names: list[str], dataset: str) -> list[str]:
    expanded: list[str] = []
    for source_name in source_names:
        if source_name == "all_live_local":
            expanded.extend(sorted(LIVE_STAGE1_TEXT_SOURCES))
        elif source_name == "all_automated_artifacts":
            expanded.extend(sorted(name for name in STAGE1_TEXT_SOURCES if name != "realworld_source_text"))
        elif source_name == "all_available":
            expanded.extend(sorted(LIVE_STAGE1_TEXT_SOURCES))
            if dataset != "realworld":
                expanded.extend(sorted(name for name in STAGE1_TEXT_SOURCES if name != "realworld_source_text"))
        else:
            expanded.append(source_name)

    seen: set[str] = set()
    unique = []
    for source_name in expanded:
        if source_name in seen:
            continue
        seen.add(source_name)
        unique.append(source_name)
    return unique


def requested_end_to_end(args: argparse.Namespace) -> list[str]:
    if args.strategies:
        names = []
        for name in args.strategies:
            resolved = LEGACY_STRATEGY_ALIASES.get(name, name)
            if resolved in END_TO_END_STRATEGIES or resolved in CLOUD_STRATEGIES:
                names.append(resolved)
        return names
    return list(args.end_to_end)


def strategy_names_for_cloud_check(args: argparse.Namespace) -> set[str]:
    if args.strategies:
        return {LEGACY_STRATEGY_ALIASES.get(name, name) for name in args.strategies}
    return set(args.end_to_end)


def qwen_adapter_needed(args: argparse.Namespace, specs: list[TwoStageSpec], end_to_end: list[str]) -> bool:
    if args.smoke_fake:
        return False
    if args.stage1_only:
        return any(
            spec.text_source in {"qwen3vl_4b_live_word_ocr", "qwen3vl_4b_verbatim_word_ocr"}
            or spec.box_source == "qwen3vl_4b_word_boxes"
            for spec in specs
        )
    if any(name == "single_qwen3vl_4b_e2e" for name in end_to_end):
        return True
    if any(spec.grader == "qwen3vl_4b_grader" for spec in specs):
        return True
    if any(spec.text_source in {"qwen3vl_4b_live_word_ocr", "qwen3vl_4b_verbatim_word_ocr"} for spec in specs):
        return True
    return any(spec.box_source == "qwen3vl_4b_word_boxes" for spec in specs)


def build_strategies(args: argparse.Namespace) -> list[SinglePassStrategy | TwoStageStrategy | Stage1OnlyStrategy]:
    requested = strategy_names_for_cloud_check(args)
    cloud_requested = requested & CLOUD_STRATEGIES
    if cloud_requested and not args.allow_cloud:
        names = ", ".join(sorted(cloud_requested))
        raise SystemExit(
            f"Cloud strategies requested ({names}) but --allow-cloud was not set. "
            "Estimate cost and get explicit approval before live cloud runs."
        )
    if cloud_requested:
        raise SystemExit(
            "Cloud Phase 4 adapters are intentionally gated and not run by default. "
            "Wire them only after explicit approval and a cost estimate."
        )

    two_stage_specs = requested_two_stage_specs(args)
    end_to_end = [] if args.stage1_only else requested_end_to_end(args)

    if args.smoke_fake:
        model_call = fake_model_call
        adapter = None
    else:
        adapter = (
            Qwen3VLAdapter(model_id=args.model_id, max_new_tokens=args.max_new_tokens)
            if qwen_adapter_needed(args, two_stage_specs, end_to_end)
            else None
        )
        model_call = (
            (lambda prompt, image_path: adapter.generate(
                prompt,
                image_path,
                max_new_tokens=args.grader_max_new_tokens,
            ))
            if adapter is not None
            else fake_model_call
        )

    strategies: list[SinglePassStrategy | TwoStageStrategy | Stage1OnlyStrategy] = []
    if "single_qwen3vl_4b_e2e" in end_to_end:
        if adapter is not None and not args.smoke_fake:
            single_call = lambda prompt, image_path: adapter.generate(
                prompt,
                image_path,
                max_new_tokens=args.max_new_tokens,
            )
        else:
            single_call = model_call
        strategies.append(SinglePassStrategy("single_qwen3vl_4b_e2e", single_call))

    for spec in two_stage_specs:
        if args.smoke_fake:
            ocr_call = fake_ocr_call
        else:
            ocr_call = make_composed_ocr_call(spec, args, adapter)
        if args.stage1_only:
            strategies.append(Stage1OnlyStrategy(f"stage1__{spec.text_source}__{spec.box_source}", ocr_call))
        else:
            strategies.append(TwoStageStrategy(strategy_name_from_spec(spec), ocr_call, model_call))

    return strategies


def load_positive_controls() -> list[dict[str, Any]]:
    if not POSITIVE_CONTROLS_PATH.exists():
        return []
    return json.loads(POSITIVE_CONTROLS_PATH.read_text())


def render_positive_control(entry: dict[str, Any], out_dir: Path) -> Path:
    from PIL import Image, ImageDraw

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{entry['id']}.png"
    if out_path.exists():
        return out_path

    lines = entry.get("lines") or [entry.get("text", "")]
    image = Image.new("RGB", (900, max(160, 70 + 42 * len(lines))), "white")
    draw = ImageDraw.Draw(image)
    y = 40
    for line in lines:
        draw.text((40, y), str(line), fill="black")
        y += 42
    image.save(out_path)
    return out_path


def positive_ocr_output(entry: dict[str, Any]) -> PipelineOutput:
    boxes = [
        TextBox.from_dict(block, index=i)
        for i, block in enumerate(entry.get("blocks", []))
    ]
    return PipelineOutput(
        strategy_name="phase4_positive_control_ocr",
        image=f"{entry['id']}.png",
        text=str(entry.get("text", "")),
        boxes=boxes,
        stage1_latency=0.0,
        total_latency=0.0,
    )


def run_positive_controls(
    strategy: SinglePassStrategy | TwoStageStrategy,
    controls: list[dict[str, Any]],
    cache_dir: Path,
) -> list[dict[str, Any]]:
    records = []
    image_dir = cache_dir / "positive_control_images"
    for entry in controls:
        image_path = render_positive_control(entry, image_dir)
        if isinstance(strategy, TwoStageStrategy):
            output = strategy.run_from_ocr(positive_ocr_output(entry), image_path=image_path)
        else:
            output = strategy.run(image_path)

        expected = set(entry.get("expected_error_types", []))
        found = {error.type for error in output.errors}
        records.append({
            "id": entry["id"],
            "valid_json": output.parse_valid,
            "emitted_errors": len(output.errors),
            "expected_error_types": sorted(expected),
            "found_error_types": sorted(found),
            "expected_type_overlap": sorted(expected & found),
            "feedback_nonempty": bool(output.feedback.summary or output.feedback.improvements),
        })
    return records


def output_record(output: PipelineOutput, include_raw: bool) -> dict[str, Any]:
    data = output.to_dict()
    if not include_raw:
        data.pop("raw_response", None)
    return data


def run_strategy(
    strategy: SinglePassStrategy | TwoStageStrategy,
    images: list[Path],
    gt_by_name: dict[str, dict[str, Any]],
    word_gt_by_name: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    print(f"[phase4] Running {strategy.name} on {len(images)} image(s)")
    image_records = []
    metric_rows = []
    latencies = []
    stage1_latencies = []
    stage2_latencies = []
    vram_peaks = []

    for run_idx in range(args.num_runs):
        for image_path in images:
            try:
                output = strategy.run(image_path)
                failed = False
                failure = ""
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                failed = True
                failure = f"{type(exc).__name__}: {exc}"
                output = PipelineOutput(
                    strategy_name=strategy.name,
                    image=image_path.name,
                    parse_valid=False,
                    notes=[failure],
                    metadata={"failed": True, "failure": failure},
                )
            metrics = evaluate_phase4_output(
                output,
                gt_by_name.get(image_path.name),
                word_gt_by_name.get(image_path.name),
            )
            metrics["stage_failed"] = failed
            if failure:
                metrics["failure"] = failure
            row = {
                "image": image_path.name,
                "run": run_idx + 1,
                **metrics,
            }
            metric_rows.append(row)
            latencies.append(output.total_latency)
            stage1_latencies.append(output.stage1_latency)
            stage2_latencies.append(output.stage2_latency)
            vram_peaks.append(get_peak_vram_mb())
            image_records.append({
                "image": image_path.name,
                "run": run_idx + 1,
                "output": output_record(output, args.include_raw),
                "metrics": metrics,
            })
            print(
                f"  {image_path.name}: total={output.total_latency:.2f}s "
                f"stage1={output.stage1_latency:.2f}s stage2={output.stage2_latency:.2f}s "
                f"valid_json={output.parse_valid} word_iou={metrics['word_iou']} "
                f"fp={metrics['false_positive_count']}"
                + (f" FAILED={failure}" if failed else "")
            )

    aggregate = aggregate_phase4_metrics(metric_rows)
    aggregate.update({
        "latency_total_avg": sum(latencies) / len(latencies) if latencies else 0.0,
        "latency_stage1_avg": sum(stage1_latencies) / len(stage1_latencies) if stage1_latencies else 0.0,
        "latency_stage2_avg": sum(stage2_latencies) / len(stage2_latencies) if stage2_latencies else 0.0,
        "vram_peak_mb": max(vram_peaks) if vram_peaks else 0,
    })

    positive_records = []
    if args.positive_controls:
        positive_records = run_positive_controls(strategy, load_positive_controls(), args.cache_dir)

    return {
        "name": strategy.name,
        "aggregate": aggregate,
        "images": image_records,
        "positive_controls": positive_records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help=(
            "Exact strategies to run. Supports legacy names plus "
            "two_stage__TEXT_SOURCE__BOX_SOURCE__GRADER."
        ),
    )
    parser.add_argument(
        "--text-sources",
        nargs="+",
        default=None,
        choices=sorted(TEXT_SOURCE_CHOICES),
        help="Saved or live Stage 1 text sources to mix into the two-stage matrix.",
    )
    parser.add_argument(
        "--box-sources",
        nargs="+",
        default=None,
        choices=sorted(STAGE1_BOX_SOURCES),
        help="Word-box sources to pair with each text source.",
    )
    parser.add_argument(
        "--graders",
        nargs="+",
        default=["qwen3vl_4b_grader"],
        choices=sorted(LOCAL_GRADERS),
        help="Stage 2 graders to run over each text/box source.",
    )
    parser.add_argument(
        "--end-to-end",
        nargs="+",
        default=["single_qwen3vl_4b_e2e"],
        choices=sorted(END_TO_END_STRATEGIES),
        help="Single-pass VLM arms to compare against the two-stage matrix.",
    )
    parser.add_argument("--dataset", choices=["iam", "realworld"], default="iam")
    parser.add_argument("--max-images", type=int, default=1, help="Number of images to run. Use 0 for all.")
    parser.add_argument("--image", dest="images", action="append", help="Specific image filename in the selected dataset image directory.")
    parser.add_argument("--num-runs", type=int, default=1, help="Repeated runs per image.")
    parser.add_argument("--model-id", default=QWEN_MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--grader-max-new-tokens", type=int, default=768)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--include-raw", action="store_true", help="Store raw model responses in the result JSON.")
    parser.add_argument("--positive-controls", action="store_true", help="Run the tiny schema/behavior positive-control probes.")
    parser.add_argument("--smoke-fake", action="store_true", help="Use fake OCR/model calls; loads no ML models.")
    parser.add_argument("--stage1-only", action="store_true", help="Run only Stage 1 OCR/localization and score truthfulness/localization metrics.")
    parser.add_argument("--continue-on-error", action="store_true", help="Record failed strategies/images instead of aborting the whole matrix.")
    parser.add_argument("--list-stage1-sources", action="store_true", help="Print the Phase 4 Stage 1 model registry and exit.")
    parser.add_argument("--allow-cloud", action="store_true", help="Allow explicitly requested cloud strategies after cost approval.")
    args = parser.parse_args()

    if args.list_stage1_sources:
        print(json.dumps(registry_metadata(PROJECT_ROOT), indent=2, ensure_ascii=False))
        raise SystemExit(0)

    if args.text_sources is None:
        args.text_sources = (
            ["realworld_source_text"]
            if args.dataset == "realworld"
            else [
                "qwen3vl_4b_wordlevel",
                "qwen3vl_8b_wordlevel_api",
                "florence2_large_wordlevel",
                "google_docai_wordlevel",
                "tesseract_wordlevel",
                "doctr_wordlevel",
                "easyocr_wordlevel",
                "locateanything_wordlevel",
            ]
        )
    args.text_sources = expand_text_sources(args.text_sources, args.dataset)
    if args.box_sources is None:
        args.box_sources = (
            ["realworld_aligned_words"]
            if args.dataset == "realworld"
            else ["qwen3vl_4b_word_boxes", "tesseract_word_boxes"]
        )

    if args.num_runs < 1:
        raise SystemExit("--num-runs must be >= 1")
    if args.allow_cloud and args.num_runs != 1:
        raise SystemExit("Cloud runs must use --num-runs 1 unless the user explicitly changes the code after approval.")
    if args.strategies:
        try:
            requested_two_stage_specs(args)
            requested_end_to_end(args)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    return args


def main() -> None:
    args = parse_args()
    image_dir, ground_truth_path, word_ground_truth_path = dataset_paths(args.dataset)
    images = select_images(args.max_images, args.images, image_dir=image_dir)
    if not images:
        raise SystemExit(f"No images found in {image_dir}")

    gt_by_name = load_ground_truth(ground_truth_path)
    word_gt_by_name = load_ground_truth(word_ground_truth_path)
    strategies = build_strategies(args)
    if not strategies:
        raise SystemExit("No strategies selected.")

    result = {
        "phase": 4,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": {
            "name": args.dataset,
            "image_dir": str(image_dir.relative_to(PROJECT_ROOT)),
            "ground_truth": str(ground_truth_path.relative_to(PROJECT_ROOT)),
            "word_localization": str(word_ground_truth_path.relative_to(PROJECT_ROOT)),
            "stage1_text_artifacts": {
                name: str(path.relative_to(PROJECT_ROOT))
                for name, path in STAGE1_TEXT_SOURCES.items()
                if path.exists()
            },
            "stage1_model_registry": registry_metadata(PROJECT_ROOT),
            "requested_text_sources": args.text_sources if not args.strategies else [],
            "requested_box_sources": args.box_sources if not args.strategies else [],
            "requested_graders": args.graders if not args.strategies else [],
            "requested_end_to_end": [] if args.stage1_only else requested_end_to_end(args),
            "stage1_only": args.stage1_only,
            "model_id": args.model_id,
            "max_new_tokens": args.max_new_tokens,
            "grader_max_new_tokens": args.grader_max_new_tokens,
            "images": [path.name for path in images],
            "positive_controls": str(POSITIVE_CONTROLS_PATH.relative_to(PROJECT_ROOT)),
            "positive_controls_enabled": args.positive_controls,
        },
        "notes": [
            "IAM handwritten crops have no intentional writing errors; error F1/IoU are not_applicable there.",
            "Real-world handwritten error localization is authoritative only for entries marked annotation_status=manual_reviewed.",
            "Phase 4 is a mix-and-match architecture benchmark: saved Stage 1 text artifacts can be paired with separate word-box sources and Stage 2 graders.",
            "Most historical word-level result artifacts store metrics, not per-word coordinates, so box sources are explicit in each strategy name.",
            "Localization is evaluated as word_iou against the active dataset word annotations.",
            "Positive controls are schema/behavior probes, not a writing-error leaderboard.",
            "Cloud strategies are disabled unless explicitly requested with approval and a cost estimate.",
            "For the real-world full-matrix run, use live Stage 1 sources; IAM Phase 2/3 artifacts are not substitutes for real-world OCR evidence.",
        ],
        "strategies": [],
    }

    for strategy in strategies:
        result["strategies"].append(run_strategy(strategy, images, gt_by_name, word_gt_by_name, args))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[phase4] Wrote {args.output}")


if __name__ == "__main__":
    main()
