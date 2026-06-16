#!/usr/bin/env python3
"""
LocateAnything-3B text-localization candidate.

This candidate evaluates NVLabs Eagle Embodied / LocateAnything as a Stage 1
localization model. Its primary value is text/word box detection, not full OCR
transcription. CER/WER should only be interpreted when the model emits actual
word labels in <ref>...</ref> spans.

Usage:
    source .venv_locateanything/bin/activate
    python scripts/eval_wordlevel_iou.py locateanything
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODEL_ID = os.environ.get("LOCATEANYTHING_MODEL", "nvidia/LocateAnything-3B")
GENERATION_MODE = os.environ.get("LOCATEANYTHING_GENERATION_MODE", "hybrid")
MAX_NEW_TOKENS = int(os.environ.get("LOCATEANYTHING_MAX_NEW_TOKENS", "8192"))
TEMPERATURE = float(os.environ.get("LOCATEANYTHING_TEMPERATURE", "0.0"))
DEVICE = os.environ.get("LOCATEANYTHING_DEVICE", "cuda")
MAX_IMAGE_SIDE = int(os.environ.get("LOCATEANYTHING_MAX_IMAGE_SIDE", "1024"))
ALLOW_CPU = os.environ.get("LOCATEANYTHING_ALLOW_CPU", "0") == "1"
VERBOSE = os.environ.get("LOCATEANYTHING_VERBOSE", "0") == "1"
TEXT_DETECTION_PROMPT = os.environ.get(
    "LOCATEANYTHING_TEXT_PROMPT",
    (
        "Detect every individual handwritten word in box format. "
        "Return one <ref>word</ref><box><x1><y1><x2><y2></box> per word, "
        "not line boxes."
    ),
)
FALLBACK_TEXT_DETECTION_PROMPT = os.environ.get(
    "LOCATEANYTHING_FALLBACK_TEXT_PROMPT",
    "Detect all the text in box format.",
)
MIN_WORD_BOXES = int(os.environ.get("LOCATEANYTHING_MIN_WORD_BOXES", "20"))

REF_BOX_PATTERN = re.compile(
    r"(?:<ref>(?P<label>.*?)</ref>\s*)?"
    r"<box><(?P<x1>\d+)><(?P<y1>\d+)><(?P<x2>\d+)><(?P<y2>\d+)></box>",
    re.DOTALL,
)
GENERIC_LABELS = {
    "",
    "text",
    "word",
    "handwriting",
    "handwritten text",
    "printed text",
    "scene text",
}


class LocateAnythingWorker:
    """Minimal stateful worker based on NVIDIA's published model-card example."""

    def __init__(self, model_path: str, device: str = "cuda") -> None:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "LocateAnything dependencies are missing. Use "
                "`source .venv_locateanything/bin/activate` and install "
                "`requirements-locateanything.txt` plus a CUDA-matched torch wheel."
            ) from exc

        if device == "cuda" and not torch.cuda.is_available() and not ALLOW_CPU:
            raise RuntimeError(
                "LocateAnything benchmark requires CUDA by default. "
                "PyTorch reports cuda.is_available() == False. Expose the GPU "
                "or set LOCATEANYTHING_ALLOW_CPU=1 for an intentionally slow CPU smoke test."
            )
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.torch = torch
        self.device = device
        self.dtype = torch.bfloat16 if device == "cuda" else torch.float32
        print("[locateanything] Loading tokenizer...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print("[locateanything] Loading processor...", flush=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        print("[locateanything] Loading model weights...", flush=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(device).eval()
        print(f"[locateanything] Model ready on {device}.", flush=True)


def _install_predict_method() -> None:
    """Attach predict after class creation so torch.no_grad is available lazily."""

    def predict(
        self: LocateAnythingWorker,
        image: Any,
        question: str,
        generation_mode: str = "hybrid",
        max_new_tokens: int = 8192,
        temperature: float = 0.0,
        verbose: bool = False,
    ) -> dict[str, Any]:
        if verbose:
            print("[locateanything] Preparing prompt/image tensors...", flush=True)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = self.processor.py_apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=images,
            videos=videos,
            return_tensors="pt",
        ).to(self.device)
        if verbose:
            print("[locateanything] Generating boxes...", flush=True)

        generation_kwargs = {
            "pixel_values": inputs["pixel_values"].to(self.dtype),
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "image_grid_hws": inputs.get("image_grid_hws", None),
            "tokenizer": self.tokenizer,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "generation_mode": generation_mode,
            "temperature": temperature,
            "verbose": verbose,
            "repetition_penalty": 1.1,
        }
        if temperature > 0:
            generation_kwargs.update({"do_sample": True, "top_p": 0.9})
        else:
            generation_kwargs.update({"do_sample": False})

        with self.torch.no_grad():
            response = self.model.generate(**generation_kwargs)

        if isinstance(response, tuple):
            answer = response[0]
            result: dict[str, Any] = {"answer": answer}
            if len(response) >= 3:
                result["history"] = response[1]
                result["stats"] = response[2]
            return result
        if not isinstance(response, str):
            try:
                answer = self.tokenizer.decode(response[0], skip_special_tokens=False)
            except Exception:
                answer = str(response)
            return {"answer": answer}
        return {"answer": response}

    LocateAnythingWorker.predict = predict  # type: ignore[method-assign]

    def detect_text(self: LocateAnythingWorker, image: Any, **kwargs: Any) -> dict[str, Any]:
        return self.predict(image, TEXT_DETECTION_PROMPT, **kwargs)

    LocateAnythingWorker.detect_text = detect_text  # type: ignore[attr-defined]


_install_predict_method()


def _clean_label(label: str | None) -> str:
    if not label:
        return ""
    text = re.sub(r"<.*?>", "", label)
    return " ".join(text.split()).strip()


def _is_transcription_label(label: str) -> bool:
    return label.lower().strip(" :;,.") not in GENERIC_LABELS


def parse_response(answer: str, img_width: int, img_height: int) -> tuple[str, list[dict], bool]:
    """Parse LocateAnything output into text labels and pixel-coordinate boxes."""
    blocks: list[dict] = []
    words: list[str] = []

    for idx, match in enumerate(REF_BOX_PATTERN.finditer(answer)):
        coords = [int(match.group(name)) for name in ("x1", "y1", "x2", "y2")]
        x1, y1, x2, y2 = coords
        x1 = int(max(0, min(img_width, x1 / 1000 * img_width)))
        y1 = int(max(0, min(img_height, y1 / 1000 * img_height)))
        x2 = int(max(0, min(img_width, x2 / 1000 * img_width)))
        y2 = int(max(0, min(img_height, y2 / 1000 * img_height)))
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        label = _clean_label(match.group("label"))
        if label:
            words.append(label)
        blocks.append(
            {
                "bbox": [x1, y1, x2, y2],
                "text": label,
                "confidence": 1.0,
                "reading_order": idx,
            }
        )

    transcription_labels = [w for w in words if _is_transcription_label(w)]
    text_is_transcription = bool(blocks) and len(transcription_labels) >= max(1, len(blocks) // 3)
    clean_text = " ".join(transcription_labels) if text_is_transcription else ""
    return clean_text, blocks, text_is_transcription


def _resize_for_inference(image: Any, max_side: int) -> Any:
    """Return a resized copy for inference while preserving the original image."""
    if max_side <= 0:
        return image
    longest_side = max(image.width, image.height)
    if longest_side <= max_side:
        return image

    resized = image.copy()
    resized.thumbnail((max_side, max_side))
    if VERBOSE:
        print(
            "[locateanything] Resized inference image "
            f"{image.width}x{image.height} -> {resized.width}x{resized.height}",
            flush=True,
        )
    return resized


def inference_fn(image_path: str | Path) -> dict[str, Any]:
    """Run LocateAnything scene-text detection on one cropped handwritten image."""
    try:
        import torch
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("LocateAnything inference requires torch and Pillow.") from exc

    if not hasattr(inference_fn, "_worker"):
        print(f"[locateanything] Loading {MODEL_ID} on {DEVICE}...")
        inference_fn._worker = LocateAnythingWorker(MODEL_ID, device=DEVICE)
        print("[locateanything] Model loaded.")

    worker = inference_fn._worker
    image = Image.open(image_path).convert("RGB")
    original_width, original_height = image.size
    inference_image = _resize_for_inference(image, MAX_IMAGE_SIDE)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = worker.detect_text(
        inference_image,
        generation_mode=GENERATION_MODE,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        verbose=VERBOSE,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    prompt_used = TEXT_DETECTION_PROMPT
    answer = str(result.get("answer", ""))
    text, blocks, text_is_transcription = parse_response(answer, original_width, original_height)
    fallback_used = False
    fallback_box_count = None

    if MIN_WORD_BOXES > 0 and len(blocks) < MIN_WORD_BOXES:
        if VERBOSE:
            print(
                "[locateanything] Sparse word-box result "
                f"({len(blocks)} < {MIN_WORD_BOXES}); retrying with fallback prompt.",
                flush=True,
            )
        fallback_used = True
        fallback_result = worker.predict(
            inference_image,
            FALLBACK_TEXT_DETECTION_PROMPT,
            generation_mode=GENERATION_MODE,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            verbose=VERBOSE,
        )
        fallback_answer = str(fallback_result.get("answer", ""))
        fallback_text, fallback_blocks, fallback_is_transcription = parse_response(
            fallback_answer,
            original_width,
            original_height,
        )
        fallback_box_count = len(fallback_blocks)
        if len(fallback_blocks) > len(blocks):
            prompt_used = FALLBACK_TEXT_DETECTION_PROMPT
            answer = fallback_answer
            text = fallback_text
            blocks = fallback_blocks
            text_is_transcription = fallback_is_transcription

    return {
        "text": text,
        "blocks": blocks,
        "raw_answer": answer,
        "text_is_transcription": text_is_transcription,
        "stage1_latency": elapsed,
        "model": MODEL_ID,
        "generation_mode": GENERATION_MODE,
        "inference_image_size": [inference_image.width, inference_image.height],
        "original_image_size": [original_width, original_height],
        "max_image_side": MAX_IMAGE_SIDE,
        "prompt_used": prompt_used,
        "fallback_used": fallback_used,
        "fallback_box_count": fallback_box_count,
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Smoke-test LocateAnything on one image.")
    parser.add_argument(
        "image",
        nargs="?",
        default=str(PROJECT_ROOT / "benchmark" / "test_dataset" / "handwritten" / "a04-039.png"),
    )
    args = parser.parse_args()
    print(json.dumps(inference_fn(args.image), indent=2))
