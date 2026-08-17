"""Score Phase 4 minimal-pair candidates with Qwen visual-gain logprobs.

This is a diagnostic scorer, not the final optical model. It estimates how much
the image changes Qwen's preference for a candidate by subtracting a blank-image
language/format prior:

    visual_gain(c) = log P(c | crop prompt) - alpha * log P(c | blank prompt)

The preferred long-term scorer is an optical-only CTC recognizer with no
language model. This script is a fast way to test whether the current local VLM
contains usable visual signal after its text prior is discounted.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from PIL import Image


QWEN_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_PAIRS = Path("benchmark/results/phase4_minimal_pair_dataset.json")
DEFAULT_OUTPUT = Path("benchmark/results/phase4_qwen_visual_gain_scores.json")

SCORING_PROMPT = (
    "Copy the visible handwritten word or short phrase exactly as written. "
    "Return only that text."
)


def compact(text: str) -> str:
    return "".join(ch.lower() for ch in str(text) if ch.isalnum())


def evidence_key(text: str) -> str:
    """Compare visible forms while preserving case and word spacing."""

    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]+", "", str(text)).strip())


def unique_label_candidates(pair: dict[str, Any]) -> list[str]:
    candidates = []
    seen: set[str] = set()
    for text in (pair.get("visible_form", ""), pair.get("normalized_form", "")):
        key = evidence_key(text)
        if text and key and key not in seen:
            seen.add(key)
            candidates.append(str(text))
    return candidates


def unique_all_candidates(pair: dict[str, Any]) -> list[str]:
    candidates = unique_label_candidates(pair)
    seen = {evidence_key(text) for text in candidates}
    for candidate in pair.get("candidates", []):
        text = str(candidate.get("text", ""))
        key = evidence_key(text)
        if text and key and key not in seen:
            seen.add(key)
            candidates.append(text)
    return candidates


class FakeScorer:
    """Small deterministic scorer for smoke tests without loading a model."""

    def score_candidate(self, crop_path: Path, candidate: str) -> dict[str, float]:
        del crop_path
        value = -float(len(compact(candidate)))
        return {
            "image_logprob": value,
            "blank_logprob": value,
            "visual_gain": 0.0,
            "token_count": float(max(1, len(compact(candidate)))),
            "char_count": float(max(1, len(compact(candidate)))),
        }


class QwenVisualGainScorer:
    def __init__(
        self,
        *,
        model_id: str,
        alpha: float,
        normalize_by: str,
        prompt: str = SCORING_PROMPT,
    ) -> None:
        self.model_id = model_id
        self.alpha = alpha
        self.normalize_by = normalize_by
        self.prompt = prompt
        self.model = None
        self.processor = None

    def _load(self) -> None:
        if self.model is not None and self.processor is not None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        print(f"[visual-gain] Loading {self.model_id} ...", flush=True)
        model_kwargs = {"dtype": "auto", "device_map": "auto"}
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
        print("[visual-gain] Model loaded.", flush=True)

    def _inputs_for(self, image: Image.Image, candidate: str) -> tuple[dict[str, Any], int]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]
        prefix = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        full = self.processor.apply_chat_template(
            messages + [{"role": "assistant", "content": [{"type": "text", "text": candidate}]}],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
        prefix_len = int(prefix["input_ids"].shape[-1])
        return full, prefix_len

    def sequence_logprob(self, image: Image.Image, candidate: str) -> dict[str, float]:
        self._load()
        import torch

        full, prefix_len = self._inputs_for(image, candidate)
        full = {key: value.to(self.model.device) for key, value in full.items()}
        input_ids = full["input_ids"]
        if prefix_len >= int(input_ids.shape[-1]):
            return {"logprob": -math.inf, "token_count": 0.0, "char_count": float(len(candidate))}

        with torch.no_grad():
            outputs = self.model(**full)
        logits = outputs.logits[0]
        ids = input_ids[0]
        end = int(ids.shape[-1])
        eos_ids = {
            value for value in (
                getattr(self.processor.tokenizer, "eos_token_id", None),
                getattr(self.processor.tokenizer, "pad_token_id", None),
            )
            if value is not None
        }
        while end > prefix_len and int(ids[end - 1]) in eos_ids:
            end -= 1
        if end <= prefix_len:
            return {"logprob": -math.inf, "token_count": 0.0, "char_count": float(len(candidate))}

        logprob = 0.0
        for pos in range(prefix_len, end):
            token_id = int(ids[pos])
            logprob += float(torch.log_softmax(logits[pos - 1], dim=-1)[token_id].item())

        token_count = float(end - prefix_len)
        char_count = float(max(1, len(compact(candidate))))
        denominator = char_count if self.normalize_by == "char" else token_count
        return {
            "logprob": logprob / max(1.0, denominator),
            "token_count": token_count,
            "char_count": char_count,
        }

    def score_candidate(self, crop_path: Path, candidate: str) -> dict[str, float]:
        with Image.open(crop_path) as image:
            crop = image.convert("RGB")
        blank = Image.new("RGB", crop.size, "white")
        image_score = self.sequence_logprob(crop, candidate)
        blank_score = self.sequence_logprob(blank, candidate)
        visual_gain = image_score["logprob"] - self.alpha * blank_score["logprob"]
        return {
            "image_logprob": image_score["logprob"],
            "blank_logprob": blank_score["logprob"],
            "visual_gain": visual_gain,
            "token_count": image_score["token_count"],
            "char_count": image_score["char_count"],
        }


def score_pair(pair: dict[str, Any], scorer: Any, *, candidate_mode: str) -> dict[str, Any]:
    crop_path = Path(pair["crop_path"])
    candidate_texts = unique_label_candidates(pair) if candidate_mode == "labels" else unique_all_candidates(pair)
    scored = []
    for text in candidate_texts:
        scores = scorer.score_candidate(crop_path, text)
        scored.append({"text": text, "norm": compact(text), **scores})
    scored.sort(key=lambda item: item["visual_gain"], reverse=True)
    top = scored[0] if scored else None
    runner_up = scored[1] if len(scored) > 1 else None
    margin = (
        float(top["visual_gain"] - runner_up["visual_gain"])
        if top is not None and runner_up is not None
        else None
    )
    visible_norm = compact(pair.get("visible_form", ""))
    normalized_norm = compact(pair.get("normalized_form", ""))
    visible_key = evidence_key(pair.get("visible_form", ""))
    normalized_key = evidence_key(pair.get("normalized_form", ""))
    top_key = evidence_key(top["text"]) if top else ""
    compact_labels_distinct = visible_norm != normalized_norm
    top_is_visible_surface = bool(top and top_key == visible_key)
    top_is_normalized_surface = bool(top and top_key == normalized_key)
    return {
        "pair_id": pair.get("pair_id"),
        "split": pair.get("split"),
        "image": pair.get("image"),
        "crop_path": pair.get("crop_path"),
        "visible_form": pair.get("visible_form"),
        "normalized_form": pair.get("normalized_form"),
        "top_text": top["text"] if top else "",
        "top_is_visible": bool(
            top_is_visible_surface or (compact_labels_distinct and top and top["norm"] == visible_norm)
        ),
        "top_is_normalized": bool(
            top_is_normalized_surface or (compact_labels_distinct and top and top["norm"] == normalized_norm)
        ),
        "top_is_visible_surface": top_is_visible_surface,
        "top_is_normalized_surface": top_is_normalized_surface,
        "margin": margin,
        "scores": scored,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [row for row in rows if row["split"] == "positive"]
    controls = [row for row in rows if row["split"] == "clean_control"]

    def rate(items: list[dict[str, Any]], key: str) -> float:
        return sum(1 for item in items if item.get(key)) / len(items) if items else 0.0

    margins = [row["margin"] for row in rows if row.get("margin") is not None]
    return {
        "pairs_scored": len(rows),
        "positive_pairs": len(positives),
        "clean_control_pairs": len(controls),
        "positive_visible_preference": rate(positives, "top_is_visible"),
        "clean_visible_preference": rate(controls, "top_is_visible"),
        "overall_visible_preference": rate(rows, "top_is_visible"),
        "margin_min": min(margins) if margins else None,
        "margin_median": sorted(margins)[len(margins) // 2] if margins else None,
        "margin_max": max(margins) if margins else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-id", default=QWEN_MODEL_ID)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--normalize-by", choices=["char", "token"], default="char")
    parser.add_argument("--candidate-mode", choices=["labels", "all"], default="labels")
    parser.add_argument("--split", choices=["all", "positive", "clean_control"], default="all")
    parser.add_argument("--max-pairs", type=int, default=0, help="Limit scored pairs; 0 means all.")
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--smoke-fake", action="store_true", help="Use deterministic fake scores.")
    args = parser.parse_args()

    data = json.loads(args.pairs.read_text())
    pairs = data.get("pairs", [])
    if args.split != "all":
        pairs = [pair for pair in pairs if pair.get("split") == args.split]
    if args.max_pairs > 0:
        pairs = pairs[:args.max_pairs]

    scorer = FakeScorer() if args.smoke_fake else QwenVisualGainScorer(
        model_id=args.model_id,
        alpha=args.alpha,
        normalize_by=args.normalize_by,
    )
    rows = []
    for offset, pair in enumerate(pairs, start=1):
        rows.append(score_pair(pair, scorer, candidate_mode=args.candidate_mode))
        if args.progress_every > 0 and offset % args.progress_every == 0:
            print(f"[visual-gain] scored {offset}/{len(pairs)} pairs", flush=True)
    result = {
        "pairs": str(args.pairs),
        "model_id": "fake" if args.smoke_fake else args.model_id,
        "alpha": args.alpha,
        "normalize_by": args.normalize_by,
        "candidate_mode": args.candidate_mode,
        "split": args.split,
        "summary": summarize(rows),
        "scores": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
