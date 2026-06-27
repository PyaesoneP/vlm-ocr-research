"""Optical-only candidate scoring helpers for Phase 4.

The first backend wraps EasyOCR's English CTC recognizer. It is not a new page
OCR source; it scores a fixed crop against fixed candidate strings so selector
policy can be calibrated independently from candidate generation.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class CandidateScore:
    text: str
    scorer_text: str
    raw_logprob: float
    normalized_logprob: float
    char_count: int
    target_length: int
    timesteps: int
    unsupported_chars: list[str]
    supported: bool
    latency_ms: float


class EasyOCRCTCCandidateScorer:
    """Score arbitrary candidate strings with EasyOCR CTC forward probability."""

    model_id = "easyocr_english_g2_ctc"

    def __init__(
        self,
        *,
        device: str = "cpu",
        img_h: int = 64,
        img_w: int = 256,
        strip_unsupported: bool = False,
    ) -> None:
        self.requested_device = device
        self.img_h = img_h
        self.img_w = img_w
        self.strip_unsupported = strip_unsupported
        self.reader = None
        self.model = None
        self.converter = None
        self.device = "cpu"
        self.alphabet: list[str] = []

    def load(self) -> None:
        if self.model is not None and self.converter is not None:
            return
        import easyocr
        import torch

        use_gpu = self.requested_device == "cuda" or (
            self.requested_device == "auto" and torch.cuda.is_available()
        )
        self.reader = easyocr.Reader(
            ["en"],
            gpu=use_gpu,
            detector=False,
            recognizer=True,
            download_enabled=False,
            verbose=False,
            quantize=False,
        )
        self.model = self.reader.recognizer
        self.converter = self.reader.converter
        self.device = str(self.reader.device)
        self.alphabet = list(self.converter.character)

    def metadata(self) -> dict[str, Any]:
        self.load()
        return {
            "scorer": self.model_id,
            "device": self.device,
            "img_h": self.img_h,
            "img_w": self.img_w,
            "strip_unsupported": self.strip_unsupported,
            "alphabet_size": len(self.alphabet),
            "blank_index": 0,
            "supports_space": " " in self.converter.dict,
            "supports_period": "." in self.converter.dict,
            "supports_apostrophe": "'" in self.converter.dict,
        }

    def _prepare_text(self, text: str) -> tuple[str, list[str]]:
        self.load()
        supported_chars = set(self.converter.dict)
        kept = []
        unsupported = []
        for char in str(text):
            if char in supported_chars:
                kept.append(char)
            else:
                unsupported.append(char)
                if not self.strip_unsupported:
                    continue
        return "".join(kept), sorted(set(unsupported))

    def _image_logits(self, crop_path: Path):
        self.load()
        import torch
        from easyocr.recognition import AlignCollate

        with Image.open(crop_path) as image:
            crop = image.convert("L")
        collate = AlignCollate(imgH=self.img_h, imgW=self.img_w, keep_ratio_with_pad=True)
        image_tensor = collate([crop]).to(self.device)
        batch_max_length = max(1, int(self.img_w / 10))
        text_for_pred = torch.LongTensor(1, batch_max_length + 1).fill_(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(image_tensor, text_for_pred)
        return logits[0]

    def greedy_decode(self, crop_path: Path) -> str:
        self.load()
        import torch

        logits = self._image_logits(crop_path)
        preds_size = torch.IntTensor([logits.size(0)])
        _, preds_index = logits.softmax(dim=1).max(1)
        return self.converter.decode_greedy(
            preds_index.cpu().numpy(),
            preds_size,
        )[0]

    def score_candidate(self, crop_path: Path, candidate: str) -> CandidateScore:
        self.load()
        import torch
        import torch.nn.functional as F

        start = time.perf_counter()
        scorer_text, unsupported = self._prepare_text(candidate)
        if not scorer_text:
            return CandidateScore(
                text=str(candidate),
                scorer_text=scorer_text,
                raw_logprob=-math.inf,
                normalized_logprob=-math.inf,
                char_count=0,
                target_length=0,
                timesteps=0,
                unsupported_chars=unsupported,
                supported=False,
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        logits = self._image_logits(crop_path)
        targets = torch.IntTensor([self.converter.dict[char] for char in scorer_text]).to(self.device)
        target_lengths = torch.IntTensor([len(scorer_text)]).to(self.device)
        input_lengths = torch.IntTensor([logits.size(0)]).to(self.device)
        log_probs = logits.log_softmax(dim=1).unsqueeze(1)
        loss = F.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=0,
            reduction="none",
            zero_infinity=True,
        )
        raw_logprob = -float(loss.item())
        char_count = max(1, len(scorer_text))
        return CandidateScore(
            text=str(candidate),
            scorer_text=scorer_text,
            raw_logprob=raw_logprob,
            normalized_logprob=raw_logprob / char_count,
            char_count=char_count,
            target_length=len(scorer_text),
            timesteps=int(logits.size(0)),
            unsupported_chars=unsupported,
            supported=not unsupported,
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    def score_candidates(self, crop_path: Path, candidates: list[str]) -> list[dict[str, Any]]:
        rows = []
        for candidate in candidates:
            score = self.score_candidate(crop_path, candidate)
            rows.append({
                "text": score.text,
                "scorer_text": score.scorer_text,
                "raw_logprob": score.raw_logprob,
                "normalized_logprob": score.normalized_logprob,
                "char_count": score.char_count,
                "target_length": score.target_length,
                "timesteps": score.timesteps,
                "unsupported_chars": score.unsupported_chars,
                "supported": score.supported,
                "latency_ms": score.latency_ms,
            })
        rows.sort(key=lambda item: item["normalized_logprob"], reverse=True)
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
            runner = rows[1] if rank == 1 and len(rows) > 1 else rows[0] if rank > 1 else None
            row["margin_to_runner_up"] = (
                float(row["normalized_logprob"] - runner["normalized_logprob"])
                if runner is not None
                else None
            )
        return rows
