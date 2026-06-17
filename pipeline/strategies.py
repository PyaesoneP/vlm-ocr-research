"""Reusable Phase 4 strategy runners."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from pipeline.contracts import PipelineOutput
from pipeline.parsing import ParseResult, parse_model_json
from pipeline.prompts import (
    SINGLE_PASS_SCHEMA,
    STAGE2_SCHEMA,
    build_repair_prompt,
    build_single_pass_prompt,
    build_stage2_prompt,
)


ModelCall = Callable[[str, Path | None], str]
OcrCall = Callable[[Path], PipelineOutput | dict]


def _sync_cuda() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(fn: Callable, *args) -> tuple[object, float]:
    _sync_cuda()
    t0 = time.perf_counter()
    result = fn(*args)
    _sync_cuda()
    return result, time.perf_counter() - t0


def _image_size(image_path: Path | None) -> tuple[int, int] | None:
    if image_path is None:
        return None
    try:
        from PIL import Image
        with Image.open(image_path) as image:
            return image.size
    except Exception:
        return None


def _parse_with_repair(
    *,
    model_call: ModelCall,
    raw: str,
    image_path: Path | None,
    image_size: tuple[int, int] | None,
    require_text: bool,
    repair_schema: dict,
) -> tuple[ParseResult, str, bool, bool, float]:
    parsed = parse_model_json(
        raw,
        image_size=image_size,
        require_text=require_text,
        require_boxes=require_text,
    )
    if parsed.valid:
        return parsed, raw, False, False, 0.0

    repair_prompt = build_repair_prompt(raw, repair_schema)
    repaired_raw_obj, repair_latency = _timed(model_call, repair_prompt, image_path)
    repaired_raw = str(repaired_raw_obj)
    repaired = parse_model_json(
        repaired_raw,
        image_size=image_size,
        require_text=require_text,
        require_boxes=require_text,
    )
    if repaired.valid:
        return repaired, repaired_raw, True, True, repair_latency
    return parsed, raw, True, False, repair_latency


class SinglePassStrategy:
    """One VLM prompt from image to OCR, error boxes, and feedback."""

    def __init__(self, name: str, model_call: ModelCall):
        self.name = name
        self.model_call = model_call

    def run(self, image_path: str | Path) -> PipelineOutput:
        path = Path(image_path)
        prompt = build_single_pass_prompt()
        raw_obj, latency = _timed(self.model_call, prompt, path)
        raw = str(raw_obj)
        parsed, final_raw, repair_attempted, repair_succeeded, repair_latency = _parse_with_repair(
            model_call=self.model_call,
            raw=raw,
            image_path=path,
            image_size=_image_size(path),
            require_text=True,
            repair_schema=SINGLE_PASS_SCHEMA,
        )
        total = latency + repair_latency
        return PipelineOutput(
            strategy_name=self.name,
            image=path.name,
            text=parsed.text,
            boxes=parsed.boxes,
            errors=parsed.errors,
            feedback=parsed.feedback,
            stage1_latency=total,
            stage2_latency=0.0,
            total_latency=total,
            parse_valid=parsed.valid,
            repair_attempted=repair_attempted,
            repair_succeeded=repair_succeeded,
            raw_response=final_raw,
            notes=parsed.problems,
            metadata={"mode": "single_pass"},
        )


class TwoStageStrategy:
    """Stage 1 OCR/localization followed by a separate grading prompt."""

    def __init__(self, name: str, ocr_call: OcrCall, grader_call: ModelCall):
        self.name = name
        self.ocr_call = ocr_call
        self.grader_call = grader_call

    def run(self, image_path: str | Path) -> PipelineOutput:
        path = Path(image_path)
        ocr_obj, measured_stage1 = _timed(self.ocr_call, path)
        ocr_output = self._coerce_ocr_output(ocr_obj, path, measured_stage1)
        return self.run_from_ocr(ocr_output, image_path=path)

    def run_from_ocr(
        self,
        ocr_output: PipelineOutput,
        *,
        image_path: str | Path | None = None,
    ) -> PipelineOutput:
        path = Path(image_path) if image_path is not None else None
        prompt = build_stage2_prompt(ocr_output.text, ocr_output.boxes)
        raw_obj, latency = _timed(self.grader_call, prompt, path)
        raw = str(raw_obj)
        parsed, final_raw, repair_attempted, repair_succeeded, repair_latency = _parse_with_repair(
            model_call=self.grader_call,
            raw=raw,
            image_path=path,
            image_size=_image_size(path),
            require_text=False,
            repair_schema=STAGE2_SCHEMA,
        )
        stage2 = latency + repair_latency
        return PipelineOutput(
            strategy_name=self.name,
            image=ocr_output.image,
            text=ocr_output.text,
            boxes=ocr_output.boxes,
            errors=parsed.errors,
            feedback=parsed.feedback,
            stage1_latency=ocr_output.stage1_latency,
            stage2_latency=stage2,
            total_latency=ocr_output.stage1_latency + stage2,
            parse_valid=parsed.valid,
            repair_attempted=repair_attempted,
            repair_succeeded=repair_succeeded,
            raw_response=final_raw,
            notes=parsed.problems,
            metadata={
                "mode": "two_stage",
                "ocr_strategy": ocr_output.strategy_name,
                **ocr_output.metadata,
            },
        )

    def _coerce_ocr_output(
        self,
        value: PipelineOutput | dict,
        image_path: Path,
        measured_stage1: float,
    ) -> PipelineOutput:
        if isinstance(value, PipelineOutput):
            if value.stage1_latency <= 0:
                value.stage1_latency = measured_stage1
                value.total_latency = measured_stage1
            return value
        output = PipelineOutput.from_ocr_dict(
            value,
            strategy_name=f"{self.name}_stage1",
            image=image_path,
            image_size=_image_size(image_path),
        )
        if output.stage1_latency <= 0:
            output.stage1_latency = measured_stage1
            output.total_latency = measured_stage1
        return output
