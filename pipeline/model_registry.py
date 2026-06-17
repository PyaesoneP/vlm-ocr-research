"""Phase 4 model registry for Stage 1 text and localization sources."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Stage1ModelSpec:
    """Metadata for a Phase 2/3 model as a Phase 4 Stage 1 candidate."""

    name: str
    display_name: str
    family: str
    source_kind: str
    artifact_relpath: str | None = None
    can_transcribe: bool = True
    can_localize: bool = False
    box_granularity: str = "none"
    prompted: bool = False
    automated_matrix: bool = False
    notes: str = ""

    def to_dict(self, project_root: Path | None = None) -> dict:
        data = asdict(self)
        if self.artifact_relpath and project_root is not None:
            path = project_root / self.artifact_relpath
            data["artifact_exists"] = path.exists()
        return data


STAGE1_MODEL_REGISTRY: dict[str, Stage1ModelSpec] = {
    "realworld_source_text": Stage1ModelSpec(
        name="realworld_source_text",
        display_name="Real-world gold source text",
        family="reference",
        source_kind="gold_reference",
        artifact_relpath="benchmark/test_dataset/realworld_writing_errors.json",
        can_localize=True,
        box_granularity="word",
        automated_matrix=False,
        notes="Diagnostic only; not a real pipeline input.",
    ),
    "qwen3vl_4b_live_word_ocr": Stage1ModelSpec(
        name="qwen3vl_4b_live_word_ocr",
        display_name="Qwen3-VL-4B live word OCR",
        family="vlm",
        source_kind="live_local",
        can_localize=True,
        box_granularity="word",
        prompted=True,
        automated_matrix=True,
    ),
    "qwen3vl_4b_verbatim_word_ocr": Stage1ModelSpec(
        name="qwen3vl_4b_verbatim_word_ocr",
        display_name="Qwen3-VL-4B live verbatim word OCR",
        family="vlm",
        source_kind="live_local",
        can_localize=True,
        box_granularity="word",
        prompted=True,
        automated_matrix=True,
        notes="Prompted to preserve the student's actual errors instead of normalizing them.",
    ),
    "qwen3vl_4b_wordlevel": Stage1ModelSpec(
        name="qwen3vl_4b_wordlevel",
        display_name="Qwen3-VL-4B word-level artifact",
        family="vlm",
        source_kind="saved_artifact",
        artifact_relpath="benchmark/results/qwen3_vl_4b_wordlevel_handwritten.json",
        can_localize=True,
        box_granularity="word",
        prompted=True,
        automated_matrix=True,
    ),
    "qwen3vl_8b_wordlevel_api": Stage1ModelSpec(
        name="qwen3vl_8b_wordlevel_api",
        display_name="Qwen3-VL-8B API word-level artifact",
        family="vlm",
        source_kind="saved_artifact_cloud_gated",
        artifact_relpath="benchmark/results/qwen3_vl_8b_wordlevel_handwritten.json",
        can_localize=True,
        box_granularity="word",
        prompted=True,
        automated_matrix=True,
        notes="Live API runs require explicit approval and cost estimate.",
    ),
    "florence2_large_wordlevel": Stage1ModelSpec(
        name="florence2_large_wordlevel",
        display_name="Florence-2-large artifact",
        family="vlm",
        source_kind="saved_artifact",
        artifact_relpath="benchmark/results/florence2_large_wordlevel_handwritten.json",
        can_localize=True,
        box_granularity="line",
        prompted=True,
        automated_matrix=True,
    ),
    "florence2_live_region_ocr": Stage1ModelSpec(
        name="florence2_live_region_ocr",
        display_name="Florence-2-large live region OCR",
        family="vlm",
        source_kind="live_local_alt_env",
        can_localize=True,
        box_granularity="line",
        prompted=True,
        automated_matrix=True,
        notes="Requires the florencetf environment; line/region boxes are diagnostic for word IoU.",
    ),
    "google_docai_wordlevel": Stage1ModelSpec(
        name="google_docai_wordlevel",
        display_name="Google Document AI word-level artifact",
        family="cloud_ocr",
        source_kind="saved_artifact_cloud_gated",
        artifact_relpath="benchmark/results/google_docai_wordlevel_handwritten.json",
        can_localize=True,
        box_granularity="word",
        automated_matrix=True,
        notes="Live cloud runs require explicit approval and cost estimate.",
    ),
    "google_docai_live_word_ocr": Stage1ModelSpec(
        name="google_docai_live_word_ocr",
        display_name="Google Document AI live word OCR",
        family="cloud_ocr",
        source_kind="cloud_gated",
        can_localize=True,
        box_granularity="word",
        automated_matrix=False,
        notes="Live cloud run; requires explicit approval and a per-page cost estimate.",
    ),
    "tesseract_wordlevel": Stage1ModelSpec(
        name="tesseract_wordlevel",
        display_name="Tesseract word-level artifact",
        family="traditional_ocr",
        source_kind="saved_artifact",
        artifact_relpath="benchmark/results/tesseract_wordlevel_handwritten.json",
        can_localize=True,
        box_granularity="word",
        automated_matrix=True,
    ),
    "tesseract_live_word_ocr": Stage1ModelSpec(
        name="tesseract_live_word_ocr",
        display_name="Tesseract live word OCR",
        family="traditional_ocr",
        source_kind="live_local",
        can_localize=True,
        box_granularity="word",
        automated_matrix=True,
    ),
    "doctr_wordlevel": Stage1ModelSpec(
        name="doctr_wordlevel",
        display_name="docTR word-level artifact",
        family="traditional_ocr",
        source_kind="saved_artifact",
        artifact_relpath="benchmark/results/doctr_wordlevel_handwritten.json",
        can_localize=True,
        box_granularity="word",
        automated_matrix=True,
    ),
    "doctr_live_word_ocr": Stage1ModelSpec(
        name="doctr_live_word_ocr",
        display_name="docTR live word OCR",
        family="traditional_ocr",
        source_kind="live_local",
        can_localize=True,
        box_granularity="word",
        automated_matrix=True,
    ),
    "easyocr_wordlevel": Stage1ModelSpec(
        name="easyocr_wordlevel",
        display_name="EasyOCR word-level artifact",
        family="traditional_ocr",
        source_kind="saved_artifact",
        artifact_relpath="benchmark/results/easyocr_wordlevel_handwritten.json",
        can_localize=True,
        box_granularity="word",
        automated_matrix=True,
    ),
    "easyocr_live_word_ocr": Stage1ModelSpec(
        name="easyocr_live_word_ocr",
        display_name="EasyOCR live word OCR",
        family="traditional_ocr",
        source_kind="live_local",
        can_localize=True,
        box_granularity="word",
        automated_matrix=True,
    ),
    "locateanything_wordlevel": Stage1ModelSpec(
        name="locateanything_wordlevel",
        display_name="LocateAnything-3B artifact",
        family="localization_vlm",
        source_kind="saved_artifact",
        artifact_relpath="benchmark/results/locateanything_wordlevel_handwritten.json",
        can_transcribe=False,
        can_localize=True,
        box_granularity="word",
        prompted=True,
        automated_matrix=True,
        notes="Localization candidate; text labels are not transcription-quality yet.",
    ),
    "hunyuan_vl_manual": Stage1ModelSpec(
        name="hunyuan_vl_manual",
        display_name="Hunyuan VL manual artifact",
        family="vlm",
        source_kind="manual_only",
        artifact_relpath="benchmark/results/hunyuan_vl_handwritten.json",
        prompted=True,
        automated_matrix=False,
        notes="Manual/lmarena ceiling result; not live automatable in this repo.",
    ),
    "hunyuan_vl_live_manual": Stage1ModelSpec(
        name="hunyuan_vl_live_manual",
        display_name="Hunyuan VL live manual entry",
        family="vlm",
        source_kind="manual_only",
        prompted=True,
        automated_matrix=False,
        notes="Manual/lmarena only; no automated live API path is available in this repo.",
    ),
    "paddleocr_vl": Stage1ModelSpec(
        name="paddleocr_vl",
        display_name="PaddleOCR-VL artifact",
        family="document_vlm",
        source_kind="saved_artifact_requires_docker",
        artifact_relpath="benchmark/results/paddleocr_vl_handwritten.json",
        can_localize=True,
        box_granularity="block",
        automated_matrix=False,
        notes="Docker-only on Blackwell; existing artifact lacks stable per-image ids.",
    ),
    "paddleocr_vl_live_ocr": Stage1ModelSpec(
        name="paddleocr_vl_live_ocr",
        display_name="PaddleOCR-VL live OCR",
        family="document_vlm",
        source_kind="live_local_requires_docker",
        can_localize=True,
        box_granularity="block",
        automated_matrix=True,
        notes="Requires the PaddleOCR-VL Docker/native Paddle environment on Blackwell.",
    ),
    "got_ocr2": Stage1ModelSpec(
        name="got_ocr2",
        display_name="GOT-OCR2.0 artifact",
        family="ocr_vlm",
        source_kind="saved_artifact",
        artifact_relpath="benchmark/results/got_ocr2_handwriting.json",
        box_granularity="none",
        prompted=True,
        automated_matrix=False,
    ),
    "got_ocr2_live_ocr": Stage1ModelSpec(
        name="got_ocr2_live_ocr",
        display_name="GOT-OCR2.0 live OCR",
        family="ocr_vlm",
        source_kind="live_local",
        can_localize=False,
        box_granularity="none",
        prompted=True,
        automated_matrix=True,
    ),
    "smoldocling": Stage1ModelSpec(
        name="smoldocling",
        display_name="SmolDocling artifact",
        family="document_vlm",
        source_kind="saved_artifact",
        artifact_relpath="benchmark/results/smoldocling_handwriting.json",
        can_localize=True,
        box_granularity="coarse",
        prompted=True,
        automated_matrix=False,
    ),
    "smoldocling_live_ocr": Stage1ModelSpec(
        name="smoldocling_live_ocr",
        display_name="SmolDocling live OCR",
        family="document_vlm",
        source_kind="live_local",
        can_localize=True,
        box_granularity="coarse",
        prompted=True,
        automated_matrix=True,
    ),
    "nemotron_ocr_v2": Stage1ModelSpec(
        name="nemotron_ocr_v2",
        display_name="Nemotron OCR v2 artifact",
        family="ocr_vlm",
        source_kind="saved_artifact",
        artifact_relpath="benchmark/results/nemotron_ocr_v2_handwriting.json",
        can_localize=True,
        box_granularity="partial",
        automated_matrix=False,
    ),
    "nemotron_ocr_v2_live_ocr": Stage1ModelSpec(
        name="nemotron_ocr_v2_live_ocr",
        display_name="Nemotron OCR v2 live OCR",
        family="ocr_vlm",
        source_kind="live_local_alt_env",
        can_localize=True,
        box_granularity="partial",
        automated_matrix=True,
        notes="Requires the aiml conda environment and local Nemotron model directory.",
    ),
    "monkeyocr": Stage1ModelSpec(
        name="monkeyocr",
        display_name="MonkeyOCR artifact",
        family="document_vlm",
        source_kind="saved_artifact",
        artifact_relpath="benchmark/results/monkeyocr_handwritten.json",
        can_localize=False,
        box_granularity="none",
        automated_matrix=False,
    ),
    "monkeyocr_live_ocr": Stage1ModelSpec(
        name="monkeyocr_live_ocr",
        display_name="MonkeyOCR live OCR",
        family="document_vlm",
        source_kind="live_local_server",
        can_localize=False,
        box_granularity="none",
        automated_matrix=True,
        notes="Requires a running local llama.cpp MonkeyOCR server.",
    ),
    "trocr_base": Stage1ModelSpec(
        name="trocr_base",
        display_name="TrOCR-base artifact",
        family="ocr_transformer",
        source_kind="saved_artifact",
        artifact_relpath="benchmark/results/trocr_base_handwritten.json",
        can_localize=True,
        box_granularity="heuristic_line",
        automated_matrix=False,
    ),
    "trocr_base_live_line_ocr": Stage1ModelSpec(
        name="trocr_base_live_line_ocr",
        display_name="TrOCR-base live line OCR",
        family="ocr_transformer",
        source_kind="live_local",
        can_localize=True,
        box_granularity="heuristic_line",
        automated_matrix=True,
        notes="Uses the existing heuristic line detector before TrOCR recognition.",
    ),
    "trocr_large": Stage1ModelSpec(
        name="trocr_large",
        display_name="TrOCR-large artifact",
        family="ocr_transformer",
        source_kind="saved_artifact",
        artifact_relpath="benchmark/results/trocr_large_handwritten.json",
        can_localize=True,
        box_granularity="heuristic_line",
        automated_matrix=False,
    ),
    "trocr_large_live_line_ocr": Stage1ModelSpec(
        name="trocr_large_live_line_ocr",
        display_name="TrOCR-large live line OCR",
        family="ocr_transformer",
        source_kind="live_local",
        can_localize=True,
        box_granularity="heuristic_line",
        automated_matrix=True,
        notes="Uses the existing heuristic line detector before TrOCR recognition.",
    ),
    "qwen3vl_8b_api_live_word_ocr": Stage1ModelSpec(
        name="qwen3vl_8b_api_live_word_ocr",
        display_name="Qwen3-VL-8B API live word OCR",
        family="vlm",
        source_kind="cloud_gated",
        can_localize=True,
        box_granularity="word",
        prompted=True,
        automated_matrix=False,
        notes="Live API runs require explicit approval and cost estimate.",
    ),
}


def live_local_text_sources() -> list[str]:
    return [
        name for name, spec in STAGE1_MODEL_REGISTRY.items()
        if spec.source_kind.startswith("live_local") and spec.can_transcribe
    ]


def automated_artifact_text_sources(project_root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, spec in STAGE1_MODEL_REGISTRY.items():
        if not spec.automated_matrix or spec.artifact_relpath is None:
            continue
        path = project_root / spec.artifact_relpath
        if path.exists():
            paths[name] = path
    return paths


def registry_metadata(project_root: Path) -> dict[str, dict]:
    return {
        name: spec.to_dict(project_root)
        for name, spec in sorted(STAGE1_MODEL_REGISTRY.items())
    }
