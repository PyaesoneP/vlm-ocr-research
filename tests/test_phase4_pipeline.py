from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pipeline.contracts import ErrorFinding, PipelineOutput, TextBox
from pipeline.localization import bbox_from_word_indices
from pipeline.metrics import NOT_APPLICABLE, compute_truthfulness_metrics, evaluate_phase4_output
from pipeline.model_registry import STAGE1_MODEL_REGISTRY
from pipeline.parsing import parse_model_json
from pipeline.strategies import SinglePassStrategy, TwoStageStrategy
from scripts.benchmark_phase4 import (
    LIVE_STAGE1_TEXT_SOURCES,
    parse_qwen_word_response,
    parse_two_stage_strategy_name,
    select_images,
)


class Phase4PipelineTests(unittest.TestCase):
    def _temp_image(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "sample.png"
        path.write_bytes(b"not a real image; tests do not inspect pixels")
        return path

    def _temp_png(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "sample.png"
        # 1x1 transparent PNG.
        path.write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4"
            b"\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05"
            b"\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        return path

    def test_parse_filters_invalid_error_types_and_clamps_boxes(self) -> None:
        raw = """```json
        {
          "errors": [
            {"type": "spelling", "bbox": [50, 20, 10, 80], "description": "typo"},
            {"type": "style", "bbox": [0, 0, 5, 5], "description": "not allowed"}
          ],
          "feedback": "Good effort."
        }
        ```"""
        parsed = parse_model_json(raw, image_size=(100, 100))

        self.assertFalse(parsed.valid)
        self.assertEqual(len(parsed.errors), 1)
        self.assertEqual(parsed.errors[0].type, "spelling")
        self.assertEqual(parsed.errors[0].bbox, [10, 20, 50, 80])
        self.assertIn("invalid type", " ".join(parsed.problems))

    def test_parse_denormalizes_qwen_1000_space_boxes(self) -> None:
        raw = json.dumps({
            "text": "Hello",
            "blocks": [{"bbox": [0, 0, 999, 500], "text": "Hello"}],
            "errors": [],
            "feedback": {"summary": "ok"},
        })
        parsed = parse_model_json(raw, image_size=(2000, 1200), require_text=True, require_boxes=True)

        self.assertTrue(parsed.valid)
        self.assertEqual(parsed.boxes[0].bbox, [0, 0, 2000, 601])

    def test_word_indices_union(self) -> None:
        boxes = [
            TextBox(index=0, bbox=[10, 10, 20, 20], text="a"),
            TextBox(index=1, bbox=[24, 12, 50, 22], text="word"),
            TextBox(index=2, bbox=[60, 15, 80, 24], text="later"),
        ]
        self.assertEqual(bbox_from_word_indices(boxes, [0, 1]), [10, 10, 50, 22])
        self.assertEqual(bbox_from_word_indices(boxes, [9]), [0, 0, 0, 0])

    def test_parse_qwen_word_response_accepts_boxes_only(self) -> None:
        try:
            import PIL  # noqa: F401
        except ImportError:
            self.skipTest("Pillow is not installed in this interpreter")
        parsed = parse_qwen_word_response("[0, 0, 999, 999], [100, 100, 200, 200]", self._temp_png())

        self.assertEqual(len(parsed["blocks"]), 2)
        self.assertFalse(parsed["_text_labels_present"])
        self.assertEqual(parsed["blocks"][0]["bbox"], [0, 0, 1, 1])

    def test_phase4_strategy_name_expresses_full_composition(self) -> None:
        spec = parse_two_stage_strategy_name(
            "two_stage__florence2_large_wordlevel__tesseract_word_boxes__qwen3vl_4b_grader"
        )

        self.assertEqual(spec.text_source, "florence2_large_wordlevel")
        self.assertEqual(spec.box_source, "tesseract_word_boxes")
        self.assertEqual(spec.grader, "qwen3vl_4b_grader")
        self.assertEqual(
            parse_two_stage_strategy_name("two_stage_qwen_text_tesseract_boxes").name,
            "two_stage__qwen3vl_4b_wordlevel__tesseract_word_boxes__qwen3vl_4b_grader",
        )

    def test_live_stage1_strategy_uses_same_stage1_boxes(self) -> None:
        spec = parse_two_stage_strategy_name(
            "two_stage__tesseract_live_word_ocr__same_stage1_boxes__qwen3vl_4b_grader"
        )

        self.assertEqual(spec.text_source, "tesseract_live_word_ocr")
        self.assertEqual(spec.box_source, "same_stage1_boxes")

    def test_verbatim_stage1_strategy_is_available(self) -> None:
        spec = parse_two_stage_strategy_name(
            "two_stage__qwen3vl_4b_verbatim_word_ocr__same_stage1_boxes__qwen3vl_4b_grader"
        )

        self.assertEqual(spec.text_source, "qwen3vl_4b_verbatim_word_ocr")
        self.assertEqual(spec.box_source, "same_stage1_boxes")

    def test_stage1_registry_covers_non_qwen_models(self) -> None:
        for name in ["paddleocr_vl", "florence2_large_wordlevel", "doctr_live_word_ocr", "easyocr_live_word_ocr", "hunyuan_vl_manual"]:
            self.assertIn(name, STAGE1_MODEL_REGISTRY)

        self.assertEqual(STAGE1_MODEL_REGISTRY["paddleocr_vl"].source_kind, "saved_artifact_requires_docker")

    def test_live_registry_covers_prior_local_candidates(self) -> None:
        for name in [
            "florence2_live_region_ocr",
            "got_ocr2_live_ocr",
            "smoldocling_live_ocr",
            "nemotron_ocr_v2_live_ocr",
            "paddleocr_vl_live_ocr",
            "monkeyocr_live_ocr",
            "trocr_base_live_line_ocr",
            "trocr_large_live_line_ocr",
        ]:
            self.assertIn(name, STAGE1_MODEL_REGISTRY)
            self.assertIn(name, LIVE_STAGE1_TEXT_SOURCES)

    def test_live_candidate_strategy_names_are_selectable(self) -> None:
        spec = parse_two_stage_strategy_name(
            "two_stage__got_ocr2_live_ocr__same_stage1_boxes__qwen3vl_4b_grader"
        )

        self.assertEqual(spec.text_source, "got_ocr2_live_ocr")
        self.assertEqual(spec.box_source, "same_stage1_boxes")

    def test_explicit_images_are_not_truncated_by_default_max(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        image_dir = Path(tmp.name)
        for name in ["a.jpg", "b.jpg"]:
            (image_dir / name).write_bytes(b"")

        images = select_images(1, ["a.jpg", "b.jpg"], image_dir=image_dir)

        self.assertEqual([path.name for path in images], ["a.jpg", "b.jpg"])

    def test_no_error_ground_truth_returns_not_applicable_f1(self) -> None:
        output = PipelineOutput(
            strategy_name="test",
            image="sample.png",
            text="Clean text.",
            boxes=[
                TextBox(index=0, bbox=[0, 0, 45, 20], text="Clean"),
                TextBox(index=1, bbox=[50, 0, 100, 20], text="text."),
            ],
            parse_valid=True,
        )
        gt = {
            "image": "sample.png",
            "text": "Clean text.",
            "blocks": [{"bbox": [0, 0, 100, 20], "text": "Clean text."}],
            "words": [
                {"bbox": [0, 0, 45, 20], "text": "Clean"},
                {"bbox": [50, 0, 100, 20], "text": "text."},
            ],
            "errors": [],
        }
        metrics = evaluate_phase4_output(output, gt, gt)

        self.assertEqual(metrics["error_detection_f1"], NOT_APPLICABLE)
        self.assertEqual(metrics["error_box_iou"], NOT_APPLICABLE)
        self.assertEqual(metrics["false_positive_count"], 0)
        self.assertEqual(metrics["word_iou"], 1.0)

    def test_error_text_f1_is_separate_from_localization(self) -> None:
        output = PipelineOutput(
            strategy_name="test",
            image="sample.png",
            text="I ate bred.",
            parse_valid=True,
        )
        output.errors = [
            # Correct type/evidence/correction, deliberately wrong bbox.
            ErrorFinding(
                type="spelling",
                bbox=[200, 200, 260, 230],
                description="Fix spelling.",
                correction="bread",
                evidence_text="bred.",
            )
        ]
        gt = {
            "image": "sample.png",
            "text": "I ate bred.",
            "words": [{"bbox": [20, 20, 80, 40], "text": "bred."}],
            "errors": [{
                "type": "spelling",
                "bbox": [20, 20, 80, 40],
                "evidence_text": "bred",
                "correction": "bread",
            }],
        }
        metrics = evaluate_phase4_output(output, gt, gt)

        self.assertEqual(metrics["error_text_f1"], 1.0)
        self.assertEqual(metrics["error_detection_f1"], 0.0)

    def test_truthfulness_metrics_detect_preserved_evidence_and_leaks(self) -> None:
        gt = {
            "text": "I bought bred and waited ten minuts.",
            "errors": [
                {"type": "spelling", "evidence_text": "bred", "correction": "bread"},
                {"type": "spelling", "evidence_text": "minuts", "correction": "minutes"},
            ],
        }

        preserved = compute_truthfulness_metrics("I bought bred and waited ten minutes.", gt)

        self.assertEqual(preserved["evidence_preserved_count"], 1)
        self.assertEqual(preserved["evidence_total"], 2)
        self.assertEqual(preserved["correction_leak_count"], 1)
        self.assertEqual(preserved["evidence_preserved_rate"], 0.5)

    def test_single_pass_repairs_invalid_json_once(self) -> None:
        image_path = self._temp_image()
        calls: list[str] = []

        def model_call(prompt: str, image_path: Path | None) -> str:
            calls.append(prompt)
            if len(calls) == 1:
                return "not json"
            return json.dumps({
                "text": "Clean text.",
                "blocks": [{"index": 0, "bbox": [0, 0, 100, 20], "text": "Clean text."}],
                "errors": [],
                "feedback": {"summary": "Looks clean."},
            })

        strategy = SinglePassStrategy("single_test", model_call)
        output = strategy.run(image_path)

        self.assertTrue(output.parse_valid)
        self.assertTrue(output.repair_attempted)
        self.assertTrue(output.repair_succeeded)
        self.assertEqual(len(calls), 2)
        self.assertEqual(output.text, "Clean text.")

    def test_two_stage_smoke(self) -> None:
        image_path = self._temp_image()

        def ocr_call(path: Path) -> PipelineOutput:
            return PipelineOutput(
                strategy_name="fake_ocr",
                image=path.name,
                text="i recieved it.",
                boxes=[TextBox(index=0, bbox=[10, 10, 90, 30], text="i recieved it.")],
                stage1_latency=0.1,
            )

        def grader_call(prompt: str, image_path: Path | None) -> str:
            return json.dumps({
                "errors": [{
                    "type": "spelling",
                    "bbox": [20, 10, 70, 30],
                    "description": "Fix spelling.",
                    "correction": "received",
                    "evidence_text": "recieved",
                }],
                "feedback": {"summary": "Fix the spelling."},
            })

        strategy = TwoStageStrategy("two_stage_test", ocr_call, grader_call)
        output = strategy.run(image_path)

        self.assertTrue(output.parse_valid)
        self.assertEqual(output.stage1_latency, 0.1)
        self.assertGreaterEqual(output.stage2_latency, 0.0)
        self.assertEqual(output.errors[0].type, "spelling")


if __name__ == "__main__":
    unittest.main()
