#!/usr/bin/env python3
"""
Hunyuan VL manual evaluation helper.

Hunyuan VL (~4B) has no API/HF access — only available via lmarena chat.
This script prepares the evaluation materials and scores the results.

Step 1 — Prepare:
    python scripts/eval_hunyuan_manual.py prepare

    This outputs:
      - A prompt to paste into lmarena chat (prints to stdout)
      - A response template file: benchmark/results/hunyuan_vl_responses_template.json

Step 2 — Evaluate (manual):
    Go to https://lmarena.ai → select Hunyuan VL model
    For each of the 5 images, upload the image and paste the prompt.
    Copy each response into the template JSON, then save as:
      benchmark/results/hunyuan_vl_responses.json

Step 3 — Score:
    python scripts/eval_hunyuan_manual.py score

    Reads the responses, computes CER/WER against ground truth, saves results.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.metrics import compute_cer_normalized, compute_wer_normalized

HANDWRITTEN_DIR = PROJECT_ROOT / "benchmark" / "test_dataset" / "handwritten"
GT_PATH = PROJECT_ROOT / "benchmark" / "test_dataset" / "ground_truth_handwritten.json"
RESULTS_DIR = PROJECT_ROOT / "benchmark" / "results"
TEMPLATE_PATH = RESULTS_DIR / "hunyuan_vl_responses_template.json"
RESPONSES_PATH = RESULTS_DIR / "hunyuan_vl_responses.json"
RESULT_PATH = RESULTS_DIR / "hunyuan_vl_handwritten.json"

# 5 representative images — mix of easy/medium/hard based on Qwen CERs
SAMPLE_IMAGES = [
    "f07-069.png",   # Easy (Qwen 4B CER 0.005, clean cursive)
    "c02-017.png",   # Easy (Qwen 4B CER 0.016, clear handwriting)
    "a04-039.png",   # Medium (Qwen 4B CER 0.026, standard IAM form)
    "b06-045.png",   # Medium (Qwen 4B CER 0.026, varied style)
    "r02-117.png",   # Hard (Qwen 4B CER 0.103, messy handwriting)
]

WORD_PROMPT = (
    "Transcribe every individual word in this handwritten text. "
    "For each word, output exactly: [x1, y1, x2, y2] the_word\n"
    "One word per line. Preserve reading order. Do not group words together."
)

LINE_PROMPT = (
    "Transcribe this handwritten text. Output the full text only, "
    "preserving line breaks. Do not add commentary."
)


def cmd_prepare():
    """Generate the prompt and response template."""
    gt_data = json.loads(GT_PATH.read_text()) if GT_PATH.exists() else []
    gt_by_name = {e["image"]: e for e in gt_data}

    # Check images exist
    missing = [img for img in SAMPLE_IMAGES if not (HANDWRITTEN_DIR / img).exists()]
    if missing:
        print(f"ERROR: Missing images: {missing}")
        sys.exit(1)

    print("=" * 70)
    print("HUNYUAN VL MANUAL EVALUATION — PREPARATION")
    print("=" * 70)
    print()
    print(f"Images selected: {len(SAMPLE_IMAGES)}")
    for i, img in enumerate(SAMPLE_IMAGES):
        gt = gt_by_name.get(img, {})
        print(f"  {i+1}. {img} — GT has {len(gt.get('text',''))} chars")
    print()
    print("---")
    print()
    print("PROMPT TO PASTE INTO LMARENA (word-level bbox):")
    print()
    print(WORD_PROMPT)
    print()
    print("---")
    print()
    print("ALTERNATIVE SIMPLE PROMPT (text-only, no bboxes):")
    print()
    print(LINE_PROMPT)
    print()
    print("---")
    print()
    print("INSTRUCTIONS:")
    print("  1. Go to https://lmarena.ai")
    print("  2. Select 'Hunyuan-VL' (or 'Hunyuan' vision model)")
    print("  3. For each of the 5 images above:")
    print("     a. Upload the image from:")
    print(f"        {HANDWRITTEN_DIR}/")
    print("     b. Paste the word-level prompt above")
    print("     c. Copy the full response into the template JSON")
    print(f"  4. Save as: {RESPONSES_PATH}")
    print()
    print(f"Template saved to: {TEMPLATE_PATH}")

    # Generate template
    template = {
        "candidate": "hunyuan_vl",
        "model": "Hunyuan-VL (~4B, lmarena chat)",
        "prompt": WORD_PROMPT,
        "images": [],
    }
    for img in SAMPLE_IMAGES:
        template["images"].append({
            "image": img,
            "response": "PASTE FULL MODEL RESPONSE HERE",
            "notes": "",
        })

    TEMPLATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEMPLATE_PATH.write_text(json.dumps(template, indent=2))
    print()
    print("Done. Fill in the template and run: python scripts/eval_hunyuan_manual.py score")


def cmd_score():
    """Score the manual responses against ground truth."""
    if not RESPONSES_PATH.exists():
        print(f"ERROR: {RESPONSES_PATH} not found.")
        print("First run: python scripts/eval_hunyuan_manual.py prepare")
        print("Then fill in the template and save as the above path.")
        sys.exit(1)

    gt_data = json.loads(GT_PATH.read_text())
    gt_by_name = {e["image"]: e for e in gt_data}
    responses = json.loads(RESPONSES_PATH.read_text())

    print("=" * 70)
    print("HUNYUAN VL MANUAL EVALUATION — SCORING")
    print("=" * 70)
    print()

    results = []
    cer_vals, wer_vals = [], []

    for entry in responses.get("images", []):
        img_name = entry["image"]
        response = entry.get("response", "").strip()

        if not response or response == "PASTE FULL MODEL RESPONSE HERE":
            print(f"  {img_name}: SKIPPED (no response)")
            continue

        gt_entry = gt_by_name.get(img_name, {})
        gt_text = gt_entry.get("text", "")

        cer = compute_cer_normalized(response, gt_text) if gt_text else 0.0
        wer = compute_wer_normalized(response, gt_text) if gt_text else 0.0

        cer_vals.append(cer)
        wer_vals.append(wer)

        results.append({
            "image": img_name,
            "response": response,
            "gt_text": gt_text,
            "cer": round(cer, 4),
            "wer": round(wer, 4),
        })

        print(f"  {img_name}: CER={cer:.4f} WER={wer:.4f}")

    if not cer_vals:
        print("No responses scored.")
        sys.exit(1)

    avg_cer = sum(cer_vals) / len(cer_vals)
    avg_wer = sum(wer_vals) / len(wer_vals)

    output = {
        "candidate": "hunyuan_vl",
        "model": "Hunyuan-VL (~4B, lmarena chat)",
        "num_images": len(results),
        "images": results,
        "aggregate": {
            "cer": round(avg_cer, 4),
            "wer": round(avg_wer, 4),
        },
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(output, indent=2))

    print(f"\nAggregate: CER={avg_cer:.4f} WER={avg_wer:.4f}")
    print(f"Saved to: {RESULT_PATH}")

    # Show comparison
    print()
    print("--- QUICK COMPARISON ---")
    print(f"  Hunyuan VL (manual, {len(results)} images): CER={avg_cer:.4f}")
    print(f"  Qwen3-VL-4B (25 images, line):      CER=0.022")
    print(f"  Qwen3-VL-8B (25 images, word):      CER=0.035")
    print(f"  Florence-2-large (25 images, line):  CER=0.061")
    print(f"  Google Doc AI (25 images, word):     CER=0.108")
    print(f"  Tesseract 5 (25 images, word):       CER=0.443")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/eval_hunyuan_manual.py <prepare|score>")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "prepare":
        cmd_prepare()
    elif cmd == "score":
        cmd_score()
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python scripts/eval_hunyuan_manual.py <prepare|score>")
        sys.exit(1)
