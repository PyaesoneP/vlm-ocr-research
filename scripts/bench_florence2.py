#!/usr/bin/env python3
"""Standalone Florence-2-large benchmark on handwritten-only IAM images."""

import json
import time
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.metrics import compute_cer_normalized, compute_wer_normalized

MODEL_ID = "microsoft/Florence-2-large"
HANDWRITTEN_DIR = PROJECT_ROOT / "benchmark" / "test_dataset" / "handwritten"
GT_FILE = PROJECT_ROOT / "benchmark" / "test_dataset" / "ground_truth_handwritten.json"
RESULTS_FILE = PROJECT_ROOT / "benchmark" / "results" / "florence2_large_handwritten.json"

# --- Load model ---
print(f"Loading {MODEL_ID}...")
t0 = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, trust_remote_code=True, torch_dtype=torch.float16
).to("cuda").eval()
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
print(f"Loaded in {time.perf_counter()-t0:.1f}s")

# --- Load ground truth ---
with open(GT_FILE) as f:
    gt_list = json.load(f)
gt_map = {g["image"]: g["text"] for g in gt_list}

# --- Get images ---
images = sorted(p for p in HANDWRITTEN_DIR.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
print(f"Running on {len(images)} images...\n")

results = {"candidate": "florence2_large", "model": MODEL_ID, "images": []}
total_cer = 0.0
total_wer = 0.0
total_time = 0.0

prompt = "<OCR>"

for i, img_path in enumerate(images):
    name = img_path.name
    image = Image.open(img_path).convert("RGB")

    # Warmup
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda", torch.float16)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=512,
        num_beams=3,
    )
    raw = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(raw, task=prompt, image_size=image.size)
    ocr_text = parsed[prompt]

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_start

    gt_text = gt_map.get(name, "")
    cer = compute_cer_normalized(ocr_text, gt_text)
    wer = compute_wer_normalized(ocr_text, gt_text)

    total_cer += cer
    total_wer += wer
    total_time += elapsed

    results["images"].append({
        "image": name,
        "text": ocr_text,
        "cer": round(cer, 4),
        "wer": round(wer, 4),
        "latency_s": round(elapsed, 2),
    })

    print(f"  [{i+1:2d}/{len(images)}] {name:20s}  CER={cer:.3f}  WER={wer:.3f}  {elapsed:.2f}s")

n = len(images)
print(f"\n{'='*50}")
print(f"FLORENCE-2-LARGE FINAL:  CER={total_cer/n:.3f}  WER={total_wer/n:.3f}  Time={total_time/n:.2f}s")
print(f"{'='*50}")

results["summary"] = {
    "mean_cer": round(total_cer / n, 4),
    "mean_wer": round(total_wer / n, 4),
    "mean_latency_s": round(total_time / n, 2),
    "total_time_s": round(total_time, 1),
    "num_images": n,
}

RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {RESULTS_FILE}")
