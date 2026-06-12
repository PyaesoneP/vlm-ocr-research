"""
Minimal smoke test for PaddleOCR-VL Docker image on Blackwell sm_120.
Run inside the container:
    docker run --rm --gpus all --network host --shm-size=8g \
      -e PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
      -v paddlex_models:/home/paddleocr/.paddlex \
      -v /home/pyaes/vlm-ocr-research/benchmark/test_dataset:/data:ro \
      -v /home/pyaes/vlm-ocr-research/scripts:/scripts:ro \
      ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-nvidia-gpu-sm120-offline \
      python3 /scripts/docker_smoke_test.py
"""

import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Test image — first curated image
# ---------------------------------------------------------------------------
DATA_DIR = Path("/data/curated")
images = sorted(p for p in DATA_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

if not images:
    print("ERROR: No test images found in /data/curated")
    sys.exit(1)

image_path = str(images[0])
print(f"Test image: {image_path}")
print(f"Image exists: {os.path.exists(image_path)}")

# ---------------------------------------------------------------------------
# Load PaddleOCR-VL
# ---------------------------------------------------------------------------
print("\n[1/3] Loading PaddleOCR-VL pipeline ...")
from paddleocr import PaddleOCRVL

pipeline = PaddleOCRVL(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
)
print("Pipeline loaded.")

# ---------------------------------------------------------------------------
# Run inference
# ---------------------------------------------------------------------------
print("\n[2/3] Running inference ...")
import paddle

paddle.device.synchronize()
t0 = time.perf_counter()

output = pipeline.predict(image_path)

paddle.device.synchronize()
elapsed = time.perf_counter() - t0

# ---------------------------------------------------------------------------
# Parse output
# ---------------------------------------------------------------------------
blocks = []
full_text = ""

if output:
    for res in output:
        # JSON path (has bboxes)
        if hasattr(res, "json") and res.json:
            j = res.json
            inner = j.get("res", j) if isinstance(j, dict) else {}
            parsing_list = inner.get("parsing_res_list", [])
            for item in parsing_list:
                text = item.get("block_content", "")
                bbox_raw = item.get("block_bbox", "")
                if text:
                    blocks.append({"text": str(text), "bbox": str(bbox_raw)[:80]})
                    full_text += str(text) + " "

        # Markdown fallback
        if not full_text and hasattr(res, "markdown") and res.markdown:
            md = res.markdown
            full_text = md.get("markdown_texts", str(md)) if isinstance(md, dict) else str(md)

        # Plain text fallback
        if not full_text and hasattr(res, "text") and res.text:
            full_text = str(res.text)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
print(f"\n[3/3] Results:")
print(f"  Latency: {elapsed:.2f}s")
print(f"  Blocks:  {len(blocks)}")
print(f"  Text:    {full_text.strip()[:200]}...")
print(f"\nPASS — PaddleOCR-VL Docker path works.")
