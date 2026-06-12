#!/usr/bin/env python3
"""
Comprehensive PaddleOCR-VL benchmark — handwriting, full-form, bbox IoU, reading order.
Zero project imports — uses PaddleOCRVL directly.

Usage (inside Docker):
    # Handwriting-only benchmark (CER, WER, bbox IoU, reading order):
    docker run --rm --gpus all --network host --shm-size=8g \
      -e PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
      -e PYTHONUNBUFFERED=1 \
      -v paddlex_models:/home/paddleocr/.paddlex \
      -v /home/pyaes/vlm-ocr-research/benchmark/test_dataset:/data:ro \
      -v /home/pyaes/vlm-ocr-research/benchmark/results:/results \
      -v /home/pyaes/vlm-ocr-research/scripts/bench_paddleocr.py:/scripts/bench_paddleocr.py:ro \
      ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-nvidia-gpu-sm120-offline \
      python3 -u /scripts/bench_paddleocr.py --mode handwriting

    # Full-form benchmark:
    ... same docker command, but:
      python3 -u /scripts/bench_paddleocr.py --mode fullform

    # Both modes:
      python3 -u /scripts/bench_paddleocr.py --mode all
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration (overridable via env vars)
# ---------------------------------------------------------------------------

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "/results"))
MODE = os.environ.get("MODE", "handwriting")


# ---------------------------------------------------------------------------
# Metrics (inline — no project imports)
# ---------------------------------------------------------------------------

def _edit_distance(s1, s2):
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for c1 in s1:
        curr = [prev[0] + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\n", " ").replace("\r", " ")).strip()


def compute_cer(ref: str, hyp: str) -> float:
    r, h = _normalize(ref), _normalize(hyp)
    if not r:
        return 1.0 if h else 0.0
    return _edit_distance(r, h) / len(r)


def compute_wer(ref: str, hyp: str) -> float:
    rw, hw = _normalize(ref).split(), _normalize(hyp).split()
    if not rw:
        return 1.0 if hw else 0.0
    return _edit_distance(rw, hw) / len(rw)


def _norm_bbox_to_xyxy(bbox):
    """Convert [x1,y1,x2,y2] or [x,y,w,h] to [x1,y1,x2,y2]."""
    if len(bbox) != 4:
        return [0, 0, 0, 0]
    # heuristic: if w < x1 or h < y1, it's xywh
    if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
        return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
    return list(bbox)


def _bbox_iou(a, b):
    """Intersection-over-Union of two [x1,y1,x2,y2] boxes."""
    xa = max(a[0], b[0]); ya = max(a[1], b[1])
    xb = min(a[2], b[2]); yb = min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = max(0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_block_iou(pred_blocks, gt_blocks, iou_threshold=0.1):
    """Greedy match predicted blocks to GT blocks by IoU."""
    if not gt_blocks:
        return {"mean_iou": 1.0 if not pred_blocks else 0.0, "matched": 0,
                "total_gt": 0, "total_pred": len(pred_blocks), "recall": 0.0, "precision": 0.0}
    matched_gt = set()
    ious = []
    for p in pred_blocks:
        p_box = _norm_bbox_to_xyxy(p.get("bbox", []))
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gt_blocks):
            if j in matched_gt:
                continue
            g_box = _norm_bbox_to_xyxy(g.get("bbox", []))
            iou = _bbox_iou(p_box, g_box)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= iou_threshold:
            ious.append(best_iou)
            matched_gt.add(best_j)
    return {
        "mean_iou": sum(ious) / len(ious) if ious else 0.0,
        "matched": len(ious),
        "total_gt": len(gt_blocks),
        "total_pred": len(pred_blocks),
        "recall": len(ious) / len(gt_blocks) if gt_blocks else 0.0,
        "precision": len(ious) / len(pred_blocks) if pred_blocks else 0.0,
    }


def extract_reading_order(blocks):
    """Sort blocks top-to-bottom, left-to-right. Returns list of original indices."""
    if not blocks:
        return []
    indexed = [(i, _norm_bbox_to_xyxy(b.get("bbox", [0, 0, 0, 0]))[1],
                _norm_bbox_to_xyxy(b.get("bbox", [0, 0, 0, 0]))[0]) for i, b in enumerate(blocks)]
    indexed.sort(key=lambda x: (x[1], x[2]))
    return [i for i, _, _ in indexed]


def compute_kendall_tau(pred_order, gt_order):
    """Kendall's tau-b between two rankings."""
    if len(pred_order) < 2:
        return 1.0
    n = len(pred_order)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = (pred_order[i] - pred_order[j]) * (gt_order[i] - gt_order[j])
            if a > 0:
                concordant += 1
            elif a < 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total > 0 else 0.0


def compute_reading_order(pred_blocks, gt_blocks, gt_order):
    """Compute reading order tau by spatially matching pred blocks to GT blocks."""
    pred_order = extract_reading_order(pred_blocks)
    if not pred_order:
        return {"kendall_tau": 0.0, "pred_order": [], "gt_order": gt_order}

    # Spatial matching: for each pred block, find best GT match
    pred_ranks = []
    matched_gt = set()
    for pi in pred_order:
        p_box = _norm_bbox_to_xyxy(pred_blocks[pi].get("bbox", []))
        best_iou, best_gt = 0.0, -1
        for gj in range(len(gt_blocks)):
            if gj in matched_gt:
                continue
            g_box = _norm_bbox_to_xyxy(gt_blocks[gj].get("bbox", []))
            iou = _bbox_iou(p_box, g_box)
            if iou > best_iou:
                best_iou, best_gt = iou, gj
        if best_gt >= 0 and best_iou > 0.05:
            pred_ranks.append(best_gt)
            matched_gt.add(best_gt)

    if not pred_ranks or not gt_order or len(gt_order) != len(gt_blocks):
        return {"kendall_tau": 0.0, "pred_order": pred_ranks, "gt_order": gt_order}

    # Map GT order to the matched subset
    gt_ranks = [gt_order[g] for g in pred_ranks]
    pred_positions = list(range(len(pred_ranks)))

    tau = compute_kendall_tau(pred_positions, sorted(range(len(gt_ranks)), key=lambda i: gt_ranks[i]))
    return {"kendall_tau": tau, "pred_order": pred_ranks, "gt_order": gt_order}


def get_gpu_name() -> str:
    try:
        import paddle
        return paddle.device.cuda.get_device_name(0)
    except Exception:
        return "Unknown GPU"


# ---------------------------------------------------------------------------
# Parse PaddleOCR-VL output
# ---------------------------------------------------------------------------

def parse_output(output) -> tuple[str, list[dict]]:
    """Extract full text and blocks from PaddleOCRVL.predict() result."""
    blocks = []
    full_text = ""
    if output:
        for res in output:
            if hasattr(res, "json") and res.json:
                inner = res.json.get("res", res.json) if isinstance(res.json, dict) else {}
                for item in inner.get("parsing_res_list", []):
                    text = item.get("block_content", "")
                    bbox_raw = item.get("block_bbox", "")
                    if text:
                        if isinstance(bbox_raw, list):
                            bbox = [float(x) for x in bbox_raw[:4]]
                        elif isinstance(bbox_raw, str):
                            try:
                                bbox = [int(float(x.strip())) for x in bbox_raw.strip("[]").split(",")]
                            except (ValueError, AttributeError):
                                bbox = [0, 0, 0, 0]
                        else:
                            bbox = [0, 0, 0, 0]
                        blocks.append({
                            "bbox": bbox if len(bbox) == 4 else [0, 0, 0, 0],
                            "text": str(text),
                        })
                        full_text += str(text) + " "
            if not full_text and hasattr(res, "markdown") and res.markdown:
                md = res.markdown
                full_text = md.get("markdown_texts", str(md)) if isinstance(md, dict) else str(md)
            if not full_text and hasattr(res, "text") and res.text:
                full_text = str(res.text)
    return full_text.strip(), blocks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    mode = MODE
    # Override from CLI arg if provided
    for arg in sys.argv[1:]:
        if arg.startswith("--mode="):
            mode = arg.split("=", 1)[1]
        elif arg in ("--mode", "-m") and sys.argv.index(arg) + 1 < len(sys.argv):
            mode = sys.argv[sys.argv.index(arg) + 1]

    modes = ["handwriting", "fullform"] if mode == "all" else [mode]
    if not all(m in ("handwriting", "fullform") for m in modes):
        print(f"ERROR: Unknown mode '{mode}'. Use handwriting, fullform, or all.", file=sys.stderr)
        return 1

    # Load pipeline once
    print("Loading PaddleOCR-VL pipeline ...")
    from paddleocr import PaddleOCRVL
    import paddle
    pipeline = PaddleOCRVL(use_doc_orientation_classify=False, use_doc_unwarping=False)
    gpu_name = get_gpu_name()
    print(f"Pipeline loaded. GPU: {gpu_name}")

    for m in modes:
        if m == "handwriting":
            eval_handwriting(pipeline, gpu_name)
        elif m == "fullform":
            eval_fullform(pipeline, gpu_name)

    return 0


def eval_handwriting(pipeline, gpu_name: str) -> None:
    """Evaluate on cropped handwritten images."""
    import paddle

    images_dir = DATA_DIR / "handwritten"
    gt_path = DATA_DIR / "ground_truth_handwritten.json"
    candidate = "paddleocr_vl_handwritten"
    label = "Handwriting-Only"

    images = sorted(p for p in images_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not images:
        print(f"ERROR: No images in {images_dir}", file=sys.stderr)
        return

    gt_data = _load_gt(gt_path)
    _run_eval(pipeline, images, gt_data, candidate, label, gpu_name)


def eval_fullform(pipeline, gpu_name: str) -> None:
    """Evaluate on full-form curated images."""
    import paddle

    images_dir = DATA_DIR / "curated"
    gt_path = DATA_DIR / "ground_truth.json"
    candidate = "paddleocr_vl_fullform"
    label = "Full-Form"

    images = sorted(p for p in images_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not images:
        print(f"ERROR: No images in {images_dir}", file=sys.stderr)
        return

    gt_data = _load_gt(gt_path)
    _run_eval(pipeline, images, gt_data, candidate, label, gpu_name)


def _load_gt(gt_path: Path) -> dict[str, dict]:
    if not gt_path.exists():
        print(f"WARNING: GT not found at {gt_path}", file=sys.stderr)
        return {}
    return {e["image"]: e for e in json.loads(gt_path.read_text())}


def _run_eval(pipeline, images: list[Path], gt_data: dict[str, dict],
              candidate_name: str, label: str, gpu_name: str) -> None:
    import paddle

    print(f"\n{'='*60}")
    print(f"PaddleOCR-VL {label} Benchmark — {len(images)} images")
    print(f"{'='*60}")

    # Warmup
    print(f"Warmup on {images[0].name} ...")
    paddle.device.synchronize()
    pipeline.predict(str(images[0]))
    paddle.device.synchronize()
    print("Warmup complete.\n")

    latencies = []
    all_texts = []
    all_blocks = []
    image_names = []
    iou_results = []
    tau_results = []

    for i, img_path in enumerate(images):
        fname = img_path.name
        paddle.device.synchronize()
        t0 = time.perf_counter()
        output = pipeline.predict(str(img_path))
        paddle.device.synchronize()
        elapsed = time.perf_counter() - t0

        full_text, blocks = parse_output(output)
        latencies.append(elapsed)
        all_texts.append(full_text)
        all_blocks.append(blocks)
        image_names.append(fname)

        # Bbox IoU and reading order (if GT has blocks)
        gt_entry = gt_data.get(fname, {})
        gt_blocks = gt_entry.get("blocks", [])
        gt_order = gt_entry.get("reading_order", [])

        iou_info = {}
        tau_info = {}
        if gt_blocks and blocks:
            iou_info = compute_block_iou(blocks, gt_blocks)
            tau_info = compute_reading_order(blocks, gt_blocks, gt_order)
        elif gt_blocks:
            iou_info = {"mean_iou": 0.0, "matched": 0, "total_gt": len(gt_blocks), "total_pred": 0, "recall": 0.0, "precision": 0.0}
            tau_info = {"kendall_tau": 0.0}
        iou_results.append(iou_info)
        tau_results.append(tau_info)

        iou_str = f"IoU={iou_info.get('mean_iou', 0):.2f}" if iou_info else "IoU=n/a"
        tau_str = f"τ={tau_info.get('kendall_tau', 0):.2f}" if tau_info else "τ=n/a"
        print(f"  [{i+1:2d}/{len(images)}] {fname}: {elapsed:.2f}s  {iou_str}  {tau_str}  text={full_text[:50]}...")

    # --- Aggregate ---
    n = len(latencies)
    avg_lat = sum(latencies) / n
    std_lat = (sum((x - avg_lat) ** 2 for x in latencies) / (n - 1)) ** 0.5 if n > 1 else 0.0

    # CER/WER
    cer_vals = [compute_cer(gt_data.get(nm, {}).get("text", ""), t) for nm, t in zip(image_names, all_texts)]
    wer_vals = [compute_wer(gt_data.get(nm, {}).get("text", ""), t) for nm, t in zip(image_names, all_texts)]
    avg_cer = sum(cer_vals) / len(cer_vals) if cer_vals else 0.0
    avg_wer = sum(wer_vals) / len(wer_vals) if wer_vals else 0.0

    # IoU aggregate
    valid_ious = [r for r in iou_results if r]
    avg_iou = sum(r["mean_iou"] for r in valid_ious) / len(valid_ious) if valid_ious else 0.0
    avg_recall = sum(r.get("recall", 0) for r in valid_ious) / len(valid_ious) if valid_ious else 0.0
    avg_precision = sum(r.get("precision", 0) for r in valid_ious) / len(valid_ious) if valid_ious else 0.0

    # Reading order aggregate
    valid_taus = [r["kendall_tau"] for r in tau_results if r]
    avg_tau = sum(valid_taus) / len(valid_taus) if valid_taus else 0.0

    # --- Build result ---
    result = {
        "candidate_name": candidate_name,
        "model_version": "PaddleOCR-VL-1.6-0.9B",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu_name": gpu_name,
        "vram_total_mb": 0,
        "latency_total_avg": avg_lat,
        "latency_total_std": std_lat,
        "latency_stage1_ocr": avg_lat,
        "latency_stage2_errors": 0.0,
        "cer": avg_cer,
        "wer": avg_wer,
        "reading_order_tau": avg_tau,
        "error_detection_f1": 0.0,
        "bounding_box_iou": avg_iou,
        "vram_peak_mb": 0,
        "throughput_ppm": 60.0 / avg_lat if avg_lat > 0 else 0.0,
        "setup_complexity": 0,
        "flexibility": 0,
        "sample_outputs": [
            {"text": all_texts[i][:200], "blocks": all_blocks[i][:3]}
            for i in range(min(3, len(all_texts)))
        ],
        "iou_details": {
            "mean_iou": avg_iou,
            "mean_recall": avg_recall,
            "mean_precision": avg_precision,
            "per_image": {nm: r for nm, r in zip(image_names, iou_results) if r},
        },
        "reading_order_details": {
            "mean_kendall_tau": avg_tau,
            "per_image": {nm: r for nm, r in zip(image_names, tau_results) if r},
        },
        "notes": f"PaddleOCR-VL-1.6 {label} benchmark. Docker sm120-offline image. "
                 f"Standalone script (no project imports, data-only mount).",
    }

    # --- Save ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{candidate_name}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out_path}")

    # --- Summary ---
    print(f"\n{'='*50}")
    print(f"PaddleOCR-VL {label} Summary")
    print(f"  Images:      {n}")
    print(f"  Avg Latency: {avg_lat:.2f}s ± {std_lat:.2f}s")
    print(f"  Throughput:  {result['throughput_ppm']:.1f} ppm")
    print(f"  CER:  {avg_cer:.4f}")
    print(f"  WER:  {avg_wer:.4f}")
    print(f"  Bbox IoU:    {avg_iou:.4f}  (recall={avg_recall:.3f}, precision={avg_precision:.3f})")
    print(f"  Read Order τ: {avg_tau:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    sys.exit(main())
