"""
Phase 1 — Baseline measurement for Google Document AI + Gemini pipeline.

Measures end-to-end latency and accuracy of the current cloud pipeline
on the test dataset.  Requires Google Cloud credentials and the
`google-cloud-documentai` and `google-genai` Python packages.

Usage:
    python benchmark/baseline.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path (needed when run as script)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Configuration (set via environment variables or edit inline)
# ---------------------------------------------------------------------------

GCP_PROJECT = os.environ.get("GCP_PROJECT", "")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us")  # Document AI location
DOCAI_PROCESSOR_ID = os.environ.get("DOCAI_PROCESSOR_ID", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")  # API key (simpler than ADC)

# Rate limiting for free tier (default: 5 RPM)
GEMINI_RPM = int(os.environ.get("GEMINI_RPM", "5"))
_GEMINI_LAST_CALL = 0.0

TEST_DATASET = _PROJECT_ROOT / "benchmark" / "test_dataset"
RESULTS_DIR = _PROJECT_ROOT / "benchmark" / "results"
NUM_RUNS = 5
NUM_IMAGES = 5  # limit to first N images from curated subset


# ---------------------------------------------------------------------------
# Document AI client
# ---------------------------------------------------------------------------

def _get_docai_client():
    """Lazy-load the Document AI client with regional endpoint."""
    try:
        from google.cloud import documentai
        from google.api_core.client_options import ClientOptions
    except ImportError:
        raise ImportError(
            "google-cloud-documentai is required. Install with:\n"
            "  pip install google-cloud-documentai"
        )

    opts = ClientOptions(
        api_endpoint=f"{GCP_LOCATION}-documentai.googleapis.com"
    ) if GCP_LOCATION != "us" else None

    if opts:
        return documentai.DocumentProcessorServiceClient(client_options=opts)
    return documentai.DocumentProcessorServiceClient()


def docai_ocr(image_path: str | Path) -> dict:
    """
    Run Google Document AI OCR on a single image.

    Returns a dict with 'text', 'blocks' (bbox + text), and 'latency'.
    """
    client = _get_docai_client()
    from google.cloud import documentai

    image_path = Path(image_path)

    t0 = time.perf_counter()

    with open(image_path, "rb") as f:
        content = f.read()

    raw_doc = documentai.RawDocument(content=content, mime_type=_mime_type(image_path))
    name = client.processor_path(GCP_PROJECT, GCP_LOCATION, DOCAI_PROCESSOR_ID)

    request = documentai.ProcessRequest(
        name=name,
        raw_document=raw_doc,
    )
    response = client.process_document(request=request)
    doc = response.document

    elapsed = time.perf_counter() - t0

    blocks = []
    for page in doc.pages:
        for block in page.blocks:
            text = _layout_text(doc, block.layout)
            bbox = _normalized_to_absolute(
                block.layout.bounding_poly, page.image.width, page.image.height
            )
            blocks.append({
                "bbox": bbox,
                "text": text,
                "confidence": block.layout.confidence,
            })

    return {
        "text": doc.text,
        "blocks": blocks,
        "stage1_latency": elapsed,
    }


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

def _get_gemini_client():
    """
    Lazy-load the Gemini client.

    Uses the official Gen AI SDK pattern with v1 API:
      client = genai.Client(http_options=HttpOptions(api_version='v1'))
    Authenticates via GOOGLE_API_KEY env var (AI Studio) or
    GEMINI_API_KEY, or falls back to Vertex AI via ADC.
    """
    try:
        from google import genai
        from google.genai.types import HttpOptions
    except ImportError:
        raise ImportError(
            "google-genai is required. Install with:\n"
            "  pip install google-genai"
        )

    # Preferred: API key auth (AI Studio) — simplest path
    api_key = GEMINI_API_KEY or os.environ.get("GOOGLE_API_KEY", "")
    if api_key:
        return genai.Client(api_key=api_key)

    # Fall back to Vertex AI via ADC (global location for preview models)
    return genai.Client(
        vertexai=True, project=GCP_PROJECT, location="global",
        http_options=HttpOptions(api_version="v1"),
    )


def gemini_error_detection(ocr_output: dict) -> dict:
    """
    Send OCR output to Gemini for error detection and feedback.

    Expects ocr_output with 'text' and 'blocks'; returns 'errors' list
    with bbox, type, and description.

    Gracefully handles quota exhaustion — returns empty errors.
    """
    try:
        client = _get_gemini_client()
    except Exception as e:
        print(f"  [warn] Gemini client unavailable: {e}")
        return {"errors": [], "feedback": "", "stage2_latency": 0.0,
                "_note": f"Gemini unavailable: {e}"}

    prompt = f"""You are an essay grader. Analyze the following handwritten essay transcription for writing errors.

Return a JSON object with an "errors" array. Each error must have:
  - "type": one of "capitalization", "spelling", "grammar", "punctuation", "structural"
  - "block_index": the index of the text block containing the error (0-based)
  - "description": a brief explanation of the error

Transcription:
{ocr_output['text']}

Blocks (index, bbox, text):
{json.dumps([{'index': i, 'bbox': b['bbox'], 'text': b['text']} for i, b in enumerate(ocr_output.get('blocks', []))])}"""

    t0 = time.perf_counter()

    # Rate limit: ensure we stay within GEMINI_RPM
    global _GEMINI_LAST_CALL
    min_interval = 60.0 / GEMINI_RPM
    elapsed_since_last = time.perf_counter() - _GEMINI_LAST_CALL
    if elapsed_since_last < min_interval:
        time.sleep(min_interval - elapsed_since_last)

    # Retry on transient errors (503, 429)
    max_retries = 8
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            _GEMINI_LAST_CALL = time.perf_counter()
            break
        except Exception as e:
            err_str = str(e)
            is_last = (attempt == max_retries - 1)

            # Quota exhaustion — wait longer
            if "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                if is_last:
                    print(f"  [warn] Gemini quota exhausted after {max_retries} retries")
                    return {"errors": [], "feedback": "", "stage2_latency": 0.0,
                            "_note": "Gemini quota exceeded"}
                wait = 30
                print(f"  [retry] Gemini quota exceeded, waiting {wait}s...")
                time.sleep(wait)
                continue

            # Model overloaded — longer backoff
            if "503" in err_str or "UNAVAILABLE" in err_str:
                if is_last:
                    print(f"  [warn] Gemini still overloaded after {max_retries} retries")
                    return {"errors": [], "feedback": "", "stage2_latency": 0.0,
                            "_note": "Gemini 503 unavailable"}
                wait = min(2 ** attempt, 30)
                print(f"  [retry] Gemini overloaded (503), waiting {wait}s...")
                time.sleep(wait)
                continue

            if is_last:
                print(f"  [warn] Gemini failed after {max_retries} attempts: {e}")
                return {"errors": [], "feedback": "", "stage2_latency": 0.0,
                        "_note": f"Gemini error: {e}"}
            wait = 2 ** attempt
            print(f"  [retry] Gemini attempt {attempt+1} failed, waiting {wait}s...")
            time.sleep(wait)

    elapsed = time.perf_counter() - t0

    # Parse JSON from Gemini response
    raw = response.text
    try:
        # Extract JSON block if wrapped in markdown
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
        parsed = json.loads(raw)
    except (json.JSONDecodeError, IndexError):
        parsed = {"errors": []}

    # Map block_index back to bounding boxes
    blocks = ocr_output.get("blocks", [])
    errors = []
    for e in parsed.get("errors", []):
        idx = e.get("block_index", -1)
        bbox = blocks[idx]["bbox"] if 0 <= idx < len(blocks) else [0, 0, 0, 0]
        errors.append({
            "type": e.get("type", "unknown"),
            "bbox": bbox,
            "description": e.get("description", ""),
        })

    return {
        "errors": errors,
        "feedback": parsed.get("feedback", ""),
        "stage2_latency": elapsed,
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def baseline_pipeline(image_path: str | Path) -> dict:
    """End-to-end Google Document AI + Gemini pipeline."""
    ocr_output = docai_ocr(image_path)
    error_output = gemini_error_detection(ocr_output)

    return {
        **ocr_output,
        **error_output,
        "stage1_latency": ocr_output["stage1_latency"],
        "stage2_latency": error_output["stage2_latency"],
        "text": ocr_output["text"],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mime_type(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
    }.get(ext, "image/png")


def _layout_text(doc, layout) -> str:
    """Extract text from a Document AI layout object."""
    text = ""
    for segment in layout.text_anchor.text_segments:
        start = int(segment.start_index) if segment.start_index else 0
        end = int(segment.end_index) if segment.end_index else len(doc.text)
        text += doc.text[start:end]
    return text


def _normalized_to_absolute(bounding_poly, img_w: int, img_h: int) -> list[int]:
    """Convert normalized vertices to absolute [x1, y1, x2, y2]."""
    xs = [v.x * img_w for v in bounding_poly.vertices]
    ys = [v.y * img_h for v in bounding_poly.vertices]
    if not xs:
        return [0, 0, 0, 0]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from benchmark.harness import BenchmarkHarness

    # Gather test images from curated subset
    curated_dir = TEST_DATASET / "curated"
    images = sorted(p for p in curated_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    images = images[:NUM_IMAGES]

    if not images:
        print(f"No test images found in {curated_dir}. "
              "Place handwritten essay images there and re-run.")
        exit(1)

    gt_path = TEST_DATASET / "ground_truth.json"

    print(f"Running baseline benchmark on {len(images)} images ({NUM_RUNS} runs each)...")
    print(f"  Project: {GCP_PROJECT}")
    print(f"  Doc AI Processor: {DOCAI_PROCESSOR_ID}")
    print(f"  Gemini Model: {GEMINI_MODEL}")

    harness = BenchmarkHarness(output_dir=RESULTS_DIR)
    result = harness.run(
        candidate_name="baseline_google_docai_gemini",
        inference_fn=baseline_pipeline,
        test_images=images,
        ground_truth=gt_path if gt_path.exists() else None,
        num_warmup=2,
        num_runs=NUM_RUNS,
    )
    result.model_version = f"Document AI + {GEMINI_MODEL}"
    result.notes = "Google Cloud baseline pipeline."

    out = harness.save_result(result)
    print(f"\nBaseline result saved to {out}")
    print(f"  Avg latency: {result.latency_total_avg:.2f}s")
    print(f"  Stage 1 (OCR): {result.latency_stage1_ocr:.2f}s")
    print(f"  Stage 2 (Errors): {result.latency_stage2_errors:.2f}s")
    print(f"  Throughput: {result.throughput_ppm:.2f} ppm")
