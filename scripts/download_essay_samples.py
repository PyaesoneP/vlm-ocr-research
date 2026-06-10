#!/usr/bin/env python3
"""
Download and prepare handwritten essay test samples.

The IAM Handwriting Database is the primary source for handwritten English text.
This script:
  1. Verifies the existing IAM page images in benchmark/test_dataset/
  2. Attempts to download IAM XML metadata (requires IAM registration)
  3. Falls back to generating a dataset manifest from image features

IAM Database: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database
Registration required for XML metadata download.

Usage:
    python scripts/download_essay_samples.py
    python scripts/download_essay_samples.py --iam-metadata-url <URL>
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DATASET = PROJECT_ROOT / "benchmark" / "test_dataset"
MANIFEST_PATH = TEST_DATASET / "dataset_manifest.json"

# Known IAM metadata mirror (requires registration token)
IAM_METADATA_URL = os.environ.get(
    "IAM_METADATA_URL",
    "https://fki.tic.heia-fr.ch/DBs/iamDB/data/xml.tar.gz",
)

# ---------------------------------------------------------------------------
# IAM XML metadata download
# ---------------------------------------------------------------------------

def download_iam_metadata(url: str | None = None) -> Optional[Path]:
    """
    Download and extract IAM XML annotations.

    Requires authentication (IAM account).  If credentials are not available,
    falls back to image-feature-based manifest generation.
    """
    if url is None:
        url = IAM_METADATA_URL

    try:
        import requests
    except ImportError:
        print("[warn] requests not installed; cannot download IAM metadata.")
        print("       Install: pip install requests")
        return None

    xml_dir = TEST_DATASET / "iam_xml"
    xml_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading IAM metadata from {url} ...")

    try:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[warn] Could not download IAM metadata: {e}")
        print("       If you have registered at fki.tic.heia-fr.ch,")
        print("       download xml.tar.gz manually and extract to:")
        print(f"       {xml_dir}")
        return None

    # Extract tar.gz
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        for chunk in resp.iter_content(chunk_size=8192):
            tmp.write(chunk)
        tmp_path = Path(tmp.name)

    try:
        shutil.unpack_archive(str(tmp_path), str(xml_dir))
        print(f"  Extracted IAM XML to {xml_dir}")
    finally:
        tmp_path.unlink(missing_ok=True)

    # Verify extraction
    xml_files = list(xml_dir.rglob("*.xml"))
    print(f"  Found {len(xml_files)} XML annotation files")
    return xml_dir if xml_files else None


# ---------------------------------------------------------------------------
# Image feature extraction (fallback when no XML metadata)
# ---------------------------------------------------------------------------

def compute_image_features(image_path: Path) -> dict:
    """Extract simple features from an image for diversity scoring."""
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        return {}

    try:
        img = Image.open(image_path).convert("L")  # Grayscale
        arr = np.array(img, dtype=np.float64)

        h, w = arr.shape
        # Text density: fraction of dark pixels
        text_density = (arr < 128).mean()
        # Contrast: standard deviation
        contrast = arr.std()
        # Estimated line count from horizontal projection
        h_proj = (arr < 128).mean(axis=1)
        threshold = h_proj.mean() + 0.5 * h_proj.std()
        line_count = max(1, int(np.sum(h_proj > threshold) / (h * 0.02)))

        return {
            "width": w,
            "height": h,
            "aspect_ratio": round(w / h, 3),
            "text_density": round(float(text_density), 4),
            "contrast": round(float(contrast), 2),
            "estimated_lines": line_count,
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  VLM-OCR Research — Test Dataset Preparation")
    print("=" * 60)
    print()

    # --- 1. Inventory existing images ---
    image_patterns = ("*.jpg", "*.jpeg", "*.png")
    images: list[Path] = []
    for pat in image_patterns:
        images.extend(sorted(TEST_DATASET.glob(pat)))

    # Filter out ground_truth_template / manifest
    images = [p for p in images if "ground_truth" not in p.name and "manifest" not in p.name]

    print(f"[1/3] Found {len(images)} images in {TEST_DATASET}")

    if not images:
        print()
        print("  No test images found.  To obtain handwritten essay samples:")
        print()
        print("  1. IAM Handwriting Database (recommended):")
        print("     https://fki.tic.heia-fr.ch/databases/iam-handwriting-database")
        print("     Register, download formsA-D.tgz + xml.tgz, extract to:")
        print(f"     {TEST_DATASET}")
        print()
        print("  2. Or place any .jpg/.png handwritten page images in:")
        print(f"     {TEST_DATASET}")
        sys.exit(1)

    # --- 2. Attempt IAM metadata download ---
    print(f"[2/3] Checking for IAM XML metadata ...")
    xml_dir = TEST_DATASET / "iam_xml"
    xml_files = xml_dir.rglob("*.xml") if xml_dir.exists() else []
    xml_count = sum(1 for _ in xml_files)

    if xml_count == 0:
        print(f"  No XML metadata found. Attempting download ...")
        result = download_iam_metadata()
        if result:
            xml_files = list(result.rglob("*.xml"))
            xml_count = len(xml_files)

    print(f"  XML annotations available: {xml_count}")

    # --- 3. Build manifest ---
    print(f"[3/3] Building dataset manifest ...")
    manifest: list[dict] = []

    for img_path in images:
        features = compute_image_features(img_path)
        entry = {
            "image": img_path.name,
            "path": str(img_path.relative_to(PROJECT_ROOT)),
            "sha256": hashlib.sha256(img_path.read_bytes()).hexdigest()[:16],
            **features,
        }
        manifest.append(entry)

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"  Wrote manifest to {MANIFEST_PATH}")
    print()
    print("Done. Run scripts/curate_test_subset.py to select a diverse subset.")


if __name__ == "__main__":
    main()
