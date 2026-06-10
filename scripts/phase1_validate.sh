#!/usr/bin/env bash
# ============================================================================
# phase1_validate.sh — End-to-end validation for Phase 1 deliverables
#
# Checks: Python env, harness imports, ground truth, baseline results.
# Exit 0 = all checks passed.
#
# Usage:
#   bash scripts/phase1_validate.sh
# ============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
PASS=0
FAIL=0
WARN=0

check() {
    local name="$1"
    shift
    echo -n "  [$name] ... "
    if "$@" &>/dev/null; then
        echo -e "${GREEN}PASS${RESET}"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}FAIL${RESET}"
        FAIL=$((FAIL + 1))
    fi
}

warn_check() {
    local name="$1"
    shift
    echo -n "  [$name] ... "
    if "$@" &>/dev/null; then
        echo -e "${GREEN}PASS${RESET}"
        PASS=$((PASS + 1))
    else
        echo -e "${YELLOW}WARN${RESET} (optional)"
        WARN=$((WARN + 1))
    fi
}

echo -e "${BOLD}${CYAN}========================================${RESET}"
echo -e "${BOLD}${CYAN}  Phase 1 — End-to-End Validation     ${RESET}"
echo -e "${BOLD}${CYAN}========================================${RESET}"
echo ""

# ---------------------------------------------------------------------------
# 1. Environment
# ---------------------------------------------------------------------------
echo -e "${BOLD}[1] Environment${RESET}"

check "Python venv exists"       test -f "$VENV_PYTHON"
check "PyTorch importable"       "$VENV_PYTHON" -c "import torch"
check "CUDA available"           "$VENV_PYTHON" -c "import torch; assert torch.cuda.is_available()"
check "GPU VRAM ≥ 8 GB"          "$VENV_PYTHON" -c "import torch; assert torch.cuda.get_device_properties(0).total_memory / 1024**3 >= 8"
check "transformers importable"  "$VENV_PYTHON" -c "import transformers"
check "Pillow importable"        "$VENV_PYTHON" -c "from PIL import Image"
# bitsandbytes does not support Blackwell (sm_120) yet — skip
# check "bitsandbytes importable"  "$VENV_PYTHON" -c "import bitsandbytes"
echo "  [bitsandbytes importable] ... ${YELLOW}SKIP${RESET} (Blackwell sm_120 not yet supported)"

echo ""

# ---------------------------------------------------------------------------
# 2. Harness & Metrics
# ---------------------------------------------------------------------------
echo -e "${BOLD}[2] Harness & Metrics${RESET}"

check "harness import"           "$VENV_PYTHON" -c "from benchmark.harness import BenchmarkHarness, BenchmarkResult"
check "harness.gpu_info"         "$VENV_PYTHON" -c "from benchmark.harness import BenchmarkHarness; n,v=BenchmarkHarness.get_gpu_info(); assert n"
check "metrics.CER"              "$VENV_PYTHON" -c "from benchmark.metrics import compute_cer; assert abs(compute_cer('abc','abc')-0)<0.001"
check "metrics.WER"              "$VENV_PYTHON" -c "from benchmark.metrics import compute_wer; assert abs(compute_wer('a b','a b')-0)<0.001"
check "metrics.tau"              "$VENV_PYTHON" -c "from benchmark.metrics import compute_reading_order_tau; assert compute_reading_order_tau([0,1,2],[0,1,2])==1.0"
check "metrics.IoU"              "$VENV_PYTHON" -c "from benchmark.metrics import _bbox_iou; assert _bbox_iou([0,0,10,10],[0,0,10,10])==1.0"

echo ""

# ---------------------------------------------------------------------------
# 3. Test Dataset
# ---------------------------------------------------------------------------
echo -e "${BOLD}[3] Test Dataset${RESET}"

check "images exist"             test "$(find "$PROJECT_ROOT/benchmark/test_dataset" -maxdepth 1 \( -name '*.jpg' -o -name '*.png' \) | wc -l)" -gt 0
warn_check "curated subset"      test -f "$PROJECT_ROOT/benchmark/test_dataset/curated_manifest.json"
warn_check "ground_truth.json"   test -f "$PROJECT_ROOT/benchmark/test_dataset/ground_truth.json"
check "template exists"          test -f "$PROJECT_ROOT/benchmark/test_dataset/ground_truth_template.json"
warn_check "DATASET.md"          test -f "$PROJECT_ROOT/benchmark/test_dataset/DATASET.md"

echo ""

# ---------------------------------------------------------------------------
# 4. Baseline Results (optional — requires GCP)
# ---------------------------------------------------------------------------
echo -e "${BOLD}[4] Baseline${RESET}"

warn_check "baseline result"     test -f "$PROJECT_ROOT/benchmark/results/baseline_google_docai_gemini.json"

echo ""

# ---------------------------------------------------------------------------
# 5. Candidate Scaffolding
# ---------------------------------------------------------------------------
echo -e "${BOLD}[5] Candidate Scaffolds${RESET}"

check "candidates import"        "$VENV_PYTHON" -c "from candidates import run_candidate"
for cand in baselines florence2 got_ocr nemotron_ocr paddleocr_vl qwen3_vl smoldocling trocr; do
    check "candidate/$cand"      test -f "$PROJECT_ROOT/candidates/$cand/eval.py"
done

echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo -e "${BOLD}========================================${RESET}"
echo -e "${BOLD}  Phase 1 Validation Summary${RESET}"
echo -e "${BOLD}========================================${RESET}"
echo -e "  ${GREEN}Passed: $PASS${RESET}"
echo -e "  ${RED}Failed: $FAIL${RESET}"
echo -e "  ${YELLOW}Warnings: $WARN${RESET} (optional / Phase 1d+)"
echo ""

if [ "$FAIL" -gt 0 ]; then
    echo -e "${RED}Some checks FAILED. Review output above.${RESET}"
    exit 1
else
    echo -e "${GREEN}All required checks PASSED. Phase 1 ready.${RESET}"
    exit 0
fi
