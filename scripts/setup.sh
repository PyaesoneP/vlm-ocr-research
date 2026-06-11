#!/usr/bin/env bash
# ============================================================================
# setup.sh — One-command environment setup for VLM-OCR research
#
# Detects CUDA, creates/activates venv, installs PyTorch + all dependencies,
# and runs validation.
#
# Usage:
#   bash scripts/setup.sh
#   bash scripts/setup.sh --no-venv   (use existing venv / system Python)
# ============================================================================

set -euo pipefail

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"
USE_VENV=true

# --- Parse args ---
for arg in "$@"; do
    case "$arg" in
        --no-venv) USE_VENV=false ;;
        -h|--help)
            echo "Usage: bash scripts/setup.sh [--no-venv]"
            exit 0
            ;;
    esac
done

echo -e "${BOLD}${CYAN}========================================${RESET}"
echo -e "${BOLD}${CYAN}  VLM-OCR Research — Environment Setup ${RESET}"
echo -e "${BOLD}${CYAN}========================================${RESET}"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Check CUDA / GPU
# ---------------------------------------------------------------------------
echo -e "${BOLD}[1/5] Checking CUDA & GPU...${RESET}"

if ! command -v nvidia-smi &>/dev/null; then
    echo -e "${RED}ERROR: nvidia-smi not found. Is the NVIDIA driver installed?${RESET}"
    exit 1
fi

NVIDIA_SMI_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")

echo -e "  ${GREEN}✓${RESET} GPU: $GPU_NAME"
echo -e "  ${GREEN}✓${RESET} VRAM: $GPU_VRAM"
echo -e "  ${GREEN}✓${RESET} Driver: $NVIDIA_SMI_VER"

# Detect CUDA toolkit version for PyTorch compatibility
CUDA_VERSION=""
if command -v nvcc &>/dev/null; then
    CUDA_VERSION=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "")
fi
if [ -z "$CUDA_VERSION" ] && [ -d /usr/local/cuda ]; then
    CUDA_VERSION=$(ls -la /usr/local/cuda 2>/dev/null | grep -oP 'cuda-\K[0-9]+\.[0-9]+' | head -1 || echo "")
fi
echo -e "  ${GREEN}✓${RESET} CUDA Toolkit: ${CUDA_VERSION:-not found (driver-only)}"

# PyTorch 2.11+ ships CUDA 13.0 wheels on PyPI directly.
# No custom index URL needed. Blackwell (sm_120) requires 2.11+.
PYTORCH_CUDA_INDEX=""
echo -e "  → PyTorch will be installed from PyPI (CUDA 13.0 wheels)"
echo -e "  → For older GPUs, use: pip install torch --index-url https://download.pytorch.org/whl/cu124"

echo ""

# ---------------------------------------------------------------------------
# Step 2: Python virtual environment
# ---------------------------------------------------------------------------
echo -e "${BOLD}[2/5] Setting up Python environment...${RESET}"

# Find Python 3.10+
PYTHON_BIN=""
for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0")
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON_BIN="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo -e "${RED}ERROR: Python 3.10+ not found. Install Python 3.10 or later.${RESET}"
    exit 1
fi

echo -e "  → Using: $PYTHON_BIN ($($PYTHON_BIN --version))"

if $USE_VENV; then
    if [ ! -d "$VENV_DIR" ]; then
        echo -e "  → Creating virtual environment at .venv/"
        $PYTHON_BIN -m venv "$VENV_DIR"
    else
        echo -e "  → Virtual environment already exists at .venv/"
    fi
    # Activate
    source "$VENV_DIR/bin/activate"
    echo -e "  ${GREEN}✓${RESET} Virtual environment active"
else
    echo -e "  ${YELLOW}⚠${RESET} Skipping venv (using system/python: $PYTHON_BIN)"
fi

# Upgrade pip
pip install --upgrade pip -q
echo -e "  ${GREEN}✓${RESET} pip upgraded"

echo ""

# ---------------------------------------------------------------------------
# Step 3: Install PyTorch with CUDA
# ---------------------------------------------------------------------------
echo -e "${BOLD}[3/5] Installing PyTorch with CUDA support...${RESET}"

# Check if CUDA PyTorch is already installed
CURRENT_TORCH=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "none")
if echo "$CURRENT_TORCH" | grep -q "+cpu"; then
    echo -e "  ${YELLOW}⚠${RESET} CPU-only PyTorch detected. Reinstalling with CUDA..."
    pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
elif echo "$CURRENT_TORCH" | grep -q "+cu"; then
    echo -e "  ${GREEN}✓${RESET} CUDA PyTorch already installed ($CURRENT_TORCH). Skipping."
else
    echo -e "  → Installing PyTorch with CUDA support..."
fi

# Only install if not already CUDA
if ! echo "$CURRENT_TORCH" | grep -q "+cu"; then
    # PyTorch 2.11+ ships CUDA wheels on PyPI (no custom index needed)
    if [ -n "$PYTORCH_CUDA_INDEX" ]; then
        pip install torch torchvision torchaudio --index-url "$PYTORCH_CUDA_INDEX"
    else
        pip install torch torchvision torchaudio
    fi
    echo -e "  ${GREEN}✓${RESET} PyTorch CUDA installed"
fi

echo ""

# ---------------------------------------------------------------------------
# Step 4: Install project dependencies
# ---------------------------------------------------------------------------
echo -e "${BOLD}[4/5] Installing project dependencies...${RESET}"

pip install -r "$PROJECT_ROOT/requirements.txt"
echo -e "  ${GREEN}✓${RESET} Dependencies installed"

# Optional packages that may be useful
echo -e "  → Installing optional utilities (lxml for IAM XML parsing, requests)..."
pip install lxml requests -q 2>/dev/null || true

echo ""

# ---------------------------------------------------------------------------
# Step 5: Validate environment
# ---------------------------------------------------------------------------
echo -e "${BOLD}[5/5] Validating environment...${RESET}"

python "$PROJECT_ROOT/scripts/validate_env.py"
VALIDATE_EXIT=$?

echo ""
if [ $VALIDATE_EXIT -eq 0 ]; then
    echo -e "${GREEN}${BOLD}════════════════════════════════════════${RESET}"
    echo -e "${GREEN}${BOLD}  ✓ Environment setup complete!        ${RESET}"
    echo -e "${GREEN}${BOLD}════════════════════════════════════════${RESET}"
    echo ""
    echo -e "  Activate the environment:"
    echo -e "    ${CYAN}source .venv/bin/activate${RESET}"
    echo ""
    echo -e "  Additional environments (Conda):"
    echo -e "    ${CYAN}conda activate aiml${RESET}       — Nemotron OCR v2"
    echo -e "    ${CYAN}conda activate florencetf${RESET}  — Florence-2 (transformers 4.40.0)"
    echo "     (see requirements-aiml.txt and requirements-florencetf.txt)"
    echo ""
    echo -e "  Next step: Phase 1b — curate test dataset"
    echo -e "    ${CYAN}python scripts/download_essay_samples.py${RESET}"
else
    echo -e "${RED}${BOLD}════════════════════════════════════════${RESET}"
    echo -e "${RED}${BOLD}  ✗ Validation failed — see errors above${RESET}"
    echo -e "${RED}${BOLD}════════════════════════════════════════${RESET}"
fi

exit $VALIDATE_EXIT
