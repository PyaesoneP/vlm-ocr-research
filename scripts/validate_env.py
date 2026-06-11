#!/usr/bin/env python3
"""
Standalone environment validation for VLM-OCR research.

Checks: CUDA, PyTorch, GPU VRAM, key packages, Python version.
Exit code 0 = all checks passed, non-zero = failures.

Usage:
    python scripts/validate_env.py
    python scripts/validate_env.py --json  # Machine-readable output
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from typing import Any

# ---------------------------------------------------------------------------
# Check registry
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    severity: str = "error"  # error | warning


def check_python_version() -> CheckResult:
    v = sys.version_info
    ok = (v.major == 3 and v.minor >= 10)
    return CheckResult(
        name="Python ≥ 3.10",
        passed=ok,
        detail=f"Python {v.major}.{v.minor}.{v.micro}",
        severity="error",
    )


def check_torch() -> CheckResult:
    try:
        import torch
        detail = f"PyTorch {torch.__version__}"
        if "+cpu" in torch.__version__:
            return CheckResult(
                name="PyTorch CUDA",
                passed=False,
                detail=f"{detail} — CPU-only! Install CUDA variant.",
                severity="error",
            )
        return CheckResult(name="PyTorch", passed=True, detail=detail)
    except ImportError:
        return CheckResult(
            name="PyTorch",
            passed=False,
            detail="Not installed.",
            severity="error",
        )


def check_cuda_available() -> CheckResult:
    try:
        import torch
        if torch.cuda.is_available():
            return CheckResult(
                name="CUDA available",
                passed=True,
                detail=f"CUDA {torch.version.cuda}, devices: {torch.cuda.device_count()}",
            )
        return CheckResult(
            name="CUDA available",
            passed=False,
            detail="torch.cuda.is_available() = False",
            severity="error",
        )
    except Exception as e:
        return CheckResult(
            name="CUDA available",
            passed=False,
            detail=str(e),
            severity="error",
        )


def check_gpu_info() -> CheckResult:
    try:
        import torch
        if not torch.cuda.is_available():
            return CheckResult(name="GPU info", passed=False, detail="CUDA not available")
        name = torch.cuda.get_device_name(0)
        total_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
        return CheckResult(
            name="GPU info",
            passed=True,
            detail=f"{name} ({total_mb} MB VRAM)",
        )
    except Exception as e:
        return CheckResult(name="GPU info", passed=False, detail=str(e))


def check_transformers() -> CheckResult:
    try:
        import transformers
        return CheckResult(
            name="transformers",
            passed=True,
            detail=f"v{transformers.__version__}",
        )
    except ImportError:
        return CheckResult(
            name="transformers",
            passed=False,
            detail="Not installed.",
            severity="error",
        )


def check_accelerate() -> CheckResult:
    try:
        import accelerate
        return CheckResult(
            name="accelerate", passed=True, detail=f"v{accelerate.__version__}"
        )
    except ImportError:
        return CheckResult(
            name="accelerate",
            passed=False,
            detail="Not installed.",
            severity="warning",
        )


def check_bitsandbytes() -> CheckResult:
    try:
        import bitsandbytes
        return CheckResult(
            name="bitsandbytes", passed=True, detail=f"v{bitsandbytes.__version__}"
        )
    except ImportError:
        return CheckResult(
            name="bitsandbytes",
            passed=False,
            detail="Not installed — blocked on Blackwell sm_120 (expected). "
                   "Uncomment in requirements.txt once NVIDIA ships support.",
            severity="warning",
        )


def check_numpy() -> CheckResult:
    try:
        import numpy
        return CheckResult(name="numpy", passed=True, detail=f"v{numpy.__version__}")
    except ImportError:
        return CheckResult(
            name="numpy", passed=False, detail="Not installed.", severity="error",
        )


def check_pillow() -> CheckResult:
    try:
        import PIL
        return CheckResult(name="Pillow", passed=True, detail=f"v{PIL.__version__}")
    except ImportError:
        return CheckResult(
            name="Pillow", passed=False, detail="Not installed.", severity="error"
        )


def check_opencv() -> CheckResult:
    try:
        import cv2
        return CheckResult(name="opencv-python", passed=True, detail=f"v{cv2.__version__}")
    except ImportError:
        return CheckResult(
            name="opencv-python",
            passed=False,
            detail="Not installed.",
            severity="warning",
        )


def check_scipy() -> CheckResult:
    try:
        import scipy
        return CheckResult(name="scipy", passed=True, detail=f"v{scipy.__version__}")
    except ImportError:
        return CheckResult(
            name="scipy", passed=False, detail="Not installed.", severity="warning"
        )


# ---------------------------------------------------------------------------
# GPU stress test (optional — validates VRAM allocation works)
# ---------------------------------------------------------------------------

def check_vram_allocation(target_mb: int = 1024) -> CheckResult:
    """Try allocating target_mb on GPU to verify VRAM is usable."""
    try:
        import torch
        if not torch.cuda.is_available():
            return CheckResult(name="VRAM allocation", passed=False, detail="CUDA not available")
        # Allocate ~1 GB to confirm VRAM works
        n_floats = (target_mb * 1024 * 1024) // 4
        t = torch.zeros(n_floats, device="cuda", dtype=torch.float32)
        del t
        torch.cuda.empty_cache()
        return CheckResult(
            name="VRAM allocation",
            passed=True,
            detail=f"Successfully allocated {target_mb} MB on GPU.",
        )
    except Exception as e:
        return CheckResult(
            name="VRAM allocation",
            passed=False,
            detail=str(e),
            severity="warning",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all_checks() -> list[CheckResult]:
    return [
        check_python_version(),
        check_torch(),
        check_cuda_available(),
        check_gpu_info(),
        check_transformers(),
        check_accelerate(),
        check_numpy(),
        check_bitsandbytes(),
        check_pillow(),
        check_opencv(),
        check_scipy(),
        check_vram_allocation(),
    ]


def main():
    parser = argparse.ArgumentParser(description="Validate VLM-OCR research environment.")
    parser.add_argument("--json", action="store_true", help="Output results as JSON.")
    args = parser.parse_args()

    results = run_all_checks()
    errors = [r for r in results if not r.passed and r.severity == "error"]
    warnings = [r for r in results if not r.passed and r.severity == "warning"]

    if args.json:
        out = {
            "all_passed": len(errors) == 0,
            "errors": len(errors),
            "warnings": len(warnings),
            "checks": [asdict(r) for r in results],
        }
        print(json.dumps(out, indent=2))
    else:
        # Pretty terminal output
        GREEN = "\033[92m"
        RED = "\033[91m"
        YELLOW = "\033[93m"
        RESET = "\033[0m"
        BOLD = "\033[1m"

        print(f"\n{BOLD}VLM-OCR Environment Validation{RESET}\n")
        print(f"{'Check':<30} {'Status':<10} Detail")
        print("-" * 80)

        for r in results:
            if r.passed:
                status = f"{GREEN}PASS{RESET}"
            elif r.severity == "warning":
                status = f"{YELLOW}WARN{RESET}"
            else:
                status = f"{RED}FAIL{RESET}"
            print(f"{r.name:<30} {status:<18} {r.detail}")

        print("-" * 80)
        if not errors:
            print(f"\n{GREEN}{BOLD}All checks passed!{RESET} Environment is ready for VLM-OCR research.\n")
        else:
            print(f"\n{RED}{BOLD}{len(errors)} error(s) found.{RESET}")
            for e in errors:
                print(f"  {RED}✗{RESET} {e.name}: {e.detail}")
            if warnings:
                print(f"\n{YELLOW}{len(warnings)} warning(s):{RESET}")
                for w in warnings:
                    print(f"  {YELLOW}⚠{RESET} {w.name}: {w.detail}")
            print()

    sys.exit(0 if len(errors) == 0 else 1)


if __name__ == "__main__":
    main()
