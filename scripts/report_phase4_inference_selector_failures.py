"""Summarize selector bottlenecks from a Phase 4 inference evidence audit.

The input audit is label-aware, but only after the graph has been built and
scored label-free. This report is for development decomposition: it separates
candidate-generation failures from selector-policy failures.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_AUDIT = Path("benchmark/results/phase4_inference_evidence_graph_audit_rw11.json")
DEFAULT_OUTPUT = Path("benchmark/results/phase4_inference_selector_failure_report.json")


def classify(row: dict[str, Any]) -> str:
    status = row.get("status")
    decision = row.get("decision")
    visible_rank = row.get("visible_top_rank")

    if status == "missing_record":
        return "missing_record"
    if status == "visible_absent":
        return "visible_absent"
    if status == "auto_supported":
        return "auto_supported"
    if visible_rank == 1 and decision == "SUPPORTED_CANONICAL":
        return "canonical_already_preserved"
    if visible_rank == 1 and not row.get("visible_supported_by_ocr", False):
        return "unsupported_visible_top_blocked"
    if status == "present_not_top" and not row.get("top_supported_by_ocr", False):
        return "noisy_unsupported_top"
    if status == "present_not_top":
        return "visible_present_not_top"
    if status == "top_but_blocked":
        return "visible_top_blocked"
    return "other"


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "image": row.get("image", ""),
        "record_id": row.get("record_id", ""),
        "evidence_text": row.get("evidence_text", ""),
        "correction": row.get("correction", ""),
        "type": row.get("type", ""),
        "category": classify(row),
        "decision": row.get("decision", ""),
        "canonical_text": row.get("canonical_text", ""),
        "top_text": row.get("top_text", ""),
        "top_source": row.get("top_source", ""),
        "top_supported_by_ocr": row.get("top_supported_by_ocr", False),
        "visible_top_rank": row.get("visible_top_rank"),
        "visible_source": row.get("visible_source", ""),
        "visible_supported_by_ocr": row.get("visible_supported_by_ocr", False),
        "alternative_margin_over_canonical": row.get("alternative_margin_over_canonical"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text())
    rows = [compact_row(row) for row in audit.get("errors", [])]
    counts = Counter(row["category"] for row in rows)
    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_type[str(row.get("type", ""))][row["category"]] += 1

    result = {
        "audit": str(args.audit),
        "summary": {
            "errors_total": len(rows),
            "categories": dict(sorted(counts.items())),
            "by_type": {key: dict(sorted(value.items())) for key, value in sorted(by_type.items())},
        },
        "errors": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
