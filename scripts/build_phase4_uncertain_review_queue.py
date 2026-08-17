"""Build an auditable review queue from CTC agreement/filter decisions.

This is the product fallback for cases where optical scoring is useful but not
safe enough to rewrite OCR automatically. The queue is label-free: it includes
all non-canonical CTC decisions and the crop/candidate evidence needed for a
human or later reviewer. If the input decision file already contains dev-label
audit fields, this script reports them only in an audit summary.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_DECISIONS = Path("benchmark/results/phase4_inference_selector_policy_ctc_filtered_stage2_spans_20image.json")
DEFAULT_WORD_GRAPH = Path("benchmark/results/phase4_inference_evidence_graph_ctc_20image.json")
DEFAULT_LINE_GRAPH = Path("benchmark/results/phase4_inference_evidence_graph_ctc_line_20image.json")
DEFAULT_OUTPUT = Path("benchmark/results/phase4_uncertain_review_queue_ctc_20image.json")
DEFAULT_MARKDOWN = Path("benchmark/results/phase4_uncertain_review_queue_ctc_20image.md")

SUPPORTED_CANONICAL_READING = "SUPPORTED_CANONICAL_READING"
REVIEW_ALTERNATIVE_READING = "REVIEW_ALTERNATIVE_READING"
REJECT_DISAGREEMENT = "REJECT_DISAGREEMENT"
UNCERTAIN_REVIEW = "UNCERTAIN_REVIEW"


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text())
    return {str(row.get("pair_id", "")): row for row in data.get("records", [])}


def load_decisions(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    return list(data.get("decisions", []))


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def same_noncanonical_top(decision: dict[str, Any]) -> bool:
    word_top = norm(decision.get("word_top_text", ""))
    line_top = norm(decision.get("line_top_text", ""))
    canonical = norm(decision.get("canonical_text", ""))
    return bool(word_top and word_top == line_top and word_top != canonical)


def finite_margin(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def min_margin(decision: dict[str, Any]) -> float | None:
    margins = [
        margin for margin in (
            finite_margin(decision.get("word_margin_over_canonical")),
            finite_margin(decision.get("line_margin_over_canonical")),
        )
        if margin is not None
    ]
    return min(margins) if margins else None


def priority(decision: dict[str, Any]) -> str:
    filtered_decision = decision.get("filtered_decision", decision.get("decision", ""))
    margin = min_margin(decision)
    if decision.get("filtered_review_kept"):
        return "P1"
    if filtered_decision == REVIEW_ALTERNATIVE_READING and same_noncanonical_top(decision):
        if margin is not None and margin >= 0.5:
            return "P1"
        return "P2"
    if decision.get("source_text_triggers"):
        return "P2"
    if filtered_decision == REJECT_DISAGREEMENT:
        return "P3"
    return "P3"


def noncanonical_queue_candidate(decision: dict[str, Any]) -> bool:
    filtered_decision = decision.get("filtered_decision", decision.get("decision", ""))
    if filtered_decision == SUPPORTED_CANONICAL_READING:
        return False
    if decision.get("decision") == SUPPORTED_CANONICAL_READING:
        return False
    return True


def top_hypotheses(record: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    rows = []
    for hyp in record.get("hypotheses", [])[:limit]:
        rows.append({
            "text": hyp.get("text", ""),
            "source": hyp.get("source", ""),
            "is_canonical": bool(hyp.get("is_canonical")),
            "supported_by_ocr": bool(hyp.get("supported_by_ocr")),
            "ctc_normalized_logprob": hyp.get("ctc_normalized_logprob"),
            "ctc_raw_logprob": hyp.get("ctc_raw_logprob"),
            "ctc_unsupported_chars": hyp.get("ctc_unsupported_chars", []),
        })
    return rows


def queue_item(
    decision: dict[str, Any],
    *,
    word_records: dict[str, dict[str, Any]],
    line_records: dict[str, dict[str, Any]],
    topn: int,
) -> dict[str, Any]:
    record_id = str(decision.get("record_id", ""))
    word_record = word_records.get(record_id, {})
    line_record = line_records.get(record_id, {})
    return {
        "queue_id": record_id,
        "priority": priority(decision),
        "image": decision.get("image", ""),
        "word_indices": decision.get("word_indices", []),
        "bbox": decision.get("bbox", []),
        "canonical_text": decision.get("canonical_text", ""),
        "review_text": decision.get("review_text", ""),
        "word_top_text": decision.get("word_top_text", ""),
        "line_top_text": decision.get("line_top_text", ""),
        "decision": decision.get("decision", ""),
        "filtered_decision": decision.get("filtered_decision", decision.get("decision", "")),
        "reason": decision.get("reason", ""),
        "filtered_review_kept": bool(decision.get("filtered_review_kept", False)),
        "filtered_review_reasons": decision.get("filtered_review_reasons", []),
        "source_text_triggers": decision.get("source_text_triggers", []),
        "word_margin_over_canonical": decision.get("word_margin_over_canonical"),
        "line_margin_over_canonical": decision.get("line_margin_over_canonical"),
        "word_crop_path": word_record.get("crop_path", ""),
        "line_crop_path": line_record.get("line_crop_path", ""),
        "line_text": line_record.get("line_text", ""),
        "suspicion_reasons": word_record.get("suspicion_reasons", []),
        "word_top_hypotheses": top_hypotheses(word_record, topn),
        "line_top_hypotheses": top_hypotheses(line_record, topn),
    }


def audit_summary(decisions: list[dict[str, Any]], queue: list[dict[str, Any]]) -> dict[str, Any]:
    queued_ids = {row["queue_id"] for row in queue}
    matched = [row for row in decisions if row.get("matched_error")]
    clean = [row for row in decisions if not row.get("matched_error")]
    queued_matched = [row for row in matched if row.get("record_id") in queued_ids]
    queued_clean = [row for row in clean if row.get("record_id") in queued_ids]
    return {
        "label_aware_audit": any("matched_error" in row for row in decisions),
        "matched_error_records_total": len(matched),
        "matched_error_records_queued": len(queued_matched),
        "matched_review_visible_queued": sum(1 for row in queued_matched if row.get("review_contains_visible")),
        "matched_top_visible_queued": sum(
            1 for row in queued_matched
            if row.get("word_top_contains_visible") or row.get("line_top_contains_visible")
        ),
        "clean_records_total": len(clean),
        "clean_records_queued": len(queued_clean),
    }


def summarize(queue: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "queue_total": len(queue),
        "priority_counts": dict(sorted(Counter(row["priority"] for row in queue).items())),
        "decision_counts": dict(sorted(Counter(row["decision"] for row in queue).items())),
        "filtered_decision_counts": dict(sorted(Counter(row["filtered_decision"] for row in queue).items())),
        "filtered_review_kept": sum(1 for row in queue if row.get("filtered_review_kept")),
        "audit": audit_summary(decisions, queue),
    }


def write_markdown(path: Path, summary: dict[str, Any], queue: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase 4 CTC Uncertain Review Queue",
        "",
        "This queue is metadata only. It does not rewrite OCR text or create automatic error claims.",
        "",
        "## Summary",
        "",
        f"- Queue records: {summary['queue_total']}",
        f"- Priority counts: `{summary['priority_counts']}`",
        f"- Decision counts: `{summary['decision_counts']}`",
        f"- Filtered review kept: {summary['filtered_review_kept']}",
        "",
        "## Records",
        "",
    ]
    for row in queue:
        lines.extend([
            f"### {row['priority']} {row['queue_id']} ({row['image']})",
            "",
            f"- Canonical: `{row['canonical_text']}`",
            f"- Word top: `{row['word_top_text']}`",
            f"- Line top: `{row['line_top_text']}`",
            f"- Review text: `{row['review_text']}`",
            f"- Decision: `{row['decision']}` -> `{row['filtered_decision']}`",
            f"- Word indices: `{row['word_indices']}`",
            f"- BBox: `{row['bbox']}`",
            f"- Word crop: `{row['word_crop_path']}`",
            f"- Line crop: `{row['line_crop_path']}`",
            f"- Reasons: `{row['suspicion_reasons']}`",
            "",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--word-graph", type=Path, default=DEFAULT_WORD_GRAPH)
    parser.add_argument("--line-graph", type=Path, default=DEFAULT_LINE_GRAPH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--topn", type=int, default=5)
    parser.add_argument(
        "--priority",
        action="append",
        dest="priorities",
        default=[],
        help="Priority to include, e.g. --priority P1 --priority P2. Defaults to all.",
    )
    parser.add_argument("--include-markdown", action="store_true")
    args = parser.parse_args()

    decisions = load_decisions(args.decisions)
    word_records = load_records(args.word_graph)
    line_records = load_records(args.line_graph)
    queue = [
        queue_item(row, word_records=word_records, line_records=line_records, topn=args.topn)
        for row in decisions
        if noncanonical_queue_candidate(row)
    ]
    if args.priorities:
        keep = set(args.priorities)
        queue = [row for row in queue if row["priority"] in keep]
    queue.sort(key=lambda row: (row["priority"], row["image"], row["queue_id"]))
    summary = summarize(queue, decisions)
    result = {
        "decisions": str(args.decisions),
        "word_graph": str(args.word_graph),
        "line_graph": str(args.line_graph),
        "label_free_queue": True,
        "summary": summary,
        "queue": queue,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.include_markdown:
        write_markdown(args.markdown, summary, queue)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output}")
    if args.include_markdown:
        print(f"Wrote {args.markdown}")


if __name__ == "__main__":
    main()
