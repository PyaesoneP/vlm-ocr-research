#!/usr/bin/env python3
"""Auto-align draft boxes to source words for the real-world dataset.

This is an annotation accelerator, not a replacement for review. It maps the
known source tokens onto draft word boxes in reading order, then attaches
intended errors to word indices where possible.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "benchmark" / "test_dataset" / "realworld_writing_errors.json"


def tokenize(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def normalize_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def union_bbox(boxes: list[list[int]]) -> list[int]:
    valid = [box for box in boxes if len(box) == 4 and box != [0, 0, 0, 0]]
    if not valid:
        return [0, 0, 0, 0]
    return [
        min(box[0] for box in valid),
        min(box[1] for box in valid),
        max(box[2] for box in valid),
        max(box[3] for box in valid),
    ]


def similarity(source_token: str, draft_words: list[dict[str, Any]]) -> float:
    source_norm = normalize_token(source_token)
    draft_norm = "".join(normalize_token(str(word.get("text", ""))) for word in draft_words)
    if not source_norm and not draft_norm:
        return 1.0
    if not source_norm or not draft_norm:
        return 0.0
    return difflib.SequenceMatcher(None, source_norm, draft_norm).ratio()


def alignment_cost(source_token: str, group: list[dict[str, Any]]) -> float:
    # Small merge penalty prevents over-merging unless the text similarity or
    # sequence constraints justify it.
    return (1.0 - similarity(source_token, group)) + 0.06 * max(0, len(group) - 1)


def align_tokens_to_drafts(
    source_tokens: list[str],
    draft_words: list[dict[str, Any]],
    max_merge: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    n = len(source_tokens)
    m = len(draft_words)
    if n == 0 or m == 0 or m < n:
        return [], {
            "method": "dp_partition",
            "status": "failed",
            "reason": "missing_source_or_too_few_draft_words",
            "source_word_count": n,
            "draft_word_count": m,
        }

    inf = float("inf")
    dp = [[inf] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[int, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for i in range(1, n + 1):
        min_j = i
        max_j = min(m, i * max_merge)
        remaining_tokens = n - i
        for j in range(min_j, max_j + 1):
            # Leave at least one draft word for each remaining source token.
            if m - j < remaining_tokens:
                continue
            for k in range(1, min(max_merge, j) + 1):
                prev_j = j - k
                if dp[i - 1][prev_j] == inf:
                    continue
                group = draft_words[prev_j:j]
                cost = dp[i - 1][prev_j] + alignment_cost(source_tokens[i - 1], group)
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    back[i][j] = (prev_j, k)

    if dp[n][m] == inf:
        return [], {
            "method": "dp_partition",
            "status": "failed",
            "reason": "no_valid_partition",
            "source_word_count": n,
            "draft_word_count": m,
        }

    groups: list[tuple[int, int]] = []
    i = n
    j = m
    while i > 0:
        item = back[i][j]
        if item is None:
            return [], {
                "method": "dp_partition",
                "status": "failed",
                "reason": "broken_backpointer",
                "source_word_count": n,
                "draft_word_count": m,
            }
        prev_j, _ = item
        groups.append((prev_j, j))
        j = prev_j
        i -= 1
    groups.reverse()

    words: list[dict[str, Any]] = []
    scores: list[float] = []
    for index, (start, end) in enumerate(groups):
        group = draft_words[start:end]
        score = similarity(source_tokens[index], group)
        scores.append(score)
        words.append({
            "index": index,
            "text": source_tokens[index],
            "bbox": union_bbox([word.get("bbox", []) for word in group]),
            "source": "auto_aligned_tesseract_draft",
            "draft_indices": [int(word.get("index", i)) for i, word in enumerate(group, start=start)],
            "draft_text": " ".join(str(word.get("text", "")) for word in group).strip(),
            "alignment_score": round(score, 4),
            "review_status": "needs_review",
        })

    low_threshold = 0.45
    return words, {
        "method": "dp_partition",
        "status": "auto_aligned_needs_review",
        "source_word_count": n,
        "draft_word_count": m,
        "word_count_delta": m - n,
        "max_merge": max_merge,
        "mean_alignment_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "low_alignment_score_count": sum(1 for score in scores if score < low_threshold),
        "low_alignment_score_threshold": low_threshold,
    }


def find_word_indices(words: list[dict[str, Any]], evidence_text: str) -> list[int]:
    evidence_tokens = [normalize_token(token) for token in tokenize(evidence_text)]
    evidence_tokens = [token for token in evidence_tokens if token]
    if not evidence_tokens:
        return []
    word_tokens = [normalize_token(str(word.get("text", ""))) for word in words]
    for start in range(0, len(word_tokens) - len(evidence_tokens) + 1):
        if word_tokens[start:start + len(evidence_tokens)] == evidence_tokens:
            return list(range(start, start + len(evidence_tokens)))
    return []


def attach_errors(entry: dict[str, Any]) -> None:
    words = entry.get("words", [])
    for error in entry.get("errors", []):
        indices = find_word_indices(words, str(error.get("evidence_text", "")))
        if indices:
            error["word_indices"] = indices
            error["bbox"] = union_bbox([words[index]["bbox"] for index in indices])
            error["annotation_status"] = "auto_aligned_needs_review"
        else:
            error["word_indices"] = []
            error["bbox"] = [0, 0, 0, 0]
            error["annotation_status"] = "needs_word_indices"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--max-merge", type=int, default=4)
    args = parser.parse_args()

    entries = json.loads(args.input.read_text())
    aligned_pages = 0
    error_links = 0
    for entry in entries:
        source_tokens = tokenize(str(entry.get("text", "")))
        draft_words = entry.get("draft_words", [])
        words, summary = align_tokens_to_drafts(source_tokens, draft_words, max_merge=args.max_merge)
        entry["alignment_summary"] = summary
        if words:
            entry["words"] = words
            entry["annotation_status"] = "auto_aligned_needs_review"
            aligned_pages += 1
            attach_errors(entry)
            error_links += sum(1 for error in entry.get("errors", []) if error.get("word_indices"))

    args.output.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {args.output}")
    print(f"auto_aligned_pages={aligned_pages}/{len(entries)} error_links={error_links}")


if __name__ == "__main__":
    main()
