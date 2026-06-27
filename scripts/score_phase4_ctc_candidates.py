"""Score Phase 4 minimal-pair candidates with an optical CTC recognizer."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.optical_candidate_scorer import EasyOCRCTCCandidateScorer


DEFAULT_PAIRS = Path("benchmark/results/phase4_minimal_pair_dataset.json")
DEFAULT_OUTPUT = Path("benchmark/results/phase4_ctc_candidate_scores_minimal_pairs.json")


def compact(text: str) -> str:
    return "".join(ch.lower() for ch in str(text) if ch.isalnum())


def evidence_key(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]+", "", str(text)).strip())


def unique_label_candidates(pair: dict[str, Any]) -> list[str]:
    candidates = []
    seen: set[str] = set()
    for text in (pair.get("visible_form", ""), pair.get("normalized_form", "")):
        key = evidence_key(text)
        if text and key and key not in seen:
            seen.add(key)
            candidates.append(str(text))
    return candidates


def unique_all_candidates(pair: dict[str, Any]) -> list[str]:
    candidates = unique_label_candidates(pair)
    seen = {evidence_key(text) for text in candidates}
    for candidate in pair.get("candidates", []):
        text = str(candidate.get("text", ""))
        key = evidence_key(text)
        if text and key and key not in seen:
            seen.add(key)
            candidates.append(text)
    return candidates


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def score_pair(
    pair: dict[str, Any],
    scorer: EasyOCRCTCCandidateScorer,
    *,
    candidate_mode: str,
) -> dict[str, Any]:
    crop_path = Path(pair["crop_path"])
    candidate_texts = unique_label_candidates(pair) if candidate_mode == "labels" else unique_all_candidates(pair)
    scored = scorer.score_candidates(crop_path, candidate_texts)
    top = scored[0] if scored else None
    runner_up = scored[1] if len(scored) > 1 else None
    margin = (
        float(top["normalized_logprob"] - runner_up["normalized_logprob"])
        if top is not None
        and runner_up is not None
        and finite(top["normalized_logprob"])
        and finite(runner_up["normalized_logprob"])
        else None
    )
    visible_norm = compact(pair.get("visible_form", ""))
    normalized_norm = compact(pair.get("normalized_form", ""))
    visible_key = evidence_key(pair.get("visible_form", ""))
    normalized_key = evidence_key(pair.get("normalized_form", ""))
    top_key = evidence_key(top["text"]) if top else ""
    compact_labels_distinct = visible_norm != normalized_norm
    top_is_visible_surface = bool(top and top_key == visible_key)
    top_is_normalized_surface = bool(top and top_key == normalized_key)
    return {
        "pair_id": pair.get("pair_id"),
        "split": pair.get("split"),
        "image": pair.get("image"),
        "crop_path": pair.get("crop_path"),
        "visible_form": pair.get("visible_form"),
        "normalized_form": pair.get("normalized_form"),
        "canonical_text": pair.get("canonical_text"),
        "error_type": pair.get("error_type"),
        "phrase_shaped": " " in evidence_key(pair.get("visible_form", "")) or " " in evidence_key(pair.get("normalized_form", "")),
        "top_text": top["text"] if top else "",
        "top_scorer_text": top["scorer_text"] if top else "",
        "top_supported": bool(top and top.get("supported")),
        "top_unsupported_chars": top.get("unsupported_chars", []) if top else [],
        "top_is_visible": bool(
            top_is_visible_surface or (compact_labels_distinct and top and compact(top["text"]) == visible_norm)
        ),
        "top_is_normalized": bool(
            top_is_normalized_surface or (compact_labels_distinct and top and compact(top["text"]) == normalized_norm)
        ),
        "top_is_visible_surface": top_is_visible_surface,
        "top_is_normalized_surface": top_is_normalized_surface,
        "margin": margin,
        "greedy_decode": scorer.greedy_decode(crop_path),
        "scores": scored,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [row for row in rows if row["split"] == "positive"]
    controls = [row for row in rows if row["split"] == "clean_control"]
    phrase_rows = [row for row in rows if row.get("phrase_shaped")]
    single_rows = [row for row in rows if not row.get("phrase_shaped")]

    def rate(items: list[dict[str, Any]], key: str) -> float:
        return sum(1 for item in items if item.get(key)) / len(items) if items else 0.0

    margins = [row["margin"] for row in rows if row.get("margin") is not None]
    unsupported_top = [row for row in rows if row.get("top_unsupported_chars")]
    return {
        "pairs_scored": len(rows),
        "positive_pairs": len(positives),
        "clean_control_pairs": len(controls),
        "phrase_shaped_pairs": len(phrase_rows),
        "single_token_pairs": len(single_rows),
        "positive_visible_preference": rate(positives, "top_is_visible"),
        "clean_visible_preference": rate(controls, "top_is_visible"),
        "single_token_visible_preference": rate(single_rows, "top_is_visible"),
        "phrase_shaped_visible_preference": rate(phrase_rows, "top_is_visible"),
        "overall_visible_preference": rate(rows, "top_is_visible"),
        "top_unsupported_count": len(unsupported_top),
        "margin_min": min(margins) if margins else None,
        "margin_median": sorted(margins)[len(margins) // 2] if margins else None,
        "margin_max": max(margins) if margins else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate-mode", choices=["labels", "all"], default="labels")
    parser.add_argument("--split", choices=["all", "positive", "clean_control"], default="all")
    parser.add_argument("--max-pairs", type=int, default=0, help="Limit scored pairs; 0 means all.")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--img-h", type=int, default=64)
    parser.add_argument("--img-w", type=int, default=256)
    parser.add_argument("--progress-every", type=int, default=5)
    args = parser.parse_args()

    data = json.loads(args.pairs.read_text())
    pairs = data.get("pairs", [])
    if args.split != "all":
        pairs = [pair for pair in pairs if pair.get("split") == args.split]
    if args.max_pairs > 0:
        pairs = pairs[:args.max_pairs]

    scorer = EasyOCRCTCCandidateScorer(device=args.device, img_h=args.img_h, img_w=args.img_w)
    rows = []
    for offset, pair in enumerate(pairs, start=1):
        rows.append(score_pair(pair, scorer, candidate_mode=args.candidate_mode))
        if args.progress_every > 0 and offset % args.progress_every == 0:
            print(f"[ctc] scored {offset}/{len(pairs)} pairs", flush=True)

    result = {
        "pairs": str(args.pairs),
        "scorer": scorer.metadata(),
        "candidate_mode": args.candidate_mode,
        "split": args.split,
        "summary": summarize(rows),
        "scores": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
