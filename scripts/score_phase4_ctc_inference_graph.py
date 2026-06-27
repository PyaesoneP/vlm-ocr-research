"""Attach CTC candidate scores to a label-free Phase 4 inference graph."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.optical_candidate_scorer import EasyOCRCTCCandidateScorer


DEFAULT_GRAPH = Path("benchmark/results/phase4_inference_evidence_graph_unscored_20image.json")
DEFAULT_OUTPUT = Path("benchmark/results/phase4_inference_evidence_graph_ctc_rw1_rw2_rw11.json")

SUPPORTED_ALTERNATIVE = "SUPPORTED_ALTERNATIVE"
SUPPORTED_CANONICAL = "SUPPORTED_CANONICAL"
UNCERTAIN_REVIEW = "UNCERTAIN_REVIEW"


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def score_record(record: dict[str, Any], scorer: EasyOCRCTCCandidateScorer) -> dict[str, Any]:
    crop_path = Path(record["crop_path"])
    scored = []
    for candidate in record.get("candidates", []):
        score = scorer.score_candidate(crop_path, str(candidate.get("text", "")))
        row = {
            **candidate,
            "ctc_raw_logprob": score.raw_logprob,
            "ctc_normalized_logprob": score.normalized_logprob,
            "ctc_char_count": score.char_count,
            "ctc_target_length": score.target_length,
            "ctc_timesteps": score.timesteps,
            "ctc_scorer_text": score.scorer_text,
            "ctc_unsupported_chars": score.unsupported_chars,
            "ctc_supported_chars": score.supported,
            "ctc_latency_ms": score.latency_ms,
            # Compatibility with the existing selector/audit vocabulary.
            "visual_gain": score.normalized_logprob,
        }
        scored.append(row)

    scored.sort(key=lambda row: row["ctc_normalized_logprob"], reverse=True)
    for rank, row in enumerate(scored, start=1):
        row["rank"] = rank

    top = scored[0] if scored else None
    runner_up = scored[1] if len(scored) > 1 else None
    canonical = next((row for row in scored if row.get("is_canonical")), None)
    best_alternative = next((row for row in scored if not row.get("is_canonical")), None)
    margin = (
        float(top["ctc_normalized_logprob"] - runner_up["ctc_normalized_logprob"])
        if top
        and runner_up
        and finite(top.get("ctc_normalized_logprob"))
        and finite(runner_up.get("ctc_normalized_logprob"))
        else None
    )
    alt_margin_over_canonical = (
        float(best_alternative["ctc_normalized_logprob"] - canonical["ctc_normalized_logprob"])
        if best_alternative
        and canonical
        and finite(best_alternative.get("ctc_normalized_logprob"))
        and finite(canonical.get("ctc_normalized_logprob"))
        else None
    )
    return {
        **record,
        "top_text": top.get("text", "") if top else "",
        "top_is_canonical": bool(top and top.get("is_canonical")),
        "top_supported_by_ocr": bool(top and top.get("supported_by_ocr")),
        "top_ctc_supported_chars": bool(top and top.get("ctc_supported_chars")),
        "top_ctc_unsupported_chars": top.get("ctc_unsupported_chars", []) if top else [],
        "margin": margin,
        "alternative_margin_over_canonical": alt_margin_over_canonical,
        "hypotheses": scored,
    }


def decide(record: dict[str, Any], *, threshold: float, require_ocr_support: bool) -> str:
    top_is_canonical = bool(record.get("top_is_canonical"))
    if top_is_canonical:
        return SUPPORTED_CANONICAL
    margin = record.get("alternative_margin_over_canonical")
    if margin is None or float(margin) < threshold:
        return UNCERTAIN_REVIEW
    if not record.get("top_ctc_supported_chars"):
        return UNCERTAIN_REVIEW
    if require_ocr_support and not record.get("top_supported_by_ocr"):
        return UNCERTAIN_REVIEW
    return SUPPORTED_ALTERNATIVE


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records_total": len(records),
        "supported_alternative": sum(1 for row in records if row.get("decision") == SUPPORTED_ALTERNATIVE),
        "supported_canonical": sum(1 for row in records if row.get("decision") == SUPPORTED_CANONICAL),
        "uncertain_review": sum(1 for row in records if row.get("decision") == UNCERTAIN_REVIEW),
        "top_alternative": sum(1 for row in records if not row.get("top_is_canonical")),
        "top_alternative_ocr_supported": sum(
            1 for row in records
            if not row.get("top_is_canonical") and row.get("top_supported_by_ocr")
        ),
        "top_alternative_ctc_supported_chars": sum(
            1 for row in records
            if not row.get("top_is_canonical") and row.get("top_ctc_supported_chars")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image", action="append", dest="images", default=[])
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--allow-unsupported-alternatives", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--img-h", type=int, default=64)
    parser.add_argument("--img-w", type=int, default=256)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    data = json.loads(args.graph.read_text())
    image_filter = set(args.images)
    records = data.get("records", [])
    if image_filter:
        records = [record for record in records if record.get("image") in image_filter]

    scorer = EasyOCRCTCCandidateScorer(device=args.device, img_h=args.img_h, img_w=args.img_w)
    scored = []
    for offset, record in enumerate(records, start=1):
        scored_record = score_record(record, scorer)
        scored_record["decision"] = decide(
            scored_record,
            threshold=args.threshold,
            require_ocr_support=not args.allow_unsupported_alternatives,
        )
        scored.append(scored_record)
        if args.progress_every > 0 and offset % args.progress_every == 0:
            print(f"[ctc-graph] scored {offset}/{len(records)} records", flush=True)

    result = {
        **{key: value for key, value in data.items() if key != "records"},
        "source_graph": str(args.graph),
        "label_free": True,
        "scored": True,
        "score_backend": "easyocr_ctc",
        "scorer": scorer.metadata(),
        "threshold": args.threshold,
        "require_ocr_support": not args.allow_unsupported_alternatives,
        "images": sorted(image_filter),
        "summary": summarize(scored),
        "records": scored,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
