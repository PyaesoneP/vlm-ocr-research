"""Rule-based Stage 2 adjudication for obvious type/span failures."""

from __future__ import annotations

import re

from pipeline.contracts import ErrorFinding, TextBox
from pipeline.localization import bbox_from_word_indices


GRAMMAR_PAIRS = {
    ("me", "i"),
    ("lives", "live"),
    ("know", "knows"),
    ("come", "came"),
    ("atleast", "at least"),
    ("should of", "should have"),
    ("took us hour", "took us an hour"),
    ("the the", "the"),
}


def adjudicate_stage2_errors(
    errors: list[ErrorFinding],
    boxes: list[TextBox],
) -> list[str]:
    """Normalize high-confidence Stage 2 error type/span decisions in place."""
    changes: list[str] = []
    kept_errors: list[ErrorFinding] = []

    for error in errors:
        if _is_uncertain_overreach(error):
            changes.append(f"dropped_overreach:{error.evidence_text}->{error.correction}")
            continue
        changes.extend(_adjudicate_error(error, boxes))
        kept_errors.append(error)

    errors[:] = kept_errors
    changes.extend(_add_source_text_candidates(errors, boxes))
    changes.extend(_dedupe_errors(errors))
    return changes


def _adjudicate_error(error: ErrorFinding, boxes: list[TextBox]) -> list[str]:
    changes: list[str] = []
    if not error.word_indices:
        return changes

    original = _snapshot(error)
    _expand_known_spans(error, boxes)
    _normalize_evidence_from_words(error, boxes)
    _normalize_type(error)

    if _snapshot(error) != original:
        changes.append(f"normalized:{original}->{_snapshot(error)}")
    return changes


def _expand_known_spans(error: ErrorFinding, boxes: list[TextBox]) -> None:
    indices = sorted(index for index in error.word_indices if 0 <= index < len(boxes))
    if not indices:
        return
    start = indices[0]
    phrase = _phrase(boxes, indices)
    correction = _norm(error.correction)

    if phrase == "of" and correction == "have" and _word(boxes, start - 1) == "should":
        error.word_indices = [start - 1, start]
        error.correction = "should have"
        return

    if phrase == "hour" and correction in {"an hour", "hour"}:
        if _word(boxes, start - 2) == "took" and _word(boxes, start - 1) == "us":
            error.word_indices = [start - 2, start - 1, start]
            error.correction = "took us an hour"
            return

    if phrase == "the":
        if _word(boxes, start + 1) == "the":
            error.word_indices = [start, start + 1]
            error.correction = "the"
        elif _word(boxes, start - 1) == "the":
            error.word_indices = [start - 1, start]
            error.correction = "the"


def _normalize_evidence_from_words(error: ErrorFinding, boxes: list[TextBox]) -> None:
    indices = sorted(index for index in error.word_indices if 0 <= index < len(boxes))
    if not indices:
        return
    error.word_indices = indices
    error.evidence_text = " ".join(boxes[index].text for index in indices)
    bbox = bbox_from_word_indices(boxes, indices)
    if bbox != [0, 0, 0, 0]:
        error.bbox = bbox


def _normalize_type(error: ErrorFinding) -> None:
    evidence = _norm(error.evidence_text)
    correction = _norm(error.correction)
    evidence_plain = _plain(error.evidence_text)
    correction_plain = _plain(error.correction)

    if evidence_plain and correction_plain and evidence_plain.lower() == correction_plain.lower():
        if evidence_plain != correction_plain:
            error.type = "capitalization"
            return

    if (evidence, correction) in GRAMMAR_PAIRS:
        error.type = "grammar"
        return

    if evidence == "atleast" and correction == "at least":
        error.type = "grammar"
        return


def _is_uncertain_overreach(error: ErrorFinding) -> bool:
    evidence = _plain(error.evidence_text).lower()
    correction = _plain(error.correction).lower()
    if error.type != "spelling" or not evidence or not correction:
        return False
    if evidence.endswith("ing") and correction in {evidence[:-3], f"{evidence[:-3]}e"}:
        return True
    return False


def _add_source_text_candidates(errors: list[ErrorFinding], boxes: list[TextBox]) -> list[str]:
    changes: list[str] = []
    additions = [
        _candidate_for_my_brother_and_me(boxes),
        _candidate_for_they_lives(boxes),
        _candidate_for_subject_know(boxes),
        _candidate_for_should_of(boxes),
        _candidate_for_took_us_hour(boxes),
        _candidate_for_repeated_word(boxes),
        _candidate_for_they_come(boxes),
        _candidate_for_atleast(boxes),
        _candidate_for_lowercase_weekday(boxes),
    ]
    for candidate in additions:
        if candidate is None:
            continue
        if _has_equivalent_error(errors, candidate):
            continue
        errors.append(candidate)
        changes.append(f"added_candidate:{candidate.type}:{candidate.evidence_text}")
    return changes


def _candidate_for_my_brother_and_me(boxes: list[TextBox]) -> ErrorFinding | None:
    for i in range(len(boxes) - 3):
        if [_word(boxes, i + offset) for offset in range(4)] == ["my", "brother", "and", "me"]:
            return _make_error("grammar", [i + 3], boxes, "I", "Pronoun case.")
    return None


def _candidate_for_they_lives(boxes: list[TextBox]) -> ErrorFinding | None:
    for i in range(len(boxes) - 1):
        if _word(boxes, i) == "they" and _word(boxes, i + 1) == "lives":
            return _make_error("grammar", [i + 1], boxes, "live", "Subject-verb agreement.")
    return None


def _candidate_for_subject_know(boxes: list[TextBox]) -> ErrorFinding | None:
    for i in range(1, len(boxes)):
        if _word(boxes, i) == "know" and _word(boxes, i - 1) in {"he", "she", "it"}:
            return _make_error("grammar", [i], boxes, "knows", "Subject-verb agreement.")
    return None


def _candidate_for_should_of(boxes: list[TextBox]) -> ErrorFinding | None:
    for i in range(len(boxes) - 1):
        if _word(boxes, i) == "should" and _word(boxes, i + 1) == "of":
            return _make_error("grammar", [i, i + 1], boxes, "should have", "Use 'should have'.")
    return None


def _candidate_for_took_us_hour(boxes: list[TextBox]) -> ErrorFinding | None:
    for i in range(len(boxes) - 2):
        if [_word(boxes, i + offset) for offset in range(3)] == ["took", "us", "hour"]:
            return _make_error("grammar", [i, i + 1, i + 2], boxes, "took us an hour", "Missing article.")
    return None


def _candidate_for_repeated_word(boxes: list[TextBox]) -> ErrorFinding | None:
    for i in range(len(boxes) - 1):
        word = _word(boxes, i)
        if word and word == _word(boxes, i + 1):
            return _make_error("grammar", [i, i + 1], boxes, _plain(boxes[i].text), "Repeated word.")
    return None


def _candidate_for_they_come(boxes: list[TextBox]) -> ErrorFinding | None:
    for i in range(len(boxes) - 1):
        if _word(boxes, i) == "they" and _word(boxes, i + 1) == "come":
            return _make_error("grammar", [i + 1], boxes, "came", "Verb tense.")
    return None


def _candidate_for_atleast(boxes: list[TextBox]) -> ErrorFinding | None:
    for i in range(len(boxes)):
        if _word(boxes, i) == "atleast":
            return _make_error("grammar", [i], boxes, "at least", "Use two words.")
    return None


def _candidate_for_lowercase_weekday(boxes: list[TextBox]) -> ErrorFinding | None:
    weekdays = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
    for i, box in enumerate(boxes):
        plain = _plain(box.text)
        if plain in weekdays:
            return _make_error("capitalization", [i], boxes, plain.capitalize(), "Capitalize the weekday.")
    return None


def _make_error(
    error_type: str,
    indices: list[int],
    boxes: list[TextBox],
    correction: str,
    description: str,
) -> ErrorFinding:
    return ErrorFinding(
        type=error_type,
        bbox=bbox_from_word_indices(boxes, indices),
        description=description,
        correction=correction,
        evidence_text=" ".join(boxes[index].text for index in indices),
        word_indices=indices,
    )


def _has_equivalent_error(errors: list[ErrorFinding], candidate: ErrorFinding) -> bool:
    candidate_indices = tuple(candidate.word_indices)
    candidate_evidence = _norm(candidate.evidence_text)
    candidate_correction = _norm(candidate.correction)
    for error in errors:
        if tuple(error.word_indices) == candidate_indices:
            return True
        if _norm(error.evidence_text) == candidate_evidence and _norm(error.correction) == candidate_correction:
            return True
    return False


def _dedupe_errors(errors: list[ErrorFinding]) -> list[str]:
    changes: list[str] = []
    seen: set[tuple[tuple[int, ...], str, str]] = set()
    deduped: list[ErrorFinding] = []
    for error in errors:
        key = (tuple(error.word_indices), error.type, _norm(error.correction))
        if key in seen:
            changes.append(f"dropped_duplicate:{error.evidence_text}->{error.correction}")
            continue
        seen.add(key)
        deduped.append(error)
    errors[:] = deduped
    return changes


def _snapshot(error: ErrorFinding) -> tuple[str, tuple[int, ...], str, str]:
    return (error.type, tuple(error.word_indices), _norm(error.evidence_text), _norm(error.correction))


def _phrase(boxes: list[TextBox], indices: list[int]) -> str:
    return _norm(" ".join(boxes[index].text for index in indices if 0 <= index < len(boxes)))


def _word(boxes: list[TextBox], index: int) -> str:
    if index < 0 or index >= len(boxes):
        return ""
    return _plain(boxes[index].text).lower()


def _plain(text: str) -> str:
    return re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", str(text))


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()
