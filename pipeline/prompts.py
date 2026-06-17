"""Strict JSON prompts for Phase 4 pipeline strategies."""

from __future__ import annotations

import json

from pipeline.contracts import ALLOWED_ERROR_TYPES, TextBox
from pipeline.localization import blocks_as_prompt_payload


ERROR_TYPES = ", ".join(sorted(ALLOWED_ERROR_TYPES))

SINGLE_PASS_SCHEMA = {
    "text": "full transcription",
    "blocks": [{"index": 0, "bbox": [0, 0, 0, 0], "text": "localized text"}],
    "errors": [{
        "type": "spelling",
        "bbox": [0, 0, 0, 0],
        "description": "brief explanation",
        "correction": "optional correction",
        "evidence_text": "exact text span",
    }],
    "feedback": {
        "summary": "short feedback for the student",
        "strengths": ["short strength"],
        "improvements": ["short improvement"],
    },
}

STAGE2_SCHEMA = {
    "errors": [{
        "type": "spelling",
        "bbox": [0, 0, 0, 0],
        "description": "brief explanation",
        "correction": "optional correction",
        "evidence_text": "exact text span",
    }],
    "feedback": {
        "summary": "short feedback for the student",
        "strengths": ["short strength"],
        "improvements": ["short improvement"],
    },
}

VERBATIM_WORD_OCR_PROMPT = (
    "You are a copyist OCR engine for handwritten student writing.\n"
    "Your only job is to copy what is visibly written, not what the sentence should say.\n"
    "Do not correct spelling, grammar, tense, punctuation, capitalization, word choice, "
    "missing words, repeated words, or homophones.\n"
    "If a student wrote a mistake, preserve the mistake exactly.\n\n"
    "Examples of required behavior:\n"
    "- If the page says 'bred', output 'bred', not 'bread'.\n"
    "- If the page says 'minuts', output 'minuts', not 'minutes'.\n"
    "- If the page says 'forgoten', output 'forgoten', not 'forgotten'.\n"
    "- If the page says 'wether', output 'wether', not 'weather'.\n"
    "- If the page says 'umbrela', output 'umbrela', not 'umbrella'.\n"
    "- If the page says 'They lives', output both words exactly as 'They' and 'lives'.\n"
    "- If the page says 'My brother and me', output 'me', not 'I'.\n\n"
    "For each visible word or punctuation-attached word, output exactly one line:\n"
    "[x1, y1, x2, y2] observed_word\n"
    "Use pixel-like coordinates in the model's normal coordinate system. "
    "Preserve reading order. Do not output explanations."
)


def build_single_pass_prompt() -> str:
    """Prompt a VLM to do OCR, localization, error detection, and feedback."""
    return (
        "You are evaluating a handwritten English essay page.\n"
        "Return only valid JSON. Do not wrap it in markdown.\n"
        "Tasks:\n"
        "1. Transcribe all handwritten text in reading order.\n"
        "2. Localize every individual word with pixel bboxes [x1, y1, x2, y2].\n"
        "3. Detect writing errors only when visible in the student's text.\n"
        "4. Produce concise student feedback.\n\n"
        f"Allowed error types: {ERROR_TYPES}.\n"
        "If there are no writing errors, return an empty errors array.\n"
        "Every error bbox must point to the smallest available word span.\n\n"
        "JSON schema:\n"
        f"{json.dumps(SINGLE_PASS_SCHEMA, indent=2)}"
    )


def build_stage2_prompt(text: str, boxes: list[TextBox]) -> str:
    """Prompt a model to grade OCR text using known text boxes."""
    return (
        "You are an English writing feedback engine for a handwritten essay pipeline.\n"
        "Return only valid JSON. Do not wrap it in markdown.\n"
        "The OCR transcription is intended to be the student's observed writing, "
        "including mistakes. Grade that observed text exactly as written.\n"
        "Use the OCR transcription and provided boxes. Do not invent errors.\n"
        "If the text is clean or copied correctly, return an empty errors array.\n\n"
        f"Allowed error types: {ERROR_TYPES}.\n"
        "For each error, choose the smallest relevant bbox from the provided boxes "
        "or the union of adjacent boxes.\n\n"
        "For each error, evidence_text must be an exact word or adjacent phrase "
        "from the OCR transcription/box text, not a corrected rewrite.\n\n"
        "Transcription:\n"
        f"{text}\n\n"
        "Boxes:\n"
        f"{json.dumps(blocks_as_prompt_payload(boxes), ensure_ascii=False)}\n\n"
        "JSON schema:\n"
        f"{json.dumps(STAGE2_SCHEMA, indent=2)}"
    )


def build_repair_prompt(raw_response: str, schema: dict) -> str:
    """Ask the model to repair a malformed JSON response."""
    return (
        "The previous response was not valid for the required schema.\n"
        "Return only corrected JSON. Do not add commentary or markdown.\n\n"
        "Required schema:\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        "Previous response:\n"
        f"{raw_response}"
    )
