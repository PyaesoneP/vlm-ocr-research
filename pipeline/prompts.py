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
        "word_indices": [0],
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

STAGE2_CONTRACT_V2_SCHEMA = {
    "errors": [{
        "type": "spelling",
        "word_indices": [0],
        "bbox": [0, 0, 0, 0],
        "evidence_text": "exact copied word or adjacent phrase from provided words",
        "correction": "corrected word or phrase",
        "description": "brief explanation grounded in the evidence_text",
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

CROP_VERBATIM_WORD_PROMPT = (
    "You are checking one small crop from a handwritten student page.\n"
    "Copy only the visible handwritten word or short adjacent phrase in this crop.\n"
    "Do not correct spelling, grammar, capitalization, homophones, repeated words, or fused words.\n"
    "If the crop says 'bred', return 'bred', not 'bread'. If it says 'forgoten', return 'forgoten'.\n"
    "Return only valid JSON with this schema:\n"
    '{"observed_text": "exact visible text", "confidence": "high|medium|low"}\n'
    "Use confidence high only when the crop is clear and the letters are visible."
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


def build_stage2_prompt(
    text: str,
    boxes: list[TextBox],
    *,
    mode: str = "baseline",
    word_alternatives: list[dict] | None = None,
) -> str:
    """Prompt a model to grade OCR text using known text boxes."""
    if mode == "contract_v2":
        return _build_stage2_contract_v2_prompt(text, boxes, verify_image=False)
    if mode == "contract_v3":
        return _build_stage2_contract_v3_prompt(text, boxes, verify_image=False)
    if mode == "contract_v3_lattice":
        return _build_stage2_contract_v3_lattice_prompt(
            text,
            boxes,
            word_alternatives or [],
            verify_image=False,
        )
    if mode == "image_verify":
        return _build_stage2_contract_v2_prompt(text, boxes, verify_image=True)
    if mode == "image_verify_v3":
        return _build_stage2_contract_v3_prompt(text, boxes, verify_image=True)
    if mode == "image_verify_v3_lattice":
        return _build_stage2_contract_v3_lattice_prompt(
            text,
            boxes,
            word_alternatives or [],
            verify_image=True,
        )

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


def _build_stage2_contract_v2_prompt(
    text: str,
    boxes: list[TextBox],
    *,
    verify_image: bool,
) -> str:
    """Stricter Stage 2 contract for reviewed real-world source-text grading."""
    image_rule = (
        "The page image is also provided. Use it only to verify that the visible "
        "handwriting supports the exact evidence_text before returning an error.\n"
        if verify_image else
        "Use only the provided transcription and word list for this source-text upper-bound test.\n"
    )
    return (
        "You are an English writing error detector for a handwritten essay feedback pipeline.\n"
        "Return only valid JSON. Do not wrap it in markdown. Do not include commentary.\n\n"
        "Your job is to find ONLY clear writing errors that are directly supported by the "
        "provided words. Do not infer intended meaning, do not rewrite the whole sentence, "
        "and do not guess. If a page is clean or you are uncertain, return an empty errors array.\n"
        f"{image_rule}\n"
        f"Allowed error types: {ERROR_TYPES}.\n\n"
        "Hard requirements for every error:\n"
        "1. evidence_text must be copied exactly from one word or adjacent words in the word list.\n"
        "2. word_indices must contain the exact provided word index or adjacent word indices that form evidence_text.\n"
        "3. bbox must equal the union of the bboxes for word_indices.\n"
        "4. correction must be the minimal corrected word or phrase, not a full sentence rewrite.\n"
        "5. Do not report punctuation, capitalization, grammar, or spelling errors unless the exact visible words prove the error.\n"
        "6. Do not report style suggestions, awkward wording, or possible improvements as errors.\n\n"
        "Examples:\n"
        "- If the word list contains bred and the correction is bread, report evidence_text bred.\n"
        "- If the word list contains should of and the correction is should have, report both word indices.\n"
        "- If the word list already contains the corrected spelling, do not invent an error.\n"
        "- If no listed words prove an error, return errors: [].\n\n"
        "Transcription:\n"
        f"{text}\n\n"
        "Word list with indices and boxes:\n"
        f"{json.dumps(blocks_as_prompt_payload(boxes), ensure_ascii=False)}\n\n"
        "JSON schema:\n"
        f"{json.dumps(STAGE2_CONTRACT_V2_SCHEMA, indent=2)}"
    )


def _build_stage2_contract_v3_prompt(
    text: str,
    boxes: list[TextBox],
    *,
    verify_image: bool,
) -> str:
    """Stage 2 contract with explicit error-type taxonomy."""
    image_rule = (
        "The page image is also provided. Use it only to verify that the visible "
        "handwriting supports the exact evidence_text before returning an error.\n"
        if verify_image else
        "Use only the provided transcription and word list for this source-text upper-bound test.\n"
    )
    return (
        "You are an English writing error detector for a handwritten essay feedback pipeline.\n"
        "Return only valid JSON. Do not wrap it in markdown. Do not include commentary.\n\n"
        "Find ONLY clear writing errors directly supported by the provided words. Do not infer "
        "intended meaning, do not rewrite whole sentences, and do not guess. If a page is clean "
        "or you are uncertain, return an empty errors array.\n"
        f"{image_rule}\n"
        "Use exactly these error type rules:\n"
        "- spelling: a single word is misspelled or nonstandard as written, such as bred->bread, "
        "wether->weather, umbrela->umbrella, There->Their, beutiful->beautiful, untill->until, "
        "wich->which, intresting->interesting, usualy->usually, or flor->floor.\n"
        "- grammar: pronoun/case, subject-verb agreement, tense, repeated words, homophone grammar, "
        "missing function words, or fused-word grammar, such as me->I, lives->live, know->knows, "
        "the the->the, should of->should have, took us hour->took us an hour, come->came, "
        "or atleast->at least. Do not label these as spelling.\n"
        "- capitalization: a word needs uppercase/lowercase, such as thursday->Thursday. "
        "Do not label capitalization as spelling.\n"
        "- punctuation: punctuation only.\n"
        "- structural: organization or paragraph-level problems only.\n\n"
        "Hard requirements for every error:\n"
        "1. evidence_text must be copied exactly from one word or adjacent words in the word list.\n"
        "2. word_indices must contain the exact provided word index or adjacent word indices that form evidence_text.\n"
        "3. bbox must equal the union of the bboxes for word_indices.\n"
        "4. correction must be the minimal corrected word or phrase, not a full sentence rewrite.\n"
        "5. Do not report style suggestions, awkward wording, or possible improvements as errors.\n\n"
        "Transcription:\n"
        f"{text}\n\n"
        "Word list with indices and boxes:\n"
        f"{json.dumps(blocks_as_prompt_payload(boxes), ensure_ascii=False)}\n\n"
        "JSON schema:\n"
        f"{json.dumps(STAGE2_CONTRACT_V2_SCHEMA, indent=2)}"
    )


def _build_stage2_contract_v3_lattice_prompt(
    text: str,
    boxes: list[TextBox],
    word_alternatives: list[dict],
    *,
    verify_image: bool,
) -> str:
    """Stage 2 contract that receives uncertainty alternatives without rewriting OCR."""
    image_rule = (
        "The page image is also provided. Use it to verify whether an alternative "
        "candidate is visibly supported before returning an error from that alternative.\n"
        if verify_image else
        "Use only the provided transcription, word list, and OCR-supported alternatives. "
        "Ignore alternatives that are not supported by an OCR source.\n"
    )
    return (
        "You are an English writing error detector for a handwritten essay feedback pipeline.\n"
        "Return only valid JSON. Do not wrap it in markdown. Do not include commentary.\n\n"
        "The canonical word list is the OCR transcript. Grade the canonical words first. "
        "Some words also include an uncertainty lattice: possible observed readings for "
        "the same word index. Alternatives are evidence hints, not replacements.\n"
        f"{image_rule}\n"
        "Use exactly these error type rules:\n"
        "- spelling: a single word is misspelled or nonstandard as written, such as bred->bread, "
        "wether->weather, umbrela->umbrella, There->Their, beutiful->beautiful, untill->until, "
        "wich->which, intresting->interesting, usualy->usually, or flor->floor.\n"
        "- grammar: pronoun/case, subject-verb agreement, tense, repeated words, homophone grammar, "
        "missing function words, or fused-word grammar, such as me->I, lives->live, know->knows, "
        "the the->the, should of->should have, took us hour->took us an hour, come->came, "
        "or atleast->at least. Do not label these as spelling.\n"
        "- capitalization: a word needs uppercase/lowercase, such as thursday->Thursday. "
        "Do not label capitalization as spelling.\n"
        "- punctuation: punctuation only.\n"
        "- structural: organization or paragraph-level problems only.\n\n"
        "Hard requirements for every error:\n"
        "1. word_indices must reference exact indices from the canonical word list.\n"
        "2. bbox must equal the union of the bboxes for word_indices.\n"
        "3. If the canonical word itself proves an error, use the canonical word as evidence_text "
        "and ignore corrected-looking alternatives for that index.\n"
        "4. Use an alternative as evidence_text only when the canonical word appears corrected "
        "or ambiguous but the alternative is a plausible erroneous observed reading for that same index.\n"
        "5. If evidence_text comes from an alternative, copy that alternative's observed_text exactly "
        "and keep word_indices anchored to the canonical word's index.\n"
        "6. Ignore unsupported lexical-neighbor alternatives unless the page image visibly supports "
        "the alternative. If uncertain, return no error for that candidate.\n"
        "7. Never return an error where evidence_text and correction are identical after punctuation "
        "is stripped.\n"
        "8. correction must be the minimal corrected word or phrase, not a full sentence rewrite.\n"
        "9. Do not report style suggestions, awkward wording, or possible improvements as errors.\n\n"
        "Transcription:\n"
        f"{text}\n\n"
        "Canonical word list with indices and boxes:\n"
        f"{json.dumps(blocks_as_prompt_payload(boxes), ensure_ascii=False)}\n\n"
        "Uncertainty alternatives by canonical word index:\n"
        f"{json.dumps(word_alternatives, ensure_ascii=False)}\n\n"
        "JSON schema:\n"
        f"{json.dumps(STAGE2_CONTRACT_V2_SCHEMA, indent=2)}"
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
