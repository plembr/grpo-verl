from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation
from typing import Any


FINAL_ANSWER_RE = re.compile(r"####\s*([^\n\r]+)")
NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def _normalize_number_text(value: str) -> str:
    text = str(value or "").strip()
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.replace("%", "")
    return text.strip()


def _to_decimal(value: str) -> Decimal | None:
    text = _normalize_number_text(value)
    if not text:
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def extract_reference_answer(answer: str) -> str:
    """Extract the GSM8K final answer after the canonical #### marker."""

    match = FINAL_ANSWER_RE.search(str(answer or ""))
    if match:
        return _normalize_number_text(match.group(1))

    numbers = NUMBER_RE.findall(str(answer or ""))
    return _normalize_number_text(numbers[-1]) if numbers else ""


def extract_model_answer(solution_str: str) -> str:
    """Extract the model's final numeric answer.

    The preferred format is `#### number`. If the model misses that format, use
    the last number in the response so early GRPO runs still get weak signal.
    """

    text = str(solution_str or "")
    match = FINAL_ANSWER_RE.search(text)
    if match:
        return _normalize_number_text(match.group(1))

    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        numbers = NUMBER_RE.findall(boxed[-1])
        if numbers:
            return _normalize_number_text(numbers[-1])

    numbers = NUMBER_RE.findall(text)
    return _normalize_number_text(numbers[-1]) if numbers else ""


def is_equivalent_number(prediction: str, reference: str) -> bool:
    pred_value = _to_decimal(prediction)
    ref_value = _to_decimal(reference)
    if pred_value is None or ref_value is None:
        return _normalize_number_text(prediction) == _normalize_number_text(reference)
    return pred_value == ref_value


def score_response(solution_str: str, ground_truth: str) -> dict[str, Any]:
    reference = extract_reference_answer(ground_truth)
    prediction = extract_model_answer(solution_str)
    has_final_marker = bool(FINAL_ANSWER_RE.search(str(solution_str or "")))

    if not prediction or not reference:
        return {
            "score": 0.0,
            "prediction": prediction,
            "reference": reference,
            "has_final_marker": has_final_marker,
            "correct": False,
        }

    correct = is_equivalent_number(prediction, reference)
    if correct:
        score = 1.0 if has_final_marker else 0.8
    else:
        score = 0.1 if has_final_marker else 0.0

    if not math.isfinite(score):
        score = 0.0

    return {
        "score": float(score),
        "prediction": prediction,
        "reference": reference,
        "has_final_marker": has_final_marker,
        "correct": correct,
    }


def compute_score(
    data_source: str,
    solution_str: str | None = None,
    ground_truth: str | None = None,
    extra_info: dict[str, Any] | None = None,
) -> float:
    """verl-compatible GSM8K rule reward.

    Also supports the shorter two-argument style for local tests:
    `compute_score(solution_str, ground_truth)`.
    """

    if ground_truth is None:
        ground_truth = solution_str or ""
        solution_str = data_source
        data_source = ""

    del data_source
    del extra_info
    return float(score_response(solution_str or "", ground_truth or "")["score"])

