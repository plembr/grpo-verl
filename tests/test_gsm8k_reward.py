from __future__ import annotations

from grpo_verl.rewards.gsm8k_reward import (
    compute_score,
    extract_model_answer,
    extract_reference_answer,
    score_response,
)


def test_extract_reference_answer() -> None:
    assert extract_reference_answer("some work\n#### 1,234") == "1234"


def test_extract_model_answer_prefers_final_marker() -> None:
    text = "First I compute 10. Then final.\n#### 42"

    assert extract_model_answer(text) == "42"


def test_extract_model_answer_handles_repeated_final_marker() -> None:
    text = "Therefore, the final answer is:\n\n#### #### 52"

    assert extract_model_answer(text) == "52"


def test_repeated_final_marker_scores_correct() -> None:
    details = score_response("Reasoning\n#### #### 18", "18")

    assert details["score"] == 1.0
    assert details["correct"] is True


def test_exact_answer_scores_one() -> None:
    assert compute_score("openai/gsm8k", "Reasoning\n#### 18", "work\n#### 18", {}) == 1.0


def test_two_argument_compatibility() -> None:
    assert compute_score("Reasoning\n#### 18", "work\n#### 18") == 1.0


def test_three_argument_compatibility() -> None:
    assert compute_score("Reasoning\n#### 18", "18", {"split": "test"}) == 1.0


def test_wrong_answer_with_marker_gets_small_format_reward() -> None:
    details = score_response("Reasoning\n#### 19", "work\n#### 18")

    assert details["score"] == 0.0
    assert details["correct"] is False


def test_correct_answer_without_marker_gets_partial_reward() -> None:
    details = score_response("Reasoning. The answer is 18.", "work\n#### 18")

    assert details["score"] == 0.7
    assert details["correct"] is True
    assert details["has_final_marker"] is False


def test_missing_number_scores_zero() -> None:
    assert compute_score("No final answer", "work\n#### 18") == 0.0
