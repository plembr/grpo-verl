from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_prepare_sample_jsonl(tmp_path: Path) -> None:
    output_dir = tmp_path / "gsm8k"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_gsm8k_verl.py",
            "--sample-data",
            "--output-dir",
            str(output_dir),
            "--format",
            "jsonl",
            "--limit",
            "4",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Prepared GSM8K verl data" in result.stdout
    train_path = output_dir / "train.jsonl"
    test_path = output_dir / "test.jsonl"
    assert train_path.exists()
    assert test_path.exists()

    first = json.loads(train_path.read_text(encoding="utf-8").splitlines()[0])
    assert first["data_source"] == "openai/gsm8k"
    assert first["ability"] == "math"
    assert first["prompt"][0]["role"] == "system"
    assert first["reward_model"]["style"] == "rule"
    assert "ground_truth" in first["reward_model"]

