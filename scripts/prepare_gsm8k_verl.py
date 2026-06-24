from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


DATA_SOURCE = "openai/gsm8k"
ABILITY = "math"
SYSTEM_PROMPT = (
    "You are a careful math reasoning assistant. Solve the problem step by step. "
    "Put the final numeric answer on a new line in the exact format: #### <answer>."
)

SAMPLE_ROWS = [
    {
        "question": "Janet has 3 apples and buys 5 more. How many apples does she have?",
        "answer": "Janet has 3 + 5 = 8 apples.\n#### 8",
    },
    {
        "question": "A pack has 12 pencils. Tom buys 4 packs. How many pencils does he buy?",
        "answer": "Tom buys 12 * 4 = 48 pencils.\n#### 48",
    },
    {
        "question": "Mia read 10 pages on Monday and twice as many on Tuesday. How many pages total?",
        "answer": "Tuesday is 10 * 2 = 20 pages. Total is 10 + 20 = 30.\n#### 30",
    },
    {
        "question": "There are 20 seats and 7 are empty. How many seats are occupied?",
        "answer": "Occupied seats are 20 - 7 = 13.\n#### 13",
    },
    {
        "question": "A ticket costs $6. How much do 9 tickets cost?",
        "answer": "The cost is 6 * 9 = 54 dollars.\n#### 54",
    },
    {
        "question": "Nora saves $15 per week for 6 weeks. How much does she save?",
        "answer": "She saves 15 * 6 = 90 dollars.\n#### 90",
    },
    {
        "question": "A class has 18 girls and 14 boys. How many students are there?",
        "answer": "There are 18 + 14 = 32 students.\n#### 32",
    },
    {
        "question": "A baker makes 45 cookies and sells 28. How many are left?",
        "answer": "Cookies left are 45 - 28 = 17.\n#### 17",
    },
]


def _load_gsm8k(split: str) -> list[dict[str, str]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: datasets. Install requirements-cloud.txt on the cloud machine, "
            "or use --sample-data for local smoke tests."
        ) from exc

    dataset = load_dataset("openai/gsm8k", "main", split=split)
    return [{"question": row["question"], "answer": row["answer"]} for row in dataset]


def _make_prompt(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": str(question).strip()},
    ]


def _to_verl_row(row: dict[str, str], split: str, index: int) -> dict[str, Any]:
    return {
        "data_source": DATA_SOURCE,
        "prompt": _make_prompt(row["question"]),
        "ability": ABILITY,
        "reward_model": {
            "style": "rule",
            "ground_truth": row["answer"],
        },
        "extra_info": {
            "split": split,
            "index": index,
            "question": row["question"],
            "answer": row["answer"],
        },
    }


def _sample_rows(rows: list[dict[str, str]], limit: int, seed: int) -> list[dict[str, str]]:
    if limit <= 0 or limit >= len(rows):
        return rows
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return rows[:limit]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Writing parquet requires pyarrow. Use --format jsonl for local smoke tests.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def _write_split(output_dir: Path, name: str, rows: list[dict[str, Any]], output_format: str) -> None:
    if output_format in {"jsonl", "both"}:
        _write_jsonl(output_dir / f"{name}.jsonl", rows)
    if output_format in {"parquet", "both"}:
        _write_parquet(output_dir / f"{name}.parquet", rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare GSM8K in verl parquet/jsonl format.")
    parser.add_argument("--output-dir", default="data/gsm8k_verl")
    parser.add_argument("--format", choices=["parquet", "jsonl", "both"], default="parquet")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="Convenience limit for both train and test.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-data", action="store_true", help="Use tiny built-in data; no network needed.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    train_limit = args.limit or args.train_limit
    test_limit = args.limit or args.test_limit

    if args.sample_data:
        train_raw = SAMPLE_ROWS
        test_raw = SAMPLE_ROWS[: max(1, min(4, len(SAMPLE_ROWS)))]
    else:
        train_raw = _load_gsm8k("train")
        test_raw = _load_gsm8k("test")

    train_raw = _sample_rows(train_raw, train_limit, args.seed)
    test_raw = _sample_rows(test_raw, test_limit, args.seed)

    train_rows = [_to_verl_row(row, "train", index) for index, row in enumerate(train_raw)]
    test_rows = [_to_verl_row(row, "test", index) for index, row in enumerate(test_raw)]

    _write_split(output_dir, "train", train_rows, args.format)
    _write_split(output_dir, "test", test_rows, args.format)
    _write_jsonl(
        output_dir / "manifest.jsonl",
        [
            {
                "data_source": DATA_SOURCE,
                "ability": ABILITY,
                "format": args.format,
                "sample_data": args.sample_data,
                "train_rows": len(train_rows),
                "test_rows": len(test_rows),
            }
        ],
    )

    print(f"Prepared GSM8K verl data: train={len(train_rows)} test={len(test_rows)} output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

