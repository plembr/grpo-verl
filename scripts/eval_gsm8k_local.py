from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grpo_verl.rewards.gsm8k_reward import score_response


def _load_rows(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
                if limit > 0 and len(rows) >= limit:
                    break
        return rows

    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Reading parquet requires pyarrow.") from exc
        rows = pq.read_table(path).to_pylist()
        return rows[:limit] if limit > 0 else rows

    raise ValueError(f"Unsupported eval file format: {path}")


def _prompt_text(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    parts = [f"{message.get('role', 'user')}:\n{message.get('content', '')}" for message in messages]
    parts.append("assistant:\n")
    return "\n".join(parts)


def _load_model(model_path: str, adapter_path: str = "") -> tuple[Any, Any, str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )

    if adapter_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("Loading a LoRA adapter requires peft.") from exc
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer, device


def _generate(model: Any, tokenizer: Any, messages: list[dict[str, str]], max_new_tokens: int) -> str:
    import torch

    prompt = _prompt_text(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a model on GSM8K-formatted verl data.")
    parser.add_argument("--model-path", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--data", default="data/gsm8k_verl/test.parquet")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = _load_rows(Path(args.data), limit=args.limit)
    model, tokenizer, device = _load_model(args.model_path, adapter_path=args.adapter_path)
    started = time.perf_counter()

    results: list[dict[str, Any]] = []
    correct = 0
    format_ok = 0
    for row in tqdm(rows, desc="eval"):
        messages = row["prompt"]
        ground_truth = row["reward_model"]["ground_truth"]
        response = _generate(model, tokenizer, messages, max_new_tokens=args.max_new_tokens)
        details = score_response(response, ground_truth)
        correct += int(bool(details["correct"]))
        format_ok += int(bool(details["has_final_marker"]))
        results.append(
            {
                "question": row.get("extra_info", {}).get("question", ""),
                "response": response,
                "ground_truth": ground_truth,
                "reward": details,
            }
        )

    total = max(1, len(rows))
    summary = {
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "data": args.data,
        "device": device,
        "total": len(rows),
        "accuracy": correct / total,
        "format_rate": format_ok / total,
        "elapsed_sec": round(time.perf_counter() - started, 3),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            for result in results:
                file.write(json.dumps(result, ensure_ascii=False) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
