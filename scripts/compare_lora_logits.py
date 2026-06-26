from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_gsm8k_local import _load_rows, _prompt_text


def _load_base_and_adapter(model_path: str, adapter_path: str) -> tuple[Any, Any, str]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    adapter_model = PeftModel.from_pretrained(base_model, adapter_path)
    adapter_model.eval()
    return adapter_model, tokenizer, device


def _last_token_logits(model: Any, tokenizer: Any, messages: list[dict[str, str]]) -> tuple[Any, int]:
    import torch

    prompt = _prompt_text(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1].detach().float().cpu()
    return logits, int(inputs["input_ids"].shape[-1])


def _top_tokens(tokenizer: Any, logits: Any, k: int) -> list[dict[str, Any]]:
    import torch

    values, ids = torch.topk(logits, k)
    return [
        {
            "token_id": int(token_id),
            "token": tokenizer.decode([int(token_id)]),
            "logit": float(value),
        }
        for value, token_id in zip(values, ids)
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare base and LoRA-adapted logits on GSM8K prompts.")
    parser.add_argument("--model-path", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--data", default="data/gsm8k_verl/test.parquet")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = _load_rows(Path(args.data), limit=args.limit)
    model, tokenizer, device = _load_base_and_adapter(args.model_path, args.adapter_path)

    results: list[dict[str, Any]] = []
    summary = {
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "data": args.data,
        "device": device,
        "total": len(rows),
        "max_abs_diff": 0.0,
        "mean_abs_diff": 0.0,
        "same_argmax": 0,
    }

    diff_sum = 0.0
    for row in tqdm(rows, desc="compare"):
        messages = row["prompt"]

        with model.disable_adapter():
            base_logits, prompt_tokens = _last_token_logits(model, tokenizer, messages)
        adapter_logits, _ = _last_token_logits(model, tokenizer, messages)
        diff = adapter_logits - base_logits

        max_abs_diff = float(diff.abs().max())
        mean_abs_diff = float(diff.abs().mean())
        base_argmax = int(base_logits.argmax())
        adapter_argmax = int(adapter_logits.argmax())
        same_argmax = base_argmax == adapter_argmax

        summary["max_abs_diff"] = max(float(summary["max_abs_diff"]), max_abs_diff)
        diff_sum += mean_abs_diff
        summary["same_argmax"] += int(same_argmax)

        results.append(
            {
                "question": row.get("extra_info", {}).get("question", ""),
                "prompt_tokens": prompt_tokens,
                "max_abs_diff": max_abs_diff,
                "mean_abs_diff": mean_abs_diff,
                "same_argmax": same_argmax,
                "base_argmax": {
                    "token_id": base_argmax,
                    "token": tokenizer.decode([base_argmax]),
                    "logit": float(base_logits[base_argmax]),
                },
                "adapter_argmax": {
                    "token_id": adapter_argmax,
                    "token": tokenizer.decode([adapter_argmax]),
                    "logit": float(adapter_logits[adapter_argmax]),
                },
                "base_top": _top_tokens(tokenizer, base_logits, args.top_k),
                "adapter_top": _top_tokens(tokenizer, adapter_logits, args.top_k),
            }
        )

    total = max(1, len(rows))
    summary["mean_abs_diff"] = diff_sum / total
    summary["same_argmax_rate"] = summary["same_argmax"] / total
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
