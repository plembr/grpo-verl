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


def _load_tokenizer(model_path: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _batched(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate GSM8K verl data with vLLM and an optional LoRA adapter.")
    parser.add_argument("--model-path", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter-path", default="", help="PEFT LoRA adapter directory. Empty evaluates base model.")
    parser.add_argument("--data", default="data/gsm8k_verl/test.parquet")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-lora-rank", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=0)
    parser.add_argument("--output", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except ImportError as exc:
        raise RuntimeError("This script requires vLLM. Run it on the cloud training environment.") from exc

    rows = _load_rows(Path(args.data), limit=args.limit)
    tokenizer = _load_tokenizer(args.model_path)
    prompts = [_prompt_text(tokenizer, row["prompt"]) for row in rows]

    llm_kwargs: dict[str, Any] = {
        "model": args.model_path,
        "trust_remote_code": True,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enable_lora": bool(args.adapter_path),
    }
    if args.adapter_path:
        llm_kwargs["max_lora_rank"] = args.max_lora_rank
    if args.max_model_len > 0:
        llm_kwargs["max_model_len"] = args.max_model_len

    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
    )
    lora_request = LoRARequest("gsm8k_adapter", 1, args.adapter_path) if args.adapter_path else None

    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    correct = 0
    format_ok = 0

    for row_batch, prompt_batch in tqdm(
        zip(_batched(rows, args.batch_size), _batched(prompts, args.batch_size)),
        total=(len(rows) + args.batch_size - 1) // args.batch_size,
        desc="vllm-eval",
    ):
        outputs = llm.generate(prompt_batch, sampling_params, lora_request=lora_request)
        # vLLM preserves input order for list generation.
        for row, output in zip(row_batch, outputs):
            response = output.outputs[0].text.strip()
            ground_truth = row["reward_model"]["ground_truth"]
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
        "backend": "vllm",
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "data": args.data,
        "total": len(rows),
        "accuracy": correct / total,
        "format_rate": format_ok / total,
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "generation": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "dtype": args.dtype,
        },
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
