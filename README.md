# Qwen2.5-1.5B GSM8K GRPO with verl

This is a small, cloud-friendly project for learning practical GRPO training
with `verl`.

Goal:

```text
Qwen2.5-1.5B-Instruct + GSM8K + verl GRPO + rule reward
```

The local machine only keeps code. The cloud machine downloads the dataset,
model, and checkpoints.

## Project Layout

```text
grpo-verl/
  grpo_verl/
    rewards/gsm8k_reward.py       # verl-compatible rule reward
  scripts/
    prepare_gsm8k_verl.py         # download/convert GSM8K to verl parquet
    eval_gsm8k_local.py           # simple baseline/checkpoint eval
    compare_lora_logits.py        # diagnose whether a LoRA adapter changes logits
  cloud/
    setup_cloud_env.sh            # optional dependency setup
    prepare_gsm8k.sh              # cloud data preparation
    train_grpo_gsm8k_qwen15b.sh   # GRPO launch script
    eval_qwen15b_gsm8k.sh         # eval launch script
  tests/
    test_gsm8k_reward.py
```

## Cloud Quickstart

Clone this repo on the cloud machine, then:

```bash
cd ~/projects/grpo-qwen/repo
bash cloud/setup_cloud_env.sh
bash cloud/prepare_gsm8k.sh
bash cloud/train_grpo_gsm8k_qwen15b.sh
```

For the first run, use a small toy subset:

```bash
TRAIN_FILES=$HOME/projects/grpo-qwen/data/gsm8k_verl/train.parquet \
TEST_FILES=$HOME/projects/grpo-qwen/data/gsm8k_verl/test.parquet \
MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct \
MAX_RESPONSE_LENGTH=512 \
ROLLOUT_N=4 \
TOTAL_EPOCHS=1 \
SAVE_FREQ=20 \
TEST_FREQ=20 \
bash cloud/train_grpo_gsm8k_qwen15b.sh
```

## Data

The cloud script downloads `openai/gsm8k` with Hugging Face `datasets` and writes
verl-compatible parquet files:

```text
$HOME/projects/grpo-qwen/data/gsm8k_verl/train.parquet
$HOME/projects/grpo-qwen/data/gsm8k_verl/test.parquet
```

Each row has:

- `data_source`
- `prompt`
- `ability`
- `reward_model.style`
- `reward_model.ground_truth`
- `extra_info`

## Reward

verl reward entrypoint:

```text
grpo_verl.rewards.gsm8k_reward.compute_score
```

Compatible signature:

```python
compute_score(data_source, solution_str, ground_truth, extra_info)
```

Scoring:

- exact final answer match: `1.0`
- numeric answer present but wrong: `0.1`
- no parseable final answer: `0.0`

Preferred answer format:

```text
... reasoning ...
#### 42
```

## Local Smoke Tests

These tests do not require GPU, model, or dataset download:

```bash
python -m pytest tests
python scripts/prepare_gsm8k_verl.py --sample-data --output-dir /tmp/gsm8k_verl_smoke --format jsonl --limit 8
```

On Windows PowerShell:

```powershell
python -m pytest tests
python scripts\prepare_gsm8k_verl.py --sample-data --output-dir D:\grpo-verl\tmp_gsm8k_verl_smoke --format jsonl --limit 8
```

## Adapter Diagnostics

If a trained LoRA adapter produces identical greedy responses to the base model,
compare logits directly:

```bash
python scripts/compare_lora_logits.py \
  --adapter-path /path/to/lora_adapter \
  --data /path/to/test.parquet \
  --limit 8
```
