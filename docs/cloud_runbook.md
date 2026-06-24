# Cloud Runbook

Use this project as a code-only repo. Keep datasets, models, checkpoints, and
logs on the cloud machine.

## Recommended Cloud Layout

```text
~/projects/grpo-qwen/
  repo/       # git clone of this project
  data/       # GSM8K parquet
  outputs/    # verl checkpoints and eval logs
```

## 1. Install Environment

The most stable route is to use a verl Docker image or a cloud image that already
has CUDA, PyTorch, vLLM, and verl installed.

If you are starting from a fresh Python environment:

```bash
cd ~/projects/grpo-qwen/repo
bash cloud/setup_cloud_env.sh
```

If cloning `verl` from GitHub fails with `RPC failed`, `early EOF`, or
`invalid index-pack output`, re-run the setup command. The setup script uses a
shallow clone and retries by default. You can also increase retries:

```bash
VERL_CLONE_RETRIES=5 bash cloud/setup_cloud_env.sh
```

Notes:

- Use a recent CUDA/PyTorch/vLLM stack supported by your installed verl version.
- If package versions fight each other, prefer the official verl Docker image.
- Keep Hugging Face dataset and model caches on the cloud disk.

## 2. Prepare GSM8K

Toy data first:

```bash
cd ~/projects/grpo-qwen/repo
TRAIN_LIMIT=128 TEST_LIMIT=64 bash cloud/prepare_gsm8k.sh
```

On cloud providers with slow or blocked Hugging Face access, try a mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
TRAIN_LIMIT=128 TEST_LIMIT=64 bash cloud/prepare_gsm8k.sh
```

If the dataset still cannot be reached, generate the built-in sample data just
to smoke-test the training pipeline:

```bash
SAMPLE_DATA=1 bash cloud/prepare_gsm8k.sh
```

Full data:

```bash
cd ~/projects/grpo-qwen/repo
bash cloud/prepare_gsm8k.sh
```

## 3. Baseline Eval

```bash
cd ~/projects/grpo-qwen/repo
LIMIT=100 bash cloud/eval_qwen15b_gsm8k.sh
```

Save the printed `accuracy` and `format_rate` before training.

## 4. Toy GRPO Run

Start with conservative settings:

```bash
cd ~/projects/grpo-qwen/repo
TRAIN_BATCH_SIZE=32 \
VAL_BATCH_SIZE=32 \
ROLLOUT_N=4 \
MAX_RESPONSE_LENGTH=512 \
TOTAL_EPOCHS=1 \
SAVE_FREQ=20 \
TEST_FREQ=20 \
EXPERIMENT_NAME=qwen15b-gsm8k-grpo-toy \
bash cloud/train_grpo_gsm8k_qwen15b.sh
```

If it OOMs, reduce in this order:

```text
MAX_RESPONSE_LENGTH -> 256
TRAIN_BATCH_SIZE -> 16
ROLLOUT_N -> 2
ROLLOUT_GPU_MEMORY_UTILIZATION -> 0.35
```

If training fails with `ModuleNotFoundError: No module named 'transfer_queue'`,
use the legacy trainer path:

```bash
TRAINER_USE_V1=false bash cloud/train_grpo_gsm8k_qwen15b.sh
```

This is the script default because it is more stable across fresh `verl`
checkouts.

If model initialization fails because `flash_attn` is not installed, use PyTorch
SDPA attention:

```bash
ATTN_IMPLEMENTATION=sdpa bash cloud/train_grpo_gsm8k_qwen15b.sh
```

If training later fails in padding utilities with the same missing `flash_attn`
module, disable remove-padding:

```bash
USE_REMOVE_PADDING=False bash cloud/train_grpo_gsm8k_qwen15b.sh
```

Both settings are script defaults. Install FlashAttention later only if you want
the extra speed and your PyTorch/CUDA stack has matching wheels.

## 5. Larger Run

After toy training works:

```bash
cd ~/projects/grpo-qwen/repo
TRAIN_BATCH_SIZE=64 \
VAL_BATCH_SIZE=64 \
ROLLOUT_N=4 \
MAX_RESPONSE_LENGTH=1024 \
TOTAL_EPOCHS=3 \
SAVE_FREQ=100 \
TEST_FREQ=100 \
EXPERIMENT_NAME=qwen15b-gsm8k-grpo-run001 \
bash cloud/train_grpo_gsm8k_qwen15b.sh
```

## 6. What To Send Back When Debugging

Do not send checkpoints. Send:

- the exact command
- the last 100-200 log lines
- GPU type and memory
- `nvidia-smi` snapshot
- changed environment variables
- eval summary JSON

## References

- verl installation: https://verl.readthedocs.io/en/latest/start/install.html
- verl data preparation: https://verl.readthedocs.io/en/latest/preparation/prepare_data.html
- verl reward functions: https://verl.readthedocs.io/en/latest/preparation/reward_function.html
- verl LoRA support: https://verl.readthedocs.io/en/latest/advance/ppo_lora.html
