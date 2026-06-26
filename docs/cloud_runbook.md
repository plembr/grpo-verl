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
module, install FlashAttention:

```bash
MAX_JOBS=8 python -m pip install flash-attn --no-build-isolation
```

The setup script can also do this when requested:

```bash
INSTALL_FLASH_ATTN=1 bash cloud/setup_cloud_env.sh
```

If compilation is slow or memory-heavy, lower `MAX_JOBS`, for example
`MAX_JOBS=4`. Make sure the installed PyTorch/CUDA stack is stable before
installing FlashAttention.

If training reaches `100%` and then prints `DataLoader worker ... is killed by
signal: Killed` during Python shutdown, reduce dataloader workers:

```bash
DATALOADER_NUM_WORKERS=0 bash cloud/train_grpo_gsm8k_qwen15b.sh
```

This is the script default for small toy runs.

If Ray reports `/tmp/ray` is over 95% full or checkpoint saving fails with
`PytorchStreamWriter failed writing file`, stop Ray, clean the old Ray session,
and put Ray temporary files on the larger output disk:

```bash
ray stop --force || true
rm -rf /tmp/ray
RAY_TMP_DIR=$HOME/projects/grpo-qwen/outputs/ray_tmp \
MAX_ACTOR_CKPT_TO_KEEP=1 \
bash cloud/train_grpo_gsm8k_qwen15b.sh
```

The training script uses these defaults, but the explicit command is useful
after a failed run has already filled `/tmp/ray`.

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

If baseline and adapter eval produce identical responses, compare logits:

```bash
python scripts/compare_lora_logits.py \
  --adapter-path /root/autodl-tmp/grpo-qwen/outputs/.../lora_adapter \
  --data /root/autodl-tmp/grpo-qwen/data/gsm8k_verl/test.parquet \
  --limit 8 \
  --output /root/autodl-tmp/grpo-qwen/outputs/logit_compare.jsonl
```

If `max_abs_diff` is nonzero but `same_argmax_rate` is high, the adapter is
loaded but the update is too small to change greedy decoding. If diffs are zero,
check adapter export/loading.

If LoRA A/B weights are nonzero but logits still do not change, inspect adapter
scaling. PEFT reports this on LoRA modules as `scaling`. A value of `0.0` means
the adapter delta is multiplied away. Repair the exported adapter config:

```bash
python scripts/fix_lora_adapter_config.py \
  --adapter-path /root/autodl-tmp/grpo-qwen/outputs/.../lora_adapter \
  --alpha 32
```

## References

- verl installation: https://verl.readthedocs.io/en/latest/start/install.html
- verl data preparation: https://verl.readthedocs.io/en/latest/preparation/prepare_data.html
- verl reward functions: https://verl.readthedocs.io/en/latest/preparation/reward_function.html
- verl LoRA support: https://verl.readthedocs.io/en/latest/advance/ppo_lora.html
