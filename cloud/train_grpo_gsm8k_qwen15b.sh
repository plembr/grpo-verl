#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
DATA_DIR="${DATA_DIR:-$HOME/projects/grpo-qwen/data/gsm8k_verl}"
TRAIN_FILES="${TRAIN_FILES:-$DATA_DIR/train.parquet}"
TEST_FILES="${TEST_FILES:-$DATA_DIR/test.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/projects/grpo-qwen/outputs}"
PROJECT_NAME="${PROJECT_NAME:-qwen15b-gsm8k-grpo}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen2_5_1_5b_grpo_lora_toy}"
REWARD_PATH="${REWARD_PATH:-$REPO_ROOT/grpo_verl/rewards/gsm8k_reward.py}"

N_GPUS="${N_GPUS:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
ROLLOUT_N="${ROLLOUT_N:-4}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-64}"
ACTOR_LR="${ACTOR_LR:-3e-5}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
SAVE_FREQ="${SAVE_FREQ:-20}"
TEST_FREQ="${TEST_FREQ:-20}"
USE_LORA="${USE_LORA:-1}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-32}"

LORA_ARGS=()
if [[ "$USE_LORA" == "1" ]]; then
  LORA_ARGS+=(
    actor_rollout_ref.model.lora_rank="$LORA_RANK"
    actor_rollout_ref.model.lora_alpha="$LORA_ALPHA"
    actor_rollout_ref.model.target_modules=all-linear
    actor_rollout_ref.rollout.load_format=safetensors
  )
fi

mkdir -p "$OUTPUT_DIR"
cd "$REPO_ROOT"

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$TRAIN_FILES" \
  data.val_files="$TEST_FILES" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.val_batch_size="$VAL_BATCH_SIZE" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_NUM_SEQS" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.ref.strategy=fsdp \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  custom_reward_function.path="$REWARD_PATH" \
  custom_reward_function.name=compute_score \
  trainer.logger="['console']" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.default_local_dir="$OUTPUT_DIR/$EXPERIMENT_NAME" \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes=1 \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  "${LORA_ARGS[@]}"

