#!/bin/bash
#SBATCH --job-name=qwen-hh-infer
#SBATCH --output=qwen-hh-infer-%j.out
#SBATCH --error=qwen-hh-infer-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=24:00:00

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/home/qu.yang1/dpo-test/dpo-eval}"
cd "$ROOT_DIR"

export SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/$USER/dynamic-dpo-v4}"
export BASE_MODEL_DIR="${BASE_MODEL_DIR:-$SCRATCH_ROOT/base_models/}"
mkdir -p "$SCRATCH_ROOT"/{hf,tmp,wandb,xdg,outputs,base_models}

export HF_HOME="${HF_HOME:-$SCRATCH_ROOT/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$SCRATCH_ROOT/hf/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$SCRATCH_ROOT/hf/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$SCRATCH_ROOT/hf/transformers}"
export TMPDIR="${TMPDIR:-$SCRATCH_ROOT/tmp}"
export WANDB_DIR="${WANDB_DIR:-$SCRATCH_ROOT/wandb}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCH_ROOT/xdg}"
export PYTHONUNBUFFERED=1

echo "Running on host: $(hostname)"
echo "Working directory: $ROOT_DIR"
echo "Scratch root: $SCRATCH_ROOT"
echo "Start time: $(date)"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. Private Hugging Face model downloads may fail." >&2
fi


configs=(
  "gpt_judge_HH/config/harmless_base/single-turn/config_eval_HH_qwen3-8b-base-beta-dpo-hh-harmless-4xh200-batch-64-20260418-012645.yaml"
  "gpt_judge_HH/config/harmless_base/multi-turn/config_eval_HH_qwen3-8b-base-beta-dpo-hh-harmless-4xh200-batch-64-20260418-012645.yaml"
  "gpt_judge_HH/config/harmless_base/single-turn/config_eval_HH_qwen3-8b-base-margin-dpo-hh-harmless-4xh200-batch-64-20260418-012645.yaml"
  "gpt_judge_HH/config/harmless_base/multi-turn/config_eval_HH_qwen3-8b-base-margin-dpo-hh-harmless-4xh200-batch-64-20260418-012645.yaml"
  "gpt_judge_HH/config/harmless_base/single-turn/config_eval_HH_qwen3-8b-base-sft-hh-harmless-4xh200-batch-64-20260417-214452.yaml"
  "gpt_judge_HH/config/harmless_base/multi-turn/config_eval_HH_qwen3-8b-base-sft-hh-harmless-4xh200-batch-64-20260417-214452.yaml"
  "gpt_judge_HH/config/helpful_base/single-turn/config_eval_HH_qwen3-8b-base-beta-dpo-hh-helpful-4xh200-batch-64-20260418-012645.yaml"
  "gpt_judge_HH/config/helpful_base/multi-turn/config_eval_HH_qwen3-8b-base-beta-dpo-hh-helpful-4xh200-batch-64-20260418-012645.yaml"
  "gpt_judge_HH/config/helpful_base/single-turn/config_eval_HH_qwen3-8b-base-margin-dpo-hh-helpful-4xh200-batch-64-20260417-214452.yaml"
  "gpt_judge_HH/config/helpful_base/multi-turn/config_eval_HH_qwen3-8b-base-margin-dpo-hh-helpful-4xh200-batch-64-20260417-214452.yaml"
  "gpt_judge_HH/config/helpful_base/single-turn/config_eval_HH_qwen3-8b-base-margin-dpo-hh-helpful-4xh200-batch-64-20260418-012645.yaml"
  "gpt_judge_HH/config/helpful_base/multi-turn/config_eval_HH_qwen3-8b-base-margin-dpo-hh-helpful-4xh200-batch-64-20260418-012645.yaml"
  "gpt_judge_HH/config/helpful_base/single-turn/config_eval_HH_qwen3-8b-base-sft-hh-helpful-4xh200-batch-64-20260417-214452.yaml"
  "gpt_judge_HH/config/helpful_base/multi-turn/config_eval_HH_qwen3-8b-base-sft-hh-helpful-4xh200-batch-64-20260417-214452.yaml"
)

for cfg in "${configs[@]}"; do
  echo
  echo "[$(date)] Running inference for $cfg"
  srun --ntasks=1 uv run --no-sync python gpt_judge_HH/generate_hh_output.py --config "$cfg"
done

echo "End time: $(date)"
