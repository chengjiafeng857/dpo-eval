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

echo "Running on host: $(hostname)"
echo "Working directory: $ROOT_DIR"
echo "Start time: $(date)"


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
