#!/usr/bin/env bash
# Open LLM Leaderboard v1 evaluation for the Qwen3 8B AlpacaEval family.
#
# Mirrors the canonical Qwen3 comparison set seen across the alpacaeval/
# configs: an UltraChat-SFT base, the beta / epsilon / margin DPO baselines,
# and the highest-scoring "new-dpo" (q_t / s_star) sweep point.
#
# Override individual paths or add/remove rows by setting CHECKPOINTS.
#
# Usage:
#   bash scripts/eval_openllm_qwen_family.sh \
#     [--tensor_parallel_size 4] [--batch_size auto] [--seed 1234] [--log_samples]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${OUTPUT_ROOT:=results/eval/openllm_v1/qwen3_8b}"

if [[ -z "${CHECKPOINTS:-}" ]]; then
  CHECKPOINTS="\
sft=jackf857/qwen3-8b-base-sft-ultrachat-4xh200-batch-128 \
margin_dpo=W-61/qwen3-8b-base-margin-dpo-ultrafeedback-4xh200-batch-128-20260423-040315 \
beta_dpo=W-61/qwen3-8b-base-beta-dpo-ultrafeedback-4xh200-batch-128-20260423-040315 \
epsilon_dpo=W-61/qwen3-8b-base-epsilon-dpo-ultrafeedback-4xh200-batch-128-20260422-131855 \
ipo=W-61/qwen3-8b-base-ipo-ultrafeedback-4xh200-batch-128-20260422-131855 \
cpo=W-61/qwen3-8b-base-cpo-ultrafeedback-4xh200-batch-128-20260422-131855 \
kto=jackf857/qwen3-8b-base-kto-ultrafeedback-4xH200-batch-128 \
orpo=jackf857/qwen3-8b-base-orpo-ultrafeedback-4xh200-batch-128 \
slic_hf=W-61/qwen3-8b-base-slic-hf-ultrafeedback-4xh200-batch-128-20260422-131855 \
r_dpo=jackf857/qwen-3-8b-base-r-dpo-ultrafeedback-4xH200-batch-128-rerun-2-runpod \
simpo=jackf857/qwen3-8b-base-simpo-ultrafeedback-4xH200-batch-128 \
our_method=W-61/qwen3-8b-base-new-dpo-ultrafeedback-4xh200-batch-128-q_t-0.43-s_star-0.4-20260429-230725"
fi

DEFAULTS=(
  --tensor_parallel_size 4
  --batch_size auto
  --seed 1234
  --log_samples
)

CHECKPOINTS="$CHECKPOINTS" OUTPUT_ROOT="$OUTPUT_ROOT" \
  bash scripts/eval_all_checkpoints.sh "${DEFAULTS[@]}" "$@"
