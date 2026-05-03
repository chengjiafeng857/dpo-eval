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
  # Pick the q_t=0.45 / s_star=0.45 new-dpo run as the canonical "our_method"
  # representative; swap to whichever sweep point you're publishing.
  CHECKPOINTS="\
sft=jackf857/qwen3-8b-base-sft-ultrachat-4xh200-batch-128 \
beta_dpo=W-61/ultrafeedback-qwen3-8b-beta-dpo \
epsilon_dpo=jackf857/Qwen3-8b-ultrafeedback-binarized-e-dpo \
margin_dpo=W-61/ultrafeedback-qwen3-8b-margin-dpo \
our_method=jackf857/qwen3-8b-base-new-dpo-ultrafeedback-4xh200-batch-128-q_t-0.45-s_star-0.4"
fi

DEFAULTS=(
  --tensor_parallel_size 4
  --batch_size auto
  --seed 1234
  --log_samples
)

CHECKPOINTS="$CHECKPOINTS" OUTPUT_ROOT="$OUTPUT_ROOT" \
  bash scripts/eval_all_checkpoints.sh "${DEFAULTS[@]}" "$@"
