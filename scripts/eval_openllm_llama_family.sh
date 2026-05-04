#!/usr/bin/env bash
# Open LLM Leaderboard v1 evaluation for the Llama-3 8B AlpacaEval family.
#
# Mirrors the canonical comparison set used in alpacaeval/config_alpacaeval_batch.yaml
# and config_alpacaeval_batch_8xh200.yaml: a UltraChat-SFT base, the major DPO
# baselines (vanilla, beta, epsilon, margin), and the "new-dpo" (q_t / s_star)
# checkpoint that is this repo's proposed method.
#
# Override individual paths or add/remove rows by setting CHECKPOINTS to the
# usual `label=hf_id ...` list.
#
# Usage:
#   bash scripts/eval_openllm_llama_family.sh \
#     [--tensor_parallel_size 4] [--batch_size auto] [--seed 1234] [--log_samples]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${OUTPUT_ROOT:=results/eval/openllm_v1/llama3_8b}"

if [[ -z "${CHECKPOINTS:-}" ]]; then
  CHECKPOINTS="\
sft=W-61/llama-3-8b-base-sft-ultrachat-8xh200 \
margin_dpo=W-61/llama-3-8b-base-margin-dpo-ultrafeedback-8xh200 \
beta_dpo=W-61/llama-3-8b-base-beta-dpo-ultrafeedback-4xh200-batch-128-20260424-044124 \
epsilon_dpo=W-61/llama-3-8b-base-epsilon-dpo-ultrafeedback-8xh200 \
ipo=jackf857/llama-3-8b-base-ipo-ultrafeedback-4xh200-batch-128-rerun \
cpo=jackf857/llama-3-8b-base-cpo-ultrafeedback-4xH200-batch-128-rerun \
kto=jackf857/llama-3-8b-base-kto-ultrafeedback-4xh200-batch-128-20260427-194056 \
orpo=jackf857/llama-3-8b-base-orpo-ultrafeedback-4xh200-rerun \
slic_hf=jackf857/llama-3-8b-base-slic-hf-ultrafeedback-4xh200-batch-128-20260428-054623 \
r_dpo=jackf857/llama-3-8b-base-r-dpo-ultrafeedback-4xH200-batch-128-rerun-2-runpod \
simpo=jackf857/llama-3-8b-base-simpo-8xh200 \
our_method=W-61/llama-3-8b-base-new-dpo-ultrafeedback-4xh200-batch-128-s_star-0.4-20260425-111846"
fi

# Default eval flags. Explicit caller flags remain authoritative.
has_arg() {
  local name="$1"
  shift
  local arg
  for arg in "$@"; do
    if [[ "$arg" == "$name" || "$arg" == "$name="* ]]; then
      return 0
    fi
  done
  return 1
}

DEFAULTS=()
if ! has_arg --tensor_parallel_size "$@" && ! has_arg --num_gpus "$@"; then
  DEFAULTS+=( --tensor_parallel_size 4 )
fi
if ! has_arg --batch_size "$@"; then
  DEFAULTS+=( --batch_size auto )
fi
if ! has_arg --seed "$@"; then
  DEFAULTS+=( --seed 1234 )
fi
if ! has_arg --log_samples "$@"; then
  DEFAULTS+=( --log_samples )
fi

CHECKPOINTS="$CHECKPOINTS" OUTPUT_ROOT="$OUTPUT_ROOT" \
  bash scripts/eval_all_checkpoints.sh "${DEFAULTS[@]}" "$@"
