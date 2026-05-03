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
dpo=princeton-nlp/Llama-3-Base-8B-SFT-DPO \
beta_dpo=W-61/llama-3-8b-base-beta-dpo-ultrafeedback-8xh200 \
epsilon_dpo=W-61/llama-3-8b-base-epsilon-dpo-ultrafeedback-8xh200 \
margin_dpo=W-61/llama-3-8b-base-margin-dpo-ultrafeedback-8xh200 \
our_method=W-61/llama-3-8b-base-new-dpo-ultrafeedback-4xh200-batch-128-q_t-0.45-s_star-0.45-20260427-221551"
fi

# Default eval flags. All flags after the script name override / extend these.
DEFAULTS=(
  --tensor_parallel_size 4
  --batch_size auto
  --seed 1234
  --log_samples
)

CHECKPOINTS="$CHECKPOINTS" OUTPUT_ROOT="$OUTPUT_ROOT" \
  bash scripts/eval_all_checkpoints.sh "${DEFAULTS[@]}" "$@"
