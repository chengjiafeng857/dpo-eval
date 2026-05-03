#!/usr/bin/env bash
# Run scripts/eval_openllm.sh over a labeled set of checkpoints (SFT, DPO,
# beta-DPO, our method, ...) using identical evaluation settings so the
# resulting summaries are directly comparable.
#
# Override the checkpoint set via the CHECKPOINTS env var:
#
#   CHECKPOINTS="sft=/path/to/sft dpo=/path/to/dpo beta_dpo=/path/to/beta our=/path/to/ours" \
#       scripts/eval_all_checkpoints.sh --backend vllm --tensor_parallel_size 4
#
# All flags after the script name are forwarded to scripts/eval_openllm.sh
# unchanged (so seed, dtype, batch_size, chat-template flags etc. are shared).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_ROOT="${OUTPUT_ROOT:-results/eval/openllm_v1}"
mkdir -p "$OUTPUT_ROOT"

# Default labeled set. Override with the CHECKPOINTS env var (space-separated
# label=path pairs). Update these defaults to match your actual checkpoint paths.
DEFAULT_CHECKPOINTS=(
  "sft=princeton-nlp/Llama-3-Base-8B-SFT"
  "dpo=princeton-nlp/Llama-3-Base-8B-SFT-DPO"
  "beta_dpo=PLACEHOLDER/beta-dpo-checkpoint"
  "our_method=PLACEHOLDER/our-method-checkpoint"
)

if [[ -n "${CHECKPOINTS:-}" ]]; then
  # shellcheck disable=SC2206
  PAIRS=( $CHECKPOINTS )
else
  PAIRS=( "${DEFAULT_CHECKPOINTS[@]}" )
fi

FORWARDED=( "$@" )

echo "[eval_all_checkpoints] output root: $OUTPUT_ROOT"
echo "[eval_all_checkpoints] forwarded args: ${FORWARDED[*]:-<none>}"
echo

for pair in "${PAIRS[@]}"; do
  label="${pair%%=*}"
  path="${pair#*=}"
  if [[ "$path" == PLACEHOLDER/* ]]; then
    echo "[eval_all_checkpoints] skipping '$label' — set CHECKPOINTS to override placeholder ($path)."
    continue
  fi
  out_dir="$OUTPUT_ROOT/$label"
  echo "=================================================================="
  echo "[eval_all_checkpoints] $label  ->  $path"
  echo "[eval_all_checkpoints] writing to $out_dir"
  echo "=================================================================="
  bash scripts/eval_openllm.sh \
    --model_path "$path" \
    --output_dir "$out_dir" \
    "${FORWARDED[@]}"
done

echo
echo "[eval_all_checkpoints] done. Per-checkpoint summaries:"
for pair in "${PAIRS[@]}"; do
  label="${pair%%=*}"
  path="${pair#*=}"
  if [[ "$path" == PLACEHOLDER/* ]]; then continue; fi
  echo "  $OUTPUT_ROOT/$label/summary.md"
done
