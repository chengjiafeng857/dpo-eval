#!/usr/bin/env bash
# Open LLM Leaderboard v1 evaluation wrapper.
#
# Forwards arguments to openllm/run_openllm_eval.py. The Python runner picks
# the built-in `openllm` task group when the installed lm-eval-harness ships
# it, otherwise falls back to openllm/tasks/openllm_v1.yaml with the same shots.
#
# By default this evaluates in BASE-LM style. Pass --apply_chat_template (and
# optionally --fewshot_as_multiturn) only when you intentionally want chat-style
# evaluation — those flags are recorded in the result metadata.

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/eval_openllm.sh [options]

Required:
  --model_path PATH                 HF model id or local checkpoint directory.
  --output_dir PATH                 Where lm-eval writes results and we write summary.{json,csv,md}.

Backend:
  --backend {hf,vllm}               Default: vllm
  --batch_size SIZE                 Default: auto
  --dtype DTYPE                     Default: auto (vllm) / bfloat16 (hf)
  --device DEV                      HF device, default: cuda:0
  --num_gpus N                      Convenience alias for --tensor_parallel_size
  --tensor_parallel_size N          vLLM tensor parallel size
  --gpu_memory_utilization F        vLLM, default: 0.85
  --max_model_len N                 vLLM, default: 4096

Evaluation protocol:
  --apply_chat_template             Off by default; turn on only for chat-style eval.
  --fewshot_as_multiturn            Off by default.
  --seed N                          Default: 1234
  --log_samples                     Save per-item generations.
  --limit N                         Smoke test with N items per task.
  --force_fallback                  Use the local openllm_v1.yaml group instead of probing.
  --dry_run                         Print the lm-eval command and exit.
  --skip_aggregate                  Skip summary.{json,csv,md} generation.
  --extra_args "..."                Raw flags forwarded to lm-eval (shell-split).

Examples:
  scripts/eval_openllm.sh \
    --model_path /scratch/$USER/checkpoints/our-method \
    --output_dir results/eval/our-method \
    --backend hf --batch_size auto --seed 1234 --log_samples

  scripts/eval_openllm.sh \
    --model_path /scratch/$USER/checkpoints/our-method \
    --output_dir results/eval/our-method-vllm \
    --backend vllm --tensor_parallel_size 4 \
    --batch_size auto --seed 1234 --log_samples
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Allow the caller to override which Python is used. When not overridden,
# prefer `uv run --group openllm python` so direct `bash scripts/...`
# invocations still use the OpenLLM project environment instead of whatever
# system Python happens to be active.
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD=( "$PYTHON_BIN" )
elif command -v uv >/dev/null 2>&1; then
  PYTHON_CMD=( uv run --group openllm python )
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON_CMD=( "$REPO_ROOT/.venv/bin/python" )
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD=( python3 )
else
  PYTHON_CMD=( python )
fi

echo "[eval_openllm] python: ${PYTHON_CMD[*]}"

# Forward every argument to the Python runner. We use `python -m` so the
# `openllm` package import inside aggregate_results works regardless of CWD.
exec "${PYTHON_CMD[@]}" -m openllm.run_openllm_eval "$@"
