#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIGS=(
  "gpt_judge_HH/config/helpful_base/multi-turn/prompts-helpful/gpt-4/config_eval_HH_llama3-hh-helpful-qt045-b0p8-20260429-085449.yaml"
  "gpt_judge_HH/config/helpful_base/multi-turn/prompts-helpful/gpt-4/config_eval_HH_llama3-hh-helpful-qt045-b0p5-20260429-085449.yaml"
  "gpt_judge_HH/config/helpful_base/multi-turn/prompts-helpful/gpt-4/config_eval_HH_llama3-hh-helpful-qt045-b0p3-20260429-085449.yaml"
  "gpt_judge_HH/config/helpful_base/multi-turn/prompts-helpful/gpt-4/config_eval_HH_llama3-hh-helpful-qt045-b0p05-20260429-085449.yaml"
  "gpt_judge_HH/config/helpful_base/multi-turn/prompts-helpful/gpt-4/config_eval_HH_llama3-hh-helpful-qt045-b0p01-20260429-085449.yaml"
  "gpt_judge_HH/config/harmless_base/multi-turn/prompts-harmless/gpt-4/config_eval_HH_llama3-hh-harmless-qt045-b0p8-20260429-085449.yaml"
  "gpt_judge_HH/config/harmless_base/multi-turn/prompts-harmless/gpt-4/config_eval_HH_llama3-hh-harmless-qt045-b0p5-20260429-085449.yaml"
  "gpt_judge_HH/config/harmless_base/multi-turn/prompts-harmless/gpt-4/config_eval_HH_llama3-hh-harmless-qt045-b0p3-20260429-085449.yaml"
  "gpt_judge_HH/config/harmless_base/multi-turn/prompts-harmless/gpt-4/config_eval_HH_llama3-hh-harmless-qt045-b0p05-20260429-085449.yaml"
  "gpt_judge_HH/config/harmless_base/multi-turn/prompts-harmless/gpt-4/config_eval_HH_llama3-hh-harmless-qt045-b0p01-20260429-085449.yaml"
)

for config in "${CONFIGS[@]}"; do
  echo "[HH-EVAL] Running inference for ${config}"
  uv run hh-generate --config "$config"
done
