#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <shard_index> [num_shards]" >&2
  echo "Example: $0 1 4" >&2
  exit 1
fi

SHARD_INDEX="$1"
NUM_SHARDS="${2:-4}"

if ! [[ "$SHARD_INDEX" =~ ^[0-9]+$ ]] || ! [[ "$NUM_SHARDS" =~ ^[0-9]+$ ]]; then
  echo "shard_index and num_shards must be positive integers." >&2
  exit 1
fi

if (( SHARD_INDEX < 1 || SHARD_INDEX > NUM_SHARDS )); then
  echo "shard_index must be between 1 and num_shards." >&2
  exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set." >&2
  exit 1
fi

ROOT="${ROOT:-outputs/qwen-hh-hyper-sweep-harmless-all}"
SKIP_FILE="${SKIP_FILE:-}"

if [[ -z "$SKIP_FILE" && -f "$ROOT/skip_judging.txt" ]]; then
  SKIP_FILE="$ROOT/skip_judging.txt"
fi

HARMLESS_CFG="gpt_judge_HH/config/harmless_base/multi-turn/prompts-general-less-harmful/gpt-4/config_eval_HH_qwen3-8b-base-new-dpo-hh-harmless-4xh200-batch-64-q_t-0.45-s_star-0.4.yaml"
HARMLESS_CHOSEN="outputs10/gpt_judge_HH/harmless_base/multi_turn/chosen_output_hh.json"
HARMLESS_SUBDIR="prompts-general-less-harmful/gpt-4"

HELPFUL_CFG="gpt_judge_HH/config/helpful_base/multi-turn/prompts-helpful/gpt-4/config_eval_HH_qwen3-8b-base-new-dpo-hh-helpful-4xh200-batch-64-q_t-0.45-s_star-0.4.yaml"
HELPFUL_CHOSEN="outputs10/gpt_judge_HH/helpful_base/multi_turn/chosen_output_hh.json"
HELPFUL_SUBDIR="prompts-helpful/gpt-4"

ROLLOUT_DIRS=()
while IFS= read -r dir; do
  ROLLOUT_DIRS+=("$dir")
done < <(find "$ROOT" -mindepth 1 -maxdepth 1 -type d | sort)

SKIP_NAMES=()
if [[ -n "$SKIP_FILE" ]]; then
  if [[ ! -f "$SKIP_FILE" ]]; then
    echo "Skip file not found: $SKIP_FILE" >&2
    exit 1
  fi
  while IFS= read -r line; do
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    if [[ -z "$line" || "$line" == \#* ]]; then
      continue
    fi
    SKIP_NAMES+=("$(basename "$line")")
  done < "$SKIP_FILE"
fi

if (( ${#ROLLOUT_DIRS[@]} == 0 )); then
  echo "No rollout directories found under $ROOT" >&2
  exit 1
fi

is_skipped() {
  local name="$1"
  local skip_name
  for skip_name in "${SKIP_NAMES[@]}"; do
    if [[ "$skip_name" == "$name" ]]; then
      return 0
    fi
  done
  return 1
}

run_rollout() {
  local dir="$1"
  local name
  local cfg
  local chosen
  local out_subdir
  local rollout_json
  local results_file
  local summary_file

  name="$(basename "$dir")"
  rollout_json="$dir/$name.json"

  if is_skipped "$name"; then
    echo "[$(date)] skipping excluded rollout: $name"
    return 0
  fi

  if [[ ! -f "$rollout_json" ]]; then
    echo "[$(date)] missing rollout json for $name: $rollout_json" >&2
    return 1
  fi

  if [[ "$name" == *"-hh-helpful-"* ]]; then
    cfg="$HELPFUL_CFG"
    chosen="$HELPFUL_CHOSEN"
    out_subdir="$HELPFUL_SUBDIR"
  elif [[ "$name" == *"-hh-harmless-"* ]]; then
    cfg="$HARMLESS_CFG"
    chosen="$HARMLESS_CHOSEN"
    out_subdir="$HARMLESS_SUBDIR"
  else
    echo "[$(date)] skipping unrecognized rollout type: $name" >&2
    return 0
  fi

  results_file="$dir/$out_subdir/chosen_vs_${name}.jsonl"
  summary_file="$dir/$out_subdir/chosen_vs_${name}_summary.json"

  if [[ -f "$summary_file" ]]; then
    echo "[$(date)] already finished: $name"
    return 0
  fi

  echo "[$(date)] judging $name"
  uv run hh-judge \
    --config "$cfg" \
    --chosen "$chosen" \
    --dpo "$rollout_json" \
    --results_file "$results_file" \
    --summary_file "$summary_file" \
    --resume
}

for idx in "${!ROLLOUT_DIRS[@]}"; do
  shard=$(( (idx % NUM_SHARDS) + 1 ))
  if (( shard != SHARD_INDEX )); then
    continue
  fi
  run_rollout "${ROLLOUT_DIRS[$idx]}"
done
