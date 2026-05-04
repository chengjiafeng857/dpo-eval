# Open LLM Leaderboard v1 — lm-eval-harness wrapper

This module evaluates SFT / DPO / β-DPO / our-method checkpoints on the
six-task Open LLM Leaderboard v1 suite using EleutherAI
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness):

| Task            | Shots | Headline metric              |
|-----------------|-------|------------------------------|
| MMLU            | 5     | acc                          |
| ARC-Challenge   | 25    | acc_norm                     |
| HellaSwag       | 10    | acc_norm                     |
| TruthfulQA      | 0     | mc2                          |
| WinoGrande      | 5     | acc                          |
| GSM8K           | 5     | exact_match                  |

The wrapper prefers the harness's built-in `openllm` task group; if that group
is missing it transparently falls back to the local
[`tasks/openllm_v1.yaml`](tasks/openllm_v1.yaml), which encodes the exact same
shot configuration.

## Install

`lm-eval` is declared in `pyproject.toml`, and `vllm` is included on Linux.
Sync the project environment on the machine that will run evaluation:

```bash
uv sync
```

The shell wrappers prefer `uv run python` automatically when `uv` is available,
so direct calls such as `bash scripts/eval_openllm_qwen_family.sh` use the
project environment instead of the system Python. You can still override the
interpreter explicitly with `PYTHON_BIN=/path/to/python`.

## Usage

### Single checkpoint, vLLM backend (default)

```bash
bash scripts/eval_openllm.sh \
  --model_path /path/to/checkpoint \
  --output_dir results/eval/my_model \
  --tensor_parallel_size 4 \
  --batch_size auto \
  --seed 1234 \
  --log_samples
```

Equivalent to:

```
lm_eval \
  --model vllm \
  --model_args pretrained=/path/to/checkpoint,tensor_parallel_size=4,dtype=auto,gpu_memory_utilization=0.85,max_model_len=4096,trust_remote_code=True \
  --tasks openllm \
  --batch_size auto \
  --output_path results/eval/my_model \
  --log_samples \
  --seed 1234
```

### Single checkpoint, HF backend

Pass `--backend hf` when you don't have vLLM available or want a deterministic
HF-only baseline:

```bash
bash scripts/eval_openllm.sh \
  --model_path /path/to/checkpoint \
  --output_dir results/eval/my_model_hf \
  --backend hf \
  --batch_size auto \
  --seed 1234 \
  --log_samples
```

Runs:

```
lm_eval \
  --model hf \
  --model_args pretrained=/path/to/checkpoint,dtype=bfloat16,trust_remote_code=True \
  --tasks openllm \
  --device cuda:0 \
  --batch_size auto \
  --output_path results/eval/my_model_hf \
  --log_samples \
  --seed 1234
```

### Batch over multiple checkpoints

`scripts/eval_all_checkpoints.sh` evaluates a labeled set of checkpoints with
identical settings so the per-model summaries are directly comparable.

```bash
CHECKPOINTS="sft=/ckpt/sft dpo=/ckpt/dpo beta_dpo=/ckpt/beta-dpo our_method=/ckpt/ours" \
OUTPUT_ROOT=results/eval/openllm_v1 \
  bash scripts/eval_all_checkpoints.sh \
    --tensor_parallel_size 4 \
    --batch_size auto --seed 1234 --log_samples
```

Override the default checkpoint list either by setting `CHECKPOINTS` (a
space-separated list of `label=path` pairs) or by editing
`DEFAULT_CHECKPOINTS` in the script.

## Evaluation protocol — base LM by default

By default we evaluate **base-LM style** (no chat template). To opt into
chat-style evaluation, pass:

```
--apply_chat_template          # render prompts through the model's chat template
--fewshot_as_multiturn         # render in-context examples as multi-turn messages
```

Both flags are **off by default** and recorded in `summary.json` /
`run_metadata.json` whenever they are set. We never silently apply a chat
template — pick one mode per comparison table and stick with it across
checkpoints.

## Output layout

```
<output_dir>/
├── run_metadata.json           # model_path, backend, dtype, batch_size, seed,
│                                #   chat-template flags, lm-eval version, command
├── results_<timestamp>.json    # raw lm-eval result blob
├── samples_<task>_<ts>.jsonl   # per-item samples when --log_samples is on
├── summary.json                # parsed six-metric summary + average
├── summary.csv                 # same, flat CSV
└── summary.md                  # same, markdown table for pasting into PRs
```

`summary.md` looks like:

```
| Task          | Shots | Metric      | Score  |
|---------------|-------|-------------|--------|
| MMLU          | 5     | acc         | 0.6312 |
| ARC-Challenge | 25    | acc_norm    | 0.5870 |
| HellaSwag     | 10    | acc_norm    | 0.8124 |
| TruthfulQA    | 0     | mc2         | 0.4905 |
| WinoGrande    | 5     | acc         | 0.7530 |
| GSM8K         | 5     | exact_match | 0.4640 |
| **Average**   |       |             | **0.6230** |
```

## Reproducibility

- Same backend across all models in a comparison table — pick `hf` or `vllm`
  once, not both.
- Same `--seed` (default `1234`) and `--batch_size` across runs.
- `run_metadata.json` captures the lm-eval version, the exact command, and
  the chat-template flag state.
- The runner prints the final command before executing it.
- `--log_samples` is recommended for any reportable run so per-item outputs
  are auditable.

## Troubleshooting

- **Built-in `openllm` group not found.** The runner prints a notice and uses
  the local fallback automatically. Use `--force_fallback` to skip the probe.
- **TruthfulQA metric naming.** The aggregator looks for `mc2` first, then
  falls back to `acc` on any `truthfulqa*` entry — both shapes are handled.
- **MMLU group vs subjects.** If only the 57 subject leaves are present, the
  aggregator averages them and reports that as `mmlu_acc`.
- **`No module named lm_eval`.** Run `uv sync` on the evaluation machine, or set
  `PYTHON_BIN` to an environment where `lm-eval` is installed. The shell wrapper
  automatically prefers `uv run python` when `uv` is available.
