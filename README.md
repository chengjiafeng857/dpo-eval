# Eval Pipeline

This repo keeps benchmark wrappers at the repo root:

- `alpacaeval/`: AlpacaEval local generation plus `alpaca-eval` scoring.
- `arenahard/`: Arena-Hard v2 wrapper around the official Arena-Hard-Auto
  runner.
- `gpt_judge_HH/`: Anthropic HH generation plus GPT-4o judging.
- `mtbench/`: MT-Bench wrapper around FastChat's original framework.

Installed entrypoints:

- `alpacaeval-infer`, `alpacaeval-eval`, `alpacaeval-batch`
- `arenahard-infer`, `arenahard-eval`, `arenahard-batch`
- `hh-generate`, `hh-judge`
- `mtbench-infer`, `mtbench-eval`, `mtbench-batch`

## Folder layout

- `alpacaeval/alpacaeval_infer.py`: single-model inference pipeline.
- `alpacaeval/alpacaeval_eval.py`: evaluation wrapper around
  `alpaca-eval`.
- `alpacaeval/batch_runner.py`: config-driven batch runner for multiple
  models.
- `alpacaeval/alpacaeval_common.py`: shared config, path, template, and
  JSON helpers.
- `alpacaeval/config_alpacaeval.yaml`: single-model example config.
- `alpacaeval/config_alpacaeval_batch.yaml`: batch config for the repo's
  UltraFeedback Qwen3 and Llama3 models.
- `alpacaeval/templates/`: custom prompt templates used when
  `use_custom_chat_template: true`.
- `alpacaeval/configs/`: checked-in AlpacaEval model-config YAMLs for the
  repo's Qwen3/Llama3 UltraFeedback and UltraChat checkpoints.
- `arenahard/`: Arena-Hard v2 configs, official-runner staging, inference,
  evaluation, and batch orchestration.
- `gpt_judge_HH/generate_hh_output.py`: HH generation and chosen-response
  export entrypoint.
- `gpt_judge_HH/judge_outputs_gpt4o.py`: GPT-4o pairwise or three-way judge.
- `gpt_judge_HH/data_utils.py`: HH transcript parsing, prompt extraction, and
  chosen-output export helpers.
- `gpt_judge_HH/config/`: checked-in helpful-base and harmless-base single-turn
  and multi-turn configs.
- `mtbench/`: MT-Bench configs, templates, inference, evaluation, and
  batch orchestration.
- `benchmark_common.py`: shared path, config, JSONL, and command helpers
  for benchmark wrappers.
- `model_generation.py`: shared tokenizer rendering and local generation
  helpers for `transformers` and `vllm`.

## Quick start

Install dependencies first:

```bash
uv sync
```

For judge-backed stages, export your OpenAI key:

```bash
export OPENAI_API_KEY=...
```

### AlpacaEval full pipeline

Run local generation first, then score the saved outputs with `alpaca-eval`:

```bash
uv run alpacaeval-infer --config alpacaeval/config_alpacaeval.yaml
uv run alpacaeval-eval --config alpacaeval/config_alpacaeval.yaml
```

Batch mode runs the same two stages for every model in the batch config:

```bash
uv run alpacaeval-batch --config alpacaeval/config_alpacaeval_batch.yaml
```

### Arena-Hard v2 full pipeline

Serve the model through an OpenAI-compatible endpoint first, then set
`arenahard.model_endpoint` in the config and run:

```bash
uv run arenahard-infer --config arenahard/config_arenahard.yaml
uv run arenahard-eval --config arenahard/config_arenahard.yaml
```

Batch mode runs both stages for every endpoint in the batch config:

```bash
uv run arenahard-batch --config arenahard/config_arenahard_batch.yaml
```

### HH GPT-judge full pipeline

Pick one HH config first. Common examples:

- `gpt_judge_HH/config/harmless_base/single-turn/config_eval_HH.yaml`
- `gpt_judge_HH/config/harmless_base/multi-turn/config_eval_HH_multiturn.yaml`
- `gpt_judge_HH/config/helpful_base/single-turn/config_eval_HH.yaml`
- `gpt_judge_HH/config/helpful_base/multi-turn/config_eval_HH_multiturn.yaml`

Then run generation once per model output you want in the comparison, export
the HH chosen baseline, and finally judge:

```bash
uv run hh-generate --config gpt_judge_HH/config/harmless_base/single-turn/config_eval_HH.yaml --model_key sft --output_key sft
uv run hh-generate --config gpt_judge_HH/config/harmless_base/single-turn/config_eval_HH.yaml --model_key beta_dpo --output_key beta_dpo
uv run hh-generate --config gpt_judge_HH/config/harmless_base/single-turn/config_eval_HH.yaml --model_key margin_dpo --output_key margin_dpo
uv run hh-generate --config gpt_judge_HH/config/harmless_base/single-turn/config_eval_HH.yaml --model_key e_dpo --output_key e_dpo
uv run hh-generate --config gpt_judge_HH/config/harmless_base/single-turn/config_eval_HH.yaml --output_key chosen --extract_chosen
uv run hh-judge --config gpt_judge_HH/config/harmless_base/single-turn/config_eval_HH.yaml --resume
```

The same command pattern works for helpful-base and for multi-turn. Swap only
the config path.

## AlpacaEval

### Full pipeline

The normal AlpacaEval workflow is two explicit stages:

1. `alpacaeval-infer` loads the AlpacaEval dataset, renders prompts, runs local
   generation with either `transformers` or `vllm`, and writes
   `model_outputs.json` plus `metadata.json`.
2. `alpacaeval-eval` reads `model_outputs.json`, calls `alpaca-eval`, and
   writes a `results/` directory under the configured output directory.

The simplest end-to-end run is:

```bash
uv run alpacaeval-infer --config alpacaeval/config_alpacaeval.yaml
uv run alpacaeval-eval --config alpacaeval/config_alpacaeval.yaml
```

### Separate stages

Inference only:

```bash
uv run alpacaeval-infer --config alpacaeval/config_alpacaeval.yaml
```

Evaluation only, using the outputs written by the earlier inference step:

```bash
uv run alpacaeval-eval --config alpacaeval/config_alpacaeval.yaml
```

Evaluation only, but against an arbitrary saved `model_outputs.json` file:

```bash
uv run alpacaeval-eval \
  --config alpacaeval/config_alpacaeval.yaml \
  --model-outputs /absolute/path/to/model_outputs.json
```

Evaluation with AlpacaEval model-config generation instead of saved outputs:

```bash
uv run alpacaeval-eval \
  --config alpacaeval/config_alpacaeval.yaml \
  --use-model-configs
```

Batch mode, full pipeline:

```bash
uv run alpacaeval-batch --config alpacaeval/config_alpacaeval_batch.yaml
```

Batch mode, separate stages:

```bash
uv run alpacaeval-batch --config alpacaeval/config_alpacaeval_batch.yaml --inference-only
uv run alpacaeval-batch --config alpacaeval/config_alpacaeval_batch.yaml --eval-only
uv run alpacaeval-batch --config alpacaeval/config_alpacaeval_batch.yaml --use-model-configs
```

### What each stage reads and writes

`alpacaeval-infer` reads:

- `alpacaeval/config_alpacaeval.yaml`
- the configured model in `alpacaeval.model_name_or_path` or top-level
  `policy_name`
- the configured dataset, defaulting to `tatsu-lab/alpaca_eval`

`alpacaeval-infer` writes into `alpacaeval.output_dir`:

- `model_outputs.json`
- `metadata.json`

`alpacaeval-eval` reads:

- the same config file
- `model_outputs.json`, unless `--model-outputs` or `--use-model-configs` is
  used

`alpacaeval-eval` writes:

- `results/`
- `alpacaeval_model_config.yaml` when `--use-model-configs` is used

### Common AlpacaEval workflow notes

- Relative paths in AlpacaEval configs are resolved relative to the config
  file, not the shell working directory.
- `alpacaeval-eval` requires `alpaca-eval` in the current environment and
  usually needs `OPENAI_API_KEY` because the default annotator config is
  OpenAI-backed.
- `alpacaeval-batch` expands a base config into one per-model run plan, applies
  model-family defaults, and can skip existing outputs/results when
  `skip_existing: true`.

## HH GPT Judge

### What the HH pipeline does

The HH pipeline is also multi-stage, but unlike AlpacaEval there is no single
batch wrapper. You run each stage directly:

1. Generate model outputs with `hh-generate`.
2. Export the dataset's chosen response with `hh-generate --extract_chosen`.
3. Compare two or three output files with `hh-judge`, which uses GPT-4o and
   writes per-example judgments plus a summary.

### Choose the right config

Use one config that matches both the dataset slice and prompt shape:

- helpful-base single-turn:
  `gpt_judge_HH/config/helpful_base/single-turn/config_eval_HH.yaml`
- helpful-base multi-turn:
  `gpt_judge_HH/config/helpful_base/multi-turn/config_eval_HH_multiturn.yaml`
- harmless-base single-turn:
  `gpt_judge_HH/config/harmless_base/single-turn/config_eval_HH.yaml`
- harmless-base multi-turn:
  `gpt_judge_HH/config/harmless_base/multi-turn/config_eval_HH_multiturn.yaml`

The HH config controls:

- `dataset.repo_id`, `dataset.data_dir`, `dataset.split`,
  `dataset.single_turn_only`, `dataset.max_instances`
- `models.<key>`: model ids or local paths for `sft`, `beta_dpo`,
  `margin_dpo`, `e_dpo`, and any other named candidates you add
- `generation.model_key`: which entry under `models` to load
- `generation.output_key`: which entry under `inputs` to write
- `inputs.<key>`: JSON paths for generated outputs and chosen exports
- `judge.candidate_keys`: which outputs the judge compares
- `output.results_file` and `output.summary_file`: judge artifacts
- `gpt4_oracle.*`: OpenAI model, prompt template, retry/backoff, and token
  budget

### Full HH pipeline

Run one generation command per model output:

```bash
uv run hh-generate --config gpt_judge_HH/config/helpful_base/multi-turn/config_eval_HH_multiturn.yaml --model_key sft --output_key sft
uv run hh-generate --config gpt_judge_HH/config/helpful_base/multi-turn/config_eval_HH_multiturn.yaml --model_key beta_dpo --output_key beta_dpo
uv run hh-generate --config gpt_judge_HH/config/helpful_base/multi-turn/config_eval_HH_multiturn.yaml --model_key margin_dpo --output_key margin_dpo
uv run hh-generate --config gpt_judge_HH/config/helpful_base/multi-turn/config_eval_HH_multiturn.yaml --model_key e_dpo --output_key e_dpo
```

Export the HH chosen reference:

```bash
uv run hh-generate --config gpt_judge_HH/config/helpful_base/multi-turn/config_eval_HH_multiturn.yaml --output_key chosen --extract_chosen
```

Run the GPT judge:

```bash
uv run hh-judge --config gpt_judge_HH/config/helpful_base/multi-turn/config_eval_HH_multiturn.yaml --resume
```

`judge.candidate_keys` decides what `hh-judge` compares. The checked-in configs
mostly use pairwise chosen-vs-model comparisons. If you want a different pair,
either:

- use one of the checked-in `chosen_vs_*.yaml` configs, or
- edit `judge.candidate_keys` in the base config, or
- override file paths on the CLI with `--sft`, `--beta_dpo`, `--margin_dpo`,
  `--chosen`, and related flags

### Separate HH stages

Generate just one model's outputs:

```bash
uv run hh-generate \
  --config gpt_judge_HH/config/harmless_base/single-turn/config_eval_HH.yaml \
  --model_key beta_dpo \
  --output_key beta_dpo
```

Export chosen responses only:

```bash
uv run hh-generate \
  --config gpt_judge_HH/config/harmless_base/single-turn/config_eval_HH.yaml \
  --output_key chosen \
  --extract_chosen
```

Judge only, using whatever output files are already present in the config:

```bash
uv run hh-judge --config gpt_judge_HH/config/harmless_base/single-turn/config_eval_HH.yaml --resume
```

Judge only, but limit to a small smoke-test subset:

```bash
uv run hh-judge \
  --config gpt_judge_HH/config/harmless_base/single-turn/config_eval_HH.yaml \
  --max_examples 20
```

Judge only, but override one or more candidate files at runtime:

```bash
uv run hh-judge \
  --config gpt_judge_HH/config/harmless_base/single-turn/config_eval_HH.yaml \
  --beta_dpo /absolute/path/to/beta_dpo_output_hh.json \
  --chosen /absolute/path/to/chosen_output_hh.json
```

### What HH stages read and write

`hh-generate` reads:

- one HH config YAML
- the configured model from `models.<model_key>`, unless `--extract_chosen` is
  used
- the Anthropic HH dataset split selected by `dataset.repo_id`,
  `dataset.data_dir`, `dataset.split`, and `dataset.single_turn_only`

`hh-generate` writes one JSON array to `inputs.<output_key>` or `--output_file`.

Generated model-output rows contain:

- `instruction`
- `raw_instruction`
- `output`
- `generator`

Chosen-export rows contain:

- `instruction`
- `output`
- `generator`

`hh-judge` reads:

- the output JSON files named by `judge.candidate_keys`
- `OPENAI_API_KEY`

`hh-judge` writes:

- `output.results_file`: JSONL with one row per judged instruction
- `output.summary_file`: JSON summary with totals, counts, and win rates

Each results row includes:

- `instruction`
- `comparison`
- `winner`
- `winner_key`
- `labels`
- `model`
- `raw_response`
- `usage`

### Common HH workflow notes

- HH config paths are used as written. They are not resolved relative to the
  config file, so run from the repo root or use absolute paths in the YAML.
- `generation.model_key` picks the model to load; `generation.output_key`
  selects where its JSON is written.
- `--extract_chosen` skips model loading entirely and exports the dataset's
  chosen response instead.
- `hh-judge --resume` skips instructions already present in the results file,
  which is useful for long GPT-judge runs.
- The judge only compares the intersection of instructions present in all
  selected candidate files.

MT-Bench:

```bash
uv run mtbench-infer --config mtbench/config_mtbench.yaml
uv run mtbench-eval --config mtbench/config_mtbench.yaml
uv run mtbench-batch --config mtbench/config_mtbench_batch.yaml
```

## Prerequisites

- Python environment managed with `uv`.
- Project dependencies installed.
- Install dependencies with `uv sync`.
- Access to the target model in `policy_name` or
  `alpacaeval.model_name_or_path`.
- Access to the AlpacaEval dataset from Hugging Face.
- The repo now declares `vllm` in both the main dependency set and the `eval`
  dependency group for Linux environments.
- The default annotator config is OpenAI-backed, so `alpacaeval-eval`
  typically needs `OPENAI_API_KEY`.
- Arena-Hard v2 clones the pinned official Arena-Hard-Auto runner, downloads
  the v2 question file and `o3-mini-2025-01-31` baseline answer, and uses
  GPT-4.1 as the default judge. It requires `OPENAI_API_KEY` for judging and
  expects the evaluated model to be reachable through `arenahard.model_endpoint`.
- HH generation requires access to the Anthropic HH dataset on Hugging Face.
- HH judging requires `OPENAI_API_KEY`.
- MT-Bench now stages the original FastChat `data/mt_bench/...` layout and
  calls FastChat's `gen_model_answer`, `gen_judgment`, and `show_result`
  modules directly.
- The MT-Bench wrapper auto-downloads the default question file,
  `judge_prompts.jsonl`, and the default reference answers when those config
  paths are left at their defaults.
- The checked-in MT-Bench defaults follow the SimPO setup: use the
  GPT-4-Turbo (`gpt-4-1106-preview`) reference answers and run both reported
  judge models, `gpt-4-1106-preview` and `gpt-4`.
- FastChat's MT-Bench modules import `anthropic` at module import time, so
  that dependency must be present even if you only use an OpenAI judge model.

## Llama 3 and chat templating

For Llama 3, apply chat templating exactly once.

Inference has two mutually exclusive prompt paths:

- `alpacaeval.use_custom_chat_template: true`: the repo formats prompts with a
  model-specific file template in `alpacaeval/templates/`.
- `alpacaeval.use_custom_chat_template: false`: the repo calls the tokenizer's
  built-in `apply_chat_template(...)`.

If prompts are built with `apply_chat_template(..., tokenize=False)`, the
follow-up tokenizer call must use `add_special_tokens=False`. This avoids the
Llama 3 double-BOS problem that happens when a rendered chat prompt is treated
like plain text and special tokens are added again.

Use the custom template path only when the checkpoint expects the repo's prompt
format or when you are using model-config evaluation.

Llama 3 template notes:

- the checked-in Llama templates omit BOS
- batch defaults select a model-specific Llama template file
- if you want model-default chat templating instead, set
  `use_custom_chat_template: false`

## Batch config defaults

`alpacaeval/config_alpacaeval_batch.yaml` is set up for the repo's eight
Qwen3/Llama3 UltraFeedback and UltraChat checkpoints.

- Backend defaults to `vllm`.
- Qwen3 models use the tokenizer default chat template and
  `stop_token_ids: [151645]`.
- Llama3 models use a model-specific custom template file and
  `stop_token_ids: [128001, 128009]`.
- `skip_existing: true` avoids rerunning inference or eval if outputs already
  exist.

MT-Bench uses the same 8-model batch matrix and the same Qwen3/Llama3 family
defaults as AlpacaEval:

- Qwen3 uses tokenizer-default chat templating and `stop_token_ids: [151645]`.
- Llama3 uses checked-in model-specific templates and
  `stop_token_ids: [128001, 128009]`.

## Key config fields

The pipeline reads the `alpacaeval` block in
`alpacaeval/config_alpacaeval.yaml`.

- `model_name_or_path`: model to load. Falls back to top-level `policy_name` if
  omitted.
- `pretty_name`: label written into the AlpacaEval payload as `generator`.
- `backend`: must be `transformers` or `vllm`.
- `output_dir`: where outputs, metadata, and results are written.
- `dataset_name`, `dataset_config`, `dataset_split`: dataset source, defaulting
  to `tatsu-lab/alpaca_eval`, `alpaca_eval`, `eval`.
- `dataset_trust_remote_code`: whether `alpacaeval-infer` should auto-accept
  Hugging Face dataset remote code. Defaults to `true`.
- `annotators_config`: AlpacaEval annotator setting passed through to
  `alpaca-eval`.
- `evaluation_mode`: `outputs` or `model_configs`.
- `use_custom_chat_template`: when `true`, prompts come from `prompt_template`;
  when `false`, the tokenizer's built-in chat template is used.
- `prompt_template`: required when `use_custom_chat_template: true`.
- `generation`: generation settings such as `batch_size`, `max_new_tokens`,
  `temperature`, `top_p`, and `stop_token_ids`.
- `transformers`: backend-specific settings like `device`, `device_map`, and
  `trust_remote_code`.
- `vllm`: backend-specific settings like `tensor_parallel_size` and
  `gpu_memory_utilization`.
- `simpo_compat`: enforces `alpaca-eval==0.6.2` during evaluation.

Arena-Hard v2 and MT-Bench use benchmark-specific config blocks:

- `arenahard.*` includes `pretty_name`, `output_dir`, `official_ref`,
  `categories`, `model_endpoint`, `judge_model`, `judge_endpoint`, and
  `control_features`. The first implementation reports `hard_prompt` only.
- `mtbench.*` includes `model_name_or_path`, `pretty_name`, `backend`,
  `output_dir`, `use_custom_chat_template`, `prompt_template`, `generation`,
  `transformers`, and `vllm`.
- `mtbench.question_file`, `mtbench.reference_answer_file`,
  `mtbench.judge_prompts_file`, `mtbench.judge_model`,
  `mtbench.judge_models`, `mtbench.mode`, `mtbench.parallel`, and
  `mtbench.baseline_model` control MT-Bench's original FastChat invocation.

HH judging uses a different config layout:

- `dataset.*` selects the HH dataset repo, split, data directory
  (`helpful-base` or `harmless-base`), and whether prompts are single-turn or
  multi-turn.
- `models.<key>` names the local or Hugging Face checkpoints for candidate
  systems.
- `generation.model_key` chooses which model entry to load for a generation
  run.
- `generation.output_key` chooses which `inputs.<key>` path receives the saved
  JSON.
- `generation.extract_chosen` switches generation into chosen-response export
  mode.
- `inputs.<key>` stores the per-candidate output JSON paths consumed later by
  `hh-judge`.
- `judge.candidate_keys` selects the two or three candidates compared in a
  given judge run.
- `gpt4_oracle.*` controls judge model name, prompts, retries, and token
  limits.
- `output.results_file` and `output.summary_file` store judge artifacts.

## Outputs

The pipeline writes into `alpacaeval.output_dir`.

Expected artifacts:

- `model_outputs.json`: one row per AlpacaEval example with the model response
  in `output`.
- `metadata.json`: run metadata including backend, prompt template source,
  dataset info, generation config, and package versions.
- `results/`: evaluation outputs produced by `alpaca-eval`.
- `alpacaeval_model_config.yaml`: only written when using model-config
  evaluation.

Arena-Hard v2 and MT-Bench write benchmark-native answer payloads:

- `model_answer.jsonl`: generated answers in the format expected by the judge
  tool.
- `metadata.json`: local generation metadata.
- `results/`: judge outputs from the configured external command.

Arena-Hard v2 additionally writes `results/show_result.txt` and copies the
official judgment file under `results/model_judgment/<judge>/<model>.jsonl`.

HH writes simpler JSON artifacts:

- `inputs.<key>`: JSON array of generated responses or chosen-response exports.
- `output.results_file`: JSONL with one row per GPT-judge decision.
- `output.summary_file`: summary JSON with totals, counts, and win rates.

Relative path handling differs by pipeline:

- AlpacaEval resolves relative paths relative to the config file.
- HH uses paths exactly as written, so relative HH paths are interpreted from
  the shell working directory.

## Tests

Relevant tests live in:

- `test/test_alpacaeval_pipeline.py`
- `test/test_alpacaeval_batch_runner.py`
- `tests/test_arenahard_v2.py`
- `test/test_mtbench_pipeline.py`
- `test/test_mtbench_batch_runner.py`

The pipeline tests cover:

- single-model inference and evaluation wiring
- batch-runner config expansion
- Llama3/Qwen3 chat-template handling
- real 3-sample AlpacaEval smoke tests with real tokenizers and stubbed model
  generation
