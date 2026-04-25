# W&B Run Export

Export the requested run with `uv`:

```bash
uv run Wandb-export/export_run.py
```

By default this exports:

- `history_sample.csv` and `history_all.csv` in long format with no sparse
  metric columns: `_step`, context columns, `metric`, `value`
- `history_sample_wide.csv` and `history_all_wide.csv` preserving W&B's raw
  sparse wide format from `run.history()` and `run.scan_history()`
- `history_all.jsonl` preserving raw `run.scan_history()` rows
- `history_train.csv` and `history_eval.csv` split by metric prefix
- `history_all_forward_filled.csv` for plotting timelines where the latest
  logged metric value should carry forward
- `run.json`, `config.json`, `summary.json`, and `metadata.json`
- `files_manifest.json` plus downloaded W&B run files under `files/`
- logged/used artifact manifests

Useful options:

```bash
uv run Wandb-export/export_run.py --no-download-files
uv run Wandb-export/export_run.py --download-artifacts
uv run Wandb-export/export_run.py --run-path /entity/project/runs/run_id
uv run Wandb-export/export_run.py --run-path entity/project/run_id
```

You need to be logged in to W&B first:

```bash
uv run wandb login
```

Or provide a key for one command:

```bash
WANDB_API_KEY=... uv run Wandb-export/export_run.py
```
