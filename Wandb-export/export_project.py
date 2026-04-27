#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2,<3",
#   "wandb>=0.17",
# ]
# ///
"""Export all W&B runs in a project.

Example:
    WANDB_API_KEY=... uv run Wandb-export/export_project.py \
      --project-path feng-cheng-northeastern-university/llama-base-hh-run-4xh200-log-q_t
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import wandb

from export_run import (
    export_artifacts,
    export_history_views,
    export_run_files,
    forward_filled_history,
    list_artifacts,
    prefix_history,
    public_run_metadata,
    safe_name,
    write_json,
    write_jsonl,
)


DEFAULT_PROJECT_PATH = (
    "feng-cheng-northeastern-university/llama-base-hh-run-4xh200-log-q_t"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-path",
        default=os.environ.get("WANDB_PROJECT_PATH", DEFAULT_PROJECT_PATH),
        help="W&B project path, e.g. entity/project.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "exports",
        help="Directory where exported data will be written.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=10_000,
        help="W&B history page size for scan_history().",
    )
    parser.add_argument(
        "--no-download-files",
        action="store_true",
        help="Skip downloading run files; still writes files_manifest.json.",
    )
    parser.add_argument(
        "--download-artifacts",
        action="store_true",
        help="Download logged and used artifacts. This can be large.",
    )
    return parser.parse_args()


def run_output_dir(out_dir: Path, project_path: str, run: Any) -> Path:
    return out_dir / safe_name(f"{project_path}/{run.name}-{run.id}")


def export_one_run(
    run: Any,
    project_path: str,
    out_dir: Path,
    page_size: int,
    download_files: bool,
    download_artifacts: bool,
) -> dict[str, Any]:
    run_dir = run_output_dir(out_dir, project_path, run)
    run_dir.mkdir(parents=True, exist_ok=True)

    sampled_history = run.history()
    print(f"\nSampled run.history() preview for {run.id} ({run.name}):")
    print(sampled_history)
    export_history_views(sampled_history, run_dir, "history_sample")

    print(f"Exporting full run.scan_history() for {run.id}...")
    history_rows = list(run.scan_history(page_size=page_size))
    write_jsonl(run_dir / "history_all.jsonl", history_rows)
    history_df = pd.DataFrame(history_rows)
    export_history_views(history_df, run_dir, "history_all")
    prefix_history(history_df, "train/").to_csv(run_dir / "history_train.csv", index=False)
    prefix_history(history_df, "eval/").to_csv(run_dir / "history_eval.csv", index=False)
    forward_filled_history(history_df).to_csv(
        run_dir / "history_all_forward_filled.csv",
        index=False,
    )

    write_json(run_dir / "run.json", public_run_metadata(run))
    write_json(run_dir / "config.json", dict(run.config))
    write_json(run_dir / "summary.json", dict(run.summary))
    write_json(run_dir / "metadata.json", getattr(run, "metadata", {}))

    files_manifest = export_run_files(run, run_dir, download=download_files)
    write_json(run_dir / "files_manifest.json", files_manifest)

    logged_artifacts = list_artifacts(run, "logged_artifacts")
    used_artifacts = list_artifacts(run, "used_artifacts")
    write_json(
        run_dir / "artifacts_logged.json",
        export_artifacts(
            logged_artifacts,
            run_dir,
            kind="logged",
            download=download_artifacts,
        ),
    )
    write_json(
        run_dir / "artifacts_used.json",
        export_artifacts(
            used_artifacts,
            run_dir,
            kind="used",
            download=download_artifacts,
        ),
    )

    return {
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "url": run.url,
        "history_rows": len(history_rows),
        "sample_history_rows": len(sampled_history),
        "output_dir": str(run_dir),
    }


def main() -> None:
    args = parse_args()
    project_path = args.project_path.strip("/")
    api = wandb.Api()
    runs = list(api.runs(project_path))

    project_dir = args.out_dir / safe_name(project_path)
    project_dir.mkdir(parents=True, exist_ok=True)
    summary_path = project_dir / "project_runs_export_summary.json"

    print(f"Found {len(runs)} runs in {project_path}")
    summaries = []
    for index, run in enumerate(runs, start=1):
        print(f"\n[{index}/{len(runs)}] Exporting {run.id} ({run.name})")
        summaries.append(
            export_one_run(
                run,
                project_path,
                args.out_dir,
                page_size=args.page_size,
                download_files=not args.no_download_files,
                download_artifacts=args.download_artifacts,
            )
        )

    summary_path.write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\nExported {len(summaries)} runs. Summary: {summary_path}")


if __name__ == "__main__":
    main()
