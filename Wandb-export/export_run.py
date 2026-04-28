#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2,<3",
#   "wandb>=0.17",
# ]
# ///
"""Export one W&B run's history, metadata, and files.

Usage:
    uv run Wandb-export/export_run.py

The default run is the one requested in this repository. Override it with
--run-path or WANDB_RUN_PATH when needed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import wandb


DEFAULT_RUN_PATH = (
    "/feng-cheng-northeastern-university/"
    "llama3-8b-base-new-method-hh-q_t-0p45/runs/yuvsexn3"
)
CONTEXT_COLUMNS = ["_step", "_runtime", "_timestamp", "train/global_step", "train/epoch"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-path",
        default=os.environ.get("WANDB_RUN_PATH", DEFAULT_RUN_PATH),
        help="W&B run path, e.g. /entity/project/runs/run_id.",
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


def safe_name(value: str) -> str:
    value = value.strip("/").replace("/runs/", "/")
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", value)


def api_run_paths(value: str) -> list[str]:
    stripped = value.strip("/")
    candidates = [value, stripped]
    parts = stripped.split("/")
    if len(parts) == 4 and parts[2] == "runs":
        candidates.append(f"{parts[0]}/{parts[1]}/{parts[3]}")
    return list(dict.fromkeys(candidates))


def get_run(api: Any, run_path: str) -> Any:
    errors: list[str] = []
    for candidate in api_run_paths(run_path):
        try:
            return api.run(candidate)
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
    raise RuntimeError(
        "Could not load W&B run. Tried:\n  " + "\n  ".join(errors)
    )


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    try:
        item = value.item
    except Exception:
        item = None
    if item is not None:
        try:
            return to_jsonable(item())
        except Exception:
            pass
    return str(value)


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(to_jsonable(data), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: to_jsonable(row.get(key)) for key in fieldnames})


def to_long_history(df: pd.DataFrame) -> pd.DataFrame:
    context_columns = [column for column in CONTEXT_COLUMNS if column in df.columns]
    metric_columns = [column for column in df.columns if column not in context_columns]
    if not metric_columns:
        return df[context_columns].copy()
    long_df = df.melt(
        id_vars=context_columns,
        value_vars=metric_columns,
        var_name="metric",
        value_name="value",
    )
    long_df = long_df.dropna(subset=["value"])
    sort_columns = [column for column in ["_step", "metric"] if column in long_df.columns]
    if sort_columns:
        long_df = long_df.sort_values(sort_columns, kind="stable")
    return long_df.reset_index(drop=True)


def prefix_history(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    context_columns = [column for column in CONTEXT_COLUMNS if column in df.columns]
    metric_columns = [column for column in df.columns if column.startswith(prefix)]
    columns = context_columns + metric_columns
    if not metric_columns:
        return pd.DataFrame(columns=columns)
    out = df.loc[df[metric_columns].notna().any(axis=1), columns].copy()
    return out.dropna(axis=1, how="all")


def forward_filled_history(df: pd.DataFrame) -> pd.DataFrame:
    if "_step" in df.columns:
        df = df.sort_values("_step", kind="stable")
    context_columns = [column for column in CONTEXT_COLUMNS if column in df.columns]
    metric_columns = [column for column in df.columns if column not in context_columns]
    out = df.copy()
    if metric_columns:
        out[metric_columns] = out[metric_columns].ffill()
    return out


def export_history_views(df: pd.DataFrame, out_dir: Path, stem: str) -> None:
    df.to_csv(out_dir / f"{stem}_wide.csv", index=False)
    df.to_json(out_dir / f"{stem}_wide.json", orient="records", lines=True)
    to_long_history(df).to_csv(out_dir / f"{stem}.csv", index=False)
    to_long_history(df).to_json(out_dir / f"{stem}.json", orient="records", lines=True)


def public_run_metadata(run: Any) -> dict[str, Any]:
    fields = [
        "id",
        "name",
        "entity",
        "project",
        "path",
        "url",
        "state",
        "created_at",
        "updated_at",
        "group",
        "job_type",
        "tags",
        "notes",
        "sweep",
    ]
    return {field: getattr(run, field, None) for field in fields}


def artifact_info(artifact: Any) -> dict[str, Any]:
    fields = [
        "name",
        "type",
        "version",
        "aliases",
        "state",
        "digest",
        "size",
        "created_at",
        "updated_at",
        "description",
    ]
    return {field: getattr(artifact, field, None) for field in fields}


def list_artifacts(run: Any, method_name: str) -> list[Any]:
    method = getattr(run, method_name, None)
    if method is None:
        return []
    try:
        return list(method())
    except Exception as exc:
        print(f"Could not list {method_name}: {exc}")
        return []


def export_run_files(run: Any, out_dir: Path, download: bool) -> list[dict[str, Any]]:
    files_dir = out_dir / "files"
    manifest: list[dict[str, Any]] = []
    for file_obj in run.files():
        info = {
            "name": getattr(file_obj, "name", None),
            "size": getattr(file_obj, "size", None),
            "md5": getattr(file_obj, "md5", None),
            "url": getattr(file_obj, "url", None),
            "downloaded_to": None,
        }
        if download:
            files_dir.mkdir(parents=True, exist_ok=True)
            downloaded = file_obj.download(root=str(files_dir), replace=True)
            info["downloaded_to"] = str(Path(downloaded.name).resolve())
        manifest.append(info)
    return manifest


def export_artifacts(
    artifacts: list[Any],
    out_dir: Path,
    kind: str,
    download: bool,
) -> list[dict[str, Any]]:
    artifact_dir = out_dir / "artifacts" / kind
    exported: list[dict[str, Any]] = []
    for artifact in artifacts:
        info = artifact_info(artifact)
        if download:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            path = Path(artifact.download(root=str(artifact_dir))).resolve()
            info["downloaded_to"] = str(path)
        exported.append(info)
    return exported


def main() -> None:
    args = parse_args()
    api = wandb.Api()
    run = get_run(api, args.run_path)

    run_dir = args.out_dir / safe_name(args.run_path)
    run_dir.mkdir(parents=True, exist_ok=True)

    print("Sampled run.history() preview:")
    sampled_history = run.history()
    print(sampled_history)
    export_history_views(sampled_history, run_dir, "history_sample")

    print("Exporting full run.scan_history()...")
    history_rows = list(run.scan_history(page_size=args.page_size))
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

    print("Exporting file manifest...")
    files_manifest = export_run_files(
        run,
        run_dir,
        download=not args.no_download_files,
    )
    write_json(run_dir / "files_manifest.json", files_manifest)

    logged_artifacts = list_artifacts(run, "logged_artifacts")
    used_artifacts = list_artifacts(run, "used_artifacts")
    write_json(
        run_dir / "artifacts_logged.json",
        export_artifacts(
            logged_artifacts,
            run_dir,
            kind="logged",
            download=args.download_artifacts,
        ),
    )
    write_json(
        run_dir / "artifacts_used.json",
        export_artifacts(
            used_artifacts,
            run_dir,
            kind="used",
            download=args.download_artifacts,
        ),
    )

    print(f"Exported {len(history_rows)} full-history rows to {run_dir}")


if __name__ == "__main__":
    main()
