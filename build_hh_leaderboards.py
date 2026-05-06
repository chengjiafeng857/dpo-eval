#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "result"
TARGET_DIRS = [
    "outputs",
    "outputs 0.2",
    "outputs 0.3",
    "outputs0",
    "outputs1",
    "outputs2",
    "outputs3",
    "outputs4",
    "outputs5",
    "outputs6",
    "outputs7",
    "outputs8",
    "outputs9",
    "outputs10",
]
CSV_COLUMNS = [
    "rank",
    "model_name",
    "source_dir",
    "judge_model",
    "judge_backend",
    "prompt_variant",
    "total",
    "model_wins",
    "chosen_wins",
    "ties",
    "errors",
    "model_win_rate",
    "chosen_win_rate",
    "summary_file",
]
PREFERRED_PROMPTS = {
    "helpful": {
        "helpful": 4,
        "helpful_base": 1,
    },
    "harmless": {
        "harmless": 5,
        "general-less-harmful": 4,
        "harmless-less-harmful": 3,
        "general": 2,
        "harmless_base": 1,
    },
}


def infer_kind(path: Path) -> str | None:
    text = path.as_posix()
    if "helpful" in text:
        return "helpful"
    if "harmless" in text:
        return "harmless"
    return None


def is_candidate(path: Path) -> bool:
    text = path.as_posix()
    top = path.parts[0]

    if not text.endswith("_summary.json"):
        return False
    if "single_turn" in text or "-single" in text:
        return False
    if "multi_turn" not in text and "-multi" not in text:
        return False
    if infer_kind(path) is None:
        return False
    if top == "outputs 0.3":
        return False
    if top == "outputs":
        return "/gpt_judge_HH/" in text and "/archive/" not in text and "/RMjudge" not in text
    if top == "outputs1":
        return "/gpt_judge_HH/" in text or "/mistral_7b_base_hh/" in text
    if top == "outputs10":
        return "/gpt_judge_HH/" in text
    return True


def extract_model_key(data: dict) -> str:
    counts = data.get("counts", {})
    for key in counts:
        if key not in {"chosen", "TIE", "Error"}:
            return key
    raise ValueError("No model key found in counts")


def extract_run_name(path: Path, model_key: str) -> str:
    for part in reversed(path.parts[:-1]):
        if part.endswith("-multi") or part.endswith("-single"):
            return part
    if "mistral_7b_base_hh" in path.parts:
        return f"mistral_7b_base_hh-{model_key}-multi"
    return path.stem.removesuffix("_summary")


def extract_prompt_variant(path: Path, kind: str) -> str:
    for part in path.parts:
        if part.startswith("prompts-"):
            return part.removeprefix("prompts-")
    return f"{kind}_base"


def dedupe_priority(row: dict) -> tuple[int, int, int, int]:
    kind = row["kind"]
    prompt_variant = row["prompt_variant"]
    summary_text = row["summary_file"]
    prompt_score = PREFERRED_PROMPTS.get(kind, {}).get(prompt_variant, 0)
    gpt4_score = 1 if "/gpt-4/" in summary_text else 0
    archive_score = 0 if "/archive/" in summary_text else 1
    depth_score = len(Path(summary_text).parts)
    return (archive_score, gpt4_score, prompt_score, depth_score)


def load_rows() -> list[dict]:
    rows: list[dict] = []

    for dirname in TARGET_DIRS:
        base = ROOT / dirname
        if not base.exists():
            continue
        for path in sorted(base.rglob("*_summary.json")):
            if not is_candidate(path.relative_to(ROOT)):
                continue

            rel_path = path.relative_to(ROOT)
            kind = infer_kind(rel_path)
            if kind is None:
                continue

            data = json.loads(path.read_text())
            model_key = extract_model_key(data)
            counts = data.get("counts", {})
            win_rates = data.get("win_rates", {})

            rows.append(
                {
                    "kind": kind,
                    "model_name": extract_run_name(rel_path, model_key),
                    "source_dir": dirname,
                    "judge_model": data.get("model", ""),
                    "judge_backend": data.get("judge_backend", ""),
                    "prompt_variant": extract_prompt_variant(rel_path, kind),
                    "total": data.get("total", 0),
                    "model_wins": counts.get(model_key, 0),
                    "chosen_wins": counts.get("chosen", 0),
                    "ties": counts.get("TIE", 0),
                    "errors": counts.get("Error", 0),
                    "model_win_rate": win_rates.get(model_key, 0.0),
                    "chosen_win_rate": win_rates.get("chosen", 0.0),
                    "summary_file": rel_path.as_posix(),
                }
            )

    return rows


def dedupe_rows(rows: list[dict]) -> list[dict]:
    best_by_key: dict[tuple[str, str], dict] = {}

    for row in rows:
        key = (row["kind"], row["model_name"])
        current = best_by_key.get(key)
        if current is None or dedupe_priority(row) > dedupe_priority(current):
            best_by_key[key] = row

    return list(best_by_key.values())


def sort_rows(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: (
            -row["model_win_rate"],
            -row["model_wins"],
            row["model_name"],
        ),
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            output_row = {column: row.get(column, "") for column in CSV_COLUMNS}
            output_row["rank"] = rank
            writer.writerow(output_row)


def write_summary(path: Path, helpful_rows: list[dict], harmless_rows: list[dict]) -> None:
    lines = [
        "# HH Leaderboards",
        "",
        "Generated from multi-turn HH judge summaries found under:",
        f"- {', '.join(TARGET_DIRS)}",
        "",
        "Deduping rules:",
        "- one row per model/run name",
        "- prefer `gpt-4` prompt-specific summaries over copied root summaries",
        "- for harmless, prefer `prompts-harmless`, then `prompts-general-less-harmful`",
        "",
        "## Helpful Top 10",
        "",
    ]

    if helpful_rows:
        for idx, row in enumerate(helpful_rows[:10], start=1):
            lines.append(
                f"{idx}. `{row['model_name']}` | win rate `{row['model_win_rate']:.4f}` | source `{row['source_dir']}` | prompt `{row['prompt_variant']}`"
            )
    else:
        lines.append("No helpful rows found.")

    lines.extend(["", "## Harmless Top 10", ""])

    if harmless_rows:
        for idx, row in enumerate(harmless_rows[:10], start=1):
            lines.append(
                f"{idx}. `{row['model_name']}` | win rate `{row['model_win_rate']:.4f}` | source `{row['source_dir']}` | prompt `{row['prompt_variant']}`"
            )
    else:
        lines.append("No harmless rows found.")

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    rows = dedupe_rows(load_rows())
    helpful_rows = sort_rows([row for row in rows if row["kind"] == "helpful"])
    harmless_rows = sort_rows([row for row in rows if row["kind"] == "harmless"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "hh_helpful_leaderboard.csv", helpful_rows)
    write_csv(OUTPUT_DIR / "hh_harmless_leaderboard.csv", harmless_rows)
    write_summary(OUTPUT_DIR / "hh_leaderboards.md", helpful_rows, harmless_rows)


if __name__ == "__main__":
    main()
