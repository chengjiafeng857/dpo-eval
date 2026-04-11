"""Leaderboard reporting for Arena-Hard v2.0."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

from benchmark_common import get_block_config, resolve_path
from config_utils import load_yaml

from .common import (
    BLOCK_NAME,
    get_benchmark_dir,
    load_model_answers,
    metadata_to_feature_vector,
    read_jsonl,
)
from .judge_settings import JUDGE_SETTINGS
from .math_utils import bootstrap_pairwise_model, one_hot_encode, to_winrate_probabilities


LABEL_TO_SCORE = {
    "A>B": [1.0],
    "A>>B": [1.0, 1.0, 1.0],
    "A=B": [0.5],
    "A<<B": [0.0, 0.0, 0.0],
    "A<B": [0.0],
    "B>A": [0.0],
    "B>>A": [0.0, 0.0, 0.0],
    "B=A": [0.5],
    "B<<A": [1.0, 1.0, 1.0],
    "B<A": [1.0],
}


def _normalize_model_name(name: str) -> str:
    return name.split("/")[-1]


def _resolve_benchmark_dir(config: Dict[str, Any], benchmark_dir: str | None) -> Path:
    if benchmark_dir is not None:
        return resolve_path(config, benchmark_dir)
    return get_benchmark_dir(config)


def load_judgments(
    benchmark_dir: Path,
    judge_names: Iterable[str],
) -> list[dict[str, Any]]:
    battles: list[dict[str, Any]] = []
    for judge_name in judge_names:
        judge_dir = benchmark_dir / "model_judgment" / judge_name
        if not judge_dir.exists():
            continue
        for file_path in sorted(judge_dir.glob("*.jsonl")):
            for row in read_jsonl(file_path):
                games = row.get("games", [])
                if len(games) < 2:
                    continue
                first = games[0]
                second = games[1]
                if first is None or second is None:
                    continue
                first_score = first.get("score")
                second_score = second.get("score")
                if first_score is None or second_score is None:
                    continue
                scores = LABEL_TO_SCORE.get(str(second_score), []) + [
                    1.0 - value for value in LABEL_TO_SCORE.get(str(first_score), [])
                ]
                for score in scores:
                    battles.append(
                        {
                            "uid": str(row["uid"]),
                            "model": _normalize_model_name(str(row["model"])),
                            "category": str(row["category"]),
                            "score": float(score),
                        }
                    )
    return battles


def get_model_style_metadata(benchmark_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    answer_dir = benchmark_dir / "model_answer"
    raw = load_model_answers(answer_dir)
    payload: dict[str, dict[str, dict[str, Any]]] = {}
    for model_name, rows in raw.items():
        payload[_normalize_model_name(model_name)] = {
            uid: dict(row.get("metadata", {}))
            for uid, row in rows.items()
        }
    return payload


def _format_rows(
    rows: list[dict[str, str | float]],
    *,
    baseline: str | None = None,
) -> str:
    output_rows = list(rows)
    if baseline and baseline not in [str(row["Model"]) for row in output_rows]:
        output_rows.append(
            {"Model": baseline, "Scores (%)": 50.0, "CI (%)": "(-0.0 / +0.0)"}
        )
    output_rows.sort(key=lambda row: float(row["Scores (%)"]), reverse=True)

    headers = ["Model", "Scores (%)", "CI (%)"]
    widths = {
        header: max(len(header), max(len(str(row[header])) for row in output_rows))
        for header in headers
    }
    lines = [
        "  ".join(header.ljust(widths[header]) for header in headers)
    ]
    for row in output_rows:
        lines.append(
            "  ".join(str(row[header]).ljust(widths[header]) for header in headers)
        )
    return "\n".join(lines)


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = position - lower_index
    return sorted_values[lower_index] * (1 - fraction) + sorted_values[upper_index] * fraction


def build_raw_leaderboard(
    battles: list[dict[str, Any]],
    *,
    baseline: str,
    bootstrap_rounds: int,
) -> list[dict[str, str | float]]:
    scores_by_model: dict[str, list[float]] = {}
    for battle in battles:
        scores_by_model.setdefault(str(battle["model"]), []).append(float(battle["score"]))

    rows: list[dict[str, str | float]] = []
    for model_name, scores in scores_by_model.items():
        if not scores:
            continue
        if bootstrap_rounds <= 0:
            bootstrap_rounds = 1
        bootstraps: list[float] = []
        score_tensor = torch.tensor(scores, dtype=torch.float32)
        for _ in range(bootstrap_rounds):
            indices = torch.randint(low=0, high=len(scores), size=(len(scores),))
            bootstraps.append(float(score_tensor[indices].mean().item()))
        bootstraps.sort()
        mean_score = sum(bootstraps) / len(bootstraps)
        lower = _quantile(bootstraps, 0.05)
        upper = _quantile(bootstraps, 0.95)
        rows.append(
            {
                "Model": model_name,
                "Scores (%)": round(mean_score * 100, 1),
                "CI (%)": f"(-{round((mean_score - lower) * 100, 1)} / +{round((upper - mean_score) * 100, 1)})",
            }
        )
    if baseline not in scores_by_model:
        rows.append({"Model": baseline, "Scores (%)": 50.0, "CI (%)": "(-0.0 / +0.0)"})
    return rows


def build_style_controlled_leaderboard(
    battles: list[dict[str, Any]],
    *,
    benchmark_dir: Path,
    baseline: str,
    control_features: list[str],
    bootstrap_rounds: int,
) -> list[dict[str, str | float]]:
    metadata = get_model_style_metadata(benchmark_dir)
    model_features = []
    baseline_features = []
    models = []
    outcomes = []

    for battle in battles:
        uid = str(battle["uid"])
        model_name = str(battle["model"])
        if model_name not in metadata or baseline not in metadata:
            continue
        if uid not in metadata[model_name] or uid not in metadata[baseline]:
            continue
        model_features.append(metadata_to_feature_vector(metadata[model_name][uid]))
        baseline_features.append(metadata_to_feature_vector(metadata[baseline][uid]))
        models.append(model_name)
        outcomes.append(float(battle["score"]))

    if not models:
        return [{"Model": baseline, "Scores (%)": 50.0, "CI (%)": "(-0.0 / +0.0)"}]

    model_feature_tensor = torch.tensor(model_features, dtype=torch.float32)
    baseline_feature_tensor = torch.tensor(baseline_features, dtype=torch.float32)
    final_feature_tensor = torch.zeros_like(model_feature_tensor)
    final_feature_tensor[:, 0] = (
        model_feature_tensor[:, 0] - baseline_feature_tensor[:, 0]
    ) / (model_feature_tensor[:, 0] + baseline_feature_tensor[:, 0]).clamp(min=1.0)

    model_md_density = model_feature_tensor[:, 1:] / (model_feature_tensor[:, :1] + 1.0)
    baseline_md_density = baseline_feature_tensor[:, 1:] / (baseline_feature_tensor[:, :1] + 1.0)
    final_feature_tensor[:, 1:] = (
        model_md_density - baseline_md_density
    ) / (model_md_density + baseline_md_density + 1.0)

    mean = torch.mean(final_feature_tensor, dim=0)
    std = torch.std(final_feature_tensor, dim=0)
    std = torch.where(std == 0, torch.ones_like(std), std)
    normalized_feature_tensor = (final_feature_tensor - mean) / std

    model_one_hot, unique_models = one_hot_encode(models, baseline=baseline)
    if "length" in control_features and "markdown" in control_features:
        style_tensor = normalized_feature_tensor
        num_style_features = 4
    elif "length" in control_features:
        style_tensor = normalized_feature_tensor[:, :1]
        num_style_features = 1
    elif "markdown" in control_features:
        style_tensor = normalized_feature_tensor[:, 1:]
        num_style_features = 3
    else:
        raise ValueError("control_features must contain 'length', 'markdown', or both.")

    all_features = torch.cat([model_one_hot, style_tensor], dim=1)
    outcomes_tensor = torch.tensor(outcomes, dtype=torch.float32)
    coefs = bootstrap_pairwise_model(all_features, outcomes_tensor, num_round=bootstrap_rounds)
    model_coefs = coefs[:, :-num_style_features]
    probabilities = to_winrate_probabilities(
        model_coefs,
        unique_models,
        baseline_model=baseline,
    )

    rows: list[dict[str, str | float]] = []
    for model_index, model_name in enumerate(unique_models):
        model_scores = sorted(float(value) for value in probabilities[:, model_index].tolist())
        median = _quantile(model_scores, 0.5)
        lower = _quantile(model_scores, 0.05)
        upper = _quantile(model_scores, 0.95)
        rows.append(
            {
                "Model": model_name,
                "Scores (%)": round(median * 100, 1),
                "CI (%)": f"(-{round((median - lower) * 100, 1)} / +{round((upper - median) * 100, 1)})",
            }
        )
    return rows


def run_arenahard_v2_report(
    config: Dict[str, Any],
    *,
    benchmark_dir: str | None = None,
    judge_names: list[str] | None = None,
    categories: list[str] | None = None,
    control_features: list[str] | None = None,
) -> dict[str, str]:
    block_cfg = get_block_config(config, BLOCK_NAME)
    resolved_benchmark_dir = _resolve_benchmark_dir(config, benchmark_dir)
    resolved_judge_names = judge_names or [str(block_cfg.get("judge_model", "gpt-4.1"))]
    resolved_categories = categories or ["hard_prompt"]
    resolved_control_features = control_features or []
    bootstrap_rounds = int(block_cfg.get("bootstrap_rounds", 100))

    battles = load_judgments(resolved_benchmark_dir, resolved_judge_names)
    if not battles:
        raise FileNotFoundError("No Arena-Hard v2.0 judgment files were found.")

    tables: dict[str, str] = {}
    for category in resolved_categories:
        category_battles = [battle for battle in battles if str(battle["category"]) == category]
        if not category_battles:
            raise ValueError(f"Invalid category or no data found: {category}")
        baseline = _normalize_model_name(JUDGE_SETTINGS[category]["baseline"])
        if resolved_control_features:
            rows = build_style_controlled_leaderboard(
                category_battles,
                benchmark_dir=resolved_benchmark_dir,
                baseline=baseline,
                control_features=resolved_control_features,
                bootstrap_rounds=bootstrap_rounds,
            )
        else:
            rows = build_raw_leaderboard(
                category_battles,
                baseline=baseline,
                bootstrap_rounds=bootstrap_rounds,
            )
        tables[category] = _format_rows(rows, baseline=baseline)
        print(f"##### Category: {category} #####")
        print(tables[category])
    return tables


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show Arena-Hard v2.0 results")
    parser.add_argument(
        "--config",
        type=str,
        default="arenahard_v2/config_arenahard_v2.yaml",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--judge-names",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--control-features",
        nargs="+",
        default=[],
    )
    parser.add_argument(
        "--category",
        nargs="+",
        default=["hard_prompt"],
    )
    args = parser.parse_args(argv)
    config = load_yaml(args.config)
    run_arenahard_v2_report(
        config,
        benchmark_dir=args.benchmark_dir,
        judge_names=args.judge_names,
        categories=args.category,
        control_features=args.control_features,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

