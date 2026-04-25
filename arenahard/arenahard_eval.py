"""Arena-Hard v2 judgment and result wrapper around the official runner."""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from benchmark_common import get_output_dir, get_pretty_name, resolve_path
from config_utils import load_yaml

from .official_runner import (
    ensure_official_runner,
    get_bench_name,
    get_categories,
    get_judge_model,
    get_judgment_filename,
    get_model_answer_filename,
    get_runner_data_dir,
    safe_copy,
    stage_arenahard_data,
    write_official_evaluation_configs,
)


def _resolve_model_answer_path(
    config: Dict[str, Any],
    model_answer_path: str | None,
) -> Path:
    if model_answer_path is not None:
        return resolve_path(config, model_answer_path)

    default_path = get_output_dir(config, "arenahard") / "model_answer.jsonl"
    if default_path.exists():
        return default_path

    raise FileNotFoundError(
        "Could not find Arena-Hard model answers. Run arenahard-infer first or "
        "pass --model-answer."
    )


def _build_judgment_command() -> list[str]:
    return [
        sys.executable,
        "gen_judgment.py",
        "--setting-file",
        "config/arena-hard-v2.0.yaml",
        "--endpoint-file",
        "config/api_config.yaml",
    ]


def _build_show_result_command(config: Dict[str, Any]) -> list[str]:
    command = [
        sys.executable,
        "show_result.py",
        "--benchmark",
        get_bench_name(config),
        "--judge-names",
        get_judge_model(config),
        "--category",
        *get_categories(config),
    ]
    control_features = get_output_control_features(config)
    if control_features:
        command.extend(["--control-features", *control_features])
    return command


def get_output_control_features(config: Dict[str, Any]) -> list[str]:
    block_cfg = config.get("arenahard", {})
    control_features = block_cfg.get("control_features", ["markdown", "length"])
    if control_features is None:
        return []
    if not isinstance(control_features, list):
        raise ValueError("arenahard.control_features must be a list.")
    return [str(feature) for feature in control_features]


def run_arenahard_evaluation(
    config: Dict[str, Any],
    *,
    model_answer_path: str | None = None,
) -> Path:
    output_dir = get_output_dir(config, "arenahard")
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    runner_dir = ensure_official_runner(config)
    staged_data = stage_arenahard_data(config, runner_dir)
    written_configs = write_official_evaluation_configs(config, runner_dir)

    resolved_answer_path = _resolve_model_answer_path(config, model_answer_path)
    runner_answer_path = (
        get_runner_data_dir(runner_dir, config)
        / "model_answer"
        / get_model_answer_filename(config)
    )
    safe_copy(resolved_answer_path, runner_answer_path)

    judgment_command = _build_judgment_command()
    print("[ArenaHard] framework=official_arena_hard_auto")
    print(f"[ArenaHard] bench_name={get_bench_name(config)}")
    print(f"[ArenaHard] judge_model={get_judge_model(config)}")
    print(f"[ArenaHard] answer_file={resolved_answer_path}")
    print(f"[ArenaHard] staged_question_file={staged_data['question_file']}")
    print(
        "[ArenaHard] judgment_command="
        f"{' '.join(shlex.quote(part) for part in judgment_command)}"
    )
    runner_judgment_dir = (
        get_runner_data_dir(runner_dir, config) / "model_judgment" / get_judge_model(config)
    )
    runner_judgment_dir.mkdir(parents=True, exist_ok=True)
    for stale_judgment_path in runner_judgment_dir.glob("*.jsonl"):
        stale_judgment_path.unlink()
    runner_judgment_path = runner_judgment_dir / get_judgment_filename(config)
    subprocess.run(judgment_command, cwd=runner_dir, check=True)

    judgment_result_path = safe_copy(
        runner_judgment_path,
        results_dir / "model_judgment" / get_judge_model(config) / get_judgment_filename(config),
    )

    show_result_command = _build_show_result_command(config)
    print(
        "[ArenaHard] show_result_command="
        f"{' '.join(shlex.quote(part) for part in show_result_command)}"
    )
    result = subprocess.run(
        show_result_command,
        cwd=runner_dir,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    show_result_path = results_dir / "show_result.txt"
    show_result_path.write_text(result.stdout, encoding="utf-8")
    print(result.stdout, end="")

    manifest_path = results_dir / "manifest.json"
    from benchmark_common import write_json

    write_json(
        manifest_path,
        {
            "bench_name": get_bench_name(config),
            "pretty_name": get_pretty_name(config, "arenahard"),
            "judge_model": get_judge_model(config),
            "categories": get_categories(config),
            "control_features": get_output_control_features(config),
            "runner_dir": str(runner_dir),
            "staged_answer_file": str(runner_answer_path),
            "runner_judgment_file": str(runner_judgment_path),
            "judgment_result_file": str(judgment_result_path),
            "show_result_file": str(show_result_path),
            "written_configs": {key: str(path) for key, path in written_configs.items()},
        },
    )
    print(f"[ArenaHard] results_dir={results_dir}")
    return results_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Arena-Hard v2 evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default="arenahard/config_arenahard.yaml",
    )
    parser.add_argument(
        "--model-answer",
        type=str,
        default=None,
    )
    args = parser.parse_args(argv)

    config = load_yaml(args.config)
    run_arenahard_evaluation(
        config,
        model_answer_path=args.model_answer,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
