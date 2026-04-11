"""Evaluation wrapper for MT-Bench using FastChat's original framework."""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from config_utils import load_yaml
from benchmark_common import (
    get_block_config,
    get_output_dir,
    get_pretty_name,
)
from .fastchat_integration import (
    FASTCHAT_BENCH_NAME,
    get_fastchat_workspace,
    resolve_mtbench_path,
    stage_fastchat_judge_prompts_file,
    stage_fastchat_model_answer_file,
    stage_fastchat_question_file,
    stage_fastchat_reference_answer_file,
)


def _resolve_model_answer_path(
    config: Dict[str, Any],
    model_answer_path: str | None,
) -> Path:
    if model_answer_path is not None:
        return resolve_mtbench_path(config, model_answer_path)

    default_path = get_output_dir(config, "mtbench") / "model_answer.jsonl"
    if default_path.exists():
        return default_path

    raise FileNotFoundError(
        "Could not find MT-Bench model answers. Run mtbench-infer first or "
        "pass --model-answer."
    )


def _get_judgment_output_path(
    workspace_dir: Path,
    *,
    judge_model: str,
    mode: str,
) -> Path:
    suffix = "single" if mode == "single" else "pair"
    return (
        workspace_dir
        / "data"
        / FASTCHAT_BENCH_NAME
        / "model_judgment"
        / f"{judge_model}_{suffix}.jsonl"
    )


def _get_judge_models(config: Dict[str, Any]) -> list[str]:
    block_cfg = get_block_config(config, "mtbench")
    judge_models = block_cfg.get("judge_models")
    if judge_models is None:
        return [str(block_cfg.get("judge_model", "gpt-4-1106-preview"))]
    if not isinstance(judge_models, list) or not judge_models:
        raise ValueError("mtbench.judge_models must be a non-empty list of strings.")
    return [str(item) for item in judge_models]


def _build_judgment_command(
    config: Dict[str, Any],
    *,
    judge_model: str,
) -> list[str]:
    block_cfg = get_block_config(config, "mtbench")
    mode = str(block_cfg.get("mode", "single"))
    command = [
        sys.executable,
        "-m",
        "fastchat.llm_judge.gen_judgment",
        "--bench-name",
        FASTCHAT_BENCH_NAME,
        "--judge-file",
        "data/judge_prompts.jsonl",
        "--judge-model",
        judge_model,
        "--mode",
        mode,
        "--model-list",
        get_pretty_name(config, "mtbench"),
    ]

    if mode == "pairwise-baseline":
        command.extend(
            [
                "--baseline-model",
                str(block_cfg.get("baseline_model", "gpt-3.5-turbo")),
            ]
        )

    parallel = block_cfg.get("parallel")
    if parallel is not None:
        command.extend(["--parallel", str(int(parallel))])

    first_n = block_cfg.get("first_n")
    if first_n is not None:
        command.extend(["--first-n", str(int(first_n))])

    return command


def _build_show_result_command(
    config: Dict[str, Any],
    *,
    judgment_path: Path,
    judge_model: str,
) -> list[str]:
    block_cfg = get_block_config(config, "mtbench")
    mode = str(block_cfg.get("mode", "single"))
    command = [
        sys.executable,
        "-m",
        "fastchat.llm_judge.show_result",
        "--bench-name",
        FASTCHAT_BENCH_NAME,
        "--input-file",
        str(judgment_path),
        "--judge-model",
        judge_model,
        "--mode",
        mode,
        "--model-list",
        get_pretty_name(config, "mtbench"),
    ]

    if mode == "pairwise-baseline":
        command.extend(
            [
                "--baseline-model",
                str(block_cfg.get("baseline_model", "gpt-3.5-turbo")),
            ]
        )

    return command


def run_mtbench_evaluation(
    config: Dict[str, Any],
    *,
    model_answer_path: str | None = None,
) -> Path:
    block_cfg = get_block_config(config, "mtbench")
    output_dir = get_output_dir(config, "mtbench")
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    resolved_answer_path = _resolve_model_answer_path(config, model_answer_path)
    workspace_dir = get_fastchat_workspace(config)
    staged_question_path = stage_fastchat_question_file(config)
    staged_prompt_path = stage_fastchat_judge_prompts_file(config)
    staged_answer_path = stage_fastchat_model_answer_file(
        config,
        pretty_name=get_pretty_name(config, "mtbench"),
        model_answer_path=resolved_answer_path,
    )

    mode = str(block_cfg.get("mode", "single"))
    judge_models = _get_judge_models(config)

    print(f"[MTBench] framework=fastchat_original")
    print(f"[MTBench] answer_file={resolved_answer_path}")
    print(f"[MTBench] staged_question_file={staged_question_path}")
    print(f"[MTBench] staged_prompt_file={staged_prompt_path}")

    for judge_index, judge_model in enumerate(judge_models):
        staged_reference_path = stage_fastchat_reference_answer_file(
            config,
            judge_model=judge_model,
        )
        staged_judgment_path = _get_judgment_output_path(
            workspace_dir,
            judge_model=judge_model,
            mode=mode,
        )
        staged_judgment_path.parent.mkdir(parents=True, exist_ok=True)
        if staged_judgment_path.exists():
            staged_judgment_path.unlink()

        command = _build_judgment_command(
            config,
            judge_model=judge_model,
        )
        print(f"[MTBench] judge_model={judge_model}")
        print(f"[MTBench] staged_reference_file={staged_reference_path}")
        print(f"[MTBench] command={' '.join(shlex.quote(part) for part in command)}")
        subprocess.run(command, cwd=workspace_dir, check=True, text=True, input="\n")

        judge_result_path = results_dir / staged_judgment_path.name
        shutil.copy2(staged_judgment_path, judge_result_path)
        if judge_index == 0:
            shutil.copy2(staged_judgment_path, results_dir / "mtbench_judgments.jsonl")

        show_result_command = _build_show_result_command(
            config,
            judgment_path=staged_judgment_path,
            judge_model=judge_model,
        )
        print(
            "[MTBench] show_result_command="
            f"{' '.join(shlex.quote(part) for part in show_result_command)}"
        )
        subprocess.run(show_result_command, cwd=workspace_dir, check=True)

    print(f"[MTBench] staged_answer_file={staged_answer_path}")
    print(f"[MTBench] results_dir={results_dir}")
    return results_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run MT-Bench evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default="mtbench/config_mtbench.yaml",
    )
    parser.add_argument(
        "--model-answer",
        type=str,
        default=None,
    )
    args = parser.parse_args(argv)

    config = load_yaml(args.config)
    run_mtbench_evaluation(
        config,
        model_answer_path=args.model_answer,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
