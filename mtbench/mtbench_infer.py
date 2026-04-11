"""Inference pipeline for MT-Bench using FastChat's original framework."""

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
    get_model_name_or_path,
    get_output_dir,
    get_package_versions,
    get_pretty_name,
    write_json,
)
from .fastchat_integration import (
    FASTCHAT_BENCH_NAME,
    get_fastchat_workspace,
    resolve_mtbench_question_file,
    stage_fastchat_question_file,
)


def _get_question_bounds(config: Dict[str, Any]) -> tuple[int | None, int | None]:
    block_cfg = get_block_config(config, "mtbench")
    question_begin = block_cfg.get("question_begin")
    question_end = block_cfg.get("question_end")
    max_instances = block_cfg.get("max_instances")

    if question_end is None and max_instances is not None:
        question_end = int(max_instances)

    return (
        None if question_begin is None else int(question_begin),
        None if question_end is None else int(question_end),
    )


def _build_fastchat_command(config: Dict[str, Any], *, answer_file: Path) -> list[str]:
    block_cfg = get_block_config(config, "mtbench")
    generation_cfg = block_cfg.get("generation", {})
    if not isinstance(generation_cfg, dict):
        raise ValueError("mtbench.generation must be a mapping.")

    question_begin, question_end = _get_question_bounds(config)
    command = [
        sys.executable,
        "-m",
        "fastchat.llm_judge.gen_model_answer",
        "--bench-name",
        FASTCHAT_BENCH_NAME,
        "--model-path",
        get_model_name_or_path(config, "mtbench"),
        "--model-id",
        get_pretty_name(config, "mtbench"),
        "--answer-file",
        str(answer_file),
        "--max-new-token",
        str(int(generation_cfg.get("max_new_tokens", 1024))),
        "--num-choices",
        str(int(block_cfg.get("num_choices", 1))),
        "--num-gpus-per-model",
        str(int(block_cfg.get("num_gpus_per_model", 1))),
        "--num-gpus-total",
        str(int(block_cfg.get("num_gpus_total", 1))),
    ]

    max_gpu_memory = block_cfg.get("max_gpu_memory")
    if max_gpu_memory:
        command.extend(["--max-gpu-memory", str(max_gpu_memory)])

    dtype = config.get("precision")
    if dtype:
        command.extend(["--dtype", str(dtype)])

    revision = block_cfg.get("revision")
    if revision:
        command.extend(["--revision", str(revision)])

    if question_begin is not None:
        command.extend(["--question-begin", str(question_begin)])
    if question_end is not None:
        command.extend(["--question-end", str(question_end)])

    return command


def run_mtbench_inference(config: Dict[str, Any]) -> Path:
    output_dir = get_output_dir(config, "mtbench")
    workspace_dir = get_fastchat_workspace(config)
    question_path = resolve_mtbench_question_file(config)
    staged_question_path = stage_fastchat_question_file(config)
    staged_answer_path = (
        workspace_dir
        / "data"
        / FASTCHAT_BENCH_NAME
        / "model_answer"
        / f"{get_pretty_name(config, 'mtbench')}.jsonl"
    )
    staged_answer_path.parent.mkdir(parents=True, exist_ok=True)
    if staged_answer_path.exists():
        staged_answer_path.unlink()

    command = _build_fastchat_command(config, answer_file=staged_answer_path)
    print(f"[MTBench] framework=fastchat_original")
    print(f"[MTBench] model={get_model_name_or_path(config, 'mtbench')}")
    print(f"[MTBench] question_file={question_path}")
    print(f"[MTBench] command={' '.join(shlex.quote(part) for part in command)}")
    subprocess.run(command, cwd=workspace_dir, check=True)

    model_answer_path = output_dir / "model_answer.jsonl"
    metadata_path = output_dir / "metadata.json"
    shutil.copy2(staged_answer_path, model_answer_path)
    write_json(
        metadata_path,
        {
            "framework": "fastchat_original",
            "model_name_or_path": get_model_name_or_path(config, "mtbench"),
            "pretty_name": get_pretty_name(config, "mtbench"),
            "question_file": str(question_path),
            "staged_question_file": str(staged_question_path),
            "staged_answer_file": str(staged_answer_path),
            "command": command,
            "package_versions": get_package_versions(
                ("torch", "transformers", "fschat", "anthropic")
            ),
        },
    )
    print(f"[MTBench] wrote_model_answer={model_answer_path}")
    print(f"[MTBench] wrote_metadata={metadata_path}")
    return model_answer_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run MT-Bench inference")
    parser.add_argument(
        "--config",
        type=str,
        default="mtbench/config_mtbench.yaml",
    )
    args = parser.parse_args(argv)

    config = load_yaml(args.config)
    run_mtbench_inference(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
