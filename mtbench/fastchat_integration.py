"""Helpers for running MT-Bench through FastChat's original framework."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict

from . import __file__ as _PACKAGE_FILE
from benchmark_common import (
    get_block_config,
    get_output_dir,
    resolve_existing_or_download_default_path,
    resolve_existing_path,
)


PACKAGE_DIR = Path(_PACKAGE_FILE).resolve().parent
FASTCHAT_BENCH_NAME = "mt_bench"

DEFAULT_MTBENCH_QUESTION_FILE = "questions.jsonl"
DEFAULT_MTBENCH_QUESTION_URL = (
    "https://raw.githubusercontent.com/lm-sys/FastChat/main/"
    "fastchat/llm_judge/data/mt_bench/question.jsonl"
)

DEFAULT_MTBENCH_REFERENCE_ANSWER_FILE = "reference_answer/gpt-4-1106-preview.jsonl"
DEFAULT_MTBENCH_REFERENCE_ANSWER_URL = (
    "https://raw.githubusercontent.com/princeton-nlp/SimPO/main/"
    "eval/mt-bench/gpt-4-1106-preview.jsonl"
)

DEFAULT_MTBENCH_JUDGE_PROMPTS_FILE = "judge_prompts.jsonl"
DEFAULT_MTBENCH_JUDGE_PROMPTS_URL = (
    "https://raw.githubusercontent.com/lm-sys/FastChat/main/"
    "fastchat/llm_judge/data/judge_prompts.jsonl"
)


def resolve_mtbench_question_file(config: Dict[str, Any]) -> Path:
    block_cfg = get_block_config(config, "mtbench")
    return resolve_existing_or_download_default_path(
        config,
        block_cfg.get("question_file", DEFAULT_MTBENCH_QUESTION_FILE),
        package_dir=PACKAGE_DIR,
        default_filename=DEFAULT_MTBENCH_QUESTION_FILE,
        download_url=DEFAULT_MTBENCH_QUESTION_URL,
    )


def resolve_mtbench_reference_answer_file(config: Dict[str, Any]) -> Path:
    block_cfg = get_block_config(config, "mtbench")
    return resolve_existing_or_download_default_path(
        config,
        block_cfg.get(
            "reference_answer_file",
            DEFAULT_MTBENCH_REFERENCE_ANSWER_FILE,
        ),
        package_dir=PACKAGE_DIR,
        default_filename=DEFAULT_MTBENCH_REFERENCE_ANSWER_FILE,
        download_url=DEFAULT_MTBENCH_REFERENCE_ANSWER_URL,
    )


def resolve_mtbench_judge_prompts_file(config: Dict[str, Any]) -> Path:
    block_cfg = get_block_config(config, "mtbench")
    return resolve_existing_or_download_default_path(
        config,
        block_cfg.get(
            "judge_prompts_file",
            DEFAULT_MTBENCH_JUDGE_PROMPTS_FILE,
        ),
        package_dir=PACKAGE_DIR,
        default_filename=DEFAULT_MTBENCH_JUDGE_PROMPTS_FILE,
        download_url=DEFAULT_MTBENCH_JUDGE_PROMPTS_URL,
    )


def get_fastchat_workspace(config: Dict[str, Any]) -> Path:
    output_dir = get_output_dir(config, "mtbench")
    workspace_dir = output_dir / "fastchat_workspace"
    (workspace_dir / "data" / FASTCHAT_BENCH_NAME).mkdir(parents=True, exist_ok=True)
    return workspace_dir


def stage_fastchat_question_file(config: Dict[str, Any]) -> Path:
    workspace_dir = get_fastchat_workspace(config)
    staged_question_path = workspace_dir / "data" / FASTCHAT_BENCH_NAME / "question.jsonl"
    source_path = resolve_mtbench_question_file(config)
    staged_question_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, staged_question_path)
    return staged_question_path


def stage_fastchat_model_answer_file(
    config: Dict[str, Any],
    *,
    pretty_name: str,
    model_answer_path: Path,
) -> Path:
    workspace_dir = get_fastchat_workspace(config)
    staged_answer_path = (
        workspace_dir / "data" / FASTCHAT_BENCH_NAME / "model_answer" / f"{pretty_name}.jsonl"
    )
    staged_answer_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_answer_path, staged_answer_path)
    return staged_answer_path


def stage_fastchat_reference_answer_file(
    config: Dict[str, Any],
    *,
    judge_model: str,
) -> Path:
    workspace_dir = get_fastchat_workspace(config)
    source_path = resolve_mtbench_reference_answer_file(config)
    staged_reference_path = (
        workspace_dir / "data" / FASTCHAT_BENCH_NAME / "reference_answer" / f"{judge_model}.jsonl"
    )
    staged_reference_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, staged_reference_path)
    return staged_reference_path


def stage_fastchat_judge_prompts_file(config: Dict[str, Any]) -> Path:
    workspace_dir = get_fastchat_workspace(config)
    source_path = resolve_mtbench_judge_prompts_file(config)
    staged_prompt_path = workspace_dir / "data" / "judge_prompts.jsonl"
    staged_prompt_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, staged_prompt_path)
    return staged_prompt_path


def resolve_mtbench_path(config: Dict[str, Any], value: Any) -> Path:
    return resolve_existing_path(config, value, package_dir=PACKAGE_DIR)
