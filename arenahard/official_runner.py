"""Helpers for staging and running the official Arena-Hard-Auto v2 runner."""

from __future__ import annotations

import os
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from benchmark_common import (
    get_block_config,
    get_output_dir,
    get_pretty_name,
    read_jsonl,
    resolve_existing_path,
    sanitize_name,
    write_jsonl,
)

from . import __file__ as _PACKAGE_FILE


PACKAGE_DIR = Path(_PACKAGE_FILE).resolve().parent

DEFAULT_OFFICIAL_REPO = "https://github.com/lmarena/arena-hard-auto.git"
DEFAULT_OFFICIAL_REF = "196f6b826783b3da7310e361a805fa36f0be83f3"
DEFAULT_BENCH_NAME = "arena-hard-v2.0"
DEFAULT_JUDGE_MODEL = "gpt-4.1"
DEFAULT_CATEGORIES = ["hard_prompt"]
DEFAULT_BASELINE_MODEL = "o3-mini-2025-01-31"
DEFAULT_QUESTION_URL = (
    "https://huggingface.co/datasets/lmarena-ai/arena-hard-auto/resolve/main/"
    "data/arena-hard-v2.0/question.jsonl"
)
DEFAULT_BASELINE_ANSWER_URL = (
    "https://huggingface.co/datasets/lmarena-ai/arena-hard-auto/resolve/main/"
    "data/arena-hard-v2.0/model_answer/o3-mini-2025-01-31.jsonl"
)
DEFAULT_REGEX_PATTERNS = [r"\[\[([AB<>=]+)\]\]", r"\[([AB<>=]+)\]"]
DEFAULT_PROMPT_TEMPLATE = (
    "<|User Prompt|>\n"
    "{QUESTION}\n\n"
    "<|The Start of Assistant A's Answer|>\n"
    "{ANSWER_A}\n"
    "<|The End of Assistant A's Answer|>\n\n"
    "<|The Start of Assistant B's Answer|>\n"
    "{ANSWER_B}\n"
    "<|The End of Assistant B's Answer|>"
)


def get_arenahard_output_dir(config: Dict[str, Any]) -> Path:
    return get_output_dir(config, "arenahard")


def get_official_runner_dir(config: Dict[str, Any]) -> Path:
    block_cfg = get_block_config(config, "arenahard")
    runner_dir = block_cfg.get("official_runner_dir")
    if runner_dir:
        return resolve_existing_path(config, runner_dir, package_dir=PACKAGE_DIR)
    return get_arenahard_output_dir(config) / "official_runner"


def _run_git(command: list[str], *, cwd: Path | None = None) -> None:
    print(f"[ArenaHard] git_command={' '.join(command)}")
    subprocess.run(command, cwd=cwd, check=True)


def ensure_official_runner(config: Dict[str, Any]) -> Path:
    block_cfg = get_block_config(config, "arenahard")
    runner_dir = get_official_runner_dir(config)
    repo_url = str(block_cfg.get("official_repo", DEFAULT_OFFICIAL_REPO))
    official_ref = str(block_cfg.get("official_ref", DEFAULT_OFFICIAL_REF))

    if (runner_dir / ".git").exists():
        _run_git(["git", "fetch", "origin"], cwd=runner_dir)
    else:
        runner_dir.parent.mkdir(parents=True, exist_ok=True)
        _run_git(["git", "clone", repo_url, str(runner_dir)])

    _run_git(["git", "checkout", "--detach", official_ref], cwd=runner_dir)
    return runner_dir


def get_bench_name(config: Dict[str, Any]) -> str:
    block_cfg = get_block_config(config, "arenahard")
    return str(block_cfg.get("bench_name", DEFAULT_BENCH_NAME))


def get_categories(config: Dict[str, Any]) -> list[str]:
    block_cfg = get_block_config(config, "arenahard")
    categories = block_cfg.get("categories", DEFAULT_CATEGORIES)
    if not isinstance(categories, list) or not categories:
        raise ValueError("arenahard.categories must be a non-empty list.")
    return [str(category) for category in categories]


def get_judge_model(config: Dict[str, Any]) -> str:
    block_cfg = get_block_config(config, "arenahard")
    return str(block_cfg.get("judge_model", DEFAULT_JUDGE_MODEL))


def get_baseline_model(config: Dict[str, Any]) -> str:
    block_cfg = get_block_config(config, "arenahard")
    return str(block_cfg.get("baseline_model", DEFAULT_BASELINE_MODEL))


def get_model_answer_filename(config: Dict[str, Any]) -> str:
    return f"{get_pretty_name(config, 'arenahard')}.jsonl"


def get_judgment_filename(config: Dict[str, Any]) -> str:
    return f"{get_pretty_name(config, 'arenahard')}.jsonl"


def filter_questions_by_category(
    rows: Iterable[Dict[str, Any]],
    categories: Iterable[str],
) -> List[Dict[str, Any]]:
    category_set = {str(category) for category in categories}
    filtered = [row for row in rows if str(row.get("category")) in category_set]
    if not filtered:
        raise ValueError(
            "Arena-Hard question filtering produced zero rows for categories: "
            f"{sorted(category_set)}"
        )
    return filtered


def _download_if_needed(url: str, target: Path) -> Path:
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        target.write_bytes(response.read())
    return target


def _resolve_resource(
    config: Dict[str, Any],
    *,
    field_name: str,
    default_url: str,
    default_filename: str,
) -> Path:
    block_cfg = get_block_config(config, "arenahard")
    value = block_cfg.get(field_name)
    if value:
        resolved = resolve_existing_path(config, value, package_dir=PACKAGE_DIR)
        if not resolved.exists():
            raise FileNotFoundError(f"Could not find arenahard.{field_name}: {resolved}")
        return resolved

    downloads_dir = get_arenahard_output_dir(config) / "downloads"
    return _download_if_needed(default_url, downloads_dir / default_filename)


def resolve_question_file(config: Dict[str, Any]) -> Path:
    return _resolve_resource(
        config,
        field_name="question_file",
        default_url=str(
            get_block_config(config, "arenahard").get("question_url", DEFAULT_QUESTION_URL)
        ),
        default_filename="question.jsonl",
    )


def resolve_baseline_answer_file(config: Dict[str, Any]) -> Path:
    baseline_model = get_baseline_model(config)
    return _resolve_resource(
        config,
        field_name="baseline_answer_file",
        default_url=str(
            get_block_config(config, "arenahard").get(
                "baseline_answer_url",
                DEFAULT_BASELINE_ANSWER_URL,
            )
        ),
        default_filename=f"{baseline_model}.jsonl",
    )


def get_runner_data_dir(runner_dir: Path, config: Dict[str, Any]) -> Path:
    return runner_dir / "data" / get_bench_name(config)


def stage_arenahard_data(config: Dict[str, Any], runner_dir: Path) -> Dict[str, Path]:
    data_dir = get_runner_data_dir(runner_dir, config)
    question_target = data_dir / "question.jsonl"
    answer_dir = data_dir / "model_answer"
    baseline_target = answer_dir / f"{get_baseline_model(config)}.jsonl"

    questions = read_jsonl(resolve_question_file(config))
    filtered_questions = filter_questions_by_category(questions, get_categories(config))
    write_jsonl(question_target, filtered_questions)

    answer_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolve_baseline_answer_file(config), baseline_target)

    return {
        "question_file": question_target,
        "baseline_answer_file": baseline_target,
    }


def _copy_endpoint_config(endpoint_cfg: Any, *, field_name: str) -> Dict[str, Any]:
    if not isinstance(endpoint_cfg, dict):
        raise ValueError(f"arenahard.{field_name} must be a mapping.")
    return dict(endpoint_cfg)


def build_api_config(
    config: Dict[str, Any],
    *,
    include_model: bool = True,
    include_judge: bool = True,
) -> Dict[str, Any]:
    block_cfg = get_block_config(config, "arenahard")
    api_config: Dict[str, Any] = {}

    if include_model:
        model_endpoint = block_cfg.get("model_endpoint")
        if model_endpoint is None:
            raise ValueError("arenahard.model_endpoint is required for Arena-Hard inference.")
        api_config[get_pretty_name(config, "arenahard")] = _copy_endpoint_config(
            model_endpoint,
            field_name="model_endpoint",
        )

    if include_judge:
        judge_endpoint = block_cfg.get(
            "judge_endpoint",
            {
                "model": get_judge_model(config),
                "endpoints": None,
                "api_type": "openai",
                "parallel": 64,
                "max_tokens": 32000,
                "temperature": 0.0,
            },
        )
        api_config[get_judge_model(config)] = _copy_endpoint_config(
            judge_endpoint,
            field_name="judge_endpoint",
        )

    return api_config


def build_gen_answer_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "bench_name": get_bench_name(config),
        "model_list": [get_pretty_name(config, "arenahard")],
    }


def build_judgment_config(config: Dict[str, Any]) -> Dict[str, Any]:
    block_cfg = get_block_config(config, "arenahard")
    return {
        "judge_model": get_judge_model(config),
        "temperature": float(block_cfg.get("judge_temperature", 0.0)),
        "max_tokens": int(block_cfg.get("judge_max_tokens", 16000)),
        "bench_name": get_bench_name(config),
        "reference": block_cfg.get("reference"),
        "regex_patterns": block_cfg.get("regex_patterns", DEFAULT_REGEX_PATTERNS),
        "prompt_template": block_cfg.get("prompt_template", DEFAULT_PROMPT_TEMPLATE),
        "model_list": [get_pretty_name(config, "arenahard")],
    }


def write_yaml(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def write_official_inference_configs(config: Dict[str, Any], runner_dir: Path) -> Dict[str, Path]:
    return {
        "gen_answer_config": write_yaml(
            runner_dir / "config" / "gen_answer_config.yaml",
            build_gen_answer_config(config),
        ),
        "api_config": write_yaml(
            runner_dir / "config" / "api_config.yaml",
            build_api_config(config, include_model=True, include_judge=True),
        ),
    }


def write_official_evaluation_configs(config: Dict[str, Any], runner_dir: Path) -> Dict[str, Path]:
    return {
        "judgment_config": write_yaml(
            runner_dir / "config" / "arena-hard-v2.0.yaml",
            build_judgment_config(config),
        ),
        "api_config": write_yaml(
            runner_dir / "config" / "api_config.yaml",
            build_api_config(config, include_model=False, include_judge=True),
        ),
    }


def safe_copy(src: Path, dst: Path) -> Path:
    if not src.exists():
        raise FileNotFoundError(f"Expected Arena-Hard artifact not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def model_output_dir(config: Dict[str, Any]) -> Path:
    return Path("../outputs/arenahard") / sanitize_name(get_pretty_name(config, "arenahard"))
