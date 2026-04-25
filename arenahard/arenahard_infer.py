"""Arena-Hard v2 endpoint-based answer generation via the official runner."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from benchmark_common import get_output_dir, get_package_versions, get_pretty_name, write_json
from config_utils import load_yaml

from .official_runner import (
    ensure_official_runner,
    get_bench_name,
    get_model_answer_filename,
    get_runner_data_dir,
    safe_copy,
    stage_arenahard_data,
    write_official_inference_configs,
)


def _build_gen_answer_command() -> list[str]:
    return [
        sys.executable,
        "gen_answer.py",
        "--config-file",
        "config/gen_answer_config.yaml",
        "--endpoint-file",
        "config/api_config.yaml",
    ]


def run_arenahard_inference(config: Dict[str, Any]) -> Path:
    output_dir = get_output_dir(config, "arenahard")
    runner_dir = ensure_official_runner(config)
    staged_data = stage_arenahard_data(config, runner_dir)
    written_configs = write_official_inference_configs(config, runner_dir)
    runner_answer_path = (
        get_runner_data_dir(runner_dir, config)
        / "model_answer"
        / get_model_answer_filename(config)
    )
    if runner_answer_path.exists():
        runner_answer_path.unlink()

    command = _build_gen_answer_command()
    print("[ArenaHard] framework=official_arena_hard_auto")
    print(f"[ArenaHard] bench_name={get_bench_name(config)}")
    print(f"[ArenaHard] model={get_pretty_name(config, 'arenahard')}")
    print(f"[ArenaHard] question_file={staged_data['question_file']}")
    print(f"[ArenaHard] command={' '.join(shlex.quote(part) for part in command)}")
    subprocess.run(command, cwd=runner_dir, check=True)

    model_answer_path = safe_copy(runner_answer_path, output_dir / "model_answer.jsonl")
    metadata_path = output_dir / "metadata.json"
    write_json(
        metadata_path,
        {
            "framework": "official_arena_hard_auto",
            "bench_name": get_bench_name(config),
            "pretty_name": get_pretty_name(config, "arenahard"),
            "runner_dir": str(runner_dir),
            "staged_question_file": str(staged_data["question_file"]),
            "staged_baseline_answer_file": str(staged_data["baseline_answer_file"]),
            "runner_answer_file": str(runner_answer_path),
            "command": command,
            "written_configs": {key: str(path) for key, path in written_configs.items()},
            "package_versions": get_package_versions(
                ("openai", "pandas", "pyyaml", "shortuuid", "tiktoken", "tqdm")
            ),
        },
    )
    print(f"[ArenaHard] wrote_model_answer={model_answer_path}")
    print(f"[ArenaHard] wrote_metadata={metadata_path}")
    return model_answer_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Arena-Hard v2 answer generation")
    parser.add_argument(
        "--config",
        type=str,
        default="arenahard/config_arenahard.yaml",
    )
    args = parser.parse_args(argv)

    config = load_yaml(args.config)
    run_arenahard_inference(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
