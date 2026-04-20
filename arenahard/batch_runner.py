"""Batch orchestration for Arena-Hard v0.1 inference and judging."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List

from benchmark_common import get_pretty_name, sanitize_name
from config_utils import load_yaml

from .common import BLOCK_NAME, get_answer_path, get_judgment_path
from .infer import run_arenahard_inference
from .judge import DEFAULT_JUDGE_MODEL, run_arenahard_judging


def _resolve_config_path(config_path: str, base_dir: Path) -> Path:
    path = Path(config_path)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _deep_update(target: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def _apply_model_family_defaults(
    config: Dict[str, Any],
    *,
    model_name_or_path: str,
    pretty_name: str,
) -> None:
    block_cfg = config.setdefault(BLOCK_NAME, {})
    generation_cfg = block_cfg.setdefault("generation", {})
    model_name = model_name_or_path.lower()
    if "qwen3" in model_name:
        block_cfg["use_custom_chat_template"] = False
        block_cfg.pop("prompt_template", None)
        generation_cfg["stop_token_ids"] = [151645]
    elif "llama3" in model_name or "llama-3" in model_name:
        block_cfg["use_custom_chat_template"] = True
        block_cfg["prompt_template"] = (
            f"../mtbench/templates/{sanitize_name(pretty_name)}.jinja"
        )
        generation_cfg["stop_token_ids"] = [128001, 128009]


def _build_model_config(
    base_config: Dict[str, Any],
    *,
    batch_config: Dict[str, Any],
    model_entry: Dict[str, Any],
    config_path: Path,
) -> Dict[str, Any]:
    config = copy.deepcopy(base_config)
    batch_overrides = batch_config.get("overrides", {})
    if batch_overrides:
        _deep_update(config, batch_overrides)

    model_name_or_path = str(model_entry["model_name_or_path"])
    pretty_name = str(model_entry.get("pretty_name", model_name_or_path))
    config["policy_name"] = model_name_or_path
    block_cfg = config.setdefault(BLOCK_NAME, {})
    block_cfg["model_name_or_path"] = model_name_or_path
    block_cfg["pretty_name"] = pretty_name
    _apply_model_family_defaults(
        config,
        model_name_or_path=model_name_or_path,
        pretty_name=pretty_name,
    )

    model_overrides = model_entry.get("overrides", {})
    if model_overrides:
        _deep_update(config, model_overrides)

    config["_config_path"] = str(config_path)
    return config


def build_run_matrix(batch_config: Dict[str, Any], *, config_path: Path) -> List[Dict[str, Any]]:
    base_config_value = batch_config.get("base_config")
    if not base_config_value:
        raise ValueError("base_config is required.")
    base_config_path = _resolve_config_path(str(base_config_value), config_path.parent)
    base_config = load_yaml(str(base_config_path))

    model_entries = batch_config.get("models", [])
    if not isinstance(model_entries, list) or not model_entries:
        raise ValueError("models must be a non-empty list.")

    run_plans = []
    for model_entry in model_entries:
        config = _build_model_config(
            base_config,
            batch_config=batch_config,
            model_entry=model_entry,
            config_path=config_path,
        )
        run_plans.append(
            {
                "pretty_name": get_pretty_name(config, BLOCK_NAME),
                "config": config,
            }
        )
    return run_plans


def run_arenahard_batch(
    batch_config: Dict[str, Any],
    *,
    config_path: str,
    run_inference: bool | None = None,
    run_judging: bool | None = None,
) -> int:
    config_file = Path(config_path).resolve()
    run_plans = build_run_matrix(batch_config, config_path=config_file)
    do_inference = (
        bool(batch_config.get("run_inference", True))
        if run_inference is None
        else run_inference
    )
    do_judging = (
        bool(batch_config.get("run_judging", True))
        if run_judging is None
        else run_judging
    )
    if not do_inference and not do_judging:
        raise ValueError("At least one of run_inference or run_judging must be enabled.")

    skip_existing = bool(batch_config.get("skip_existing", True))
    continue_on_error = bool(batch_config.get("continue_on_error", False))
    failures: list[str] = []

    for run_index, run_plan in enumerate(run_plans, start=1):
        config = run_plan["config"]
        pretty_name = run_plan["pretty_name"]
        judge_model = str(config[BLOCK_NAME].get("judge_model", DEFAULT_JUDGE_MODEL))
        answer_path = get_answer_path(config)
        judgment_path = get_judgment_path(config, judge_model)
        print(f"[ArenaHard-BATCH] ({run_index}/{len(run_plans)}) model={pretty_name}")
        try:
            if do_inference:
                if skip_existing and answer_path.exists():
                    print(f"[ArenaHard-BATCH] skipping inference; existing_answers={answer_path}")
                else:
                    run_arenahard_inference(config)
            if do_judging:
                if skip_existing and judgment_path.exists():
                    print(
                        "[ArenaHard-BATCH] existing_judgments found; "
                        f"resuming_or_verifying={judgment_path}"
                    )
                run_arenahard_judging(config)
        except Exception as exc:
            message = f"{pretty_name}: {exc}"
            print(f"[ArenaHard-BATCH] failed: {message}")
            failures.append(message)
            if not continue_on_error:
                break

    if failures:
        print("[ArenaHard-BATCH] failed_models=")
        for message in failures:
            print(f"  - {message}")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Arena-Hard v0.1 for a model batch")
    parser.add_argument(
        "--config",
        type=str,
        default="arenahard/configs/config_arenahard_batch.yaml",
    )
    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument("--judging-only", action="store_true")
    args = parser.parse_args(argv)
    if args.inference_only and args.judging_only:
        raise ValueError("Choose at most one of --inference-only or --judging-only.")

    batch_config = load_yaml(args.config)
    run_inference = None if not args.judging_only else False
    run_judging = None if not args.inference_only else False
    return run_arenahard_batch(
        batch_config,
        config_path=args.config,
        run_inference=run_inference,
        run_judging=run_judging,
    )


if __name__ == "__main__":
    raise SystemExit(main())
