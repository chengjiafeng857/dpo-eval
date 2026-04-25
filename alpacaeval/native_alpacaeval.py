"""Run AlpacaEval's native model-generation path from repo configs."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

from config_utils import load_yaml, resolve_torch_dtype
from .alpacaeval_common import (
    DEFAULT_ALPACA_EVAL_ANNOTATOR,
    get_alpacaeval_config,
    get_generation_config,
    get_model_name_or_path,
    get_output_dir,
    get_pretty_name,
    load_prompt_template,
    resolve_path,
    sanitize_name,
    use_custom_chat_template,
)


def _torch_dtype_name(config: Dict[str, Any]) -> str | None:
    torch_dtype = resolve_torch_dtype(config.get("precision", "fp32"))
    if torch_dtype is None:
        return None
    return str(torch_dtype).replace("torch.", "")


def build_native_model_config(config: Dict[str, Any]) -> Path:
    """Write an AlpacaEval model config for evaluate_from_model."""
    if not use_custom_chat_template(config):
        raise ValueError(
            "AlpacaEval's native local decoders require a prompt_template. "
            "Set alpacaeval.use_custom_chat_template=true and provide "
            "alpacaeval.prompt_template."
        )

    alpacaeval_cfg = get_alpacaeval_config(config)
    generation_cfg = get_generation_config(config)
    backend = str(alpacaeval_cfg.get("backend", "transformers")).lower()
    output_dir = get_output_dir(config)
    prompt_template_path, _ = load_prompt_template(config)

    if backend == "vllm":
        fn_completions = "vllm_local_completions"
    elif backend == "transformers":
        fn_completions = "huggingface_local_completions"
    else:
        raise ValueError("alpacaeval.backend must be 'transformers' or 'vllm'.")

    completions_kwargs: Dict[str, Any] = {
        "model_name": get_model_name_or_path(config),
        "max_new_tokens": int(generation_cfg.get("max_new_tokens", 1024)),
        "temperature": float(generation_cfg.get("temperature", 0.0)),
        "top_p": float(generation_cfg.get("top_p", 1.0)),
        "do_sample": bool(generation_cfg.get("do_sample", False)),
        "batch_size": int(generation_cfg.get("batch_size", 1)),
    }

    stop_token_ids = generation_cfg.get("stop_token_ids")
    if stop_token_ids:
        completions_kwargs["stop_token_ids"] = [
            int(token_id) for token_id in stop_token_ids
        ]

    model_kwargs: Dict[str, Any] = {}
    torch_dtype_name = _torch_dtype_name(config)
    if torch_dtype_name is not None:
        model_kwargs["torch_dtype"] = torch_dtype_name

    if backend == "vllm":
        vllm_cfg = alpacaeval_cfg.get("vllm", {})
        if not isinstance(vllm_cfg, dict):
            raise ValueError("alpacaeval.vllm must be a mapping.")
        model_kwargs["tp"] = int(vllm_cfg.get("tensor_parallel_size", 1))
        model_kwargs["trust_remote_code"] = bool(
            vllm_cfg.get("trust_remote_code", False)
        )
        tokenizer_mode = vllm_cfg.get("tokenizer_mode")
        if tokenizer_mode is not None:
            model_kwargs["tokenizer_mode"] = str(tokenizer_mode)
    else:
        transformers_cfg = alpacaeval_cfg.get("transformers", {})
        if not isinstance(transformers_cfg, dict):
            raise ValueError("alpacaeval.transformers must be a mapping.")
        trust_remote_code = bool(transformers_cfg.get("trust_remote_code", False))
        model_kwargs["trust_remote_code"] = trust_remote_code
        device_map = transformers_cfg.get("device_map")
        if device_map is not None:
            model_kwargs["device_map"] = device_map

    if model_kwargs:
        completions_kwargs["model_kwargs"] = model_kwargs

    model_key = sanitize_name(get_pretty_name(config))
    payload = {
        model_key: {
            "prompt_template": str(prompt_template_path),
            "fn_completions": fn_completions,
            "completions_kwargs": completions_kwargs,
            "pretty_name": get_pretty_name(config),
        }
    }

    config_path = output_dir / "alpacaeval_native_model_config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def build_native_command(
    config: Dict[str, Any],
    *,
    model_config_path: Path,
    max_instances: int | None = None,
    chunksize: int | None = None,
    is_load_outputs: bool | None = None,
) -> list[str]:
    alpacaeval_cfg = get_alpacaeval_config(config)
    output_dir = get_output_dir(config)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "alpaca_eval.main",
        "evaluate_from_model",
        "--model_configs",
        str(model_config_path),
        "--annotators_config",
        str(alpacaeval_cfg.get("annotators_config", DEFAULT_ALPACA_EVAL_ANNOTATOR)),
        "--output_path",
        str(results_dir),
    ]

    reference_model_configs = alpacaeval_cfg.get("reference_model_configs")
    if reference_model_configs:
        command.extend(
            ["--reference_model_configs", str(resolve_path(config, reference_model_configs))]
        )
    if max_instances is not None:
        command.extend(["--max_instances", str(max_instances)])
    if chunksize is not None:
        command.extend(["--chunksize", str(chunksize)])
    if is_load_outputs is not None:
        command.extend(["--is_load_outputs", str(is_load_outputs)])
    return command


def run_native_generation_only(
    config: Dict[str, Any],
    *,
    model_config_path: Path,
    max_instances: int | None = None,
    chunksize: int | None = 64,
    is_load_outputs: bool = True,
) -> Path:
    """Generate model_outputs.json with AlpacaEval prompting and decoders."""
    import pandas as pd

    from alpaca_eval import constants, decoders, utils

    df_dataset = utils.load_or_convert_to_dataframe(constants.ALPACAEVAL_REFERENCE_OUTPUTS)
    if chunksize is not None and max_instances is not None:
        chunksize = None
    if max_instances is not None:
        df_dataset = df_dataset.iloc[:max_instances]

    model_configs = utils.load_configs(
        model_config_path,
        relative_to=constants.MODELS_CONFIG_DIR,
    )
    if len(model_configs) != 1:
        raise ValueError("AlpacaEval native generation expects exactly one model config.")

    generator = next(iter(model_configs.keys()))
    model_config = next(iter(model_configs.values()))
    output_path = get_output_dir(config) / "model_outputs.json"

    for df_chunk in utils.dataframe_chunk_generator(
        df_dataset,
        chunksize=chunksize,
        tqdm_desc="Chunking for native generation",
    ):
        columns_to_keep = ["dataset", "instruction", "output", "generator"]
        columns_to_keep = [column for column in columns_to_keep if column in df_chunk.columns]
        curr_outputs = df_chunk[columns_to_keep].copy()

        old_outputs = None
        if is_load_outputs and output_path.exists():
            old_outputs = utils.load_or_convert_to_dataframe(output_path)
            found_old_outputs = curr_outputs["instruction"].isin(old_outputs["instruction"])
            curr_outputs = curr_outputs[~found_old_outputs]

        if len(curr_outputs) > 0:
            prompt_template = Path(model_config["prompt_template"])
            if not prompt_template.is_absolute():
                prompt_template = constants.MODELS_CONFIG_DIR / prompt_template
            prompts, _ = utils.make_prompts(
                curr_outputs,
                template=utils.read_or_return(prompt_template),
            )
            fn_completions = decoders.get_fn_completions(model_config["fn_completions"])
            completions = fn_completions(
                prompts=prompts,
                **model_config["completions_kwargs"],
            )["completions"]
            curr_outputs["output"] = [completion.strip() for completion in completions]
            curr_outputs["generator"] = generator

        if old_outputs is not None:
            curr_outputs = pd.concat([old_outputs, curr_outputs], axis=0)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        curr_outputs.to_json(output_path, orient="records", indent=2)

    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run AlpacaEval's native evaluate_from_model pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="alpacaeval/config_alpacaeval.yaml",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--generation-only",
        action="store_true",
        help="Use AlpacaEval prompt/decoder code and stop after model_outputs.json.",
    )
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--chunksize", type=int, default=None)
    parser.add_argument(
        "--no-load-outputs",
        action="store_true",
        help="Pass --is_load_outputs False to AlpacaEval.",
    )
    args = parser.parse_args(argv)

    config = load_yaml(args.config)
    model_config_path = build_native_model_config(config)
    command = build_native_command(
        config,
        model_config_path=model_config_path,
        max_instances=args.max_instances,
        chunksize=args.chunksize,
        is_load_outputs=False if args.no_load_outputs else None,
    )

    print(f"[AlpacaEval-native] model_config={model_config_path}")
    print(
        "[AlpacaEval-native] command="
        + " ".join(shlex.quote(part) for part in command)
    )
    if args.dry_run:
        return 0
    if args.generation_only:
        output_path = run_native_generation_only(
            config,
            model_config_path=model_config_path,
            max_instances=args.max_instances,
            chunksize=args.chunksize,
            is_load_outputs=not args.no_load_outputs,
        )
        print(f"[AlpacaEval-native] wrote_model_outputs={output_path}")
        return 0

    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
