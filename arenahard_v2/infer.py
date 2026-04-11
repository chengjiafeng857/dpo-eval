"""Inference pipeline for Arena-Hard v2.0."""

from __future__ import annotations

import argparse
import concurrent.futures
from pathlib import Path
from typing import Any, Dict, List, Sequence

from benchmark_common import (
    get_block_config,
    get_generation_config,
    get_model_name_or_path,
    get_package_versions,
    get_pretty_name,
)
from config_utils import load_yaml
from model_generation import (
    generate_with_transformers,
    generate_with_vllm,
    load_render_tokenizer,
    render_chat_prompts,
)

from . import __file__ as _PACKAGE_FILE
from .common import (
    BLOCK_NAME,
    build_answer_row,
    get_answer_path,
    get_metadata_path,
    load_endpoint_catalog_from_config,
    load_jsonl_map,
    load_questions,
    write_json,
    write_jsonl,
)
from .endpoint import EndpointPool, create_chat_completion, get_endpoint_settings


PACKAGE_DIR = Path(_PACKAGE_FILE).resolve().parent


def _generate_local_answers(
    config: Dict[str, Any],
    questions: Sequence[Dict[str, Any]],
) -> List[str]:
    backend = str(get_block_config(config, BLOCK_NAME).get("backend", "vllm")).lower()
    generation_cfg = get_generation_config(config, BLOCK_NAME)
    tokenizer = load_render_tokenizer(config, BLOCK_NAME)
    system_prompt = get_block_config(config, BLOCK_NAME).get("system_prompt")
    conversations = []
    for question in questions:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        messages.append({"role": "user", "content": str(question["prompt"])})
        conversations.append(messages)

    prompts, _ = render_chat_prompts(
        config,
        BLOCK_NAME,
        tokenizer=tokenizer,
        conversations=conversations,
        package_dir=PACKAGE_DIR,
    )

    if backend == "vllm":
        return generate_with_vllm(config, BLOCK_NAME, prompts, generation_cfg)
    if backend == "transformers":
        return generate_with_transformers(config, BLOCK_NAME, prompts, generation_cfg)
    raise ValueError("arenahard_v2.backend must be 'transformers' or 'vllm'.")


def _generate_endpoint_answers(
    config: Dict[str, Any],
    questions: Sequence[Dict[str, Any]],
    *,
    endpoint_file: str | None = None,
) -> List[str]:
    block_cfg = get_block_config(config, BLOCK_NAME)
    endpoint_catalog = load_endpoint_catalog_from_config(config, endpoint_file)
    endpoint_name = str(block_cfg.get("endpoint_name", get_pretty_name(config, BLOCK_NAME)))
    settings = get_endpoint_settings(endpoint_catalog, endpoint_name)
    pool = EndpointPool(settings.get("endpoints"))
    parallel = int(settings.get("parallel", 1))
    system_prompt = block_cfg.get("system_prompt") or settings.get("sys_prompt")

    outputs: list[str] = [""] * len(questions)

    def generate_one(index: int, question: Dict[str, Any]) -> tuple[int, str]:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        messages.append({"role": "user", "content": str(question["prompt"])})
        result = create_chat_completion(
            settings=settings,
            pool=pool,
            messages=messages,
            temperature=settings.get("temperature"),
            max_tokens=settings.get("max_tokens"),
        )
        return index, str(result["answer"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = [
            executor.submit(generate_one, index, question)
            for index, question in enumerate(questions)
        ]
        for future in concurrent.futures.as_completed(futures):
            index, output = future.result()
            outputs[index] = output
    return outputs


def run_arenahard_v2_inference(
    config: Dict[str, Any],
    *,
    endpoint_file: str | None = None,
) -> Path:
    questions = load_questions(config)
    answer_path = get_answer_path(config)
    metadata_path = get_metadata_path(config)
    existing = load_jsonl_map(answer_path, key_field="uid")
    pending_questions = [question for question in questions if str(question["uid"]) not in existing]

    block_cfg = get_block_config(config, BLOCK_NAME)
    mode = str(block_cfg.get("mode", "local")).lower()
    if pending_questions:
        if mode == "local":
            outputs = _generate_local_answers(config, pending_questions)
        elif mode == "endpoint":
            outputs = _generate_endpoint_answers(config, pending_questions, endpoint_file=endpoint_file)
        else:
            raise ValueError("arenahard_v2.mode must be 'local' or 'endpoint'.")
        system_prompt = block_cfg.get("system_prompt")
        model_name = get_pretty_name(config, BLOCK_NAME)
        for question, output in zip(pending_questions, outputs, strict=True):
            existing[str(question["uid"])] = build_answer_row(
                model_name=model_name,
                question=question,
                answer_text=output,
                system_prompt=str(system_prompt) if system_prompt else None,
            )

    ordered_rows = [existing[str(question["uid"])] for question in questions if str(question["uid"]) in existing]
    write_jsonl(answer_path, ordered_rows)
    write_json(
        metadata_path,
        {
            "bench_name": str(block_cfg.get("bench_name", "arena-hard-v2.0")),
            "mode": mode,
            "model_name_or_path": get_model_name_or_path(config, BLOCK_NAME),
            "pretty_name": get_pretty_name(config, BLOCK_NAME),
            "question_count": len(questions),
            "answer_file": str(answer_path),
            "question_file": str(Path(block_cfg.get("question_file", "question.jsonl"))),
            "package_versions": get_package_versions(("torch", "transformers", "vllm", "tiktoken", "openai")),
        },
    )
    print(f"[ArenaHardV2] model={get_pretty_name(config, BLOCK_NAME)}")
    print(f"[ArenaHardV2] mode={mode}")
    print(f"[ArenaHardV2] wrote_answer_file={answer_path}")
    return answer_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Arena-Hard v2.0 inference")
    parser.add_argument(
        "--config",
        type=str,
        default="arenahard_v2/config_arenahard_v2.yaml",
    )
    parser.add_argument(
        "--endpoint-file",
        type=str,
        default=None,
    )
    args = parser.parse_args(argv)
    config = load_yaml(args.config)
    run_arenahard_v2_inference(config, endpoint_file=args.endpoint_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

