"""Judging pipeline for Arena-Hard v2.0."""

from __future__ import annotations

import argparse
import concurrent.futures
import re
from pathlib import Path
from typing import Any, Dict, List

from benchmark_common import get_block_config, get_pretty_name
from config_utils import load_yaml

from .common import (
    BLOCK_NAME,
    get_answer_dir,
    get_answer_path,
    get_judgment_path,
    load_endpoint_catalog_from_config,
    load_jsonl_map,
    load_model_answers,
    load_questions,
    write_jsonl,
)
from .endpoint import EndpointPool, create_chat_completion, get_endpoint_settings
from .judge_settings import DEFAULT_PROMPT_TEMPLATE, DEFAULT_REGEX_PATTERNS, JUDGE_SETTINGS


def _extract_answer_text(answer_row: Dict[str, Any]) -> str:
    messages = answer_row.get("messages", [])
    if not messages:
        return ""
    last_message = messages[-1]
    content = last_message.get("content", "")
    if isinstance(content, dict):
        return str(content.get("answer", ""))
    return str(content)


def _parse_score(judgment_text: str, regex_patterns: List[str]) -> str | None:
    for pattern_text in regex_patterns:
        pattern = re.compile(pattern_text)
        matches = pattern.findall(judgment_text.upper())
        matches = [match for match in matches if match]
        if matches:
            return str(matches[-1]).strip()
    return None


def _pairwise_judgment(
    *,
    question: Dict[str, Any],
    baseline_row: Dict[str, Any],
    answer_row: Dict[str, Any],
    settings: Dict[str, Any],
    pool: EndpointPool,
    prompt_template: str,
    regex_patterns: List[str],
) -> Dict[str, Any] | None:
    category = str(question["category"])
    prompt = prompt_template.format(
        QUESTION=str(question["prompt"]),
        ANSWER_A=_extract_answer_text(baseline_row),
        ANSWER_B=_extract_answer_text(answer_row),
    )
    messages = [
        {
            "role": "system",
            "content": JUDGE_SETTINGS[category]["system_prompt"],
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]
    result = create_chat_completion(
        settings=settings,
        pool=pool,
        messages=messages,
        temperature=float(settings.get("temperature", 0.0)),
        max_tokens=int(settings.get("max_tokens", 16000)),
    )
    return {
        "score": _parse_score(str(result["answer"]), regex_patterns),
        "judgment": result,
        "prompt": messages,
    }


def run_arenahard_v2_judging(
    config: Dict[str, Any],
    *,
    endpoint_file: str | None = None,
) -> Path:
    block_cfg = get_block_config(config, BLOCK_NAME)
    questions = load_questions(config)
    target_model = get_pretty_name(config, BLOCK_NAME)
    answer_path = get_answer_path(config)
    if not answer_path.exists():
        raise FileNotFoundError(
            "Could not find target model answers. Run arenahard-v2-infer first."
        )

    judge_model = str(block_cfg.get("judge_model", "gpt-4.1"))
    judge_endpoint_name = str(block_cfg.get("judge_endpoint_name", judge_model))
    prompt_template = str(block_cfg.get("prompt_template", DEFAULT_PROMPT_TEMPLATE))
    regex_patterns = [
        str(pattern)
        for pattern in block_cfg.get("regex_patterns", DEFAULT_REGEX_PATTERNS)
    ]

    endpoint_catalog = load_endpoint_catalog_from_config(config, endpoint_file)
    judge_settings = get_endpoint_settings(endpoint_catalog, judge_endpoint_name)
    judge_settings = dict(judge_settings)
    judge_settings["temperature"] = float(block_cfg.get("judge_temperature", judge_settings.get("temperature", 0.0)))
    judge_settings["max_tokens"] = int(block_cfg.get("judge_max_tokens", judge_settings.get("max_tokens", 16000)))
    pool = EndpointPool(judge_settings.get("endpoints"))

    answers_by_model = load_model_answers(get_answer_dir(config))
    if target_model not in answers_by_model:
        raise FileNotFoundError(f"Answers for target model '{target_model}' are missing.")

    missing_baselines = sorted(
        {
            JUDGE_SETTINGS[str(question["category"])]["baseline"]
            for question in questions
            if JUDGE_SETTINGS[str(question["category"])]["baseline"] not in answers_by_model
            and JUDGE_SETTINGS[str(question["category"])]["baseline"] != target_model
        }
    )
    if missing_baselines:
        raise FileNotFoundError(
            "Missing baseline answer files required for Arena-Hard v2.0 judging: "
            + ", ".join(missing_baselines)
        )

    output_path = get_judgment_path(config, judge_model)
    existing = load_jsonl_map(output_path, key_field="uid")
    if target_model in {
        JUDGE_SETTINGS[str(question["category"])]["baseline"] for question in questions
    } and all(
        JUDGE_SETTINGS[str(question["category"])]["baseline"] == target_model for question in questions
    ):
        write_jsonl(output_path, [])
        return output_path

    def judge_one(question: Dict[str, Any]) -> Dict[str, Any]:
        category = str(question["category"])
        uid = str(question["uid"])
        baseline_model = str(JUDGE_SETTINGS[category]["baseline"])
        answer_row = answers_by_model[target_model][uid]
        baseline_row = answers_by_model[baseline_model][uid]
        round_one = _pairwise_judgment(
            question=question,
            baseline_row=baseline_row,
            answer_row=answer_row,
            settings=judge_settings,
            pool=pool,
            prompt_template=prompt_template,
            regex_patterns=regex_patterns,
        )
        round_two = _pairwise_judgment(
            question=question,
            baseline_row=answer_row,
            answer_row=baseline_row,
            settings=judge_settings,
            pool=pool,
            prompt_template=prompt_template,
            regex_patterns=regex_patterns,
        )
        return {
            "uid": uid,
            "category": category,
            "judge": judge_model,
            "model": target_model,
            "baseline": baseline_model,
            "games": [round_one, round_two],
        }

    pending_questions = [
        question for question in questions
        if str(question["uid"]) not in existing
        and get_pretty_name(config, BLOCK_NAME) != JUDGE_SETTINGS[str(question["category"])]["baseline"]
    ]
    if pending_questions:
        parallel = int(judge_settings.get("parallel", 1))
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = [executor.submit(judge_one, question) for question in pending_questions]
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                existing[str(row["uid"])] = row

    ordered = [existing[str(question["uid"])] for question in questions if str(question["uid"]) in existing]
    write_jsonl(output_path, ordered)
    print(f"[ArenaHardV2] judge={judge_model}")
    print(f"[ArenaHardV2] wrote_judgment_file={output_path}")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Arena-Hard v2.0 judging")
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
    run_arenahard_v2_judging(config, endpoint_file=args.endpoint_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

