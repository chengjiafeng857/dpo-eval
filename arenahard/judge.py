"""Judging pipeline for Arena-Hard v0.1."""

from __future__ import annotations

import argparse
import concurrent.futures
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from benchmark_common import get_block_config, get_pretty_name
from config_utils import load_yaml
from tqdm import tqdm

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
from .endpoint import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    EndpointPool,
    create_chat_completion,
    get_endpoint_settings,
)
from .judge_settings import DEFAULT_PROMPT_TEMPLATE, DEFAULT_REGEX_PATTERNS, JUDGE_SETTINGS


DEFAULT_JUDGE_MODEL = "gpt-4-1106-preview"


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


def _preview_text(text: str, *, limit: int = 240) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _build_verdict_repair_messages(
    *,
    category: str,
    judgment_text: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are repairing a judge output format. "
                "Return exactly one label and nothing else: "
                "[[A>>B]], [[A>B]], [[A=B]], [[B>A]], or [[B>>A]]."
            ),
        },
        {
            "role": "user",
            "content": (
                "The previous Arena-Hard judge analysis did not include a parseable final label. "
                "Based only on the analysis below, return exactly one verdict label.\n\n"
                f"Category: {category}\n"
                "Analysis:\n"
                f"{judgment_text}"
            ),
        },
    ]


def _repair_score(
    *,
    question: Dict[str, Any],
    settings: Dict[str, Any],
    pool: EndpointPool,
    regex_patterns: List[str],
    judgment_text: str,
    round_label: str,
) -> tuple[str | None, Dict[str, Any] | None]:
    repair_messages = _build_verdict_repair_messages(
        category=str(question["category"]),
        judgment_text=judgment_text,
    )
    repair_result = create_chat_completion(
        settings=settings,
        pool=pool,
        messages=repair_messages,
        temperature=0.0,
        max_tokens=32,
    )
    repair_score = _parse_score(str(repair_result["answer"]), regex_patterns)
    if repair_score is None:
        raise ValueError(
            "Judge verdict could not be repaired "
            f"for uid={question['uid']} round={round_label} "
            f"analysis_excerpt={_preview_text(judgment_text)} "
            f"repair_excerpt={_preview_text(str(repair_result['answer']))}"
        )
    return repair_score, {"prompt": repair_messages, "judgment": repair_result}


def _pairwise_judgment(
    *,
    question: Dict[str, Any],
    baseline_row: Dict[str, Any],
    answer_row: Dict[str, Any],
    settings: Dict[str, Any],
    pool: EndpointPool,
    prompt_template: str,
    regex_patterns: List[str],
    round_label: str,
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
        max_tokens=int(settings.get("max_tokens", 4096)),
    )
    score = _parse_score(str(result["answer"]), regex_patterns)
    if score is None:
        score, repair = _repair_score(
            question=question,
            settings=settings,
            pool=pool,
            regex_patterns=regex_patterns,
            judgment_text=str(result["answer"]),
            round_label=round_label,
        )
    else:
        repair = None
    return {
        "score": score,
        "judgment": result,
        "prompt": messages,
        "repair": repair,
    }


def _format_judge_failure(
    *,
    question: Dict[str, Any],
    target_model: str,
    judge_model: str,
) -> str:
    category = str(question["category"])
    baseline_model = str(JUDGE_SETTINGS[category]["baseline"])
    return (
        "[ArenaHard][judge-error] "
        f"uid={question['uid']} "
        f"category={category} "
        f"model={target_model} "
        f"baseline={baseline_model} "
        f"judge={judge_model}"
    )


def _ordered_rows(
    *,
    questions: List[Dict[str, Any]],
    existing: Dict[str, Dict[str, Any]],
) -> list[Dict[str, Any]]:
    return [
        existing[str(question["uid"])]
        for question in questions
        if str(question["uid"]) in existing
    ]


def _is_complete_judgment_row(row: Dict[str, Any]) -> bool:
    games = row.get("games", [])
    if len(games) < 2:
        return False
    for game in games[:2]:
        if not isinstance(game, dict):
            return False
        if game.get("score") is None:
            return False
    return True


def _collect_completed_rows(
    *,
    future_to_question: Dict[concurrent.futures.Future[Dict[str, Any]], Dict[str, Any]],
    existing: Dict[str, Dict[str, Any]],
) -> int:
    collected = 0
    for future in future_to_question:
        if not future.done() or future.cancelled():
            continue
        exception = future.exception()
        if exception is not None:
            continue
        row = future.result()
        uid = str(row["uid"])
        if uid in existing:
            continue
        existing[uid] = row
        collected += 1
    return collected


def _validate_answer_coverage(
    *,
    questions: List[Dict[str, Any]],
    answers_by_model: Dict[str, Dict[str, Dict[str, Any]]],
    target_model: str,
) -> None:
    missing_rows: list[str] = []
    target_answers = answers_by_model.get(target_model, {})
    for question in questions:
        uid = str(question["uid"])
        category = str(question["category"])
        baseline_model = str(JUDGE_SETTINGS[category]["baseline"])
        if uid not in target_answers:
            missing_rows.append(f"uid={uid} model={target_model}")
        baseline_answers = answers_by_model.get(baseline_model, {})
        if baseline_model != target_model and uid not in baseline_answers:
            missing_rows.append(f"uid={uid} model={baseline_model}")
    if missing_rows:
        preview = ", ".join(missing_rows[:8])
        remainder = len(missing_rows) - min(len(missing_rows), 8)
        if remainder > 0:
            preview += f", ... (+{remainder} more)"
        raise FileNotFoundError(
            "Missing per-question answer rows required for Arena-Hard v0.1 judging: "
            + preview
        )


def run_arenahard_judging(
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
            "Could not find target model answers. Run arenahard-infer first."
        )

    judge_model = str(block_cfg.get("judge_model", DEFAULT_JUDGE_MODEL))
    judge_endpoint_name = str(block_cfg.get("judge_endpoint_name", judge_model))
    prompt_template = str(
        block_cfg.get("judge_prompt_template", DEFAULT_PROMPT_TEMPLATE)
    )
    regex_patterns = [
        str(pattern)
        for pattern in block_cfg.get("regex_patterns", DEFAULT_REGEX_PATTERNS)
    ]

    endpoint_catalog = load_endpoint_catalog_from_config(config, endpoint_file)
    judge_settings = get_endpoint_settings(endpoint_catalog, judge_endpoint_name)
    judge_settings = dict(judge_settings)
    judge_settings["temperature"] = float(block_cfg.get("judge_temperature", judge_settings.get("temperature", 0.0)))
    judge_settings["max_tokens"] = int(block_cfg.get("judge_max_tokens", judge_settings.get("max_tokens", 4096)))
    judge_settings["timeout"] = float(
        block_cfg.get("judge_timeout", judge_settings.get("timeout", DEFAULT_REQUEST_TIMEOUT_SECONDS))
    )
    judge_settings["max_retries"] = int(
        block_cfg.get("judge_max_retries", judge_settings.get("max_retries", 0))
    )
    judge_settings["initial_backoff"] = float(
        block_cfg.get("judge_initial_backoff", judge_settings.get("initial_backoff", 1.0))
    )
    judge_settings["max_backoff"] = float(
        block_cfg.get("judge_max_backoff", judge_settings.get("max_backoff", 60.0))
    )
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
            "Missing baseline answer files required for Arena-Hard v0.1 judging: "
            + ", ".join(missing_baselines)
        )
    _validate_answer_coverage(
        questions=questions,
        answers_by_model=answers_by_model,
        target_model=target_model,
    )

    output_path = get_judgment_path(config, judge_model)
    raw_existing = load_jsonl_map(output_path, key_field="uid")
    existing = {
        uid: row for uid, row in raw_existing.items() if _is_complete_judgment_row(row)
    }
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
            round_label="baseline_vs_candidate",
        )
        round_two = _pairwise_judgment(
            question=question,
            baseline_row=answer_row,
            answer_row=baseline_row,
            settings=judge_settings,
            pool=pool,
            prompt_template=prompt_template,
            regex_patterns=regex_patterns,
            round_label="candidate_vs_baseline",
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
        parallel = max(1, int(block_cfg.get("judge_parallel", judge_settings.get("parallel", 1))))
        checkpoint_every = max(1, int(block_cfg.get("judge_checkpoint_every", 1)))
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=parallel)
        future_to_question = {
            executor.submit(judge_one, question): question for question in pending_questions
        }
        progress = tqdm(total=len(future_to_question), desc=f"Judging {target_model}", unit="question")
        wait_for_shutdown = True
        completed_since_checkpoint = 0
        try:
            for future in concurrent.futures.as_completed(future_to_question):
                question = future_to_question[future]
                try:
                    row = future.result()
                except Exception as exc:
                    completed_since_checkpoint += _collect_completed_rows(
                        future_to_question=future_to_question,
                        existing=existing,
                    )
                    if completed_since_checkpoint > 0:
                        write_jsonl(
                            output_path,
                            _ordered_rows(questions=questions, existing=existing),
                        )
                        completed_since_checkpoint = 0
                    failure_context = _format_judge_failure(
                        question=question,
                        target_model=target_model,
                        judge_model=judge_model,
                    )
                    print(
                        f"{failure_context} error={type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    wait_for_shutdown = False
                    raise RuntimeError(
                        f"Arena-Hard v0.1 judging aborted. {failure_context}"
                    ) from exc
                existing[str(row["uid"])] = row
                progress.update(1)
                completed_since_checkpoint += 1
                if completed_since_checkpoint >= checkpoint_every:
                    write_jsonl(
                        output_path,
                        _ordered_rows(questions=questions, existing=existing),
                    )
                    completed_since_checkpoint = 0
        finally:
            progress.close()
            executor.shutdown(wait=wait_for_shutdown, cancel_futures=True)

    ordered = _ordered_rows(questions=questions, existing=existing)
    write_jsonl(output_path, ordered)
    print(f"[ArenaHard] judge={judge_model}")
    print(f"[ArenaHard] wrote_judgment_file={output_path}")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Arena-Hard v0.1 judging")
    parser.add_argument(
        "--config",
        type=str,
        default="arenahard/configs/config_arenahard.yaml",
    )
    parser.add_argument(
        "--endpoint-file",
        type=str,
        default=None,
    )
    args = parser.parse_args(argv)
    config = load_yaml(args.config)
    run_arenahard_judging(config, endpoint_file=args.endpoint_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
