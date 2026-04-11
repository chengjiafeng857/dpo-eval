"""Common helpers for Arena-Hard v2.0."""

from __future__ import annotations

import io
import json
import re
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List

from benchmark_common import (
    get_block_config,
    get_model_name_or_path,
    get_pretty_name,
    read_jsonl,
    resolve_path,
    sanitize_name,
)
from config_utils import load_yaml


BLOCK_NAME = "arenahard_v2"
DEFAULT_BENCH_NAME = "arena-hard-v2.0"
DEFAULT_QUESTION_FILE = "question.jsonl"
DEFAULT_QUESTION_URL = (
    "https://huggingface.co/datasets/lmarena-ai/arena-hard-auto/resolve/main/"
    "data/arena-hard-v2.0/question.jsonl"
)
CODE_BLOCK_PATTERN = re.compile(r"```([^`]*)```")


def get_benchmark_dir(config: Dict[str, Any]) -> Path:
    block_cfg = get_block_config(config, BLOCK_NAME)
    bench_name = str(block_cfg.get("bench_name", DEFAULT_BENCH_NAME))
    benchmark_dir = block_cfg.get("benchmark_dir", Path("../data") / bench_name)
    resolved = resolve_path(config, benchmark_dir)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def get_question_path(config: Dict[str, Any]) -> Path:
    block_cfg = get_block_config(config, BLOCK_NAME)
    question_file = str(block_cfg.get("question_file", DEFAULT_QUESTION_FILE))
    path = Path(question_file)
    if path.is_absolute():
        return path
    if path.parent != Path("."):
        return resolve_path(config, path)
    return get_benchmark_dir(config) / question_file


def ensure_question_file(config: Dict[str, Any]) -> Path:
    path = get_question_path(config)
    if path.exists():
        return path

    block_cfg = get_block_config(config, BLOCK_NAME)
    configured_name = str(block_cfg.get("question_file", DEFAULT_QUESTION_FILE))
    if configured_name != DEFAULT_QUESTION_FILE:
        raise FileNotFoundError(
            "Could not find Arena-Hard v2.0 question file. "
            "Set arenahard_v2.question_file to a valid JSONL path."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DEFAULT_QUESTION_URL, timeout=60) as response:
        path.write_bytes(response.read())
    return path


def load_questions(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    question_path = ensure_question_file(config)
    rows = read_jsonl(question_path)
    normalized: list[dict[str, Any]] = []
    for row in rows:
        uid = row.get("uid")
        prompt = row.get("prompt")
        category = row.get("category")
        if uid is None or not isinstance(prompt, str) or not prompt.strip():
            continue
        payload = dict(row)
        payload["uid"] = str(uid)
        payload["prompt"] = prompt
        if category is not None:
            payload["category"] = str(category)
        normalized.append(payload)

    if not normalized:
        raise ValueError("Arena-Hard v2.0 question file is empty after normalization.")

    max_instances = get_block_config(config, BLOCK_NAME).get("max_instances")
    if max_instances is not None:
        normalized = normalized[: int(max_instances)]
    return normalized


def get_answer_dir(config: Dict[str, Any]) -> Path:
    path = get_benchmark_dir(config) / "model_answer"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_judgment_dir(config: Dict[str, Any], judge_name: str) -> Path:
    path = get_benchmark_dir(config) / "model_judgment" / judge_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_metadata_dir(config: Dict[str, Any]) -> Path:
    path = get_benchmark_dir(config) / "run_metadata"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_answer_path(config: Dict[str, Any]) -> Path:
    pretty_name = get_pretty_name(config, BLOCK_NAME)
    return get_answer_dir(config) / f"{sanitize_name(pretty_name)}.jsonl"


def get_judgment_path(config: Dict[str, Any], judge_name: str) -> Path:
    pretty_name = get_pretty_name(config, BLOCK_NAME)
    return get_judgment_dir(config, judge_name) / f"{sanitize_name(pretty_name)}.jsonl"


def get_metadata_path(config: Dict[str, Any]) -> Path:
    pretty_name = get_pretty_name(config, BLOCK_NAME)
    return get_metadata_dir(config) / f"{sanitize_name(pretty_name)}.json"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = ""
    for row in rows:
        text += json.dumps(row, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")


def load_jsonl_map(path: Path, *, key_field: str) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    return {str(row[key_field]): row for row in rows if key_field in row}


def load_model_answers(answer_dir: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    payload: dict[str, dict[str, dict[str, Any]]] = {}
    if not answer_dir.exists():
        return payload
    for path in sorted(answer_dir.glob("*.jsonl")):
        rows = read_jsonl(path)
        if not rows:
            continue
        model_name = str(rows[0].get("model", path.stem))
        payload[model_name] = {
            str(row["uid"]): row
            for row in rows
            if "uid" in row
        }
    return payload


def load_endpoint_catalog_from_config(config: Dict[str, Any], endpoint_file: str | None) -> dict[str, Any]:
    block_cfg = get_block_config(config, BLOCK_NAME)
    resolved = resolve_path(config, endpoint_file or block_cfg.get("endpoint_file", "api_config.yaml"))
    return load_yaml(resolved)


def remove_pattern(text: str, pattern: re.Pattern[str]) -> str:
    cleaned = text
    blocks = pattern.findall(text)
    for block in blocks:
        cleaned = cleaned.replace(block, "")
    return cleaned


def count_markdown_elements(markdown_text: str, *, suffix: str = "") -> dict[str, Any]:
    return {
        f"header_count{suffix}": {
            "h1": len(re.findall(r"^#{1}\s", markdown_text, re.MULTILINE)),
            "h2": len(re.findall(r"^#{2}\s", markdown_text, re.MULTILINE)),
            "h3": len(re.findall(r"^#{3}\s", markdown_text, re.MULTILINE)),
            "h4": len(re.findall(r"^#{4}\s", markdown_text, re.MULTILINE)),
            "h5": len(re.findall(r"^#{5}\s", markdown_text, re.MULTILINE)),
            "h6": len(re.findall(r"^#{6}\s", markdown_text, re.MULTILINE)),
        },
        f"list_count{suffix}": {
            "ordered": len(re.findall(r"^\s*\d+\.\s", markdown_text, re.MULTILINE)),
            "unordered": len(re.findall(r"^\s*[-*+]\s", markdown_text, re.MULTILINE)),
        },
        f"bold_count{suffix}": {
            "**": len(re.findall(r"\*\*[^*\n]+\*\*", markdown_text)),
            "__": len(re.findall(r"__[^_\n]+__", markdown_text)),
        },
    }


def _token_len(text: str) -> int:
    try:
        import tiktoken

        encoder = tiktoken.encoding_for_model("gpt-4o")
        return len(encoder.encode(text, disallowed_special=()))
    except Exception:
        return len(text.split())


def build_style_metadata(answer_text: str) -> dict[str, Any]:
    cleaned = remove_pattern(answer_text, CODE_BLOCK_PATTERN)
    return {"token_len": _token_len(answer_text)} | count_markdown_elements(cleaned)


def metadata_to_feature_vector(metadata: Dict[str, Any]) -> list[float]:
    header_count = metadata.get("header_count", {})
    list_count = metadata.get("list_count", {})
    bold_count = metadata.get("bold_count", {})
    return [
        float(metadata.get("token_len", 0)),
        float(sum(float(value) for value in header_count.values())),
        float(sum(float(value) for value in list_count.values())),
        float(sum(float(value) for value in bold_count.values())),
    ]


def build_answer_row(
    *,
    model_name: str,
    question: Dict[str, Any],
    answer_text: str,
    system_prompt: str | None = None,
) -> Dict[str, Any]:
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question["prompt"]})
    messages.append({"role": "assistant", "content": {"answer": answer_text}})
    return {
        "uid": str(question["uid"]),
        "ans_id": uuid.uuid4().hex,
        "model": model_name,
        "messages": messages,
        "tstamp": float(question.get("tstamp", 0.0)) or __import__("time").time(),
        "metadata": build_style_metadata(answer_text),
    }

