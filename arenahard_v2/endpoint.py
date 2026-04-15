"""Endpoint helpers for Arena-Hard v2.0."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Sequence

import httpx
from openai import OpenAI

from config_utils import load_yaml


DEFAULT_REQUEST_TIMEOUT_SECONDS = 180.0
DEFAULT_GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
SUPPORTED_API_TYPES = {"openai", "gemini", "vertex"}
DEFAULT_GEMINI_SAFETY_SETTINGS = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]


def _format_endpoint_label(endpoint: Dict[str, Any]) -> str:
    base_url = endpoint.get("api_base")
    if base_url:
        return str(base_url)
    if endpoint.get("project_id"):
        region = endpoint.get("region") or endpoint.get("regions") or "unknown-region"
        return f"vertex:{region}"
    api_type = endpoint.get("_api_type")
    return f"default-{api_type}" if api_type else "default-endpoint"


def _log_endpoint_error(
    *,
    model: str,
    endpoint: Dict[str, Any],
    attempt: int,
    total_attempts: int,
    exc: Exception,
    retry_delay: float | None,
    timeout_seconds: float,
) -> None:
    error_text = str(exc).strip() or repr(exc)
    message = (
        "[ArenaHardV2][endpoint-error] "
        f"model={model} "
        f"endpoint={_format_endpoint_label(endpoint)} "
        f"attempt={attempt}/{total_attempts} "
        f"timeout={timeout_seconds:.1f}s "
        f"error={type(exc).__name__}: {error_text}"
    )
    if retry_delay is not None:
        message += f" retry_in={retry_delay:.1f}s"
    print(message, file=sys.stderr, flush=True)


def _log_endpoint_fallback(
    *,
    model: str,
    endpoint: Dict[str, Any],
    reason: str,
) -> None:
    print(
        "[ArenaHardV2][endpoint-fallback] "
        f"model={model} "
        f"endpoint={_format_endpoint_label(endpoint)} "
        f"reason={reason}",
        file=sys.stderr,
        flush=True,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, str):
        return value.encode("utf-8", errors="replace").decode("utf-8")
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def _stringify_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        answer = content.get("answer")
        if answer is not None:
            return str(answer)
        return json.dumps(_json_safe(content), ensure_ascii=False)
    if isinstance(content, list):
        return "\n".join(_stringify_message_content(item) for item in content)
    return str(content)


def _should_use_raw_http_fallback(exc: Exception) -> bool:
    error_text = str(exc)
    return (
        type(exc).__name__ == "BadRequestError"
        and "could not parse the json body" in error_text.lower()
    )


def _create_chat_completion_via_httpx(
    *,
    model: str,
    endpoint: Dict[str, Any],
    messages: Sequence[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    base_url = str(
        endpoint.get("api_base")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    ).rstrip("/")
    api_key = endpoint.get("api_key") or os.environ.get("OPENAI_API_KEY")
    payload = {
        "model": model,
        "messages": _json_safe(list(messages)),
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    with httpx.Client(timeout=timeout) as client:
        response = client.post(
            f"{base_url}/chat/completions",
            content=body,
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()

    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("Endpoint response did not include any choices.")
    message = choices[0].get("message", {})
    usage = data.get("usage", {})
    return {"answer": str(message.get("content") or ""), "usage": dict(usage)}


def _build_gemini_payload(
    *,
    messages: Sequence[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    include_safety_settings: bool,
    safety_settings: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"contents": []}
    working_messages = list(messages)
    if working_messages and working_messages[0].get("role") == "system":
        payload["systemInstruction"] = {
            "parts": [{"text": _stringify_message_content(working_messages[0].get("content", ""))}]
        }
        working_messages = working_messages[1:]

    role_map = {"user": "user", "assistant": "model"}
    contents: list[dict[str, Any]] = []
    for message in working_messages:
        role = role_map.get(str(message.get("role", "user")), "user")
        contents.append(
            {
                "role": role,
                "parts": [{"text": _stringify_message_content(message.get("content", ""))}],
            }
        )
    payload["contents"] = contents
    payload["generationConfig"] = {
        "temperature": float(temperature),
        "maxOutputTokens": int(max_tokens),
    }
    if include_safety_settings:
        payload["safetySettings"] = list(
            safety_settings or DEFAULT_GEMINI_SAFETY_SETTINGS
        )
    return payload


def _extract_gemini_response(data: dict[str, Any]) -> dict[str, Any]:
    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError("Endpoint response did not include any candidates.")
    content = candidates[0].get("content", {})
    parts = content.get("parts", [])
    answer = ""
    for part in parts:
        text = part.get("text")
        if text:
            answer += str(text)
    if not answer:
        raise RuntimeError("Endpoint response did not include any text parts.")
    usage = data.get("usageMetadata", {})
    return {"answer": answer, "usage": dict(usage)}


def _create_chat_completion_via_gemini_http(
    *,
    model: str,
    endpoint: Dict[str, Any],
    settings: dict[str, Any],
    messages: Sequence[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    api_key = (
        endpoint.get("api_key")
        or settings.get("api_key")
        or os.environ.get("GEMINI_API_KEY")
    )
    if not api_key:
        raise ValueError(
            "Gemini endpoints require `api_key` or the `GEMINI_API_KEY` environment variable."
        )
    base_url = str(
        endpoint.get("api_base")
        or settings.get("api_base")
        or DEFAULT_GEMINI_API_BASE
    ).rstrip("/")
    payload = _build_gemini_payload(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        include_safety_settings=True,
        safety_settings=settings.get("safety_settings"),
    )
    with httpx.Client(timeout=timeout) as client:
        response = client.post(
            f"{base_url}/models/{model}:generateContent",
            params={"key": api_key},
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
    return _extract_gemini_response(data)


def _resolve_vertex_region(endpoint: Dict[str, Any], settings: dict[str, Any]) -> str:
    region = (
        endpoint.get("region")
        or endpoint.get("regions")
        or settings.get("region")
        or settings.get("regions")
        or os.environ.get("VERTEX_REGION")
    )
    if isinstance(region, (list, tuple)):
        if not region:
            raise ValueError("Vertex endpoint region list cannot be empty.")
        return str(region[0])
    if region is None or str(region).strip() == "":
        raise ValueError(
            "Vertex endpoints require `region`/`regions` or the `VERTEX_REGION` environment variable."
        )
    return str(region)


def _get_vertex_access_token(endpoint: Dict[str, Any], settings: dict[str, Any]) -> str:
    token = endpoint.get("access_token") or settings.get("access_token")
    if token:
        return str(token)
    return subprocess.check_output(
        ["gcloud", "auth", "application-default", "print-access-token"],
        text=True,
    ).strip()


def _create_chat_completion_via_vertex(
    *,
    model: str,
    endpoint: Dict[str, Any],
    settings: dict[str, Any],
    messages: Sequence[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    project_id = (
        endpoint.get("project_id")
        or settings.get("project_id")
        or os.environ.get("VERTEX_PROJECT_ID")
    )
    if not project_id:
        raise ValueError(
            "Vertex endpoints require `project_id` or the `VERTEX_PROJECT_ID` environment variable."
        )
    region = _resolve_vertex_region(endpoint, settings)
    access_token = _get_vertex_access_token(endpoint, settings)
    base_url = str(
        endpoint.get("api_base")
        or settings.get("api_base")
        or f"https://{region}-aiplatform.googleapis.com/v1"
    ).rstrip("/")
    payload = _build_gemini_payload(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        include_safety_settings=False,
    )
    with httpx.Client(timeout=timeout) as client:
        response = client.post(
            (
                f"{base_url}/projects/{project_id}/locations/{region}/publishers/google/"
                f"models/{model}:generateContent"
            ),
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
    return _extract_gemini_response(data)


class EndpointPool:
    def __init__(self, endpoints: Sequence[Dict[str, Any]] | None) -> None:
        self._endpoints = list(endpoints or [{}])
        self._lock = threading.Lock()
        self._index = 0

    def next(self) -> Dict[str, Any]:
        with self._lock:
            endpoint = dict(self._endpoints[self._index % len(self._endpoints)])
            self._index += 1
        return endpoint


def load_endpoint_catalog(path: str | Path) -> dict[str, Any]:
    return load_yaml(path)


def get_endpoint_settings(
    endpoint_catalog: dict[str, Any],
    endpoint_name: str,
) -> dict[str, Any]:
    settings = endpoint_catalog.get(endpoint_name)
    if not isinstance(settings, dict):
        raise ValueError(f"Endpoint '{endpoint_name}' not found in endpoint catalog.")
    api_type = str(settings.get("api_type", "openai")).lower()
    if api_type not in SUPPORTED_API_TYPES:
        raise ValueError(
            "Unsupported Arena-Hard v2.0 endpoint type "
            f"'{api_type}'. Supported types: {', '.join(sorted(SUPPORTED_API_TYPES))}."
        )
    return settings


def create_chat_completion(
    *,
    settings: dict[str, Any],
    pool: EndpointPool,
    messages: Sequence[dict[str, Any]],
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_retries: int = 5,
    initial_backoff: float = 1.0,
    max_backoff: float = 60.0,
) -> dict[str, Any]:
    model = str(settings.get("model") or settings.get("model_name") or "")
    if not model:
        raise ValueError("Endpoint settings must define a model name.")
    api_type = str(settings.get("api_type", "openai")).lower()

    chosen_temperature = (
        float(temperature)
        if temperature is not None
        else float(settings.get("temperature", 0.0))
    )
    chosen_max_tokens = (
        int(max_tokens)
        if max_tokens is not None
        else int(settings.get("max_tokens", 4096))
    )
    timeout = float(settings.get("timeout", DEFAULT_REQUEST_TIMEOUT_SECONDS))
    chosen_max_retries = int(settings.get("max_retries", max_retries))
    chosen_initial_backoff = float(settings.get("initial_backoff", initial_backoff))
    chosen_max_backoff = float(settings.get("max_backoff", max_backoff))

    attempt = 0
    backoff = chosen_initial_backoff
    last_error: Exception | None = None
    total_attempts = chosen_max_retries + 1
    while attempt <= chosen_max_retries:
        endpoint = pool.next()
        endpoint["_api_type"] = api_type
        try:
            if api_type == "openai":
                api_key = endpoint.get("api_key") or os.environ.get("OPENAI_API_KEY")
                base_url = endpoint.get("api_base") or os.environ.get("OPENAI_BASE_URL")
                client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=timeout,
                )
                response = client.chat.completions.create(
                    model=model,
                    messages=list(messages),
                    temperature=chosen_temperature,
                    max_tokens=chosen_max_tokens,
                )
                content = response.choices[0].message.content or ""
                usage: dict[str, Any] = {}
                if response.usage:
                    usage = (
                        response.usage.model_dump()
                        if hasattr(response.usage, "model_dump")
                        else dict(response.usage)
                    )
                return {"answer": str(content), "usage": usage}
            if api_type == "gemini":
                return _create_chat_completion_via_gemini_http(
                    model=model,
                    endpoint=endpoint,
                    settings=settings,
                    messages=messages,
                    temperature=chosen_temperature,
                    max_tokens=chosen_max_tokens,
                    timeout=timeout,
                )
            if api_type == "vertex":
                return _create_chat_completion_via_vertex(
                    model=model,
                    endpoint=endpoint,
                    settings=settings,
                    messages=messages,
                    temperature=chosen_temperature,
                    max_tokens=chosen_max_tokens,
                    timeout=timeout,
                )
            raise ValueError(f"Unsupported endpoint api_type '{api_type}'.")
        except Exception as exc:  # Broad for API/network/service errors.
            if api_type == "openai" and _should_use_raw_http_fallback(exc):
                _log_endpoint_fallback(
                    model=model,
                    endpoint=endpoint,
                    reason="sdk_json_body_parse_error",
                )
                try:
                    return _create_chat_completion_via_httpx(
                        model=model,
                        endpoint=endpoint,
                        messages=messages,
                        temperature=chosen_temperature,
                        max_tokens=chosen_max_tokens,
                        timeout=timeout,
                    )
                except Exception as fallback_exc:
                    _log_endpoint_error(
                        model=model,
                        endpoint=endpoint,
                        attempt=attempt + 1,
                        total_attempts=total_attempts,
                        exc=fallback_exc,
                        retry_delay=None if attempt + 1 > chosen_max_retries else backoff,
                        timeout_seconds=timeout,
                    )
                    last_error = fallback_exc
                    attempt += 1
                    if attempt > chosen_max_retries:
                        break
                    time.sleep(backoff)
                    backoff = min(backoff * 2, chosen_max_backoff)
                    continue
            last_error = exc
            attempt += 1
            retry_delay = None if attempt > chosen_max_retries else backoff
            _log_endpoint_error(
                model=model,
                endpoint=endpoint,
                attempt=attempt,
                total_attempts=total_attempts,
                exc=exc,
                retry_delay=retry_delay,
                timeout_seconds=timeout,
            )
            if attempt > chosen_max_retries:
                break
            time.sleep(backoff)
            backoff = min(backoff * 2, chosen_max_backoff)

    raise RuntimeError(
        "Endpoint request failed "
        f"for model '{model}' after {total_attempts} attempts "
        f"(timeout={timeout:.1f}s, endpoint={_format_endpoint_label(endpoint)})"
    ) from last_error
