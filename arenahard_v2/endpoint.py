"""OpenAI-compatible endpoint helpers for Arena-Hard v2.0."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Dict, Sequence

from openai import OpenAI

from config_utils import load_yaml


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
    if api_type != "openai":
        raise ValueError(
            "Only OpenAI-compatible endpoints are supported in Arena-Hard v2.0 v1."
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
    timeout = settings.get("timeout")

    attempt = 0
    backoff = initial_backoff
    last_error: Exception | None = None
    while attempt <= max_retries:
        endpoint = pool.next()
        api_key = endpoint.get("api_key")
        base_url = endpoint.get("api_base")
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
        try:
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
        except Exception as exc:  # Broad for API/network/service errors.
            last_error = exc
            attempt += 1
            if attempt > max_retries:
                break
            time.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)

    raise RuntimeError(f"OpenAI-compatible endpoint failed after {max_retries} retries") from last_error

