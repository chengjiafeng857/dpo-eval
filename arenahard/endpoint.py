"""Endpoint helpers for Arena-Hard v0.1.

Prefer the shared v2 implementation when that package exists in the checkout.
Fall back to a local implementation so v0.1 remains runnable in repos that do
not vendor `arenahard_v2`.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

try:  # pragma: no cover - exercised only when arenahard_v2 is available.
    from arenahard_v2.endpoint import (  # type: ignore[attr-defined] # noqa: F401
        DEFAULT_GEMINI_API_BASE,
        DEFAULT_GEMINI_SAFETY_SETTINGS,
        DEFAULT_REQUEST_TIMEOUT_SECONDS,
        SUPPORTED_API_TYPES,
        EndpointPool,
        create_chat_completion,
        get_endpoint_settings,
    )
except ModuleNotFoundError:
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - openai installs httpx transitively.
        raise RuntimeError("httpx is required for Arena-Hard endpoint requests.") from exc

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is required for Arena-Hard endpoint requests.") from exc

    DEFAULT_REQUEST_TIMEOUT_SECONDS = 180.0
    DEFAULT_GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
    DEFAULT_GEMINI_SAFETY_SETTINGS = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    SUPPORTED_API_TYPES = {"openai", "gemini", "vertex"}

    class EndpointPool:
        """Round-robin endpoint selector safe for concurrent judge workers."""

        def __init__(self, endpoints: Any) -> None:
            self._lock = threading.Lock()
            normalized = _normalize_endpoints(endpoints)
            self._endpoints = normalized or [{}]
            self._index = 0

        def acquire(self) -> dict[str, Any]:
            with self._lock:
                endpoint = dict(self._endpoints[self._index % len(self._endpoints)])
                self._index += 1
                return endpoint

    def _normalize_endpoints(endpoints: Any) -> list[dict[str, Any]]:
        if endpoints is None:
            return []
        if not isinstance(endpoints, Sequence) or isinstance(endpoints, (str, bytes, bytearray)):
            raise ValueError("Endpoint settings 'endpoints' must be a sequence of mappings.")
        normalized: list[dict[str, Any]] = []
        for endpoint in endpoints:
            if endpoint is None:
                normalized.append({})
                continue
            if not isinstance(endpoint, Mapping):
                raise ValueError("Each endpoint entry must be a mapping.")
            normalized.append(dict(endpoint))
        return normalized

    def get_endpoint_settings(endpoint_catalog: Mapping[str, Any], endpoint_name: str) -> dict[str, Any]:
        if endpoint_name not in endpoint_catalog:
            raise KeyError(f"Unknown endpoint config: {endpoint_name}")
        raw_settings = endpoint_catalog[endpoint_name]
        if not isinstance(raw_settings, Mapping):
            raise ValueError(f"Endpoint config '{endpoint_name}' must be a mapping.")
        settings = dict(raw_settings)
        api_type = str(settings.get("api_type", "openai")).lower()
        if api_type not in SUPPORTED_API_TYPES:
            raise ValueError(
                f"Unsupported endpoint api_type '{api_type}'. "
                f"Expected one of {sorted(SUPPORTED_API_TYPES)}."
            )
        settings["api_type"] = api_type
        settings["timeout"] = float(settings.get("timeout", DEFAULT_REQUEST_TIMEOUT_SECONDS))
        settings["max_retries"] = int(settings.get("max_retries", 0))
        settings["initial_backoff"] = float(settings.get("initial_backoff", 1.0))
        settings["max_backoff"] = float(settings.get("max_backoff", 60.0))
        settings["endpoints"] = _normalize_endpoints(settings.get("endpoints"))
        return settings

    def _extract_text_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
            parts: list[str] = []
            for part in content:
                if isinstance(part, Mapping) and part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
                elif isinstance(part, str):
                    parts.append(part)
            return "".join(parts)
        return str(content or "")

    def _serialize_usage(usage: Any) -> dict[str, Any] | None:
        if usage is None:
            return None
        if isinstance(usage, Mapping):
            return dict(usage)
        if hasattr(usage, "model_dump"):
            dumped = usage.model_dump()
            if isinstance(dumped, Mapping):
                return dict(dumped)
        if hasattr(usage, "__dict__"):
            return {
                key: value
                for key, value in vars(usage).items()
                if not key.startswith("_")
            }
        return None

    def _openai_payload(
        *,
        settings: Mapping[str, Any],
        endpoint: Mapping[str, Any],
        messages: Sequence[Mapping[str, Any]],
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": str(endpoint.get("model") or settings["model"]),
            "messages": [dict(message) for message in messages],
        }
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        return payload

    def _call_openai(
        *,
        settings: Mapping[str, Any],
        endpoint: Mapping[str, Any],
        messages: Sequence[Mapping[str, Any]],
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        api_key = endpoint.get("api_key") or settings.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for Arena-Hard OpenAI-compatible judging.")

        client = OpenAI(
            api_key=str(api_key),
            base_url=endpoint.get("api_base") or settings.get("api_base"),
            timeout=float(settings.get("timeout", DEFAULT_REQUEST_TIMEOUT_SECONDS)),
        )
        payload = _openai_payload(
            settings=settings,
            endpoint=endpoint,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        response = client.chat.completions.create(**payload)
        choice = response.choices[0]
        answer = _extract_text_content(choice.message.content if choice.message is not None else "")
        return {
            "answer": answer,
            "usage": _serialize_usage(getattr(response, "usage", None)),
            "raw_response": response,
        }

    def _call_openai_raw_http(
        *,
        settings: Mapping[str, Any],
        endpoint: Mapping[str, Any],
        messages: Sequence[Mapping[str, Any]],
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        api_key = endpoint.get("api_key") or settings.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for Arena-Hard OpenAI-compatible judging.")

        api_base = str(endpoint.get("api_base") or settings.get("api_base") or "https://api.openai.com/v1")
        url = f"{api_base.rstrip('/')}/chat/completions"
        payload = _openai_payload(
            settings=settings,
            endpoint=endpoint,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        with httpx.Client(timeout=float(settings.get("timeout", DEFAULT_REQUEST_TIMEOUT_SECONDS))) as client:
            response = client.post(
                url,
                content=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
        response.raise_for_status()
        body = response.json()
        choices = body.get("choices", [])
        message = choices[0].get("message", {}) if choices else {}
        answer = _extract_text_content(message.get("content", ""))
        return {
            "answer": answer,
            "usage": body.get("usage"),
            "raw_response": body,
        }

    def _gemini_contents(messages: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        system_instruction: dict[str, Any] | None = None
        contents: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role", "user"))
            text = _extract_text_content(message.get("content", ""))
            if role == "system":
                system_instruction = {"parts": [{"text": text}]}
                continue
            if role == "assistant":
                gemini_role = "model"
            else:
                gemini_role = "user"
            contents.append({"role": gemini_role, "parts": [{"text": text}]})
        return system_instruction, contents

    def _call_gemini(
        *,
        settings: Mapping[str, Any],
        endpoint: Mapping[str, Any],
        messages: Sequence[Mapping[str, Any]],
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        api_key = endpoint.get("api_key") or settings.get("api_key") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required for native Gemini requests.")

        model_name = str(endpoint.get("model") or settings["model"])
        base_url = str(endpoint.get("api_base") or settings.get("api_base") or DEFAULT_GEMINI_API_BASE).rstrip("/")
        url = f"{base_url}/models/{model_name}:generateContent"
        system_instruction, contents = _gemini_contents(messages)
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": float(temperature if temperature is not None else settings.get("temperature", 0.0)),
                "maxOutputTokens": int(max_tokens if max_tokens is not None else settings.get("max_tokens", 4096)),
            },
            "safetySettings": settings.get("safety_settings", DEFAULT_GEMINI_SAFETY_SETTINGS),
        }
        if system_instruction is not None:
            payload["systemInstruction"] = system_instruction

        with httpx.Client(timeout=float(settings.get("timeout", DEFAULT_REQUEST_TIMEOUT_SECONDS))) as client:
            response = client.post(url, params={"key": str(api_key)}, json=payload)
        response.raise_for_status()
        body = response.json()
        candidates = body.get("candidates", [])
        candidate = candidates[0] if candidates else {}
        answer = ""
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            answer += str(part.get("text", ""))
        return {
            "answer": answer,
            "usage": body.get("usageMetadata"),
            "raw_response": body,
        }

    def _vertex_headers() -> dict[str, str]:
        token = subprocess.check_output(
            ["gcloud", "auth", "print-access-token"],
            text=True,
        ).strip()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def _call_vertex(
        *,
        settings: Mapping[str, Any],
        endpoint: Mapping[str, Any],
        messages: Sequence[Mapping[str, Any]],
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        project_id = str(endpoint.get("project_id") or settings.get("project_id") or "")
        if not project_id:
            raise RuntimeError("Vertex Gemini requests require 'project_id'.")
        regions_value = endpoint.get("regions") or settings.get("regions") or "us-central1"
        if isinstance(regions_value, str):
            regions = [part.strip() for part in regions_value.split(",") if part.strip()]
        elif isinstance(regions_value, Sequence):
            regions = [str(part).strip() for part in regions_value if str(part).strip()]
        else:
            raise ValueError("'regions' must be a string or sequence.")
        if not regions:
            raise RuntimeError("Vertex Gemini requests require at least one region.")
        region = random.choice(regions)
        model_name = str(endpoint.get("model") or settings["model"])
        url = (
            f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}"
            f"/locations/{region}/publishers/google/models/{model_name}:generateContent"
        )
        system_instruction, contents = _gemini_contents(messages)
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": float(temperature if temperature is not None else settings.get("temperature", 0.0)),
                "maxOutputTokens": int(max_tokens if max_tokens is not None else settings.get("max_tokens", 4096)),
            },
        }
        if system_instruction is not None:
            payload["systemInstruction"] = system_instruction

        with httpx.Client(timeout=float(settings.get("timeout", DEFAULT_REQUEST_TIMEOUT_SECONDS))) as client:
            response = client.post(url, headers=_vertex_headers(), json=payload)
        response.raise_for_status()
        body = response.json()
        candidates = body.get("candidates", [])
        candidate = candidates[0] if candidates else {}
        answer = ""
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            answer += str(part.get("text", ""))
        return {
            "answer": answer,
            "usage": body.get("usageMetadata"),
            "raw_response": body,
        }

    def create_chat_completion(
        *,
        settings: Mapping[str, Any],
        pool: EndpointPool,
        messages: Sequence[Mapping[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        api_type = str(settings.get("api_type", "openai")).lower()
        timeout = float(settings.get("timeout", DEFAULT_REQUEST_TIMEOUT_SECONDS))
        max_retries = int(settings.get("max_retries", 0))
        initial_backoff = float(settings.get("initial_backoff", 1.0))
        max_backoff = float(settings.get("max_backoff", 60.0))
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            endpoint = pool.acquire()
            try:
                if api_type == "openai":
                    try:
                        return _call_openai(
                            settings=settings,
                            endpoint=endpoint,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                    except Exception as exc:
                        exc_name = exc.__class__.__name__
                        exc_text = str(exc)
                        if exc_name == "BadRequestError" and "parse the JSON body" in exc_text:
                            print(
                                "[ArenaHard][endpoint-fallback] "
                                "retrying request with raw HTTP JSON body.",
                                file=sys.stderr,
                            )
                            return _call_openai_raw_http(
                                settings=settings,
                                endpoint=endpoint,
                                messages=messages,
                                temperature=temperature,
                                max_tokens=max_tokens,
                            )
                        raise
                if api_type == "gemini":
                    return _call_gemini(
                        settings=settings,
                        endpoint=endpoint,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                if api_type == "vertex":
                    return _call_vertex(
                        settings=settings,
                        endpoint=endpoint,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                raise ValueError(f"Unsupported endpoint api_type '{api_type}'.")
            except Exception as exc:
                last_error = exc
                total_attempts = max_retries + 1
                print(
                    "[ArenaHard][endpoint-error] "
                    f"attempt={attempt + 1}/{total_attempts} "
                    f"model={settings.get('model')} api_type={api_type} "
                    f"timeout={timeout:.1f}s error={exc}",
                    file=sys.stderr,
                )
                if attempt >= max_retries:
                    break
                delay = min(max_backoff, initial_backoff * (2**attempt))
                if delay > 0:
                    time.sleep(delay)

        raise RuntimeError(
            f"Arena-Hard endpoint request failed after {max_retries + 1} attempts: {last_error}"
        ) from last_error
