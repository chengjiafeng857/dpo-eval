"""Minimal config helpers for the standalone eval pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and attach its source path for relative resolution."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level mapping in config file: {config_path}")
    payload["_config_path"] = str(config_path)
    return payload


def resolve_torch_dtype(precision: Any) -> torch.dtype | None:
    """Translate common precision strings into torch dtypes."""
    if precision is None:
        return None

    normalized = str(precision).strip().lower()
    if normalized in {"auto", "none", ""}:
        return None
    if normalized in {"fp32", "float32", "float"}:
        return torch.float32
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported precision value: {precision}")
