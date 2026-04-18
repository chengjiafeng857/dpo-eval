"""Pinned Arena-Hard v0.1 judge settings."""

from __future__ import annotations

from arenahard_v2.judge_settings import (
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_REGEX_PATTERNS,
    OG_ARENA_HARD_PROMPT,
)


JUDGE_SETTINGS = {
    "arena-hard-v0.1": {
        "baseline": "gpt-4-0314",
        "system_prompt": OG_ARENA_HARD_PROMPT,
    },
}

