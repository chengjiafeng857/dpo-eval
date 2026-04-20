"""Pinned Arena-Hard v0.1 judge settings."""

from __future__ import annotations

OG_ARENA_HARD_PROMPT = (
    "You are an impartial judge evaluating the quality of two assistant answers "
    "to the same user prompt. Compare helpfulness, correctness, depth, "
    "relevance, and safety. Avoid position bias and do not let verbosity alone "
    "determine the outcome. After your analysis, end with exactly one verdict "
    "label: [[A>>B]], [[A>B]], [[A=B]], [[B>A]], or [[B>>A]]."
)

DEFAULT_PROMPT_TEMPLATE = """<|User Prompt|>
{QUESTION}

<|The Start of Assistant A's Answer|>
{ANSWER_A}
<|The End of Assistant A's Answer|>

<|The Start of Assistant B's Answer|>
{ANSWER_B}
<|The End of Assistant B's Answer|>"""

DEFAULT_REGEX_PATTERNS = [
    r"\[\[(A>>B|A>B|A=B|B>A|B>>A)\]\]",
    r"\[(A>>B|A>B|A=B|B>A|B>>A)\]",
]


JUDGE_SETTINGS = {
    "arena-hard-v0.1": {
        "baseline": "gpt-4-0314",
        "system_prompt": OG_ARENA_HARD_PROMPT,
    },
}
