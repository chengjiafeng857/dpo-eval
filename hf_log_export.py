#!/usr/bin/env python3
"""Export a Hugging Face dataset split to a JSON array."""

from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

from datasets import load_dataset


DEFAULT_DATASET = (
    "jackf857/llama-3-8b-base-scheduled-beta-margin-dpo-hh-helpful-margin-log"
)
DEFAULT_OUTPUT = Path("scheduled-beta-margin.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Hugging Face dataset repo ID.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to export.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSON path.",
    )
    return parser.parse_args()


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    try:
        item = value.item
    except AttributeError:
        item = None
    if item is not None:
        return to_jsonable(item())
    return value


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset, split=args.split)
    rows = [{key: to_jsonable(value) for key, value in row.items()} for row in dataset]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} rows to {args.output.resolve()}")


if __name__ == "__main__":
    main()
