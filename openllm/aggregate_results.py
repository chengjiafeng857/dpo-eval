"""Aggregate lm-eval-harness Open LLM v1 result JSON into summary.{json,csv,md}.

lm-eval writes one or more `results_*.json` files (and optionally per-sample logs)
into the output directory tree. Metric keys vary across harness versions
(e.g. `acc,none` vs `acc`, with `_stderr` suffixes), and MMLU is reported either
as a group score or as 57 sub-task entries. This module finds the latest results
JSON and extracts the six headline metrics defensively.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Optional


# Headline metric per task. Order of preference matters: pick the first key
# that the result dict actually exposes.
METRIC_PREFS: dict[str, tuple[str, ...]] = {
    "arc_challenge": ("acc_norm", "acc"),
    "hellaswag": ("acc_norm", "acc"),
    "truthfulqa_mc2": ("acc", "mc2"),
    "winogrande": ("acc",),
    "gsm8k": ("exact_match", "acc"),
    # MMLU: reported as a group on most versions; we also handle the per-subject
    # average via _mmlu_average() below.
    "mmlu": ("acc",),
}


def _strip_suffix(key: str) -> str:
    # lm-eval often appends ",none" or ",strict-match" to metric keys.
    return key.split(",", 1)[0]


def _normalize_metrics(d: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        if not isinstance(v, (int, float)):
            continue
        base = _strip_suffix(k)
        if base.endswith("_stderr"):
            continue
        # Keep first occurrence; harness sometimes lists raw and aliased keys.
        out.setdefault(base, float(v))
    return out


def find_latest_results_json(output_dir: Path) -> Optional[Path]:
    candidates = sorted(output_dir.rglob("results_*.json"))
    if not candidates:
        # Some versions write `results.json` directly.
        alt = sorted(output_dir.rglob("results.json"))
        candidates = alt
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _pick(metrics: dict[str, float], prefs: tuple[str, ...]) -> Optional[float]:
    for key in prefs:
        if key in metrics:
            return metrics[key]
    return None


def _mmlu_average(results: dict[str, dict[str, Any]]) -> Optional[float]:
    """Average MMLU sub-task accuracies when the group score is missing."""
    sub_scores: list[float] = []
    for task_name, raw in results.items():
        if not task_name.startswith("mmlu_"):
            continue
        if task_name == "mmlu_pro":  # different benchmark
            continue
        norm = _normalize_metrics(raw)
        v = _pick(norm, ("acc",))
        if v is not None:
            sub_scores.append(v)
    if not sub_scores:
        return None
    return sum(sub_scores) / len(sub_scores)


def _truthfulqa_mc2(results: dict[str, dict[str, Any]]) -> Optional[float]:
    """Pull the MC2 accuracy from any of the truthfulqa_* entries."""
    candidates = [
        "truthfulqa_mc2",
        "truthfulqa",
        "truthfulqa_mc",
    ]
    for name in candidates:
        if name in results:
            norm = _normalize_metrics(results[name])
            v = _pick(norm, METRIC_PREFS["truthfulqa_mc2"])
            if v is not None:
                return v
    # Some harness versions name the subtask `truthfulqa_mc2` only when it's a
    # leaf, otherwise look for any key containing 'mc2'.
    for name, raw in results.items():
        if "truthfulqa" not in name:
            continue
        norm = _normalize_metrics(raw)
        if "mc2" in norm:
            return norm["mc2"]
        if "acc" in norm and "mc2" in name:
            return norm["acc"]
    return None


def extract_summary(results_blob: dict[str, Any]) -> dict[str, Optional[float]]:
    results = results_blob.get("results") or results_blob.get("groups") or {}
    if "results" in results_blob and "groups" in results_blob:
        # Merge so group-level scores (e.g. mmlu, openllm) are visible alongside leaf tasks.
        results = {**results_blob.get("groups", {}), **results_blob["results"]}

    summary: dict[str, Optional[float]] = {}

    # MMLU: prefer group-level, fall back to per-subject mean.
    mmlu_val: Optional[float] = None
    if "mmlu" in results:
        mmlu_val = _pick(_normalize_metrics(results["mmlu"]), METRIC_PREFS["mmlu"])
    if mmlu_val is None:
        mmlu_val = _mmlu_average(results)
    summary["mmlu_acc"] = mmlu_val

    # ARC-Challenge normalized accuracy.
    if "arc_challenge" in results:
        summary["arc_challenge_acc_norm"] = _pick(
            _normalize_metrics(results["arc_challenge"]), METRIC_PREFS["arc_challenge"]
        )
    else:
        summary["arc_challenge_acc_norm"] = None

    # HellaSwag normalized accuracy.
    if "hellaswag" in results:
        summary["hellaswag_acc_norm"] = _pick(
            _normalize_metrics(results["hellaswag"]), METRIC_PREFS["hellaswag"]
        )
    else:
        summary["hellaswag_acc_norm"] = None

    # TruthfulQA MC2.
    summary["truthfulqa_mc2"] = _truthfulqa_mc2(results)

    # WinoGrande accuracy.
    if "winogrande" in results:
        summary["winogrande_acc"] = _pick(
            _normalize_metrics(results["winogrande"]), METRIC_PREFS["winogrande"]
        )
    else:
        summary["winogrande_acc"] = None

    # GSM8K (exact_match preferred).
    if "gsm8k" in results:
        summary["gsm8k_exact_match"] = _pick(
            _normalize_metrics(results["gsm8k"]), METRIC_PREFS["gsm8k"]
        )
    else:
        summary["gsm8k_exact_match"] = None

    present = [v for v in summary.values() if v is not None]
    summary["openllm_v1_average"] = sum(present) / len(present) if present else None
    return summary


def aggregate(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    results_path = find_latest_results_json(output_dir)
    if results_path is None:
        raise FileNotFoundError(f"No results_*.json found under {output_dir}")

    blob = json.loads(results_path.read_text())
    summary = extract_summary(blob)

    meta_path = output_dir / "run_metadata.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    payload = {
        "model_path": metadata.get("model_path"),
        "backend": metadata.get("backend"),
        "dtype": metadata.get("dtype"),
        "batch_size": metadata.get("batch_size"),
        "seed": metadata.get("seed"),
        "apply_chat_template": metadata.get("apply_chat_template", False),
        "fewshot_as_multiturn": metadata.get("fewshot_as_multiturn", False),
        "lm_eval_version": metadata.get("lm_eval_version"),
        "results_json": str(results_path),
        "scores": summary,
    }

    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    _write_csv(output_dir / "summary.csv", payload)
    _write_md(output_dir / "summary.md", payload)
    return payload


def _fmt(v: Optional[float]) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"


def _write_csv(path: Path, payload: dict[str, Any]) -> None:
    scores = payload["scores"]
    rows = [
        ("metric", "value"),
        ("MMLU acc (5-shot)", _fmt(scores.get("mmlu_acc"))),
        ("ARC-Challenge acc_norm (25-shot)", _fmt(scores.get("arc_challenge_acc_norm"))),
        ("HellaSwag acc_norm (10-shot)", _fmt(scores.get("hellaswag_acc_norm"))),
        ("TruthfulQA MC2 (0-shot)", _fmt(scores.get("truthfulqa_mc2"))),
        ("WinoGrande acc (5-shot)", _fmt(scores.get("winogrande_acc"))),
        ("GSM8K exact_match (5-shot)", _fmt(scores.get("gsm8k_exact_match"))),
        ("Open LLM v1 average", _fmt(scores.get("openllm_v1_average"))),
    ]
    with path.open("w", newline="") as f:
        csv.writer(f).writerows(rows)


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    s = payload["scores"]
    lines = [
        f"# Open LLM Leaderboard v1 — `{payload.get('model_path')}`",
        "",
        f"- backend: `{payload.get('backend')}`  dtype: `{payload.get('dtype')}`  "
        f"batch_size: `{payload.get('batch_size')}`  seed: `{payload.get('seed')}`",
        f"- apply_chat_template: `{payload.get('apply_chat_template')}`  "
        f"fewshot_as_multiturn: `{payload.get('fewshot_as_multiturn')}`",
        f"- lm-eval version: `{payload.get('lm_eval_version')}`",
        "",
        "| Task | Shots | Metric | Score |",
        "|------|-------|--------|-------|",
        f"| MMLU            | 5  | acc        | {_fmt(s.get('mmlu_acc'))} |",
        f"| ARC-Challenge   | 25 | acc_norm   | {_fmt(s.get('arc_challenge_acc_norm'))} |",
        f"| HellaSwag       | 10 | acc_norm   | {_fmt(s.get('hellaswag_acc_norm'))} |",
        f"| TruthfulQA      | 0  | mc2        | {_fmt(s.get('truthfulqa_mc2'))} |",
        f"| WinoGrande      | 5  | acc        | {_fmt(s.get('winogrande_acc'))} |",
        f"| GSM8K           | 5  | exact_match| {_fmt(s.get('gsm8k_exact_match'))} |",
        f"| **Average**     |    |            | **{_fmt(s.get('openllm_v1_average'))}** |",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Aggregate lm-eval Open LLM v1 results.")
    p.add_argument("output_dir", type=Path)
    args = p.parse_args()
    payload = aggregate(args.output_dir)
    print(json.dumps(payload["scores"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
