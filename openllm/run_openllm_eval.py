"""Open LLM Leaderboard v1 evaluation runner backed by EleutherAI lm-evaluation-harness.

Wraps the `lm_eval` CLI for the six-task v1 suite (MMLU 5-shot, ARC-Challenge 25-shot,
HellaSwag 10-shot, TruthfulQA 0-shot, WinoGrande 5-shot, GSM8K 5-shot).

Prefers the built-in `openllm` task group; falls back to the local
`tasks/openllm_v1.yaml` group when the installed harness does not ship `openllm`.
Prints the final command before executing, records run metadata, and aggregates
results into summary.json / summary.csv / summary.md.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

OPENLLM_DIR = Path(__file__).resolve().parent
FALLBACK_TASK_DIR = OPENLLM_DIR / "tasks"
FALLBACK_GROUP_NAME = "openllm_v1"


def _run_capture(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _detect_lm_eval_entrypoint() -> list[str]:
    """Return the argv prefix used to invoke lm-eval-harness."""
    if shutil.which("lm_eval") is not None:
        return ["lm_eval"]
    if shutil.which("lm-eval") is not None:
        return ["lm-eval"]
    # Module-form fallback works whenever the package is importable.
    return [sys.executable, "-m", "lm_eval"]


def _lm_eval_version() -> str:
    try:
        from importlib.metadata import version

        return version("lm_eval")
    except Exception:
        try:
            from importlib.metadata import version

            return version("lm-eval")
        except Exception:
            return "unknown"


def _has_builtin_openllm_group(entrypoint: list[str]) -> bool:
    """Best-effort probe to see whether the installed harness ships an `openllm` group."""
    proc = _run_capture(entrypoint + ["--tasks", "list"])
    blob = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if not blob.strip():
        # `--tasks list` failed; try `--show_config_groups` style probe via help.
        proc2 = _run_capture(entrypoint + ["--help"])
        blob = (proc2.stdout or "") + "\n" + (proc2.stderr or "")
    # Match `openllm` as a standalone token to avoid hitting tasks like `openllm_xyz`.
    for line in blob.splitlines():
        token = line.strip().split()[0] if line.strip() else ""
        if token == "openllm":
            return True
    return False


def _build_model_args(args: argparse.Namespace) -> str:
    parts: list[str] = [f"pretrained={args.model_path}"]
    if args.backend == "hf":
        parts.append(f"dtype={args.dtype}")
        parts.append("trust_remote_code=True")
    else:  # vllm
        tp = args.tensor_parallel_size or args.num_gpus or 1
        parts.append(f"tensor_parallel_size={tp}")
        parts.append(f"dtype={args.dtype}")
        parts.append(f"gpu_memory_utilization={args.gpu_memory_utilization}")
        parts.append(f"max_model_len={args.max_model_len}")
        parts.append("trust_remote_code=True")
    return ",".join(parts)


def build_command(args: argparse.Namespace, *, tasks_arg: str, include_path: Optional[Path]) -> list[str]:
    entrypoint = _detect_lm_eval_entrypoint()
    cmd = list(entrypoint) + [
        "--model", args.backend,
        "--model_args", _build_model_args(args),
        "--tasks", tasks_arg,
        "--batch_size", str(args.batch_size),
        "--output_path", str(args.output_dir),
        "--seed", str(args.seed),
    ]
    if args.backend == "hf":
        cmd += ["--device", args.device]
    if args.log_samples:
        cmd.append("--log_samples")
    if args.apply_chat_template:
        cmd.append("--apply_chat_template")
    if args.fewshot_as_multiturn:
        cmd.append("--fewshot_as_multiturn")
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if include_path is not None:
        cmd += ["--include_path", str(include_path)]
    if args.extra_args:
        cmd += shlex.split(args.extra_args)
    return cmd


def write_run_metadata(args: argparse.Namespace, *, cmd: list[str], tasks_arg: str) -> None:
    meta = {
        "model_path": args.model_path,
        "backend": args.backend,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": args.device if args.backend == "hf" else None,
        "tensor_parallel_size": args.tensor_parallel_size or args.num_gpus,
        "gpu_memory_utilization": args.gpu_memory_utilization if args.backend == "vllm" else None,
        "max_model_len": args.max_model_len if args.backend == "vllm" else None,
        "apply_chat_template": bool(args.apply_chat_template),
        "fewshot_as_multiturn": bool(args.fewshot_as_multiturn),
        "log_samples": bool(args.log_samples),
        "limit": args.limit,
        "tasks": tasks_arg,
        "lm_eval_version": _lm_eval_version(),
        "command": cmd,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Open LLM Leaderboard v1 evaluation via lm-eval-harness.")
    p.add_argument("--model_path", required=True, help="HF model id or local checkpoint path.")
    p.add_argument("--output_dir", required=True, type=Path, help="Where lm-eval writes results and we write summaries.")
    p.add_argument("--backend", choices=["hf", "vllm"], default="vllm")
    p.add_argument("--batch_size", default="auto", help='Defaults to "auto". Pass an int for fixed batching.')
    p.add_argument("--dtype", default=None, help="Defaults to auto for vllm, bfloat16 for hf.")
    p.add_argument("--num_gpus", type=int, default=None, help="Convenience alias for --tensor_parallel_size.")
    p.add_argument("--tensor_parallel_size", type=int, default=None)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--device", default="cuda:0", help="HF backend device. Ignored for vllm.")
    p.add_argument("--apply_chat_template", action="store_true",
                   help="Off by default: evaluate base-LM style. Recorded in metadata when on.")
    p.add_argument("--fewshot_as_multiturn", action="store_true")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--log_samples", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="Optional task-item limit for smoke tests.")
    p.add_argument("--force_fallback", action="store_true",
                   help="Skip openllm-builtin probe and use the local YAML task group.")
    p.add_argument("--dry_run", action="store_true", help="Print the command and exit without running.")
    p.add_argument("--skip_aggregate", action="store_true", help="Do not produce summary.{json,csv,md}.")
    p.add_argument("--extra_args", default="", help="Extra raw flags passed through to lm-eval (shell-split).")
    args = p.parse_args()

    if args.dtype is None:
        args.dtype = "bfloat16" if args.backend == "hf" else "auto"
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def main() -> int:
    args = parse_args()
    entrypoint = _detect_lm_eval_entrypoint()

    if args.force_fallback or not _has_builtin_openllm_group(entrypoint):
        tasks_arg = FALLBACK_GROUP_NAME
        include_path = FALLBACK_TASK_DIR
        if args.force_fallback:
            print(f"[openllm-eval] forcing fallback task group '{tasks_arg}' from {include_path}")
        else:
            print(f"[openllm-eval] built-in 'openllm' group not detected; using fallback "
                  f"'{tasks_arg}' from {include_path}")
    else:
        tasks_arg = "openllm"
        include_path = None
        print("[openllm-eval] using built-in 'openllm' task group")

    cmd = build_command(args, tasks_arg=tasks_arg, include_path=include_path)

    pretty = " ".join(shlex.quote(c) for c in cmd)
    print("[openllm-eval] command:")
    print(f"  {pretty}")

    write_run_metadata(args, cmd=cmd, tasks_arg=tasks_arg)

    if args.dry_run:
        print("[openllm-eval] --dry_run set, not executing.")
        return 0

    rc = subprocess.call(cmd, env=os.environ.copy())
    print(f"[openllm-eval] lm-eval exited with code {rc}")

    if rc != 0:
        return rc

    if not args.skip_aggregate:
        try:
            from openllm.aggregate_results import aggregate
        except ModuleNotFoundError:
            # Allow direct `python openllm/run_openllm_eval.py` invocation by
            # importing the sibling module via its file path.
            sys.path.insert(0, str(OPENLLM_DIR))
            from aggregate_results import aggregate  # type: ignore

        try:
            aggregate(args.output_dir)
            print(f"[openllm-eval] summary written under {args.output_dir}")
        except Exception as e:
            print(f"[openllm-eval] aggregation failed: {e}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
