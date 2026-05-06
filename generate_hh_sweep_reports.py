#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "result"


def parse_number(token: str) -> float:
    return float(token.replace("p", "."))


def format_number(value: float) -> str:
    text = f"{value:.12g}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def parse_param(text: str, param: str) -> float | None:
    patterns = {
        "beta": [
            r"beta-([0-9]+p[0-9]+|[0-9]+(?:\.[0-9]+)?)",
            r"\bb([0-9]+p[0-9]+)\b",
            r"beta([0-9]+(?:\.[0-9]+)?)",
        ],
        "q_t": [
            r"q_t-([0-9]+(?:\.[0-9]+)?)",
            r"\bqt([0-9]{3})\b",
        ],
        "s_star": [
            r"s_star-([0-9]+(?:\.[0-9]+)?)",
            r"s_star([0-9]+(?:\.[0-9]+)?)",
        ],
        "eta": [
            r"eta-([0-9]+(?:\.[0-9]+)?)",
        ],
    }
    for pattern in patterns[param]:
        match = re.search(pattern, text)
        if not match:
            continue
        token = match.group(1)
        if param == "q_t" and token.isdigit() and len(token) == 3:
            return int(token) / 100
        return parse_number(token)
    return None


@dataclass
class EvalRecord:
    value: float
    win_rate: float
    total: int
    source: Path


@dataclass
class SectionSpec:
    title: str
    param: str
    fixed: str
    eval_globs: list[str]
    raw_globs: list[str]
    note: str | None = None


@dataclass
class PartSpec:
    title: str
    intro: str
    sections: list[SectionSpec]


@dataclass
class ReportSpec:
    title: str
    file_name: str
    overview: str
    parts: list[PartSpec]


def load_eval_records(spec: SectionSpec) -> dict[float, EvalRecord]:
    records: dict[float, EvalRecord] = {}
    for pattern in spec.eval_globs:
        for path in sorted(ROOT.glob(pattern)):
            value = parse_param(path.as_posix(), spec.param)
            if value is None:
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            record = EvalRecord(
                value=value,
                win_rate=float(data["win_rates"]["dpo"]),
                total=int(data["total"]),
                source=path,
            )
            current = records.get(value)
            if current is None or record.source.as_posix() < current.source.as_posix():
                records[value] = record
    return records


def load_raw_values(spec: SectionSpec) -> set[float]:
    values: set[float] = set()
    for pattern in spec.raw_globs:
        for path in sorted(ROOT.glob(pattern)):
            value = parse_param(path.name, spec.param)
            if value is not None:
                values.add(value)
    return values


def render_section(spec: SectionSpec) -> list[str]:
    eval_records = load_eval_records(spec)
    raw_values = load_raw_values(spec)
    values = sorted(set(eval_records) | raw_values)

    lines = [f"### {spec.title}", f"Fixed settings: `{spec.fixed}`."]
    if spec.note:
        lines.append(spec.note)

    if not values:
        lines.append("No local runs or evaluation summaries found.")
        lines.append("")
        return lines

    lines.append("")
    lines.append("| Value | DPO Win Rate | Judgments | Notes |")
    lines.append("| --- | ---: | ---: | --- |")
    for value in values:
        record = eval_records.get(value)
        if record is None:
            note = "Run found locally, but no local GPT-4 summary was found."
            lines.append(f"| `{format_number(value)}` | — | — | {note} |")
        else:
            note = f"`{os.path.relpath(record.source.resolve(), ROOT)}`"
            lines.append(
                f"| `{format_number(value)}` | {percent(record.win_rate)} | {record.total} | {note} |"
            )

    judged = sorted(eval_records.values(), key=lambda item: (-item.win_rate, item.value))
    if judged:
        best = judged[0]
        lines.append("")
        lines.append(
            f"Best judged value: `{format_number(best.value)}` at {percent(best.win_rate)} over {best.total} judgments."
        )
    else:
        lines.append("")
        lines.append("No judged summary was found for this section; the table only reflects discovered run folders.")

    lines.append("")
    return lines


def render_report(spec: ReportSpec) -> str:
    lines = [f"# {spec.title}", "", spec.overview, ""]
    for part in spec.parts:
        lines.append(f"## {part.title}")
        lines.append(part.intro)
        lines.append("")
        for section in part.sections:
            lines.extend(render_section(section))
    return "\n".join(lines).rstrip() + "\n"


QWEN_REPORT = ReportSpec(
    title="Qwen HH Sweep Report",
    file_name="qwen_hh_sweeps_report.md",
    overview=(
        "This report summarizes the locally available HH sweep artifacts for Qwen. "
        "Helpful `q_t` uses GPT-4 judged summaries from `outputs0`, while helpful `s_star` and `eta` use the locally downloaded run folders from `../wandb_logging_data` because no local judged summaries were found for those slices. "
        "Harmless scores come from the GPT-4 `prompts-general-less-harmful` summaries in `outputs0`. "
        "Per request, Qwen beta sweep is omitted."
    ),
    parts=[
        PartSpec(
            title="Helpful",
            intro="Qwen helpful sweeps.",
            sections=[
                SectionSpec(
                    title="q_t Sweep",
                    param="q_t",
                    fixed="beta=0.1, s_star=0.4, eta=0.1",
                    eval_globs=[
                        "outputs0/qwen-hh-hyper-sweep-harmless-all/qwen3-8b-base-new-dpo-hh-helpful-4xh200-batch-64-s_star-0.4-eta-0.1-q_t-*-multi/prompts-helpful/gpt-4/*summary.json",
                    ],
                    raw_globs=[
                        "../wandb_logging_data/qwen-hh-helpful-q_t-sweep/*",
                    ],
                ),
                SectionSpec(
                    title="s_star Sweep",
                    param="s_star",
                    fixed="beta=0.1, q_t=0.45, eta=0.1",
                    eval_globs=[],
                    raw_globs=[
                        "../wandb_logging_data/qwen-hh-new-dpo-multi-hyperparamter-sweep/helpful/beta-0.1-q_t-0.45-eta-0.1/*",
                    ],
                ),
                SectionSpec(
                    title="eta Sweep",
                    param="eta",
                    fixed="beta=0.1, q_t=0.45, s_star=0.4",
                    eval_globs=[],
                    raw_globs=[
                        "../wandb_logging_data/qwen-hh-new-dpo-multi-hyperparamter-sweep/helpful/beta-0.1-q_t-0.45-s_star-0.4/*",
                    ],
                ),
            ],
        ),
        PartSpec(
            title="Harmless",
            intro="Qwen harmless sweeps judged on the GPT-4 less-harmful prompt set.",
            sections=[
                SectionSpec(
                    title="q_t Sweep",
                    param="q_t",
                    fixed="beta=0.1, s_star=0.4, eta=0.1",
                    eval_globs=[
                        "outputs0/qwen-hh-hyper-sweep-harmless-all/qwen3-8b-base-new-dpo-hh-harmless-4xh200-batch-64-s_star-0.4-eta-0.1-q_t-*-multi/prompts-general-less-harmful/gpt-4/*summary.json",
                    ],
                    raw_globs=[
                        "../wandb_logging_data/qwen-hh-new-dpo-multi-hyperparamter-sweep/harmless/beta-0.1-s_star-0.4-eta-0.1/*",
                    ],
                ),
                SectionSpec(
                    title="s_star Sweep",
                    param="s_star",
                    fixed="beta=0.1, q_t=0.45, eta=0.1",
                    eval_globs=[
                        "outputs0/qwen-hh-hyper-sweep-harmless-all/qwen3-8b-base-new-dpo-hh-harmless-4xh200-batch-64-q_t-0.45-eta-0.1-s_star-*-multi/prompts-general-less-harmful/gpt-4/*summary.json",
                    ],
                    raw_globs=[
                        "../wandb_logging_data/qwen-hh-new-dpo-multi-hyperparamter-sweep/harmless/beta-0.1-q_t-0.45-eta-0.1/*",
                    ],
                ),
                SectionSpec(
                    title="eta Sweep",
                    param="eta",
                    fixed="beta=0.1, q_t=0.45, s_star=0.4",
                    eval_globs=[
                        "outputs0/qwen-hh-hyper-sweep-harmless-all/qwen3-8b-base-new-dpo-hh-harmless-4xh200-batch-64-q_t-0.45-s_star-0.4-eta-*-multi/prompts-general-less-harmful/gpt-4/*summary.json",
                    ],
                    raw_globs=[
                        "../wandb_logging_data/qwen-hh-new-dpo-multi-hyperparamter-sweep/harmless/beta-0.1-q_t-0.45-s_star-0.4/*",
                    ],
                ),
            ],
        ),
    ],
)


LLAMA_REPORT = ReportSpec(
    title="Llama HH Sweep Report",
    file_name="llama_hh_sweeps_report.md",
    overview=(
        "This report summarizes the locally available HH sweep artifacts for Llama. "
        "Helpful beta uses `outputs10/gpt_judge_HH`, helpful `q_t`/`s_star`/`eta` use `outputs 0.2`, "
        "and harmless beta uses `outputs10/gpt_judge_HH` while harmless `q_t`/`s_star`/`eta` use `outputs/gpt_judge_HH`."
    ),
    parts=[
        PartSpec(
            title="Helpful",
            intro="Llama helpful sweeps judged with GPT-4 helpful prompts.",
            sections=[
                SectionSpec(
                    title="beta Sweep",
                    param="beta",
                    fixed="q_t=0.45, s_star=0.4, eta=0.1",
                    eval_globs=[
                        "outputs10/gpt_judge_HH/helpful_base/multi_turn/llama3-hh-helpful-qt045-b0p*-20260429-085449-multi/prompts-helpful/gpt-4/*summary.json",
                    ],
                    raw_globs=[
                        "../wandb_logging_data/good/llama3-new-dpo-multi-hyperparamter-sweep/beta-sweep/helpful/*",
                        "../wandb_logging_data/archieved/wandb_llama3_hh_new_dpo_multi_beta_sweep/llama-3-8b-base-new-dpo-hh-helpful-4xh200-batch-64-s_star-0.4-eta-0.1-q_t-0.45-beta-*",
                    ],
                ),
                SectionSpec(
                    title="q_t Sweep",
                    param="q_t",
                    fixed="beta=0.1, s_star=0.4, eta=0.1",
                    eval_globs=[
                        "outputs 0.2/llama-3-8b-base-new-dpo-hh-helpful-4xh200-batch-64-s_star-0.4-eta-0.1-q_t-*-multi/prompts-helpful/gpt-4/*summary.json",
                    ],
                    raw_globs=[],
                ),
                SectionSpec(
                    title="s_star Sweep",
                    param="s_star",
                    fixed="beta=0.1, q_t=0.45, eta=0.1",
                    eval_globs=[
                        "outputs 0.2/llama-3-8b-base-new-dpo-hh-helpful-4xh200-batch-64-q_t-0.45-eta-0.1-s_star-*-multi/prompts-helpful/gpt-4/*summary.json",
                    ],
                    raw_globs=[
                        "../wandb_logging_data/good/llama3-new-dpo-multi-hyperparamter-sweep/s_star-sweep/helpful/*",
                        "../wandb_logging_data/archieved/wandb_llama3_hh_new_dpo_hyperparamter_sweep/llama-3-8b-base-new-dpo-hh-helpful-4xh200-batch-64-q_t-0.45-eta-0.1-s_star-*",
                    ],
                ),
                SectionSpec(
                    title="eta Sweep",
                    param="eta",
                    fixed="beta=0.1, q_t=0.45, s_star=0.4",
                    eval_globs=[
                        "outputs 0.2/llama-3-8b-base-new-dpo-hh-helpful-4xh200-batch-64-q_t-0.45-s_star-0.4-eta-*-multi/prompts-helpful/gpt-4/*summary.json",
                    ],
                    raw_globs=[],
                ),
            ],
        ),
        PartSpec(
            title="Harmless",
            intro="Llama harmless sweeps judged with GPT-4 harmless prompts.",
            sections=[
                SectionSpec(
                    title="beta Sweep",
                    param="beta",
                    fixed="q_t=0.45, s_star=0.4, eta=0.1",
                    eval_globs=[
                        "outputs10/gpt_judge_HH/harmless_base/multi_turn/llama3-hh-harmless-qt045-b0p*-20260429-085449-multi/prompts-harmless/gpt-4/*summary.json",
                    ],
                    raw_globs=[
                        "../wandb_logging_data/good/llama3-new-dpo-multi-hyperparamter-sweep/beta-sweep/harmless/*",
                        "../wandb_logging_data/archieved/wandb_llama3_hh_new_dpo_multi_beta_sweep/llama-3-8b-base-new-dpo-hh-harmless-4xh200-batch-64-s_star-0.4-eta-0.1-q_t-0.45-beta-*",
                    ],
                ),
                SectionSpec(
                    title="q_t Sweep",
                    param="q_t",
                    fixed="beta=0.1, s_star=0.4, eta=0.1",
                    eval_globs=[
                        "outputs/gpt_judge_HH/harmless_base/multi_turn/llama-3-8b-base-new-dpo-hh-harmless-4xh200-batch-64-s_star-0.4-eta-0.1-q_t-*-multi/prompts-harmless/gpt-4/*summary.json",
                    ],
                    raw_globs=[],
                ),
                SectionSpec(
                    title="s_star Sweep",
                    param="s_star",
                    fixed="beta=0.1, q_t=0.45, eta=0.1",
                    eval_globs=[
                        "outputs/gpt_judge_HH/harmless_base/multi_turn/llama-3-8b-base-new-dpo-hh-harmless-4xh200-batch-64-q_t-0.45-eta-0.1-s_star-*-multi/prompts-harmless/gpt-4/*summary.json",
                    ],
                    raw_globs=[
                        "../wandb_logging_data/good/llama3-new-dpo-multi-hyperparamter-sweep/s_star-sweep/harmless/*",
                        "../wandb_logging_data/archieved/wandb_llama3_hh_new_dpo_hyperparamter_sweep/llama-3-8b-base-new-dpo-hh-harmless-4xh200-batch-64-q_t-0.45-eta-0.1-s_star-*",
                    ],
                ),
                SectionSpec(
                    title="eta Sweep",
                    param="eta",
                    fixed="beta=0.1, q_t=0.45, s_star=0.4",
                    eval_globs=[
                        "outputs/gpt_judge_HH/harmless_base/multi_turn/llama-3-8b-base-new-dpo-hh-harmless-4xh200-batch-64-q_t-0.45-s_star-0.4-eta-*-multi/prompts-harmless/gpt-4/*summary.json",
                    ],
                    raw_globs=[],
                ),
            ],
        ),
    ],
)


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for spec in (QWEN_REPORT, LLAMA_REPORT):
        output_path = REPORTS_DIR / spec.file_name
        output_path.write_text(render_report(spec), encoding="utf-8")
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
