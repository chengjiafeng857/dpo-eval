from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from gpt_judge_HH.data_utils import parse_hh_to_messages


HH_ROLE_RE = re.compile(r"(?:^|\n\n)(Human|Assistant):")
FINAL_ASSISTANT_CUE_RE = re.compile(r"(?:\n\n)?Assistant:\s*$")


@dataclass
class JudgeResult:
    winner_label: str
    scores: dict[str, Any]
    raw_backend_output: dict[str, Any]
    comparison: str | None = None


def instruction_to_chat_messages(instruction: str) -> list[dict[str, str]]:
    """Convert an HH judge instruction into chat messages for chat-template RMs."""
    text = str(instruction).replace("\r\n", "\n").replace("\r", "\n").strip()
    if HH_ROLE_RE.search(text):
        messages = parse_hh_to_messages(text)
        if messages:
            return messages

    cleaned = FINAL_ASSISTANT_CUE_RE.sub("", text).strip()
    return [{"role": "user", "content": cleaned}]


def _float_list(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (int, float, bool)):
        return [float(value)]
    result: list[float] = []
    for item in value:
        if isinstance(item, list):
            if not item:
                result.append(0.0)
            else:
                result.append(float(item[0]))
        else:
            result.append(float(item))
    return result


class PairRMJudge:
    """Pairwise Reward Model judge using the official LLM-Blender wrapper."""

    def __init__(
        self,
        model_name: str = "llm-blender/PairRM",
        *,
        batch_size: int = 8,
        tie_epsilon: float = 0.0,
    ) -> None:
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.tie_epsilon = float(tie_epsilon)
        self._blender = None

    @property
    def blender(self) -> Any:
        if self._blender is None:
            try:
                import llm_blender
            except ImportError as exc:
                raise ImportError(
                    "PairRM judging requires llm-blender. Install the project "
                    "dependencies or run: pip install "
                    "git+https://github.com/yuchenlin/LLM-Blender.git"
                ) from exc

            blender = llm_blender.Blender()
            blender.loadranker(self.model_name)
            self._blender = blender
        return self._blender

    def _compare_pairs(
        self,
        prompt: str,
        labeled_outputs: dict[str, str],
        label_pairs: list[tuple[str, str]],
    ) -> list[float]:
        inputs = [prompt] * len(label_pairs)
        candidates_a = [labeled_outputs[left] for left, _ in label_pairs]
        candidates_b = [labeled_outputs[right] for _, right in label_pairs]

        try:
            logits = self.blender.compare(
                inputs,
                candidates_a,
                candidates_b,
                return_logits=True,
                mode="[A,B]",
                batch_size=self.batch_size,
            )
        except TypeError:
            logits = self.blender.compare(
                inputs,
                candidates_a,
                candidates_b,
                return_logits=True,
                mode="[A,B]",
            )
        return _float_list(logits)

    def judge(self, prompt: str, labeled_outputs: dict[str, str]) -> JudgeResult:
        labels = list(labeled_outputs.keys())
        if len(labels) not in (2, 3):
            raise ValueError("PairRMJudge supports two or three labeled candidates.")

        pair_indices = [
            (labels[left], labels[right])
            for left in range(len(labels))
            for right in range(left + 1, len(labels))
        ]
        logits = self._compare_pairs(prompt, labeled_outputs, pair_indices)

        wins = {label: 0 for label in labels}
        margins = {label: 0.0 for label in labels}
        pairwise: list[dict[str, Any]] = []
        for (left, right), logit in zip(pair_indices, logits):
            if logit > self.tie_epsilon:
                pair_winner = left
                wins[left] += 1
                margins[left] += logit
                margins[right] -= logit
            elif logit < -self.tie_epsilon:
                pair_winner = right
                wins[right] += 1
                margins[left] += logit
                margins[right] -= logit
            else:
                pair_winner = "TIE"

            pairwise.append(
                {
                    "left": left,
                    "right": right,
                    "logit_left_minus_right": logit,
                    "winner": pair_winner,
                }
            )

        if len(labels) == 2:
            winner = pairwise[0]["winner"]
            if winner == "TIE":
                comparison = "PairRM judged the pair as tied."
            else:
                comparison = (
                    f"PairRM selected {winner} with margin "
                    f"{abs(pairwise[0]['logit_left_minus_right']):.6g}."
                )
            return JudgeResult(
                winner_label=winner,
                comparison=comparison,
                scores={labels[0]: logits[0], labels[1]: -logits[0]},
                raw_backend_output={
                    "backend": "pairrm",
                    "model": self.model_name,
                    "pairwise": pairwise,
                    "tie_epsilon": self.tie_epsilon,
                },
            )

        max_wins = max(wins.values())
        win_candidates = [label for label, value in wins.items() if value == max_wins]
        if len(win_candidates) == 1:
            winner = win_candidates[0]
        else:
            best_margin = max(margins[label] for label in win_candidates)
            margin_candidates = [
                label
                for label in win_candidates
                if margins[label] == best_margin
            ]
            winner = margin_candidates[0] if len(margin_candidates) == 1 else "TIE"

        if winner == "TIE":
            comparison = "PairRM judged the candidates as tied after pairwise aggregation."
        else:
            comparison = (
                f"PairRM selected {winner} by pairwise win aggregation "
                f"({wins[winner]} wins)."
            )
        return JudgeResult(
            winner_label=winner,
            comparison=comparison,
            scores={
                label: {"wins": wins[label], "margin_sum": margins[label]}
                for label in labels
            },
            raw_backend_output={
                "backend": "pairrm",
                "model": self.model_name,
                "pairwise": pairwise,
                "wins": wins,
                "margin_sums": margins,
                "tie_epsilon": self.tie_epsilon,
            },
        )


class ArmoRMJudge:
    """Scalar reward-model judge using ArmoRM chat-message scoring."""

    def __init__(
        self,
        model_name: str = "RLHFlow/ArmoRM-Llama3-8B-v0.1",
        *,
        precision: str | None = "bf16",
        device_map: str | None = "auto",
        max_length: int = 4096,
        truncation: bool = True,
        trust_remote_code: bool = True,
        tie_epsilon: float = 0.0,
    ) -> None:
        self.model_name = model_name
        self.precision = precision
        self.device_map = device_map
        self.max_length = int(max_length)
        self.truncation = bool(truncation)
        self.trust_remote_code = bool(trust_remote_code)
        self.tie_epsilon = float(tie_epsilon)
        self._model = None
        self._tokenizer = None
        self._device = None

    @staticmethod
    def _ensure_llama_docstring() -> None:
        try:
            from transformers.models.llama import modeling_llama

            if not hasattr(modeling_llama, "LLAMA_INPUTS_DOCSTRING"):
                modeling_llama.LLAMA_INPUTS_DOCSTRING = ""
            if not hasattr(modeling_llama, "LLAMA_START_DOCSTRING"):
                modeling_llama.LLAMA_START_DOCSTRING = ""
        except Exception:
            pass

    def _load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "ArmoRM judging requires torch and transformers."
            ) from exc

        self._ensure_llama_docstring()

        dtype = None
        if self.precision:
            precision = self.precision.lower()
            if precision == "bf16":
                dtype = torch.bfloat16
            elif precision == "fp16":
                dtype = torch.float16
            elif precision in ("fp32", "float32"):
                dtype = torch.float32

        kwargs: dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
        if dtype is not None:
            kwargs["dtype"] = dtype
        if self.device_map:
            kwargs["device_map"] = self.device_map

        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            **kwargs,
        )
        self._model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        if not self.device_map:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(device)
            self._device = device
        else:
            try:
                self._device = self._model.device
            except AttributeError:
                self._device = next(self._model.parameters()).device

    def _score_messages(self, messages: list[dict[str, str]]) -> float:
        self._load()
        assert self._model is not None
        assert self._tokenizer is not None

        input_ids = self._tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length,
        )
        if self._device is not None:
            input_ids = input_ids.to(self._device)

        import torch

        with torch.inference_mode():
            output = self._model(input_ids)
            if hasattr(output, "score"):
                score = output.score
            else:
                score = output.logits.squeeze(-1)
        return float(score.detach().float().cpu().reshape(-1)[0].item())

    def judge(self, prompt: str, labeled_outputs: dict[str, str]) -> JudgeResult:
        labels = list(labeled_outputs.keys())
        if len(labels) not in (2, 3):
            raise ValueError("ArmoRMJudge supports two or three labeled candidates.")

        prompt_messages = instruction_to_chat_messages(prompt)
        scores: dict[str, float] = {}
        for label, output in labeled_outputs.items():
            messages = prompt_messages + [
                {"role": "assistant", "content": output},
            ]
            scores[label] = self._score_messages(messages)

        max_score = max(scores.values())
        winners = [
            label
            for label, score in scores.items()
            if abs(score - max_score) <= self.tie_epsilon
        ]
        winner = winners[0] if len(winners) == 1 else "TIE"
        if winner == "TIE":
            comparison = "ArmoRM judged the candidates as tied by scalar reward score."
        else:
            comparison = f"ArmoRM selected {winner} with score {scores[winner]:.6g}."

        return JudgeResult(
            winner_label=winner,
            comparison=comparison,
            scores=scores,
            raw_backend_output={
                "backend": "armorm",
                "model": self.model_name,
                "scores": scores,
                "prompt_messages": prompt_messages,
                "max_length": self.max_length,
                "truncation": self.truncation,
                "tie_epsilon": self.tie_epsilon,
            },
        )


def build_local_judge(backend: str, judge_cfg: dict[str, Any], model_name: str) -> Any:
    normalized = backend.strip().lower()
    if normalized == "pairrm":
        return PairRMJudge(
            model_name=model_name,
            batch_size=int(judge_cfg.get("batch_size", 8)),
            tie_epsilon=float(judge_cfg.get("tie_epsilon", 0.0)),
        )
    if normalized == "armorm":
        return ArmoRMJudge(
            model_name=model_name,
            precision=judge_cfg.get("precision", "bf16"),
            device_map=judge_cfg.get("device_map", "auto"),
            max_length=int(judge_cfg.get("max_length", 4096)),
            truncation=bool(judge_cfg.get("truncation", True)),
            trust_remote_code=bool(judge_cfg.get("trust_remote_code", True)),
            tie_epsilon=float(judge_cfg.get("tie_epsilon", 0.0)),
        )
    raise ValueError(f"Unsupported local judge backend: {backend}")
