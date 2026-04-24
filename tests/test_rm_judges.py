from __future__ import annotations

import contextlib
import io
import json
import unittest
from pathlib import Path
import sys
import tempfile

import yaml

from gpt_judge_HH import judge_outputs_gpt4o
from gpt_judge_HH.judge_outputs_gpt4o import _resolve_prompt_text
from gpt_judge_HH.rm_judges import (
    ArmoRMJudge,
    JudgeResult,
    PairRMJudge,
    instruction_to_chat_messages,
)


class FakePairRMJudge(PairRMJudge):
    def __init__(self, logits: list[float]) -> None:
        super().__init__(model_name="fake-pairrm", tie_epsilon=0.0)
        self.logits = logits
        self.seen_prompt: str | None = None
        self.seen_pairs: list[tuple[str, str]] | None = None

    def _compare_pairs(
        self,
        prompt: str,
        labeled_outputs: dict[str, str],
        label_pairs: list[tuple[str, str]],
    ) -> list[float]:
        self.seen_prompt = prompt
        self.seen_pairs = label_pairs
        return self.logits


class FakeArmoRMJudge(ArmoRMJudge):
    def __init__(self, scores: list[float]) -> None:
        super().__init__(model_name="fake-armorm", tie_epsilon=0.0)
        self.pending_scores = list(scores)
        self.seen_messages: list[list[dict[str, str]]] = []

    def _score_messages(self, messages: list[dict[str, str]]) -> float:
        self.seen_messages.append(messages)
        return self.pending_scores.pop(0)


class RMJudgeTests(unittest.TestCase):
    def test_prompt_field_defaults_to_instruction_not_raw_instruction(self) -> None:
        prompt = _resolve_prompt_text(
            "canonical prompt",
            [
                (
                    "candidate",
                    {
                        "instruction": "canonical prompt",
                        "raw_instruction": "<chat-template>wrong prompt",
                        "output": "answer",
                    },
                )
            ],
            "instruction",
        )

        self.assertEqual(prompt, "canonical prompt")

    def test_pairrm_two_way_uses_instruction_as_source(self) -> None:
        judge = FakePairRMJudge([1.25])
        result = judge.judge("judge prompt", {"A": "good", "B": "bad"})

        self.assertEqual(judge.seen_prompt, "judge prompt")
        self.assertEqual(judge.seen_pairs, [("A", "B")])
        self.assertEqual(result.winner_label, "A")
        self.assertEqual(result.scores, {"A": 1.25, "B": -1.25})

    def test_pairrm_three_way_aggregates_pairwise_wins(self) -> None:
        judge = FakePairRMJudge([1.0, -2.0, -3.0])
        result = judge.judge("judge prompt", {"A": "a", "B": "b", "C": "c"})

        self.assertEqual(judge.seen_pairs, [("A", "B"), ("A", "C"), ("B", "C")])
        self.assertEqual(result.winner_label, "C")
        self.assertEqual(result.scores["C"]["wins"], 2)

    def test_instruction_to_chat_messages_single_turn(self) -> None:
        messages = instruction_to_chat_messages("What is tea?")

        self.assertEqual(messages, [{"role": "user", "content": "What is tea?"}])

    def test_instruction_to_chat_messages_hh_multiturn_strips_empty_assistant_cue(
        self,
    ) -> None:
        messages = instruction_to_chat_messages(
            "Human: Hi\n\nAssistant: Hello\n\nHuman: Continue\n\nAssistant:"
        )

        self.assertEqual(
            messages,
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "Continue"},
            ],
        )

    def test_armorm_appends_candidate_as_final_assistant_message(self) -> None:
        judge = FakeArmoRMJudge([0.1, 0.7])
        result = judge.judge("Human: Hi\n\nAssistant:", {"A": "hello", "B": "hi"})

        self.assertEqual(result.winner_label, "B")
        self.assertEqual(judge.seen_messages[0][-1], {"role": "assistant", "content": "hello"})
        self.assertEqual(judge.seen_messages[1][-1], {"role": "assistant", "content": "hi"})
        self.assertEqual(judge.seen_messages[0][0], {"role": "user", "content": "Hi"})

    def test_hh_judge_cli_uses_mocked_local_backend(self) -> None:
        class MockLocalJudge:
            def judge(self, prompt: str, labeled_outputs: dict[str, str]) -> JudgeResult:
                self.prompt = prompt
                self.labeled_outputs = labeled_outputs
                return JudgeResult(
                    winner_label="A",
                    comparison="mock comparison",
                    scores={"A": 1.0, "B": 0.0},
                    raw_backend_output={"backend": "mock"},
                )

        original_argv = sys.argv
        original_builder = judge_outputs_gpt4o.build_local_judge
        mock_judge = MockLocalJudge()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                chosen_path = tmp_path / "chosen.json"
                model_path = tmp_path / "model.json"
                results_path = tmp_path / "results.jsonl"
                summary_path = tmp_path / "summary.json"
                config_path = tmp_path / "config.yaml"

                chosen_path.write_text(
                    json.dumps(
                        [
                            {
                                "instruction": "prompt",
                                "raw_instruction": "wrong raw prompt",
                                "output": "chosen",
                            }
                        ]
                    ),
                    encoding="utf-8",
                )
                model_path.write_text(
                    json.dumps([{"instruction": "prompt", "output": "model"}]),
                    encoding="utf-8",
                )
                config_path.write_text(
                    yaml.safe_dump(
                        {
                            "inputs": {
                                "chosen": str(chosen_path),
                                "beta_dpo": str(model_path),
                            },
                            "judge": {
                                "backend": "pairrm",
                                "model": "mock-model",
                                "candidate_keys": ["chosen", "beta_dpo"],
                                "prompt_field": "instruction",
                            },
                            "output": {
                                "results_file": str(results_path),
                                "summary_file": str(summary_path),
                            },
                        }
                    ),
                    encoding="utf-8",
                )

                judge_outputs_gpt4o.build_local_judge = lambda *args, **kwargs: mock_judge
                sys.argv = ["hh-judge", "--config", str(config_path)]
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                    io.StringIO()
                ):
                    judge_outputs_gpt4o.main()

                rows = [
                    json.loads(line)
                    for line in results_path.read_text(encoding="utf-8").splitlines()
                ]
                summary = json.loads(summary_path.read_text(encoding="utf-8"))

                self.assertEqual(mock_judge.prompt, "prompt")
                self.assertEqual(rows[0]["judge_backend"], "pairrm")
                self.assertEqual(rows[0]["scores"], {"A": 1.0, "B": 0.0})
                self.assertEqual(rows[0]["raw_backend_output"], {"backend": "mock"})
                self.assertEqual(summary["judge_backend"], "pairrm")
        finally:
            judge_outputs_gpt4o.build_local_judge = original_builder
            sys.argv = original_argv

    def test_hh_judge_cli_reports_missing_candidate_input(self) -> None:
        original_argv = sys.argv
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                chosen_path = tmp_path / "chosen.json"
                actual_model_path = tmp_path / "llama-3-8b-base-new-dpo-hh-harmless-s_star0.6-4xh200-batch-64-20260421-213851-multi.json"
                missing_model_path = tmp_path / "llama-3-8b-base-new-dpo-hh-harmless-s_star0.6-4xh200-batch-64-20260422-051621-multi.json"
                results_path = tmp_path / "results.jsonl"
                summary_path = tmp_path / "summary.json"
                config_path = tmp_path / "config.yaml"

                chosen_path.write_text(
                    json.dumps([{"instruction": "prompt", "output": "chosen"}]),
                    encoding="utf-8",
                )
                actual_model_path.write_text(
                    json.dumps([{"instruction": "prompt", "output": "model"}]),
                    encoding="utf-8",
                )
                config_path.write_text(
                    yaml.safe_dump(
                        {
                            "inputs": {
                                "chosen": str(chosen_path),
                                "beta_dpo": str(missing_model_path),
                            },
                            "judge": {
                                "backend": "pairrm",
                                "model": "mock-model",
                                "candidate_keys": ["chosen", "beta_dpo"],
                            },
                            "output": {
                                "results_file": str(results_path),
                                "summary_file": str(summary_path),
                            },
                        }
                    ),
                    encoding="utf-8",
                )

                sys.argv = ["hh-judge", "--config", str(config_path)]
                with self.assertRaises(FileNotFoundError) as exc:
                    judge_outputs_gpt4o.main()

                message = str(exc.exception)
                self.assertIn("beta_dpo", message)
                self.assertIn(str(missing_model_path), message)
                self.assertIn(str(actual_model_path), message)
        finally:
            sys.argv = original_argv


if __name__ == "__main__":
    unittest.main()
