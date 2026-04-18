from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from config_utils import load_yaml

from arenahard.batch_runner import build_run_matrix
from arenahard.common import BLOCK_NAME, build_answer_row, load_questions, read_jsonl, write_jsonl
from arenahard.infer import run_arenahard_inference
from arenahard.judge import run_arenahard_judging
from arenahard.report import run_arenahard_report


def _write_yaml(path: Path, payload: str) -> None:
    path.write_text(payload, encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    text = ""
    for row in rows:
        text += json.dumps(row) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class ArenaHardTests(unittest.TestCase):
    def test_question_download_and_max_instances(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            _write_yaml(
                config_path,
                f"""
policy_name: demo/model
arenahard:
  benchmark_dir: {root / "data"}
  question_file: question.jsonl
  max_instances: 1
""",
            )
            config = load_yaml(config_path)
            payload = (
                json.dumps(
                    {
                        "uid": "q1",
                        "prompt": "One",
                        "category": "arena-hard-v0.1",
                        "cluster": "demo",
                    }
                )
                + "\n"
                + json.dumps({"uid": "q2", "prompt": "Two", "category": "arena-hard-v0.1"})
                + "\n"
            )

            class _Response:
                def __enter__(self) -> io.BytesIO:
                    return io.BytesIO(payload.encode("utf-8"))

                def __exit__(self, exc_type, exc, tb) -> bool:
                    return False

            with mock.patch("arenahard.common.urllib.request.urlopen", return_value=_Response()):
                questions = load_questions(config)

            self.assertEqual(len(questions), 1)
            self.assertEqual(questions[0]["category"], "arena-hard-v0.1")
            self.assertEqual(questions[0]["cluster"], "demo")
            self.assertTrue((root / "data" / "question.jsonl").exists())

    def test_inference_local_writes_official_answer_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            question_file = root / "question.jsonl"
            _write_jsonl(
                question_file,
                [
                    {"uid": "q1", "prompt": "Prompt 1", "category": "arena-hard-v0.1"},
                    {"uid": "q2", "prompt": "Prompt 2", "category": "arena-hard-v0.1"},
                ],
            )
            config_path = root / "config.yaml"
            _write_yaml(
                config_path,
                f"""
policy_name: demo/model
arenahard:
  benchmark_dir: {root / "data"}
  pretty_name: demo-model
  model_name_or_path: demo/model
  mode: local
  backend: transformers
  question_file: {question_file}
""",
            )
            config = load_yaml(config_path)
            with mock.patch("arenahard.infer._generate_local_answers", return_value=["alpha", "beta"]):
                answer_path = run_arenahard_inference(config)

            rows = read_jsonl(answer_path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["uid"], "q1")
            self.assertEqual(rows[0]["model"], "demo-model")
            self.assertEqual(rows[0]["messages"][-1]["content"]["answer"], "alpha")
            self.assertIn("token_len", rows[0]["metadata"])
            self.assertIn("header_count", rows[0]["metadata"])

    def test_judging_uses_v1_baseline_and_judge_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            question = {"uid": "q1", "prompt": "Prompt 1", "category": "arena-hard-v0.1"}
            question_file = root / "question.jsonl"
            _write_jsonl(question_file, [question])
            config_path = root / "config.yaml"
            _write_yaml(
                config_path,
                f"""
policy_name: demo/model
arenahard:
  benchmark_dir: {root / "data"}
  pretty_name: demo-model
  model_name_or_path: demo/model
  question_file: {question_file}
  endpoint_file: {root / "api.yaml"}
  judge_parallel: 1
  judge_checkpoint_every: 1
""",
            )
            _write_yaml(
                root / "api.yaml",
                """
gpt-4-1106-preview:
  model: gpt-4-1106-preview
  endpoints: null
  api_type: openai
  parallel: 1
  max_tokens: 4096
  temperature: 0.0
""",
            )
            config = load_yaml(config_path)
            answer_dir = root / "data" / "model_answer"
            write_jsonl(
                answer_dir / "demo-model.jsonl",
                [
                    build_answer_row(
                        model_name="demo-model",
                        question=question,
                        answer_text="candidate",
                    )
                ],
            )
            write_jsonl(
                answer_dir / "gpt-4-0314.jsonl",
                [
                    build_answer_row(
                        model_name="gpt-4-0314",
                        question=question,
                        answer_text="baseline",
                    )
                ],
            )

            responses = [
                {"answer": "Assistant B is better. [[B>A]]"},
                {"answer": "Assistant A is better. [[A>B]]"},
            ]
            with mock.patch("arenahard.judge.create_chat_completion", side_effect=responses) as mocked_create:
                judgment_path = run_arenahard_judging(config)

            rows = read_jsonl(judgment_path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["judge"], "gpt-4-1106-preview")
            self.assertEqual(rows[0]["baseline"], "gpt-4-0314")
            self.assertEqual(rows[0]["games"][0]["score"], "B>A")
            self.assertEqual(rows[0]["games"][1]["score"], "A>B")
            first_call = mocked_create.call_args_list[0].kwargs
            self.assertEqual(first_call["settings"]["model"], "gpt-4-1106-preview")
            self.assertIn("gpt-4-0314", rows[0]["baseline"])

    def test_report_builds_v1_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            _write_yaml(
                config_path,
                f"""
policy_name: demo/model
arenahard:
  benchmark_dir: {root / "data"}
  pretty_name: demo-model
  model_name_or_path: demo/model
  bootstrap_rounds: 1
""",
            )
            judgment_dir = root / "data" / "model_judgment" / "gpt-4-1106-preview"
            _write_jsonl(
                judgment_dir / "demo-model.jsonl",
                [
                    {
                        "uid": "q1",
                        "category": "arena-hard-v0.1",
                        "judge": "gpt-4-1106-preview",
                        "model": "demo-model",
                        "baseline": "gpt-4-0314",
                        "games": [
                            {"score": "B>A"},
                            {"score": "A>B"},
                        ],
                    }
                ],
            )
            config = load_yaml(config_path)
            tables = run_arenahard_report(
                config,
                judge_names=["gpt-4-1106-preview"],
                categories=["arena-hard-v0.1"],
            )

            self.assertIn("demo-model", tables["arena-hard-v0.1"])
            self.assertIn("gpt-4-0314", tables["arena-hard-v0.1"])

    def test_batch_matrix_uses_arenahard_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_config = root / "base.yaml"
            batch_config = root / "batch.yaml"
            _write_yaml(
                base_config,
                """
policy_name: base/model
arenahard:
  benchmark_dir: ../data/arena-hard-v0.1
  pretty_name: base-model
  model_name_or_path: base/model
  generation:
    max_new_tokens: 16
""",
            )
            _write_yaml(
                batch_config,
                """
base_config: base.yaml
models:
  - model_name_or_path: org/qwen3-demo
    pretty_name: qwen3-demo
""",
            )
            config = load_yaml(batch_config)
            run_matrix = build_run_matrix(config, config_path=batch_config)

            self.assertEqual(run_matrix[0]["pretty_name"], "qwen3-demo")
            self.assertEqual(run_matrix[0]["config"][BLOCK_NAME]["model_name_or_path"], "org/qwen3-demo")
            self.assertFalse(run_matrix[0]["config"][BLOCK_NAME]["use_custom_chat_template"])


if __name__ == "__main__":
    unittest.main()
