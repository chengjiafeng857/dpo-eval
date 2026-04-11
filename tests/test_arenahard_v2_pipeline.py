from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from config_utils import load_yaml

from arenahard_v2.batch_runner import build_run_matrix
from arenahard_v2.common import BLOCK_NAME, load_questions, read_jsonl
from arenahard_v2.infer import run_arenahard_v2_inference
from arenahard_v2.judge import run_arenahard_v2_judging
from arenahard_v2.report import run_arenahard_v2_report


def _write_yaml(path: Path, payload: str) -> None:
    path.write_text(payload, encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    text = ""
    for row in rows:
        text += json.dumps(row) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class ArenaHardV2Tests(unittest.TestCase):
    def test_question_download_and_max_instances(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            _write_yaml(
                config_path,
                f"""
policy_name: demo/model
arenahard_v2:
  benchmark_dir: {root / "data"}
  question_file: question.jsonl
  max_instances: 1
""",
            )
            config = load_yaml(config_path)
            payload = (
                json.dumps({"uid": "q1", "prompt": "One", "category": "hard_prompt"})
                + "\n"
                + json.dumps({"uid": "q2", "prompt": "Two", "category": "creative_writing"})
                + "\n"
            )

            class _Response:
                def __enter__(self) -> io.BytesIO:
                    return io.BytesIO(payload.encode("utf-8"))

                def __exit__(self, exc_type, exc, tb) -> bool:
                    return False

            with mock.patch("arenahard_v2.common.urllib.request.urlopen", return_value=_Response()):
                questions = load_questions(config)

            self.assertEqual(len(questions), 1)
            self.assertEqual(questions[0]["category"], "hard_prompt")
            self.assertTrue((root / "data" / "question.jsonl").exists())

    def test_inference_local_writes_official_answer_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            question_file = root / "question.jsonl"
            _write_jsonl(
                question_file,
                [
                    {"uid": "q1", "prompt": "Prompt 1", "category": "hard_prompt"},
                    {"uid": "q2", "prompt": "Prompt 2", "category": "creative_writing"},
                ],
            )
            config_path = root / "config.yaml"
            _write_yaml(
                config_path,
                f"""
policy_name: demo/model
arenahard_v2:
  benchmark_dir: {root / "data"}
  pretty_name: demo-model
  model_name_or_path: demo/model
  mode: local
  backend: transformers
  question_file: {question_file}
""",
            )
            config = load_yaml(config_path)
            with mock.patch("arenahard_v2.infer._generate_local_answers", return_value=["alpha", "beta"]):
                answer_path = run_arenahard_v2_inference(config)

            rows = read_jsonl(answer_path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["uid"], "q1")
            self.assertEqual(rows[0]["model"], "demo-model")
            self.assertEqual(rows[0]["messages"][-1]["content"]["answer"], "alpha")
            self.assertIn("token_len", rows[0]["metadata"])
            self.assertIn("header_count", rows[0]["metadata"])

    def test_inference_endpoint_mode_uses_endpoint_generator(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            question_file = root / "question.jsonl"
            _write_jsonl(
                question_file,
                [{"uid": "q1", "prompt": "Prompt 1", "category": "hard_prompt"}],
            )
            config_path = root / "config.yaml"
            _write_yaml(
                config_path,
                f"""
policy_name: demo/model
arenahard_v2:
  benchmark_dir: {root / "data"}
  pretty_name: endpoint-model
  model_name_or_path: demo/model
  mode: endpoint
  endpoint_file: {root / "api.yaml"}
  endpoint_name: endpoint-model
  question_file: {question_file}
""",
            )
            _write_yaml(
                root / "api.yaml",
                """
endpoint-model:
  model: endpoint-model
  endpoints:
    - api_base: http://127.0.0.1:8000/v1
      api_key: token
  api_type: openai
  parallel: 2
  max_tokens: 16
  temperature: 0.0
""",
            )
            config = load_yaml(config_path)
            with mock.patch("arenahard_v2.infer._generate_endpoint_answers", return_value=["remote-answer"]):
                answer_path = run_arenahard_v2_inference(config)

            rows = read_jsonl(answer_path)
            self.assertEqual(rows[0]["messages"][-1]["content"]["answer"], "remote-answer")

    def test_judging_parses_both_rounds_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            benchmark_dir = root / "data"
            question_file = root / "question.jsonl"
            _write_jsonl(
                question_file,
                [{"uid": "q1", "prompt": "Prompt 1", "category": "hard_prompt"}],
            )
            answer_dir = benchmark_dir / "model_answer"
            _write_jsonl(
                answer_dir / "candidate.jsonl",
                [
                    {
                        "uid": "q1",
                        "ans_id": "a1",
                        "model": "candidate",
                        "messages": [
                            {"role": "user", "content": "Prompt 1"},
                            {"role": "assistant", "content": {"answer": "candidate answer"}},
                        ],
                        "tstamp": 0.0,
                        "metadata": {"token_len": 2, "header_count": {}, "list_count": {}, "bold_count": {}},
                    }
                ],
            )
            _write_jsonl(
                answer_dir / "o3-mini-2025-01-31.jsonl",
                [
                    {
                        "uid": "q1",
                        "ans_id": "a2",
                        "model": "o3-mini-2025-01-31",
                        "messages": [
                            {"role": "user", "content": "Prompt 1"},
                            {"role": "assistant", "content": {"answer": "baseline answer"}},
                        ],
                        "tstamp": 0.0,
                        "metadata": {"token_len": 2, "header_count": {}, "list_count": {}, "bold_count": {}},
                    }
                ],
            )
            config_path = root / "config.yaml"
            _write_yaml(
                config_path,
                f"""
policy_name: candidate
arenahard_v2:
  benchmark_dir: {benchmark_dir}
  pretty_name: candidate
  model_name_or_path: candidate
  question_file: {question_file}
  endpoint_file: {root / "api.yaml"}
  judge_model: gpt-4.1
  judge_endpoint_name: gpt-4.1
""",
            )
            _write_yaml(
                root / "api.yaml",
                """
gpt-4.1:
  model: gpt-4.1
  endpoints: null
  api_type: openai
  parallel: 2
  max_tokens: 128
  temperature: 0.0
""",
            )
            config = load_yaml(config_path)
            responses = [{"answer": "reason [[A>B]]"}, {"answer": "reason [A<B]"}]
            with mock.patch("arenahard_v2.judge.create_chat_completion", side_effect=responses):
                judgment_path = run_arenahard_v2_judging(config)

            rows = read_jsonl(judgment_path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["games"][0]["score"], "A>B")
            self.assertEqual(rows[0]["games"][1]["score"], "A<B")

            with mock.patch("arenahard_v2.judge.create_chat_completion", side_effect=RuntimeError("should not be called")):
                second_path = run_arenahard_v2_judging(config)
            self.assertEqual(judgment_path, second_path)

    def test_report_raw_and_style_controlled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            benchmark_dir = root / "data"
            config_path = root / "config.yaml"
            _write_yaml(
                config_path,
                f"""
policy_name: candidate
arenahard_v2:
  benchmark_dir: {benchmark_dir}
  bootstrap_rounds: 4
""",
            )
            answer_dir = benchmark_dir / "model_answer"
            _write_jsonl(
                answer_dir / "candidate.jsonl",
                [
                    {
                        "uid": "q1",
                        "model": "candidate",
                        "messages": [{"role": "assistant", "content": {"answer": "a"}}],
                        "metadata": {
                            "token_len": 10,
                            "header_count": {"h1": 1},
                            "list_count": {"ordered": 0, "unordered": 1},
                            "bold_count": {"**": 1, "__": 0},
                        },
                    },
                    {
                        "uid": "q2",
                        "model": "candidate",
                        "messages": [{"role": "assistant", "content": {"answer": "b"}}],
                        "metadata": {
                            "token_len": 12,
                            "header_count": {"h1": 0},
                            "list_count": {"ordered": 1, "unordered": 0},
                            "bold_count": {"**": 0, "__": 1},
                        },
                    },
                ],
            )
            _write_jsonl(
                answer_dir / "o3-mini-2025-01-31.jsonl",
                [
                    {
                        "uid": "q1",
                        "model": "o3-mini-2025-01-31",
                        "messages": [{"role": "assistant", "content": {"answer": "base"}}],
                        "metadata": {
                            "token_len": 8,
                            "header_count": {"h1": 0},
                            "list_count": {"ordered": 0, "unordered": 0},
                            "bold_count": {"**": 0, "__": 0},
                        },
                    },
                    {
                        "uid": "q2",
                        "model": "o3-mini-2025-01-31",
                        "messages": [{"role": "assistant", "content": {"answer": "base"}}],
                        "metadata": {
                            "token_len": 8,
                            "header_count": {"h1": 0},
                            "list_count": {"ordered": 0, "unordered": 0},
                            "bold_count": {"**": 0, "__": 0},
                        },
                    },
                ],
            )
            judgment_dir = benchmark_dir / "model_judgment" / "gpt-4.1"
            _write_jsonl(
                judgment_dir / "candidate.jsonl",
                [
                    {
                        "uid": "q1",
                        "category": "hard_prompt",
                        "judge": "gpt-4.1",
                        "model": "candidate",
                        "baseline": "o3-mini-2025-01-31",
                        "games": [{"score": "A>B"}, {"score": "A<B"}],
                    },
                    {
                        "uid": "q2",
                        "category": "hard_prompt",
                        "judge": "gpt-4.1",
                        "model": "candidate",
                        "baseline": "o3-mini-2025-01-31",
                        "games": [{"score": "A>>B"}, {"score": "A<<B"}],
                    },
                ],
            )
            config = load_yaml(config_path)
            raw_tables = run_arenahard_v2_report(config, judge_names=["gpt-4.1"], categories=["hard_prompt"])
            self.assertIn("candidate", raw_tables["hard_prompt"])
            style_tables = run_arenahard_v2_report(
                config,
                judge_names=["gpt-4.1"],
                categories=["hard_prompt"],
                control_features=["markdown", "length"],
            )
            self.assertIn("candidate", style_tables["hard_prompt"])
            self.assertIn("o3-mini-2025-01-31", style_tables["hard_prompt"])

    def test_batch_run_matrix_applies_family_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_config = root / "base.yaml"
            batch_config = root / "batch.yaml"
            _write_yaml(
                base_config,
                """
policy_name: demo/model
arenahard_v2:
  benchmark_dir: ../data/arena-hard-v2.0
  judge_model: gpt-4.1
  generation:
    max_new_tokens: 256
""",
            )
            _write_yaml(
                batch_config,
                """
base_config: base.yaml
run_inference: true
run_judging: true
models:
  - model_name_or_path: demo/llama-3-8b
    pretty_name: ultrachat-llama-3-8b-sft
  - model_name_or_path: demo/qwen3-8b
    pretty_name: ultrachat-qwen3-8b-sft
""",
            )
            matrix = build_run_matrix(load_yaml(batch_config), config_path=batch_config.resolve())
            llama_cfg = matrix[0]["config"][BLOCK_NAME]
            qwen_cfg = matrix[1]["config"][BLOCK_NAME]
            self.assertTrue(llama_cfg["use_custom_chat_template"])
            self.assertIn("../mtbench/templates/ultrachat-llama-3-8b-sft.jinja", llama_cfg["prompt_template"])
            self.assertFalse(qwen_cfg["use_custom_chat_template"])
            self.assertEqual(qwen_cfg["generation"]["stop_token_ids"], [151645])

    def test_smoke_infer_judge_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            question_file = root / "question.jsonl"
            _write_jsonl(
                question_file,
                [
                    {"uid": "q1", "prompt": "Prompt 1", "category": "hard_prompt"},
                    {"uid": "q2", "prompt": "Prompt 2", "category": "hard_prompt"},
                ],
            )
            benchmark_dir = root / "data"
            answer_dir = benchmark_dir / "model_answer"
            _write_jsonl(
                answer_dir / "o3-mini-2025-01-31.jsonl",
                [
                    {
                        "uid": "q1",
                        "ans_id": "base1",
                        "model": "o3-mini-2025-01-31",
                        "messages": [{"role": "assistant", "content": {"answer": "base1"}}],
                        "tstamp": 0.0,
                        "metadata": {"token_len": 5, "header_count": {}, "list_count": {}, "bold_count": {}},
                    },
                    {
                        "uid": "q2",
                        "ans_id": "base2",
                        "model": "o3-mini-2025-01-31",
                        "messages": [{"role": "assistant", "content": {"answer": "base2"}}],
                        "tstamp": 0.0,
                        "metadata": {"token_len": 5, "header_count": {}, "list_count": {}, "bold_count": {}},
                    },
                ],
            )
            config_path = root / "config.yaml"
            _write_yaml(
                config_path,
                f"""
policy_name: candidate
arenahard_v2:
  benchmark_dir: {benchmark_dir}
  pretty_name: candidate
  model_name_or_path: candidate
  mode: local
  question_file: {question_file}
  endpoint_file: {root / "api.yaml"}
  judge_model: gpt-4.1
  judge_endpoint_name: gpt-4.1
  bootstrap_rounds: 3
""",
            )
            _write_yaml(
                root / "api.yaml",
                """
gpt-4.1:
  model: gpt-4.1
  endpoints: null
  api_type: openai
  parallel: 2
  max_tokens: 32
  temperature: 0.0
""",
            )
            config = load_yaml(config_path)
            with mock.patch("arenahard_v2.infer._generate_local_answers", return_value=["cand1", "cand2"]):
                run_arenahard_v2_inference(config)
            with mock.patch(
                "arenahard_v2.judge.create_chat_completion",
                side_effect=[{"answer": "[[A>B]]"}, {"answer": "[A<B]"}, {"answer": "[[A=B]]"}, {"answer": "[A=B]"}],
            ):
                run_arenahard_v2_judging(config)
            tables = run_arenahard_v2_report(config, judge_names=["gpt-4.1"], categories=["hard_prompt"])
            self.assertIn("candidate", tables["hard_prompt"])


if __name__ == "__main__":
    unittest.main()
