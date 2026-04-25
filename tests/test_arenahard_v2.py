from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from arenahard import arenahard_eval, arenahard_infer, batch_runner
from arenahard.official_runner import (
    build_api_config,
    build_gen_answer_config,
    build_judgment_config,
    filter_questions_by_category,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _base_config(tmp_path: Path) -> dict:
    question_file = tmp_path / "question.jsonl"
    baseline_file = tmp_path / "o3-mini-2025-01-31.jsonl"
    _write_jsonl(
        question_file,
        [
            {"uid": "hard-1", "category": "hard_prompt", "prompt": "Hard prompt"},
            {
                "uid": "creative-1",
                "category": "creative_writing",
                "prompt": "Creative prompt",
            },
        ],
    )
    _write_jsonl(
        baseline_file,
        [
            {
                "uid": "hard-1",
                "model": "o3-mini-2025-01-31",
                "messages": [{"role": "assistant", "content": {"answer": "base"}}],
                "metadata": {"token_len": 1},
            }
        ],
    )
    return {
        "_config_path": str(tmp_path / "config.yaml"),
        "arenahard": {
            "pretty_name": "my-model",
            "output_dir": str(tmp_path / "outputs" / "my-model"),
            "question_file": str(question_file),
            "baseline_answer_file": str(baseline_file),
            "categories": ["hard_prompt"],
            "model_endpoint": {
                "model": "my-model",
                "endpoints": [{"api_base": "http://127.0.0.1:8000/v1", "api_key": "-"}],
                "api_type": "openai",
                "parallel": 4,
                "max_tokens": 8192,
                "temperature": 0.0,
            },
            "judge_model": "gpt-4.1",
            "judge_endpoint": {
                "model": "gpt-4.1",
                "endpoints": None,
                "api_type": "openai",
                "parallel": 8,
                "max_tokens": 32000,
                "temperature": 0.0,
            },
        },
    }


class ArenaHardV2Tests(unittest.TestCase):
    def test_config_generation_matches_official_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _base_config(Path(tmp))

            self.assertEqual(
                build_gen_answer_config(config),
                {"bench_name": "arena-hard-v2.0", "model_list": ["my-model"]},
            )

            api_config = build_api_config(config)
            self.assertEqual(api_config["my-model"]["api_type"], "openai")
            self.assertEqual(api_config["gpt-4.1"]["parallel"], 8)

            judgment_config = build_judgment_config(config)
            self.assertEqual(judgment_config["judge_model"], "gpt-4.1")
            self.assertEqual(judgment_config["temperature"], 0.0)
            self.assertEqual(judgment_config["max_tokens"], 16000)
            self.assertEqual(judgment_config["model_list"], ["my-model"])
            self.assertIn("prompt_template", judgment_config)

    def test_question_filtering_keeps_hard_prompt_only(self) -> None:
        filtered = filter_questions_by_category(
            [
                {"uid": "hard", "category": "hard_prompt"},
                {"uid": "creative", "category": "creative_writing"},
            ],
            ["hard_prompt"],
        )

        self.assertEqual(filtered, [{"uid": "hard", "category": "hard_prompt"}])

    def test_inference_stages_runner_and_copies_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            runner_dir = tmp_path / "runner"
            runner_dir.mkdir()
            config = _base_config(tmp_path)

            def fake_run(command, *, cwd, check):
                answer_path = (
                    Path(cwd)
                    / "data"
                    / "arena-hard-v2.0"
                    / "model_answer"
                    / "my-model.jsonl"
                )
                _write_jsonl(answer_path, [{"uid": "hard-1", "model": "my-model"}])
                return subprocess.CompletedProcess(command, 0)

            with mock.patch.object(
                arenahard_infer,
                "ensure_official_runner",
                return_value=runner_dir,
            ), mock.patch.object(arenahard_infer.subprocess, "run", side_effect=fake_run):
                output_path = arenahard_infer.run_arenahard_inference(config)

            self.assertTrue(output_path.exists())
            staged_questions = [
                json.loads(line)
                for line in (
                    runner_dir / "data" / "arena-hard-v2.0" / "question.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["uid"] for row in staged_questions], ["hard-1"])
            self.assertTrue((runner_dir / "config" / "gen_answer_config.yaml").exists())
            self.assertTrue((runner_dir / "config" / "api_config.yaml").exists())

    def test_evaluation_captures_judgment_and_show_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            runner_dir = tmp_path / "runner"
            runner_dir.mkdir()
            config = _base_config(tmp_path)
            model_answer_path = tmp_path / "model_answer.jsonl"
            _write_jsonl(model_answer_path, [{"uid": "hard-1", "model": "my-model"}])

            def fake_run(command, *, cwd, check, text=False, stdout=None, stderr=None):
                if command[1] == "gen_judgment.py":
                    judgment_path = (
                        Path(cwd)
                        / "data"
                        / "arena-hard-v2.0"
                        / "model_judgment"
                        / "gpt-4.1"
                        / "my-model.jsonl"
                    )
                    _write_jsonl(judgment_path, [{"uid": "hard-1", "model": "my-model"}])
                    return subprocess.CompletedProcess(command, 0)
                return subprocess.CompletedProcess(command, 0, stdout="leaderboard\n")

            with mock.patch.object(
                arenahard_eval,
                "ensure_official_runner",
                return_value=runner_dir,
            ), mock.patch.object(arenahard_eval.subprocess, "run", side_effect=fake_run):
                results_dir = arenahard_eval.run_arenahard_evaluation(
                    config,
                    model_answer_path=str(model_answer_path),
                )

            self.assertTrue((results_dir / "show_result.txt").exists())
            self.assertEqual(
                (results_dir / "show_result.txt").read_text(encoding="utf-8"),
                "leaderboard\n",
            )
            self.assertTrue(
                (
                    results_dir
                    / "model_judgment"
                    / "gpt-4.1"
                    / "my-model.jsonl"
                ).exists()
            )

    def test_batch_matrix_uses_model_endpoint_and_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            base_config_path = tmp_path / "config_arenahard.yaml"
            batch_config_path = tmp_path / "config_arenahard_batch.yaml"
            base_config = _base_config(tmp_path)
            base_config_path.write_text(yaml.safe_dump(base_config), encoding="utf-8")
            batch_config = {
                "base_config": "config_arenahard.yaml",
                "models": [
                    {
                        "pretty_name": "model-a",
                        "model_endpoint": {
                            "model": "model-a",
                            "endpoints": [{"api_base": "http://localhost:8001/v1"}],
                            "api_type": "openai",
                        },
                    }
                ],
            }
            batch_config_path.write_text(yaml.safe_dump(batch_config), encoding="utf-8")

            matrix = batch_runner.build_run_matrix(
                batch_config,
                config_path=batch_config_path,
            )

            self.assertEqual(matrix[0]["pretty_name"], "model-a")
            self.assertIn("../outputs/arenahard/model-a", matrix[0]["config"]["arenahard"]["output_dir"])
            self.assertEqual(
                matrix[0]["config"]["arenahard"]["model_endpoint"]["model"],
                "model-a",
            )

    def test_cli_parsing_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(yaml.safe_dump(_base_config(Path(tmp))), encoding="utf-8")

            with mock.patch.object(arenahard_infer, "run_arenahard_inference") as run_infer:
                self.assertEqual(arenahard_infer.main(["--config", str(config_path)]), 0)
                self.assertTrue(run_infer.called)

            with mock.patch.object(arenahard_eval, "run_arenahard_evaluation") as run_eval:
                self.assertEqual(arenahard_eval.main(["--config", str(config_path)]), 0)
                self.assertTrue(run_eval.called)


if __name__ == "__main__":
    unittest.main()
