from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from config_utils import load_yaml

from arenahard_v2.batch_runner import build_run_matrix
from arenahard_v2.common import BLOCK_NAME, load_questions, read_jsonl
from arenahard_v2.endpoint import EndpointPool, create_chat_completion
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

    def test_endpoint_logs_retries_and_raises_with_context(self) -> None:
        class _FailingCompletions:
            def create(self, **kwargs: object) -> object:
                raise RuntimeError("simulated endpoint outage")

        class _FailingChat:
            completions = _FailingCompletions()

        class _FailingClient:
            chat = _FailingChat()

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with mock.patch("arenahard_v2.endpoint.OpenAI", return_value=_FailingClient()):
                with self.assertRaises(RuntimeError) as ctx:
                    create_chat_completion(
                        settings={
                            "model": "gpt-4.1",
                            "timeout": 7,
                            "max_retries": 1,
                            "initial_backoff": 0,
                            "max_backoff": 0,
                        },
                        pool=EndpointPool([{}]),
                        messages=[{"role": "user", "content": "Hello"}],
                    )

        log_output = stderr.getvalue()
        self.assertIn("[ArenaHardV2][endpoint-error]", log_output)
        self.assertIn("attempt=1/2", log_output)
        self.assertIn("attempt=2/2", log_output)
        self.assertIn("timeout=7.0s", log_output)
        self.assertIn("simulated endpoint outage", log_output)
        self.assertIn("after 2 attempts", str(ctx.exception))

    def test_endpoint_falls_back_to_raw_http_on_sdk_json_body_error(self) -> None:
        class _FakeBadRequestError(Exception):
            pass

        _FakeBadRequestError.__name__ = "BadRequestError"

        class _FailingCompletions:
            def create(self, **kwargs: object) -> object:
                raise _FakeBadRequestError("We could not parse the JSON body of your request.")

        class _FailingChat:
            completions = _FailingCompletions()

        class _FailingClient:
            chat = _FailingChat()

        class _HttpxResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, object]:
                return {
                    "choices": [{"message": {"content": "fallback answer"}}],
                    "usage": {"total_tokens": 12},
                }

        class _HttpxClient:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

            def __enter__(self) -> "_HttpxClient":
                return self

            def __exit__(self, exc_type, exc, tb) -> bool:
                return False

            def post(self, url: str, *, content: bytes, headers: dict[str, str]) -> _HttpxResponse:
                self.url = url
                self.content = content
                self.headers = headers
                return _HttpxResponse()

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with mock.patch("arenahard_v2.endpoint.OpenAI", return_value=_FailingClient()):
                with mock.patch("arenahard_v2.endpoint.httpx.Client", return_value=_HttpxClient()):
                    result = create_chat_completion(
                        settings={"model": "gpt-4.1", "timeout": 7, "max_retries": 0},
                        pool=EndpointPool([{}]),
                        messages=[{"role": "user", "content": "Hello"}],
                    )

        self.assertEqual(result["answer"], "fallback answer")
        self.assertEqual(result["usage"]["total_tokens"], 12)
        self.assertIn("[ArenaHardV2][endpoint-fallback]", stderr.getvalue())

    def test_endpoint_uses_native_gemini_api(self) -> None:
        class _HttpxResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, object]:
                return {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": "gemini answer"},
                                ]
                            }
                        }
                    ],
                    "usageMetadata": {"totalTokenCount": 42},
                }

        class _HttpxClient:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs
                self.calls: list[dict[str, object]] = []

            def __enter__(self) -> "_HttpxClient":
                return self

            def __exit__(self, exc_type, exc, tb) -> bool:
                return False

            def post(
                self,
                url: str,
                *,
                params: dict[str, str],
                json: dict[str, object],
            ) -> _HttpxResponse:
                self.calls.append({"url": url, "params": params, "json": json})
                return _HttpxResponse()

        client = _HttpxClient()
        with mock.patch("arenahard_v2.endpoint.httpx.Client", return_value=client):
            result = create_chat_completion(
                settings={
                    "model": "gemini-2.5-pro-preview-03-25",
                    "api_type": "gemini",
                    "api_key": "gemini-key",
                    "timeout": 7,
                    "max_retries": 0,
                },
                pool=EndpointPool([{}]),
                messages=[
                    {"role": "system", "content": "Judge carefully"},
                    {"role": "user", "content": "Prompt 1"},
                    {"role": "assistant", "content": "Prior turn"},
                ],
                temperature=0.3,
                max_tokens=256,
            )

        self.assertEqual(result["answer"], "gemini answer")
        self.assertEqual(result["usage"]["totalTokenCount"], 42)
        self.assertEqual(
            client.calls[0]["url"],
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro-preview-03-25:generateContent",
        )
        self.assertEqual(client.calls[0]["params"], {"key": "gemini-key"})
        payload = client.calls[0]["json"]
        self.assertEqual(
            payload["systemInstruction"]["parts"][0]["text"],
            "Judge carefully",
        )
        self.assertEqual(payload["contents"][0]["role"], "user")
        self.assertEqual(payload["contents"][0]["parts"][0]["text"], "Prompt 1")
        self.assertEqual(payload["contents"][1]["role"], "model")
        self.assertEqual(
            payload["generationConfig"],
            {"temperature": 0.3, "maxOutputTokens": 256},
        )
        self.assertIn("safetySettings", payload)

    def test_endpoint_uses_vertex_gemini_api(self) -> None:
        class _HttpxResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, object]:
                return {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": "vertex answer"},
                                ]
                            }
                        }
                    ],
                    "usageMetadata": {"totalTokenCount": 21},
                }

        class _HttpxClient:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs
                self.calls: list[dict[str, object]] = []

            def __enter__(self) -> "_HttpxClient":
                return self

            def __exit__(self, exc_type, exc, tb) -> bool:
                return False

            def post(
                self,
                url: str,
                *,
                headers: dict[str, str],
                json: dict[str, object],
            ) -> _HttpxResponse:
                self.calls.append({"url": url, "headers": headers, "json": json})
                return _HttpxResponse()

        client = _HttpxClient()
        with mock.patch("arenahard_v2.endpoint.subprocess.check_output", return_value="vertex-token\n"):
            with mock.patch("arenahard_v2.endpoint.httpx.Client", return_value=client):
                result = create_chat_completion(
                    settings={
                        "model": "gemini-2.5-pro-preview-03-25",
                        "api_type": "vertex",
                        "project_id": "demo-project",
                        "regions": "us-central1",
                        "timeout": 7,
                        "max_retries": 0,
                    },
                    pool=EndpointPool([{}]),
                    messages=[
                        {"role": "system", "content": "Judge carefully"},
                        {"role": "user", "content": "Prompt 1"},
                    ],
                    temperature=0.0,
                    max_tokens=512,
                )

        self.assertEqual(result["answer"], "vertex answer")
        self.assertEqual(result["usage"]["totalTokenCount"], 21)
        self.assertEqual(
            client.calls[0]["url"],
            "https://us-central1-aiplatform.googleapis.com/v1/projects/demo-project/locations/us-central1/publishers/google/models/gemini-2.5-pro-preview-03-25:generateContent",
        )
        self.assertEqual(
            client.calls[0]["headers"]["Authorization"],
            "Bearer vertex-token",
        )
        self.assertNotIn("safetySettings", client.calls[0]["json"])
        self.assertEqual(
            client.calls[0]["json"]["generationConfig"],
            {"temperature": 0.0, "maxOutputTokens": 512},
        )

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
  prompt_template: ../mtbench/templates/should-not-be-used-for-judging.jinja
  judge_model: gpt-4.1
  judge_endpoint_name: gpt-4.1
  judge_parallel: 1
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
            with mock.patch("arenahard_v2.judge.create_chat_completion", side_effect=responses) as mocked_create:
                judgment_path = run_arenahard_v2_judging(config)

            rows = read_jsonl(judgment_path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["games"][0]["score"], "A>B")
            self.assertEqual(rows[0]["games"][1]["score"], "A<B")
            call_messages = mocked_create.call_args_list[0].kwargs["messages"]
            self.assertEqual(call_messages[1]["content"], "<|User Prompt|>\nPrompt 1\n\n<|The Start of Assistant A's Answer|>\nbaseline answer\n<|The End of Assistant A's Answer|>\n\n<|The Start of Assistant B's Answer|>\ncandidate answer\n<|The End of Assistant B's Answer|>")

            with mock.patch("arenahard_v2.judge.create_chat_completion", side_effect=RuntimeError("should not be called")):
                second_path = run_arenahard_v2_judging(config)
            self.assertEqual(judgment_path, second_path)

    def test_judging_repairs_missing_verdict_label(self) -> None:
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
  judge_parallel: 1
""",
            )
            _write_yaml(
                root / "api.yaml",
                """
gpt-4.1:
  model: gpt-4.1
  endpoints: null
  api_type: openai
  parallel: 1
  max_tokens: 128
  temperature: 0.0
""",
            )
            config = load_yaml(config_path)
            with mock.patch(
                "arenahard_v2.judge.create_chat_completion",
                side_effect=[
                    {"answer": "analysis without label"},
                    {"answer": "[[A>B]]"},
                    {"answer": "analysis without label again"},
                    {"answer": "[[A<B]]"},
                ],
            ):
                judgment_path = run_arenahard_v2_judging(config)

            rows = read_jsonl(judgment_path)
            self.assertEqual(rows[0]["games"][0]["score"], "A>B")
            self.assertEqual(rows[0]["games"][1]["score"], "A<B")
            self.assertIsNotNone(rows[0]["games"][0]["repair"])
            self.assertIsNotNone(rows[0]["games"][1]["repair"])

    def test_judging_reruns_existing_rows_with_missing_scores(self) -> None:
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
                        "games": [{"score": None}, {"score": "A<B"}],
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
  judge_parallel: 1
""",
            )
            _write_yaml(
                root / "api.yaml",
                """
gpt-4.1:
  model: gpt-4.1
  endpoints: null
  api_type: openai
  parallel: 1
  max_tokens: 128
  temperature: 0.0
""",
            )
            config = load_yaml(config_path)
            with mock.patch(
                "arenahard_v2.judge.create_chat_completion",
                side_effect=[
                    {"answer": "[[A>B]]"},
                    {"answer": "[[A<B]]"},
                ],
            ) as mocked_create:
                judgment_path = run_arenahard_v2_judging(config)

            self.assertEqual(mocked_create.call_count, 2)
            rows = read_jsonl(judgment_path)
            self.assertEqual(rows[0]["games"][0]["score"], "A>B")
            self.assertEqual(rows[0]["games"][1]["score"], "A<B")

    def test_judging_logs_question_context_and_aborts(self) -> None:
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
  judge_parallel: 1
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
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                with mock.patch(
                    "arenahard_v2.judge.create_chat_completion",
                    side_effect=RuntimeError("judge endpoint unavailable"),
                ):
                    with self.assertRaises(RuntimeError) as ctx:
                        run_arenahard_v2_judging(config)

            log_output = stderr.getvalue()
            self.assertIn("[ArenaHardV2][judge-error]", log_output)
            self.assertIn("uid=q1", log_output)
            self.assertIn("category=hard_prompt", log_output)
            self.assertIn("model=candidate", log_output)
            self.assertIn("baseline=o3-mini-2025-01-31", log_output)
            self.assertIn("judge=gpt-4.1", log_output)
            self.assertIn("judge endpoint unavailable", log_output)
            self.assertIn("Arena-Hard v2.0 judging aborted.", str(ctx.exception))
            self.assertFalse((benchmark_dir / "model_judgment" / "gpt-4.1" / "candidate.jsonl").exists())

    def test_judging_checkpoints_progress_and_resumes_after_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            benchmark_dir = root / "data"
            question_file = root / "question.jsonl"
            _write_jsonl(
                question_file,
                [
                    {"uid": "q1", "prompt": "Prompt 1", "category": "hard_prompt"},
                    {"uid": "q2", "prompt": "Prompt 2", "category": "hard_prompt"},
                ],
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
                            {"role": "assistant", "content": {"answer": "candidate answer 1"}},
                        ],
                        "tstamp": 0.0,
                        "metadata": {"token_len": 2, "header_count": {}, "list_count": {}, "bold_count": {}},
                    },
                    {
                        "uid": "q2",
                        "ans_id": "a2",
                        "model": "candidate",
                        "messages": [
                            {"role": "user", "content": "Prompt 2"},
                            {"role": "assistant", "content": {"answer": "candidate answer 2"}},
                        ],
                        "tstamp": 0.0,
                        "metadata": {"token_len": 2, "header_count": {}, "list_count": {}, "bold_count": {}},
                    },
                ],
            )
            _write_jsonl(
                answer_dir / "o3-mini-2025-01-31.jsonl",
                [
                    {
                        "uid": "q1",
                        "ans_id": "b1",
                        "model": "o3-mini-2025-01-31",
                        "messages": [
                            {"role": "user", "content": "Prompt 1"},
                            {"role": "assistant", "content": {"answer": "baseline answer 1"}},
                        ],
                        "tstamp": 0.0,
                        "metadata": {"token_len": 2, "header_count": {}, "list_count": {}, "bold_count": {}},
                    },
                    {
                        "uid": "q2",
                        "ans_id": "b2",
                        "model": "o3-mini-2025-01-31",
                        "messages": [
                            {"role": "user", "content": "Prompt 2"},
                            {"role": "assistant", "content": {"answer": "baseline answer 2"}},
                        ],
                        "tstamp": 0.0,
                        "metadata": {"token_len": 2, "header_count": {}, "list_count": {}, "bold_count": {}},
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
  question_file: {question_file}
  endpoint_file: {root / "api.yaml"}
  judge_model: gpt-4.1
  judge_endpoint_name: gpt-4.1
  judge_parallel: 1
""",
            )
            _write_yaml(
                root / "api.yaml",
                """
gpt-4.1:
  model: gpt-4.1
  endpoints: null
  api_type: openai
  parallel: 1
  max_tokens: 128
  temperature: 0.0
""",
            )
            config = load_yaml(config_path)

            with mock.patch(
                "arenahard_v2.judge.create_chat_completion",
                side_effect=[
                    {"answer": "ok [[A>B]]"},
                    {"answer": "ok [[A<B]]"},
                    RuntimeError("boom on q2"),
                ],
            ):
                with self.assertRaises(RuntimeError):
                    run_arenahard_v2_judging(config)

            partial_path = benchmark_dir / "model_judgment" / "gpt-4.1" / "candidate.jsonl"
            partial_rows = read_jsonl(partial_path)
            self.assertEqual([row["uid"] for row in partial_rows], ["q1"])

            with mock.patch(
                "arenahard_v2.judge.create_chat_completion",
                side_effect=[
                    {"answer": "ok [[A=B]]"},
                    {"answer": "ok [[A=B]]"},
                ],
            ) as mocked_create:
                run_arenahard_v2_judging(config)

            self.assertEqual(mocked_create.call_count, 2)
            final_rows = read_jsonl(partial_path)
            self.assertEqual([row["uid"] for row in final_rows], ["q1", "q2"])

    def test_report_raises_on_invalid_judgment_rows(self) -> None:
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
                        "games": [{"score": "A>B"}, {"score": None}],
                    }
                ],
            )
            config = load_yaml(config_path)
            with self.assertRaises(ValueError) as ctx:
                run_arenahard_v2_report(config, judge_names=["gpt-4.1"], categories=["hard_prompt"])
            self.assertIn("Invalid Arena-Hard v2.0 judgment rows found", str(ctx.exception))

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
