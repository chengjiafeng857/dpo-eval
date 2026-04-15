from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmark_common import read_jsonl


class BenchmarkCommonTests(unittest.TestCase):
    def test_read_jsonl_preserves_unicode_line_separator_inside_string(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.jsonl"
            payload = {
                "uid": "q1",
                "prompt": "before\u2028after",
            }
            path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
            rows = read_jsonl(path)
            self.assertEqual(rows, [payload])


if __name__ == "__main__":
    unittest.main()
