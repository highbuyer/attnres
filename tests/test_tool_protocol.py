from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tool_protocol import execute_tool, parse_tool_call, strip_tool_markup, validate_tool_sample  # noqa: E402


class ToolProtocolTest(unittest.TestCase):
    def test_parse_tool_call(self) -> None:
        parsed = parse_tool_call('<|tool_call_start|><|tool_name_search_code|>{"query":"rope_theta"}<|tool_call_end|>')
        self.assertEqual(parsed, ("search_code", {"query": "rope_theta"}))
        self.assertIsNone(parse_tool_call("plain text"))

    def test_parse_tool_call_prefers_latest_complete_call(self) -> None:
        transcript = (
            '<|tool_call_start|><|tool_name_search_code|>{"query":"rope_theta"}<|tool_call_end|>'
            '<|tool_result_start|>src/train.py:59:    rope_theta: float = 10000.0<|tool_result_end|>'
            '<|tool_call_start|><|tool_name_read_file|>{"path":"src/train.py","offset":55,"limit":3}<|tool_call_end|>'
        )
        self.assertEqual(
            parse_tool_call(transcript),
            ("read_file", {"path": "src/train.py", "offset": 55, "limit": 3}),
        )

    def test_strip_tool_markup_removes_tool_transcript(self) -> None:
        transcript = (
            '<|tool_call_start|><|tool_name_search_code|>{"query":"rope_theta"}<|tool_call_end|>'
            '<|tool_result_start|>src/train.py:59: rope_theta<|tool_result_end|>'
            "rope_theta 默认值是 10000。"
        )
        self.assertEqual(strip_tool_markup(transcript), "rope_theta 默认值是 10000。")

    def test_execute_read_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "demo.txt"
            file_path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
            result = execute_tool("read_file", {"path": "demo.txt", "offset": 2, "limit": 2}, tmp_dir)
            self.assertEqual(result, "2 beta\n3 gamma")

    def test_execute_read_file_blocks_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            outside = root.parent / "secret.txt"
            outside.write_text("top-secret\n", encoding="utf-8")
            result = execute_tool("read_file", {"path": "../secret.txt", "offset": 1, "limit": 1}, tmp_dir)
            self.assertEqual(result, "(路径越界: ../secret.txt)")

    def test_validate_tool_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "app.py"
            file_path.write_text("def hello():\n    return 1\n", encoding="utf-8")
            sample = {
                "messages": [
                    {"role": "user", "content": "读取 app.py"},
                    {"role": "assistant", "content": '<|tool_call_start|><|tool_name_read_file|>{"path":"app.py","offset":1,"limit":2}<|tool_call_end|>'},
                    {"role": "assistant", "content": "<|tool_result_start|>1 def hello():\n2     return 1<|tool_result_end|>"},
                    {"role": "assistant", "content": "这是 app.py 的内容。"},
                ]
            }
            self.assertEqual(validate_tool_sample(sample, tmp_dir), (True, "ok"))


if __name__ == "__main__":
    unittest.main()
