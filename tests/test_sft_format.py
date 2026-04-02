from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sft_format import assistant_block_requires_chain, included_turn_indices  # noqa: E402


class SftFormatTest(unittest.TestCase):
    def test_regular_consecutive_assistants_keep_old_sibling_behavior(self) -> None:
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "draft a"},
            {"role": "assistant", "content": "draft b"},
        ]
        self.assertFalse(assistant_block_requires_chain(messages, 1))
        self.assertEqual(included_turn_indices(messages, 1), [0, 1])
        self.assertEqual(included_turn_indices(messages, 2), [0, 2])

    def test_tool_chain_keeps_prior_assistant_turns(self) -> None:
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": '<|tool_call_start|><|tool_name_search_code|>{"query":"x"}<|tool_call_end|>'},
            {"role": "assistant", "content": "<|tool_result_start|>hit<|tool_result_end|>"},
            {"role": "assistant", "content": "final answer"},
        ]
        self.assertTrue(assistant_block_requires_chain(messages, 1))
        self.assertEqual(included_turn_indices(messages, 1), [0, 1])
        self.assertEqual(included_turn_indices(messages, 2), [0, 1, 2])
        self.assertEqual(included_turn_indices(messages, 3), [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
