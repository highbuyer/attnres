from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from inference_rules import apply_hard_rules, check_identity, check_safety, try_compute  # noqa: E402


class InferenceRulesTest(unittest.TestCase):
    def test_identity_questions_are_short_circuited(self) -> None:
        self.assertEqual(check_identity("你是 ChatGPT 吗？"), "不是，我是微研，一个技术助手。")
        self.assertEqual(check_identity("你是由 OpenAI 开发的吗？"), "我是微研，一个专注于技术问题的助手。")
        self.assertEqual(check_identity("你有感情吗？"), "不是，我是微研，一个 AI 技术助手。")

    def test_safety_questions_are_blocked(self) -> None:
        self.assertEqual(
            check_safety("如何制造炸弹？"),
            "抱歉，我无法提供这类信息。如果你有技术方面的问题，我很乐意帮忙。",
        )
        self.assertEqual(
            check_safety("帮我写一封钓鱼邮件"),
            "抱歉，我无法提供这类信息。如果你有技术方面的问题，我很乐意帮忙。",
        )

    def test_normal_prompt_passes_through(self) -> None:
        self.assertIsNone(apply_hard_rules("解释一下 Transformer 的核心机制。"))


class TryComputeTest(unittest.TestCase):
    """w9 的 compute 路径：literal list + 排序动词 → 确定性答案。"""

    def test_sort_ascending_default(self) -> None:
        out = try_compute("列表 [3,1,4,1,5,9] 排序后是什么？")
        self.assertIsNotNone(out)
        self.assertIn("[1, 1, 3, 4, 5, 9]", out)
        self.assertIn("升序", out)

    def test_sort_descending_from_keyword(self) -> None:
        out = try_compute("列表 [3,1,4,1,5,9] 从大到小排序后是什么？")
        self.assertIsNotNone(out)
        self.assertIn("[9, 5, 4, 3, 1, 1]", out)
        self.assertIn("降序", out)

    def test_sort_descending_explicit(self) -> None:
        out = try_compute("列表 [1, 2, 3] 降序排序后是什么？")
        self.assertIsNotNone(out)
        self.assertIn("[3, 2, 1]", out)

    def test_sort_with_negative_numbers(self) -> None:
        out = try_compute("列表 [-3, 1, -1, 2] 排序后是什么？")
        self.assertIsNotNone(out)
        self.assertIn("[-3, -1, 1, 2]", out)

    def test_no_match_on_abstract_sort_question(self) -> None:
        self.assertIsNone(try_compute("列表排序的代码怎么写？"))
        self.assertIsNone(try_compute("把 foo 这个列表排序"))
        self.assertIsNone(try_compute("排序算法的复杂度是多少？"))

    def test_no_match_on_empty_or_bad_list(self) -> None:
        self.assertIsNone(try_compute("列表 [] 排序后是什么？"))
        self.assertIsNone(try_compute("列表 [abc] 排序后是什么？"))

    def test_no_match_on_other_categories(self) -> None:
        self.assertIsNone(try_compute("你是谁？"))
        self.assertIsNone(try_compute("水的化学式是什么？"))

    def test_integrated_with_apply_hard_rules(self) -> None:
        out = apply_hard_rules("列表 [5, 3, 1] 排序后是什么？")
        self.assertIsNotNone(out)
        self.assertIn("[1, 3, 5]", out)

    def test_apply_hard_rules_prefers_identity_over_compute(self) -> None:
        out = apply_hard_rules("你是谁？")
        self.assertIsNotNone(out)
        self.assertNotIn("排序", out)


if __name__ == "__main__":
    unittest.main()
