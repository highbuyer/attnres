from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from research_fallback import (  # noqa: E402
    _core_tokens,
    _is_related_loose,
    candidate_titles,
    is_related,
    is_unknown_answer,
)


class IsRelatedStrictTest(unittest.TestCase):
    """w5 发现的 wiki 误匹 case 不能再通过 is_related。"""

    def test_world_tallest_vs_worst_person(self) -> None:
        # "世界上最高的山峰" 不应和 "世界上最糟糕的人" 词条判定相关
        ok = is_related(
            "世界上最高的山峰是哪座？",
            "世界上最糟糕的人",
            "《世界上最糟糕的人》 是2021年黑色浪漫喜剧剧情片",
        )
        self.assertFalse(ok)

    def test_list_sort_vs_table(self) -> None:
        # "列表 [3,1,4,1,5,9] 排序后是什么？" 不应和 "列表（表格）" 词条判定相关
        ok = is_related(
            "列表 [3,1,4,1,5,9] 排序后是什么？",
            "列表",
            "表格，是把信息或數據整理成行與列，或是更為複雜的結構。",
        )
        self.assertFalse(ok)

    def test_beijing_landmarks_vs_tram(self) -> None:
        # "北京有什么著名景点" 不应和 "北京有轨电车" 词条判定相关
        ok = is_related(
            "北京有什么著名景点？",
            "北京有轨电车",
            "北京有轨电车的历史始于1899年由西门子公司建设的连接永定门至京奉铁路马家堡站的电车线路",
        )
        self.assertFalse(ok)


class IsRelatedAllowBorderlineTest(unittest.TestCase):
    """保留的擦边 case：虽然不完全精确但 user 能接受；不应过度过滤。"""

    def test_water_formula_vs_chemical_formula(self) -> None:
        # 水的化学式 → "化学式" 通用词条，半相关，允许通过
        ok = is_related(
            "水的化学式是什么？",
            "化学式",
            "化學式 是用化学元素符号、数字、其他符号来表示组成特定化合物或分子的原子化学比例信息的一种公式。",
        )
        self.assertTrue(ok)

    def test_diamond_composition_vs_chemical_composition(self) -> None:
        ok = is_related(
            "金刚石的化学成分是什么？",
            "化学成分",
            "化學成份是化學中的一個概念，在純物質及混合物中有不同（但相近的）意義。",
        )
        self.assertTrue(ok)

    def test_python_vs_java_wiki_python(self) -> None:
        # Python 和 Java 的对比 → Python 词条只覆盖半边，但算相关
        ok = is_related(
            "Python 和 Java 哪个更适合初学者？",
            "Python",
            "Python，是一种广泛使用的解释型、高级和通用的编程语言。",
        )
        self.assertTrue(ok)


class IsRelatedShortQueryFallbackTest(unittest.TestCase):
    """短 query 无核心串时退回 loose 判定，避免过度过滤。"""

    def test_short_query_fallback_to_loose(self) -> None:
        # "鲁迅" 长度 2，不满足 ≥3 字符 core；退 loose
        cores = _core_tokens("鲁迅")
        self.assertEqual(cores, [])
        # loose 判定下应能匹上含 "鲁迅" 的 extract
        self.assertTrue(_is_related_loose("鲁迅", "鲁迅（1881年—1936年），原名周樹人"))


class CoreTokensTest(unittest.TestCase):
    def test_core_tokens_extraction(self) -> None:
        self.assertIn("著名景点", _core_tokens("北京有什么著名景点？"))
        self.assertNotIn("北京有", _core_tokens("北京有什么著名景点？"))  # "有" 现在是 stopword
        self.assertNotIn("山峰", _core_tokens("世界上最高的山峰是哪座？"))  # 山峰长 2 < 3，被 core 规则过滤
        self.assertIn("世界上最高", _core_tokens("世界上最高的山峰是哪座？"))
        self.assertIn("Python", _core_tokens("Python 和 Java 哪个更适合初学者？"))
        self.assertIn("Java", _core_tokens("Python 和 Java 哪个更适合初学者？"))


class IsUnknownAnswerTest(unittest.TestCase):
    def test_unknown_templates_match(self) -> None:
        self.assertTrue(is_unknown_answer("这个问题我不太确定答案，建议查阅相关资料获取准确信息。"))
        self.assertTrue(is_unknown_answer("我不确定这个问题的准确答案，不想给你错误的信息。"))

    def test_factual_answer_does_not_match(self) -> None:
        self.assertFalse(is_unknown_answer("水的化学式是 H₂O。"))
        self.assertFalse(is_unknown_answer("地球绕太阳转一圈约 365.25 天。"))

    def test_w7a_new_templates_match(self) -> None:
        # w6 smoke test 在 weiyan-api 上看到的实际漏掉的"不会答"模板，
        # 都应触发 fallback。
        self.assertTrue(
            is_unknown_answer("对不起，我无法确认这个信息的准确性，建议你通过其他渠道核实。"),
            "对不起 + 无法确认 + 建议你…核实 应触发 fallback",
        )
        self.assertTrue(
            is_unknown_answer("这超出了我的知识范围，建议咨询专业人士或查阅权威资料。"),
            "超出知识范围 应触发 fallback",
        )
        self.assertTrue(
            is_unknown_answer("不好意思，这个问题我了解有限，暂时没法给你准确答案。"),
            "不好意思 + 了解有限 应触发 fallback",
        )
        self.assertTrue(
            is_unknown_answer("这些信息我并不太确定，建议你查证一下。"),
            "不太确定…信息 + 建议你查证 应触发 fallback",
        )

    def test_w7a_safety_refuse_not_matched(self) -> None:
        # 扩大模板后仍不能吃掉安全拒绝——"无法提供这类信息"语义与"不会答"
        # 是两件事，fallback 去 wiki 查"制造炸弹"明显是错的。
        self.assertFalse(
            is_unknown_answer("抱歉，我无法提供这类信息。如果你有技术方面的问题，我很乐意帮忙。"),
            "安全拒绝的'无法提供这类信息'不能触发 fallback",
        )
        self.assertFalse(
            is_unknown_answer("这类请求我无法处理，请换一个话题。"),
            "安全拒绝的'无法处理'不能触发 fallback",
        )

    def test_w7a_declarative_suggestion_not_matched(self) -> None:
        # "根据 PEP-8，建议咨询团队约定"是陈述性建议，不是自我拒答；
        # 新正则要求"建议/请"后必须带人称代词才算拒答语气。
        self.assertFalse(
            is_unknown_answer("根据 PEP-8，建议咨询团队约定。"),
            "陈述性建议（无人称代词）不应触发 fallback",
        )
        self.assertFalse(
            is_unknown_answer("查询到 3 条结果：foo.py, bar.py, baz.py。"),
            "工具返回'查询到...'不应触发 fallback",
        )


class CandidateTitlesTest(unittest.TestCase):
    def test_titles_include_english_tokens(self) -> None:
        cands = candidate_titles("Python 和 Java 哪个更适合初学者？")
        self.assertIn("Python", cands)
        self.assertIn("Java", cands)


if __name__ == "__main__":
    unittest.main()
