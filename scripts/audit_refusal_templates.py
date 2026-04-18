#!/usr/bin/env python3
"""从 sft_tool_summary_v5.jsonl 导出所有命中 4 个拒答模板的训练样本，
供人工/下游 pipeline 审查。输出 runs/refusal_templates_audit.jsonl，
每行带 line_idx + 启发性可答度标记（heuristic_benign）。

背景：docs/REFUSAL_ANALYSIS_v1.md。模型 over_refusal 6/64 的根因是训练数据里
128 条 (1.80%) 被错误标成拒答模板——其中 81 条是常识/逻辑可答题。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TEMPLATES = {
    "T1": "我不确定这个问题的准确答案",
    "T2": "无法确认这个信息的准确性",
    "T3": "超出了我的知识范围",
    "T4": "这个问题我不太确定答案",
}

BENIGN_KEYWORDS = (
    "列出", "列举", "哪些", "描述", "选择", "判断", "分类", "定义", "解释",
    "什么是", "如何", "为什么", "举例", "总结", "写一", "给出", "提供",
    "计算", "比较", "区别", "特点", "名称", "历史", "原理", "类别",
    "编写", "生成", "根据", "提取", "改写", "翻译",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-data", default="sft_tool_summary_v5.jsonl")
    parser.add_argument("--out", default="runs/refusal_templates_audit.jsonl")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    hits_by_tpl = {k: 0 for k in TEMPLATES}
    total = 0
    matched_total = 0
    benign_total = 0

    with open(args.in_data) as fin, open(out_path, "w") as fout:
        for idx, line in enumerate(fin):
            total += 1
            d = json.loads(line)
            for msg in d.get("messages", []):
                if msg.get("role") != "assistant":
                    continue
                content = msg.get("content", "")
                matched_tpl = None
                for tid, tpl in TEMPLATES.items():
                    if tpl in content:
                        matched_tpl = tid
                        break
                if matched_tpl is None:
                    continue

                user_q = next(
                    (m["content"] for m in d["messages"] if m.get("role") == "user"),
                    "",
                )
                benign = any(kw in user_q for kw in BENIGN_KEYWORDS)
                rec = {
                    "line_idx": idx,
                    "template_hit": matched_tpl,
                    "heuristic_benign": benign,
                    "user_q": user_q,
                    "assistant_a": content,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                hits_by_tpl[matched_tpl] += 1
                matched_total += 1
                if benign:
                    benign_total += 1
                break  # 一条样本只算一次

    print(f"总训练样本: {total}")
    print(f"命中任一拒答模板: {matched_total} ({100*matched_total/total:.2f}%)")
    print(f"  启发性可答题（应替换成正确答案）: {benign_total}")
    print(f"  其他（可能真的该拒或 prompt 本身模糊）: {matched_total - benign_total}")
    print("\n按模板:")
    for tid, tpl in TEMPLATES.items():
        print(f"  {tid} '{tpl}' : {hits_by_tpl[tid]}")
    print(f"\n结果写入 {out_path}")


if __name__ == "__main__":
    main()
