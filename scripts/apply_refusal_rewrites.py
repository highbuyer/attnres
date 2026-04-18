#!/usr/bin/env python3
"""把 DeepSeek 改写的 101 条回贴到 sft_tool_summary_v5.jsonl，生成 clean 版。

- 输入：sft_tool_summary_v5.jsonl (7126) + runs/refusal_rewrites.jsonl (101)
- 输出：data/sft_tool_summary_v5_clean.jsonl (7126，101 条 assistant content 被替换)
- 27 条 non-benign（heuristic 可能漏判）不动——避免误删

替换逻辑：按 line_idx 找到原样本，把 messages 里第一个命中拒答模板的
assistant response 换成 rewritten_a。其他 assistant message（如果有）不动。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


TEMPLATES = [
    "我不确定这个问题的准确答案",
    "无法确认这个信息的准确性",
    "超出了我的知识范围",
    "这个问题我不太确定答案",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-data", default="sft_tool_summary_v5.jsonl")
    parser.add_argument("--rewrites", default="runs/refusal_rewrites.jsonl")
    parser.add_argument("--out", default="data/sft_tool_summary_v5_clean.jsonl")
    args = parser.parse_args()

    # 读改写
    rewrites = {}
    with open(args.rewrites) as f:
        for line in f:
            r = json.loads(line)
            rewrites[r["line_idx"]] = r["rewritten_a"]
    print(f"载入改写 {len(rewrites)} 条")

    # 逐行替换
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_total = 0
    n_replaced = 0
    n_skipped_no_template = 0

    with open(args.in_data) as fin, open(out_path, "w") as fout:
        for idx, line in enumerate(fin):
            n_total += 1
            d = json.loads(line)
            if idx in rewrites:
                # 找到第一个命中拒答模板的 assistant message，替换
                replaced = False
                for msg in d.get("messages", []):
                    if msg.get("role") != "assistant":
                        continue
                    content = msg.get("content", "") or ""
                    if any(tpl in content for tpl in TEMPLATES):
                        msg["content"] = rewrites[idx]
                        replaced = True
                        break
                if replaced:
                    n_replaced += 1
                else:
                    n_skipped_no_template += 1
                    print(f"  warn: line {idx} 改写列表里但原文没找到模板？")
            fout.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\n总样本: {n_total}")
    print(f"成功替换: {n_replaced}")
    print(f"未找到模板（异常）: {n_skipped_no_template}")
    print(f"不变（含 27 条 non-benign）: {n_total - n_replaced - n_skipped_no_template}")
    print(f"\n输出: {out_path}")


if __name__ == "__main__":
    main()
