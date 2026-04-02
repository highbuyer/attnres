#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tool_protocol import validate_tool_sample  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审计工具调用训练数据的真实性")
    parser.add_argument("path", help="待审计的 JSONL 文件")
    parser.add_argument("--repo-root", default=str(ROOT), help="用于执行工具验证的仓库根目录")
    parser.add_argument("--out", default=None, help="可选：写出过滤后的有效样本")
    parser.add_argument("--show-bad", type=int, default=5, help="最多展示多少条坏样本")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_path = Path(args.path).expanduser()
    repo_root = Path(args.repo_root).expanduser()
    rows = [json.loads(line) for line in src_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    stats = Counter()
    bad_examples: list[tuple[int, str, str]] = []
    kept: list[dict] = []

    for idx, sample in enumerate(rows, start=1):
        ok, reason = validate_tool_sample(sample, repo_root)
        stats[reason] += 1
        if ok:
            kept.append(sample)
            continue
        if len(bad_examples) < args.show_bad:
            prompt = sample.get("messages", [{}])[0].get("content", "")
            bad_examples.append((idx, reason, prompt))

    tool_rows = sum(count for reason, count in stats.items() if reason != "no_tool_call")
    valid_tool_rows = stats["ok"]
    invalid_tool_rows = tool_rows - valid_tool_rows

    print(f"文件: {src_path}")
    print(f"仓库: {repo_root}")
    print(f"总样本: {len(rows)}")
    print(f"工具样本: {tool_rows}")
    print(f"有效工具样本: {valid_tool_rows}")
    print(f"无效工具样本: {invalid_tool_rows}")
    for reason, count in sorted(stats.items()):
        print(f"  {reason}: {count}")

    if bad_examples:
        print("\n坏样本示例:")
        for idx, reason, prompt in bad_examples:
            print(f"  line {idx}: {reason} | {prompt}")

    if args.out:
        out_path = Path(args.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            for sample in kept:
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"\n已写出过滤结果: {out_path} ({len(kept)} 条)")


if __name__ == "__main__":
    main()
