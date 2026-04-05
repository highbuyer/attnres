#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 tool-policy-only SFT 数据")
    parser.add_argument(
        "--input",
        default="docs/tool_call_samples_repo.jsonl",
        help="输入工具样本 JSONL",
    )
    parser.add_argument(
        "--out",
        default="sft_tool_policy_v1.jsonl",
        help="输出 tool-policy-only 数据集 JSONL",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=512,
        help="对原始样本重复多少倍，用于放大 tool-call 监督",
    )
    return parser.parse_args()


def convert_sample(sample: dict) -> dict | None:
    messages = sample.get("messages", [])
    if len(messages) < 2:
        return None

    user_msg = messages[0]
    if user_msg.get("role") != "user":
        return None

    tool_call_msg = next(
        (
            message
            for message in messages
            if message.get("role") == "assistant"
            and "<|tool_call_start|>" in message.get("content", "")
        ),
        None,
    )
    if tool_call_msg is None:
        return None

    return {
        "messages": [
            {"role": "user", "content": user_msg["content"]},
            {"role": "assistant", "content": tool_call_msg["content"]},
        ]
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    out_path = Path(args.out).expanduser()

    raw_samples = []
    with input_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            sample = convert_sample(json.loads(line))
            if sample is not None:
                raw_samples.append(sample)

    repeated = raw_samples * args.repeat
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for sample in repeated:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"输入样本: {len(raw_samples)}")
    print(f"重复倍数: {args.repeat}")
    print(f"输出样本: {len(repeated)}")
    print(f"输出文件: {out_path}")


if __name__ == "__main__":
    main()
