#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


TOOL_MARKERS = ("<|tool_call_start|>", "<|tool_result_start|>")


def has_tool_markup(text: str) -> bool:
    return any(marker in text for marker in TOOL_MARKERS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审计 SFT 混合数据里的工具样本分布")
    parser.add_argument("path", help="SFT JSONL 路径")
    parser.add_argument("--show-prompts", type=int, default=12, help="展示多少条工具 prompt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.path).expanduser()

    raw_total = 0
    raw_tool = 0
    conv_with_consecutive_asst = 0
    conv_with_tool_and_consecutive = 0
    assistant_blocks = 0
    assistant_blocks_tool = 0
    tool_prompts: list[str] = []
    prompt_shapes = Counter()

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw_total += 1
            sample = json.loads(line)
            messages = sample["messages"]
            is_tool_conv = any(has_tool_markup(message["content"]) for message in messages)
            if is_tool_conv:
                raw_tool += 1
                if messages and messages[0]["role"] == "user" and len(tool_prompts) < args.show_prompts:
                    tool_prompts.append(messages[0]["content"])
                if messages and messages[0]["role"] == "user":
                    prompt = messages[0]["content"]
                    if "读取" in prompt or "读一下" in prompt or "给我看" in prompt:
                        prompt_shapes["read_like"] += 1
                    elif "哪里" in prompt or "在哪" in prompt or "有没有" in prompt:
                        prompt_shapes["search_like"] += 1
                    else:
                        prompt_shapes["other"] += 1

            i = 0
            has_consecutive = False
            while i < len(messages):
                j = i + 1
                while j < len(messages) and messages[j]["role"] == messages[i]["role"]:
                    j += 1
                if messages[i]["role"] == "assistant" and j - i > 1:
                    has_consecutive = True
                    assistant_blocks += 1
                    if any(has_tool_markup(message["content"]) for message in messages[i:j]):
                        assistant_blocks_tool += 1
                i = j

            if has_consecutive and is_tool_conv:
                conv_with_tool_and_consecutive += 1
            if has_consecutive:
                conv_with_consecutive_asst += 1

    print(f"文件: {path}")
    print(f"raw_total={raw_total}")
    print(f"raw_tool={raw_tool} ({raw_tool / max(raw_total, 1):.4%})")
    print(f"conv_with_consecutive_asst={conv_with_consecutive_asst}")
    print(f"conv_with_tool_and_consecutive={conv_with_tool_and_consecutive}")
    print(f"assistant_blocks_gt1={assistant_blocks}")
    print(f"assistant_blocks_gt1_tool={assistant_blocks_tool}")
    print(f"tool_prompt_shapes={dict(prompt_shapes)}")
    if tool_prompts:
        print("\n示例工具 prompt:")
        for prompt in tool_prompts:
            print(f"- {prompt}")


if __name__ == "__main__":
    main()
