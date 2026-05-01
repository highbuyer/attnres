#!/usr/bin/env python3
"""把 attnres 老格式 SFT 数据转换为 Qwen 风格 (与 sft_format_v2 配套)。

老格式（v6/v21 时代）：
  assistant content 内嵌  <|tool_call_start|><|tool_name_X|>{json}<|tool_call_end|>
  assistant content 内嵌  <|tool_result_start|>...<|tool_result_end|>

新格式 (Qwen / sft_claude_trajectories.jsonl 同款)：
  {role: assistant, content: text, tool_calls: [{name, arguments}]}
  {role: tool, content: text}

输入文件示例（含 inline marker 的）:
  - sft_archive/sft_toolheavy_v1.jsonl
  - sft_archive/sft_tool_policy_v1.jsonl
  - sft_archive/sft_tool_gated_v1.jsonl / v2
  - sft_archive/sft_tool_summary_v*.jsonl

用法:
  .venv/bin/python -u scripts/convert_inline_to_qwen.py \\
    --in datasets/sft_archive/sft_toolheavy_v1.jsonl \\
    --out datasets/sft_toolheavy_qwen.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter

# inline marker 正则
RE_TOOL_CALL = re.compile(
    r"<\|tool_call_start\|>"      # 起
    r"<\|tool_name_(?P<name>[^|]+)\|>"  # 工具名
    r"(?P<args>.*?)"              # 参数 JSON
    r"<\|tool_call_end\|>",       # 终
    re.DOTALL,
)
RE_TOOL_RESULT = re.compile(
    r"<\|tool_result_start\|>(?P<body>.*?)<\|tool_result_end\|>",
    re.DOTALL,
)


def parse_inline_assistant(content: str) -> tuple[str, list[dict]] | tuple[None, None]:
    """解析 assistant content：返回 (text, tool_calls) 或 (None, None) 表示是 tool_result。"""
    # 先看是否纯 tool_result 包装
    rr = RE_TOOL_RESULT.search(content)
    if rr and rr.group(0) == content.strip():
        return None, None  # 整条是 tool_result

    # 解析里面的 tool_call (可能多个)
    tool_calls: list[dict] = []
    text_parts: list[str] = []
    last_end = 0
    found_call = False
    for m in RE_TOOL_CALL.finditer(content):
        if m.start() > last_end:
            txt = content[last_end:m.start()].strip()
            if txt:
                text_parts.append(txt)
        name = m.group("name").strip()
        args_raw = m.group("args").strip()
        try:
            args = json.loads(args_raw) if args_raw else {}
        except json.JSONDecodeError:
            args = {"_raw": args_raw}
        tool_calls.append({"name": name, "arguments": args})
        last_end = m.end()
        found_call = True
    if last_end < len(content):
        tail = content[last_end:].strip()
        if tail:
            text_parts.append(tail)

    text = "\n".join(text_parts)
    if not found_call:
        return content.strip(), []  # 普通 assistant 文本
    return text, tool_calls


def extract_tool_result(content: str) -> str | None:
    rr = RE_TOOL_RESULT.search(content)
    if rr:
        return rr.group("body").strip()
    return None


def convert_sample(messages: list[dict]) -> list[dict] | None:
    """单条 messages -> 转换后 messages。返回 None 表示无效。"""
    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "") or ""
        if role == "user":
            out.append({"role": "user", "content": content})
            continue
        if role != "assistant":
            # 老数据没 system/tool 角色（都是 assistant 内嵌）
            continue

        # 检查是否纯 tool_result 包装
        tr = extract_tool_result(content)
        if tr is not None and content.strip().startswith("<|tool_result_start|>"):
            out.append({"role": "tool", "content": tr})
            continue

        # 解析 inline tool_call + 普通文本
        text, tool_calls = parse_inline_assistant(content)
        if text is None:
            # extract_tool_result 已处理
            continue
        msg = {"role": "assistant", "content": text}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        out.append(msg)

    if not out:
        return None
    # 必须以 assistant 收尾才有学习信号
    while out and out[-1]["role"] != "assistant":
        out.pop()
    if not out or not any(m["role"] == "user" for m in out):
        return None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    n_in = n_out = 0
    n_with_tool = 0
    n_dropped = 0
    role_counter = Counter()
    tool_name_counter = Counter()

    with open(args.src, "r", encoding="utf-8") as f, \
         open(args.out, "w", encoding="utf-8") as fout:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                n_dropped += 1
                continue
            n_in += 1
            msgs = obj.get("messages")
            if not msgs:
                n_dropped += 1
                continue
            converted = convert_sample(msgs)
            if not converted:
                n_dropped += 1
                continue
            n_out += 1
            for m in converted:
                role_counter[m["role"]] += 1
                if m.get("tool_calls"):
                    n_with_tool += 1
                    for tc in m["tool_calls"]:
                        tool_name_counter[tc["name"]] += 1
            fout.write(json.dumps({"messages": converted}, ensure_ascii=False) + "\n")

    print(f"[convert] in={n_in}  out={n_out}  dropped={n_dropped}")
    print(f"[convert] role dist: {dict(role_counter)}")
    print(f"[convert] assistant with tool_call: {n_with_tool}")
    print(f"[convert] top tool names:")
    for name, cnt in tool_name_counter.most_common(10):
        print(f"  {cnt:>5}  {name}")


if __name__ == "__main__":
    main()
