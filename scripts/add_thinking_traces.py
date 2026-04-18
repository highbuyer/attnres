#!/usr/bin/env python3
"""把现有 SFT 样本加上 thinking-trace 包装，让下一轮 SFT 学到"先想再答"。

输入：标准 SFT JSONL（messages 格式，含 user/assistant/tool）
输出：同格式 JSONL，但 assistant 消息前插入一段
    <think>简短推理</think>
    ...（原内容）

不注册新 special token（避免破坏现有 tokenizer）。用普通 `<think>` / `</think>`
XML 标签，模型 SFT 时会学到输出这两个文本标记；推理端 strip_tool_markup 那层
再加一个正则把 <think>...</think> 剥掉返回给用户就行。

启发式规则（离线静态推导，不调模型）：
  - 若 assistant 消息含 <|tool_call_start|> → 插入"需要查代码/文件来回答"
  - 若 user 消息匹配"X 是什么" / "什么是 X" / "解释 X" 且不含项目元词
    → 插入"这是通用知识题，凭训练直接答"
  - 若 user 明显要读某个文件（命中 \w+\.py/md/json 等）
    → 插入"用户指定了具体文件，直接 read_file"
  - 否则默认"按用户问题推理"

用法：
  uv run python scripts/add_thinking_traces.py \\
      --src sft_tool_summary_v5.jsonl \\
      --dst sft_tool_summary_v5_think.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


TOOL_CALL_START = "<|tool_call_start|>"
TOOL_RESULT_START = "<|tool_result_start|>"
FILE_REF_RE = re.compile(r"[A-Za-z0-9_./-]+\.(?:py|md|json|jsonl|toml|yaml|yml|txt|sh|cfg)", re.IGNORECASE)
WHAT_IS_RE = re.compile(r"(是什么|什么是|解释一下|介绍一下|简述)", re.IGNORECASE)
PROJECT_HINT_RE = re.compile(r"(项目|仓库|代码|文件|函数|模块|attnres|sft|train|infer|tokenizer|checkpoint)", re.IGNORECASE)


def _infer_trace(user_msg: str, asst_msg: str) -> str:
    """对 (user, assistant) 消息对推理一段简短的 think 内容。"""
    if TOOL_CALL_START in asst_msg:
        # 工具调用分支
        if "search_code" in asst_msg:
            return "用户问了具体代码/符号位置，先搜索代码库。"
        if "read_file" in asst_msg:
            if FILE_REF_RE.search(user_msg):
                return "用户指定了具体文件，直接读内容。"
            return "用户要看文件内容，先定位路径再读。"
        return "需要用工具去仓库里验证，不能凭空答。"
    if TOOL_RESULT_START in asst_msg:
        # tool 结果回合（runtime 注入，不走 think）
        return ""
    # 纯自然语言答
    if WHAT_IS_RE.search(user_msg):
        if PROJECT_HINT_RE.search(user_msg):
            return "问的是项目里的概念，结合已知的代码结构回答。"
        return "这是通用知识题，直接回答。"
    if FILE_REF_RE.search(user_msg):
        return "用户提到了文件但没明确要读，先判断是否需要调 read_file。"
    if PROJECT_HINT_RE.search(user_msg):
        return "和项目有关，若确定能直接答就答，否则查代码。"
    return "按用户问题推理并回答。"


def _wrap_with_think(content: str, trace: str) -> str:
    if not trace:
        return content
    # 保留 tool_call/result 标记在 think 之后
    return f"<think>{trace}</think>\n{content}"


def augment_sample(sample: dict) -> dict:
    msgs = sample.get("messages", [])
    if not msgs:
        return sample
    out_msgs = []
    last_user = ""
    for i, m in enumerate(msgs):
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            last_user = content
            out_msgs.append(m)
            continue
        if role == "assistant":
            trace = _infer_trace(last_user, content)
            new_content = _wrap_with_think(content, trace)
            out_msgs.append({**m, "content": new_content})
            continue
        out_msgs.append(m)
    return {**sample, "messages": out_msgs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--limit", type=int, default=None, help="可选：只处理前 N 条（调试用）")
    ap.add_argument("--preview", action="store_true", help="只打印前 3 条增强结果，不写盘")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        raise SystemExit(f"源文件不存在: {src}")

    processed = 0
    think_inserted = 0
    with src.open("r", encoding="utf-8") as fin:
        out_fp = None if args.preview else dst.open("w", encoding="utf-8")
        try:
            for line_no, line in enumerate(fin):
                if args.limit and processed >= args.limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"[skip] line {line_no}: {exc}")
                    continue
                aug = augment_sample(sample)
                # 统计插入数
                for m in aug["messages"]:
                    if m.get("role") == "assistant" and "<think>" in m.get("content", ""):
                        think_inserted += 1
                if args.preview:
                    if processed < 3:
                        print(f"=== sample {processed} ===")
                        for m in aug["messages"][:4]:
                            preview = m["content"][:240].replace("\n", " ")
                            print(f"  [{m.get('role')}] {preview}")
                        print()
                else:
                    out_fp.write(json.dumps(aug, ensure_ascii=False) + "\n")
                processed += 1
        finally:
            if out_fp:
                out_fp.close()

    print(f"Processed {processed} samples, inserted <think> on {think_inserted} assistant turns")
    if not args.preview:
        print(f"Saved to {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
