#!/usr/bin/env python3
"""把 ~/.claude/projects/*/*.jsonl 转换成 Qwen 风格 SFT 数据。

输出 schema (Qwen / minimind 主线格式):
  {
    "messages": [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "...",
       "reasoning_content": "...",            # 可选，来自 thinking block
       "tool_calls": [{"name":..,"arguments":..}]},  # 可选
      {"role": "tool", "content": "..."},
      ...
    ],
    "meta": {"project": "...", "session": "...", "n_tools": 3}
  }

用法:
  .venv/bin/python -u scripts/build_claude_traj_sft.py \\
    --root ~/.claude/projects \\
    --out datasets/sft_claude_trajectories.jsonl \\
    --min-tool-use 1 --max-tool-result-chars 4000
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# 清洗 user content（去 CLI 噪声）
# ---------------------------------------------------------------------------

NOISE_TAGS = [
    re.compile(r"<command-name>.*?</command-name>", re.S),
    re.compile(r"<command-message>.*?</command-message>", re.S),
    re.compile(r"<command-args>.*?</command-args>", re.S),
    re.compile(r"<local-command-stdout>.*?</local-command-stdout>", re.S),
    re.compile(r"<local-command-stderr>.*?</local-command-stderr>", re.S),
    re.compile(r"<local-command-caveat>.*?</local-command-caveat>", re.S),
    re.compile(r"<system-reminder>.*?</system-reminder>", re.S),
    re.compile(r"<bash-input>.*?</bash-input>", re.S),
    re.compile(r"<bash-stdout>.*?</bash-stdout>", re.S),
    re.compile(r"<bash-stderr>.*?</bash-stderr>", re.S),
]

# 终端 UI 渲染：`●` 行 + 连续多行（这通常是上一轮 assistant 输出被裹进 user）
TERMINAL_BULLET_RE = re.compile(r"^[●○•]\s.*?(?=\n\n|\Z)", re.M | re.S)

# 已知的"非真实回答"：Claude API 错误、占位回应
ASSISTANT_GARBAGE_PATTERNS = [
    re.compile(r"^API Error:", re.I),
    re.compile(r"^Request was aborted", re.I),
    re.compile(r"^No response requested\.?\s*$", re.I),
    re.compile(r"^\(no content\)\s*$", re.I),
    re.compile(r"^Prompt is too long", re.I),
]


def clean_user_text(s: str) -> str:
    for pat in NOISE_TAGS:
        s = pat.sub("", s)
    # 去 `●` 起头的终端渲染段（保留 user 实际输入）
    # 仅当出现在内容中段且后跟 ≥30 字内容时认为是被混入的 assistant 输出
    if "\n●" in s or s.startswith("●"):
        # 切到第一个 ● 之前
        idx = s.find("●")
        before = s[:idx].rstrip()
        if before:
            s = before
        else:
            # 整段都是 ● 渲染，丢弃
            return ""
    return s.strip()


def is_assistant_garbage(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    return any(p.search(t) for p in ASSISTANT_GARBAGE_PATTERNS)


# ---------------------------------------------------------------------------
# tool_result content 抽取
# ---------------------------------------------------------------------------

def _extract_tool_result_content(c) -> str:
    """tool_result 的 content 可能是 str 或 [{type:'text',text:...}, ...]"""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        out = []
        for blk in c:
            if isinstance(blk, dict):
                if blk.get("type") == "text":
                    out.append(blk.get("text", "") or "")
                elif blk.get("type") == "image":
                    out.append("[image omitted]")
                else:
                    out.append(str(blk))
        return "\n".join(out)
    return str(c) if c else ""


# ---------------------------------------------------------------------------
# 单 session 转换
# ---------------------------------------------------------------------------

def _post_process_session(msgs: list[dict]) -> list[dict]:
    """会话级清理：去掉孤立 user / 合并连续同 role / 修剪头尾"""
    if not msgs:
        return msgs

    # 1. 合并连续相同 role：user/tool 简单拼 content；assistant 还要合并 tool_calls / reasoning
    merged: list[dict] = []
    for m in msgs:
        if merged and merged[-1]["role"] == m["role"]:
            prev = merged[-1]
            if m["role"] in ("user", "tool"):
                prev_c = prev.get("content", "") or ""
                cur_c = m.get("content", "") or ""
                prev["content"] = (prev_c + "\n\n" + cur_c).strip()
            elif m["role"] == "assistant":
                # text 拼接
                prev_c = prev.get("content", "") or ""
                cur_c = m.get("content", "") or ""
                if prev_c and cur_c:
                    prev["content"] = prev_c + "\n\n" + cur_c
                elif cur_c:
                    prev["content"] = cur_c
                # reasoning 拼接
                prev_r = prev.get("reasoning_content", "") or ""
                cur_r = m.get("reasoning_content", "") or ""
                if prev_r and cur_r:
                    prev["reasoning_content"] = prev_r + "\n\n" + cur_r
                elif cur_r:
                    prev["reasoning_content"] = cur_r
                # tool_calls 拼接
                prev_tc = prev.get("tool_calls") or []
                cur_tc = m.get("tool_calls") or []
                if prev_tc or cur_tc:
                    prev["tool_calls"] = list(prev_tc) + list(cur_tc)
            continue
        merged.append(dict(m))
    msgs = merged

    # 2. 去掉开头连续的 tool / assistant（应该 user 起始）
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    # 3. 去掉结尾的 user / tool（必须以 assistant 收尾才有学习信号）
    while msgs and msgs[-1]["role"] != "assistant":
        msgs.pop()
    return msgs


def convert_session(entries: list[dict], max_tool_result_chars: int) -> list[dict]:
    """entries: 一个 session 的所有 jsonl 行（已解析）.
    返回 Qwen 风格 messages list。
    """
    msgs: list[dict] = []
    pending_user_tool_results: list[str] = []  # tool_use_id 顺序无关，直接拼

    for o in entries:
        t = o.get("type")
        if t not in ("user", "assistant"):
            continue
        m = o.get("message")
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role not in ("user", "assistant"):
            continue

        # === user 消息 ===
        if role == "user":
            if isinstance(content, str):
                txt = clean_user_text(content)
                if txt:
                    msgs.append({"role": "user", "content": txt})
            elif isinstance(content, list):
                # 可能含 tool_result（人类工具回灌）+ 普通 text
                tool_results = []
                user_texts = []
                for c in content:
                    if not isinstance(c, dict):
                        continue
                    ctype = c.get("type")
                    if ctype == "tool_result":
                        ts = _extract_tool_result_content(c.get("content", ""))
                        if len(ts) > max_tool_result_chars:
                            ts = ts[:max_tool_result_chars] + f"\n... [truncated, original {len(ts)} chars]"
                        tool_results.append(ts)
                    elif ctype == "text":
                        ut = clean_user_text(c.get("text", "") or "")
                        if ut:
                            user_texts.append(ut)
                    elif ctype == "image":
                        user_texts.append("[image omitted]")
                # 先把 tool_result 当 tool 角色 push
                for tr in tool_results:
                    msgs.append({"role": "tool", "content": tr})
                # 然后是真正的人类 user 文本
                if user_texts:
                    msgs.append({"role": "user", "content": "\n".join(user_texts)})
            continue

        # === assistant 消息 ===
        if role == "assistant":
            if isinstance(content, str):
                txt = content.strip()
                if is_assistant_garbage(txt):
                    continue
                if txt:
                    msgs.append({"role": "assistant", "content": txt})
                continue
            if not isinstance(content, list):
                continue

            text_parts = []
            thinking_parts = []
            tool_calls = []
            for c in content:
                if not isinstance(c, dict):
                    continue
                ctype = c.get("type")
                if ctype == "text":
                    txt = (c.get("text", "") or "").strip()
                    if txt and not is_assistant_garbage(txt):
                        text_parts.append(txt)
                elif ctype == "thinking":
                    th = (c.get("thinking", "") or "").strip()
                    if th:
                        thinking_parts.append(th)
                elif ctype == "tool_use":
                    name = c.get("name", "?")
                    inp = c.get("input", {}) or {}
                    if not isinstance(inp, dict):
                        inp = {"value": inp}
                    tool_calls.append({"name": name, "arguments": inp})
                # tool_result 在 assistant 消息里出现则忽略（不应该出现）
                # image 跳过

            asst_msg: dict = {"role": "assistant"}
            asst_msg["content"] = "\n\n".join(text_parts) if text_parts else ""
            if thinking_parts:
                asst_msg["reasoning_content"] = "\n\n".join(thinking_parts)
            if tool_calls:
                asst_msg["tool_calls"] = tool_calls
            # 完全空（没 text 没 tool_call 也没 thinking）则跳过
            if not asst_msg["content"] and not tool_calls and "reasoning_content" not in asst_msg:
                continue
            msgs.append(asst_msg)

    # 后处理：合并连续 tool 角色（同一 assistant 多个 tool_use → 多个 tool_result，
    # 我们按出现顺序保留）
    return _post_process_session(msgs)


def session_quality_ok(msgs: list[dict], min_tool_use: int, min_assistant: int = 2) -> bool:
    n_tool_use = sum(1 for m in msgs if m["role"] == "assistant" and m.get("tool_calls"))
    if n_tool_use < min_tool_use:
        return False
    n_assistant = sum(1 for m in msgs if m["role"] == "assistant")
    if n_assistant < min_assistant:
        return False
    # 至少要有一个 user
    if not any(m["role"] == "user" for m in msgs):
        return False
    # 至少要有一个 assistant 文本回答（非纯 tool_call）
    if not any(m["role"] == "assistant" and m.get("content") for m in msgs):
        return False
    return True


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/.claude/projects"))
    ap.add_argument("--out", default="datasets/sft_claude_trajectories.jsonl")
    ap.add_argument("--min-tool-use", type=int, default=1, help="session 至少含多少次 tool_use 才保留")
    ap.add_argument("--max-tool-result-chars", type=int, default=4000,
                    help="单条 tool_result 内容上限，超过截断（避免单 session 爆 200k）")
    ap.add_argument("--include-projects", default=None,
                    help="逗号分隔的项目名（basename 子串匹配），不指定 = 全部")
    ap.add_argument("--exclude-projects", default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.root, "*", "*.jsonl")))
    if args.include_projects:
        keep = [k.strip() for k in args.include_projects.split(",") if k.strip()]
        files = [f for f in files if any(k in os.path.basename(os.path.dirname(f)) for k in keep)]
    if args.exclude_projects:
        skip = [k.strip() for k in args.exclude_projects.split(",") if k.strip()]
        files = [f for f in files if not any(k in os.path.basename(os.path.dirname(f)) for k in skip)]
    print(f"[build] scanning {len(files)} session files under {args.root}")

    n_total = 0
    n_kept = 0
    n_skipped_short = 0
    tool_use_dist = Counter()
    project_count = Counter()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as fout:
        for fp in files:
            entries = []
            with open(fp, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            if not entries:
                continue
            n_total += 1
            msgs = convert_session(entries, args.max_tool_result_chars)
            if not session_quality_ok(msgs, args.min_tool_use):
                n_skipped_short += 1
                continue
            project = os.path.basename(os.path.dirname(fp))
            session = os.path.basename(fp).replace(".jsonl", "")
            n_tool_use = sum(1 for m in msgs if m.get("tool_calls"))
            tool_use_dist[n_tool_use] += 1
            project_count[project] += 1
            for m in msgs:
                if m.get("tool_calls"):
                    for tc in m["tool_calls"]:
                        pass  # 仅占位，可日后收集
            obj = {
                "messages": msgs,
                "meta": {"project": project, "session": session, "n_tool_use": n_tool_use},
            }
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_kept += 1

    print(f"\n[build] {n_total} sessions scanned, {n_kept} kept, {n_skipped_short} skipped (low quality)")
    print(f"[build] tool_use per kept session: "
          f"min={min(tool_use_dist) if tool_use_dist else 0} "
          f"med={sorted(tool_use_dist.elements())[len(list(tool_use_dist.elements()))//2] if tool_use_dist else 0} "
          f"max={max(tool_use_dist) if tool_use_dist else 0}")
    print(f"[build] top 10 projects by kept sessions:")
    for proj, cnt in project_count.most_common(10):
        print(f"  {cnt:>4}  {proj}")
    print(f"\n[build] wrote {out_path} ({out_path.stat().st_size/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
