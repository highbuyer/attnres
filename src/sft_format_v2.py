"""Qwen 风格消息渲染器（不扩 vocab，沿用现有 BOS/USER/ASST/EOS）。

输出 schema 协议（单条 sample）：
  [BOS]
  [USER]   user content
  [ASST]   asst content + <tool_call>{...}</tool_call> ...
  [USER]   <tool_response>...</tool_response>
  [ASST]   ...
  [EOS]

约定：
  - BOS = <|reserved_0|>     仅样本开头（loss=0）
  - USER = <|reserved_1|>    user 角色 / tool_response 容器（loss=0）
  - ASST = <|reserved_2|>    assistant 角色（loss=1 起算）
  - EOS = <|reserved_3|>     assistant turn 终止符（loss=1）

为何不扩 vocab：保留 v2 ckpt 不动，<tool_call> / <tool_response> 走纯字符串
（BPE 会拆，但模型从训练数据自学边界）。后续若需可再加 special token。
"""
from __future__ import annotations

import json
from typing import Any


def _render_tool_calls(tool_calls: list[dict]) -> str:
    """Render tool_calls list → text segment to append after assistant content."""
    parts = []
    for tc in tool_calls:
        name = tc.get("name", "?")
        args = tc.get("arguments", {})
        if not isinstance(args, (dict, list)):
            args = {"value": args}
        try:
            args_str = json.dumps(args, ensure_ascii=False)
        except Exception:
            args_str = str(args)
        parts.append(f'<tool_call>\n{{"name": "{name}", "arguments": {args_str}}}\n</tool_call>')
    return "\n".join(parts)


def _render_assistant_text(msg: dict) -> str:
    """assistant text body = (reasoning?) + content + tool_calls (if any)."""
    chunks = []
    rc = msg.get("reasoning_content")
    if rc:
        chunks.append(f"<think>\n{rc}\n</think>")
    c = (msg.get("content") or "").strip()
    if c:
        chunks.append(c)
    tcs = msg.get("tool_calls") or []
    if tcs:
        chunks.append(_render_tool_calls(tcs))
    return "\n".join(chunks)


def _render_tool_text(msg: dict) -> str:
    c = (msg.get("content") or "").strip()
    return f"<tool_response>\n{c}\n</tool_response>"


def render_messages_to_samples(
    messages: list[dict],
    tokenizer: Any,
    max_seq_len: int,
    bos_id: int,
    user_id: int,
    asst_id: int,
    eos_id: int,
) -> list[tuple[list[int], list[int]]]:
    """
    把多轮 messages → 多个 (ids, mask) sample。
    每个 assistant 终点切一个 sample，sample 内保留：
      [BOS] + (history turns) + [ASST] + (this asst body) + [EOS]
    """
    enc = tokenizer.enc

    # 先把每个 message 编码成 (turn_role_id, content_ids, content_mask_per_token)
    # mask=1 表示该 token 计 loss
    turns: list[tuple[str, list[int], list[int]]] = []
    for m in messages:
        role = m.get("role")
        if role == "user":
            text = (m.get("content") or "").strip()
            if not text:
                continue
            ids = [user_id] + enc.encode(text, allowed_special="all")
            mask = [0] * len(ids)
            turns.append(("user", ids, mask))
        elif role == "tool":
            text = _render_tool_text(m)
            ids = [user_id] + enc.encode(text, allowed_special="all")
            mask = [0] * len(ids)
            turns.append(("tool", ids, mask))
        elif role == "assistant":
            text = _render_assistant_text(m)
            if not text:
                continue
            body = enc.encode(text, allowed_special="all")
            ids = [asst_id] + body + [eos_id]
            mask = [0] + [1] * len(body) + [1]   # ASST_ID 不计 loss，body+EOS 计
            turns.append(("assistant", ids, mask))

    # 切片：从每个 assistant 终点回溯，把 [BOS]+history+this_asst 拼起来 ≤ max_seq_len
    samples: list[tuple[list[int], list[int]]] = []
    for j, (role, _ids, _mask) in enumerate(turns):
        if role != "assistant":
            continue
        cur_ids: list[int] = []
        cur_mask: list[int] = []
        # 从 turn[j] 开始向左回溯（包括自己），不超过 max_seq_len-1（留 BOS）
        for k in range(j, -1, -1):
            cand = turns[k][1] + cur_ids
            cand_mask = turns[k][2] + cur_mask
            if 1 + len(cand) > max_seq_len:
                # 装不下了，停（保留之前的 cur_ids）
                if not cur_ids:
                    # 当前 assistant turn 自己就太长，硬截
                    # 保留 ASST_ID + 末尾 (max_seq_len-2) + EOS
                    asst_ids = turns[j][1]
                    asst_mask = turns[j][2]
                    if len(asst_ids) > max_seq_len - 1:
                        # ASST_ID + 部分 body + EOS
                        body_budget = max_seq_len - 1 - 2
                        cur_ids = [asst_ids[0]] + asst_ids[-(body_budget + 1):]
                        cur_mask = [asst_mask[0]] + asst_mask[-(body_budget + 1):]
                break
            cur_ids = cand
            cur_mask = cand_mask
        if not cur_ids:
            continue
        ids = [bos_id] + cur_ids
        mask = [0] + cur_mask
        if sum(mask) == 0:
            continue
        samples.append((ids, mask))
    return samples
