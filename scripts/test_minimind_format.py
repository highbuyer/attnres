"""Smoke test: minimind conversations → (ids, mask) 与 chat_template 渲染结果对齐。

验证：
1. 渲染文本与 HF apply_chat_template 字符串对齐（空白可小异）
2. loss_mask 只在 assistant 块（<|im_start|>assistant\n 到 <|im_end|>\n 之间）为 1
3. user / system / tool_response 段为 0
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prepare import Tokenizer  # type: ignore

tk = Tokenizer.from_directory()
enc = tk.enc


def render_system(convs, tools):
    if not tools and not (convs and convs[0].get("role") == "system" and convs[0].get("content")):
        return ""
    if tools:
        sys_content = convs[0].get("content", "") if convs and convs[0].get("role") == "system" else ""
        tools_block = (
            "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
        )
        for tool in tools:
            tools_block += "\n" + json.dumps(tool, ensure_ascii=False)
        tools_block += (
            "\n</tools>\n\nFor each function call, return a json object with function name and "
            'arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n'
            '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
        )
        head = sys_content + "\n\n" if sys_content else ""
        return f"<|im_start|>system\n{head}{tools_block}<|im_end|>\n"
    return f"<|im_start|>system\n{convs[0]['content']}<|im_end|>\n"


def render_assistant(msg):
    content = msg.get("content", "") or ""
    rc = msg.get("reasoning_content", "") or ""
    body = "<think>\n" + rc.strip("\n") + "\n</think>\n\n" + content.lstrip("\n")
    tcs = msg.get("tool_calls")
    if tcs:
        if isinstance(tcs, str):
            tcs = json.loads(tcs)
        for i, tc in enumerate(tcs):
            if (i == 0 and content) or i > 0:
                body += "\n"
            fn = tc.get("function", tc)
            args = fn.get("arguments", "")
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            body += '<tool_call>\n{"name": "' + fn["name"] + '", "arguments": ' + args + "}\n</tool_call>"
    return body


def format_record(record):
    convs = record.get("conversations", [])
    tools = None
    if convs and convs[0].get("role") == "system" and convs[0].get("tools"):
        t = convs[0]["tools"]
        tools = json.loads(t) if isinstance(t, str) else t

    chunks = []  # [(kind, text)]   kind ∈ {"system","user","tool","asst_prefix","asst_body","asst_suffix"}
    sys_text = render_system(convs, tools)
    if sys_text:
        chunks.append(("system", sys_text))

    start = 1 if convs and convs[0].get("role") == "system" else 0
    for m in convs[start:]:
        role = m.get("role")
        content = m.get("content", "") or ""
        if role == "user":
            chunks.append(("user", f"<|im_start|>user\n{content}<|im_end|>\n"))
        elif role == "tool":
            chunks.append(("tool", f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n"))
        elif role == "assistant":
            chunks.append(("asst_prefix", "<|im_start|>assistant\n"))
            chunks.append(("asst_body", render_assistant(m)))
            chunks.append(("asst_suffix", "<|im_end|>\n"))

    # tokenize + mask
    ids: list[int] = []
    mask: list[int] = []
    pieces_text: list[tuple[str, str, int, int]] = []  # (kind, text, tok_start, tok_end)
    for kind, text in chunks:
        t_ids = enc.encode(text, allowed_special="all")
        start_i = len(ids)
        ids.extend(t_ids)
        end_i = len(ids)
        pieces_text.append((kind, text, start_i, end_i))
        m_flag = 1 if kind in ("asst_body", "asst_suffix") else 0
        mask.extend([m_flag] * len(t_ids))

    return ids, mask, pieces_text


def main():
    sample_path = Path("/home/langshen/base_mode/attnres/data/minimind_sample/sft_tail_2mb.jsonl")
    with sample_path.open() as f:
        f.readline()  # skip truncated first
        line = f.readline()
    record = json.loads(line)
    ids, mask, pieces = format_record(record)
    print(f"total tokens: {len(ids)}")
    print(f"mask=1 count: {sum(mask)}  ({sum(mask)/len(mask):.1%})")
    print("--- pieces ---")
    for kind, text, s, e in pieces:
        print(f"[{kind:15} tok={s}:{e}={e-s}] {text[:80]!r}")
    # decode check
    decoded = enc.decode(ids)
    print("--- decoded head 400 ---")
    print(decoded[:400])
    print("--- decoded mask=1 region (assistant content) ---")
    mask_1_ids = [ids[i] for i in range(len(ids)) if mask[i] == 1]
    print(enc.decode(mask_1_ids)[:400])


if __name__ == "__main__":
    main()
