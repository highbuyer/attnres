#!/usr/bin/env python3
"""从 Claude Code 的 session 归档（~/.claude/projects/…attnres/*.jsonl）
提取 Read/Grep-only 的轨迹，映射到 attnres 格式并用 execute_tool 重算 tool_result，
闭环 validate 通过才算 gold。

dry-run 默认；--write 才落盘到 data/cc_gold_trajectories_v2.jsonl。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/home/langshen/base_mode/attnres")
sys.path.insert(0, str(ROOT / "src"))

from tool_protocol import (  # noqa: E402
    TOOL_CALL_START,
    TOOL_CALL_END,
    TOOL_RESULT_START,
    TOOL_RESULT_END,
    execute_tool,
    parse_tool_call,
    validate_tool_sample,
)

SESSION_DIR = Path.home() / ".claude/projects/-home-langshen-base-mode-attnres"
OUT_PATH = ROOT / "data" / "cc_gold_trajectories_v2.jsonl"


def render_tool_call(tool_name: str, params: dict) -> str:
    tag = f"<|tool_name_{tool_name}|>"
    inner = tag + json.dumps(params, ensure_ascii=False, separators=(",", ":"))
    return f"{TOOL_CALL_START}{inner}{TOOL_CALL_END}"


def render_tool_result(result: str) -> str:
    return f"{TOOL_RESULT_START}{result}{TOOL_RESULT_END}"


PATH_CANDIDATE_PREFIXES = ("", "src/", "scripts/", "docs/", "tests/")


def remap_to_current_repo(rel: str, root: Path) -> str | None:
    """历史 session 用老布局路径（如 'sft.py'），尝试重映射到当前 repo 位置。
    只在唯一候选存在时替换；多个候选/零候选返回 None。"""
    direct = (root / rel).resolve()
    try:
        if direct.exists() and direct.is_relative_to(root):
            return rel  # 路径原样可用
    except (AttributeError, ValueError):
        pass

    basename = rel.lstrip("./")
    if "/" in basename:
        # 带目录的路径，没法猜；只试顶级 prefix 替换
        return None

    hits = []
    for prefix in PATH_CANDIDATE_PREFIXES:
        cand = (root / f"{prefix}{basename}").resolve()
        try:
            if cand.exists() and cand.is_relative_to(root):
                hits.append(f"{prefix}{basename}")
        except (AttributeError, ValueError):
            continue
    if len(hits) == 1:
        return hits[0]
    return None


def map_claude_code_tool(name: str, inp: dict, root: Path) -> tuple[str, dict] | None:
    """把 Claude Code 的 Read/Grep/Glob 参数映射到 attnres 的 read_file/search_code。
    无法干净映射的返回 None。"""
    if name == "Read":
        fp = inp.get("file_path") or ""
        if not fp:
            return None
        try:
            p = Path(fp).resolve()
            # 允许相对路径 / 绝对路径：先试 resolve 是否在 root 内
            if p.is_absolute():
                if not p.is_relative_to(root):
                    return None
                rel = str(p.relative_to(root))
            else:
                rel = fp.lstrip("./")
        except (ValueError, AttributeError):
            return None
        # 老布局路径 -> 当前 repo 位置的 basename 候选映射
        remapped = remap_to_current_repo(rel, root)
        if remapped is None:
            return None
        rel = remapped
        params: dict[str, object] = {"path": rel}
        if "offset" in inp:
            try:
                params["offset"] = int(inp["offset"])
            except (TypeError, ValueError):
                return None
        if "limit" in inp:
            try:
                params["limit"] = int(inp["limit"])
            except (TypeError, ValueError):
                return None
        return ("read_file", params)

    if name == "Grep":
        pattern = inp.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            return None
        # 丢掉带 path 限定、multiline、regex 特性的——attnres search_code 只是子串
        if inp.get("path"):
            return None
        if inp.get("multiline"):
            return None
        if inp.get("-i"):
            return None
        # 明显的正则元字符（.*+|）也跳过，attnres 是 fixed-strings
        if any(ch in pattern for ch in r".*+|()[]{}^$\\"):
            return None
        return ("search_code", {"query": pattern})

    # Glob / Bash / Edit / Write / Task* 一律不映射
    return None


def extract_final_text(blocks: list) -> str | None:
    """取最后一个 type=text 的文本块。"""
    for blk in reversed(blocks):
        if isinstance(blk, dict) and blk.get("type") == "text":
            t = (blk.get("text") or "").strip()
            if t:
                return t
    return None


def is_meta_user(msg_content) -> bool:
    if not isinstance(msg_content, str):
        return False
    markers = (
        "<command-name>",
        "<local-command-caveat>",
        "<command-message>",
        "<system-reminder>",
    )
    return any(m in msg_content for m in markers)


def is_substantive(text: str, min_chars: int) -> bool:
    """用户 prompt / final 必须有实质内容，过滤 '好'/'嗯'/'行' 这类"""
    stripped = text.strip()
    if len(stripped) < min_chars:
        return False
    trivial = {"好", "嗯", "行", "对", "是", "好的", "嗯嗯", "收到", "ok", "OK"}
    if stripped in trivial:
        return False
    return True


TOOL_RESULT_ERROR_PREFIXES = (
    "(文件不存在",
    "(路径越界",
    "(未知工具",
    "(无效 offset",
    "(无效 limit",
    "(工具执行错误",
    "(空查询",
    "(未找到匹配",
)


def is_error_tool_result(result: str) -> bool:
    return result.startswith(TOOL_RESULT_ERROR_PREFIXES)


def iter_trajectories(session_file: Path):
    """yield (user_prompt, [(tool_name, tool_input), ...], final_text)。
    final_text 必须是最后一个 tool_use 之后的 assistant text block。"""
    current_user = None
    tool_calls: list[tuple[str, dict]] = []
    final_text: str | None = None
    saw_tool_since_last_text = False  # 追踪是否 tool_use 之后有新 text

    def flush():
        nonlocal current_user, tool_calls, final_text, saw_tool_since_last_text
        pending = None
        if current_user and tool_calls and final_text and not saw_tool_since_last_text:
            pending = (current_user, list(tool_calls), final_text)
        current_user, tool_calls, final_text, saw_tool_since_last_text = None, [], None, False
        return pending

    for raw in session_file.open(encoding="utf-8"):
        try:
            ev = json.loads(raw)
        except json.JSONDecodeError:
            continue
        etype = ev.get("type")
        msg = ev.get("message") or {}

        if etype == "user":
            c = msg.get("content")
            if isinstance(c, str) and not is_meta_user(c) and not ev.get("isMeta"):
                pending = flush()
                if pending is not None:
                    yield pending
                current_user = c.strip()
                tool_calls = []
                final_text = None
                saw_tool_since_last_text = False

        elif etype == "assistant":
            c = msg.get("content")
            if not isinstance(c, list):
                continue
            # 一条 assistant event 可能同时有 tool_use 和 text；按顺序扫：
            # 遇到 tool_use -> 记到 tool_calls，saw_tool_since_last_text = True（清掉之前的 final_text）
            # 遇到 text     -> 如果在所有 tool 之后出现，更新 final_text
            for blk in c:
                if not isinstance(blk, dict):
                    continue
                t = blk.get("type")
                if t == "tool_use":
                    tool_calls.append((blk.get("name"), blk.get("input") or {}))
                    final_text = None
                    saw_tool_since_last_text = True
                elif t == "text":
                    txt = (blk.get("text") or "").strip()
                    if txt:
                        final_text = txt
                        saw_tool_since_last_text = False

    pending = flush()
    if pending is not None:
        yield pending


def multi_step_validate(messages: list[dict], root: Path) -> tuple[bool, str]:
    """逐对 (tool_call, tool_result) 比对：每一步的 call 用 execute_tool 重执行，
    与紧接着的 tool_result 的内容（剥 start/end tag 后）逐字节比对。"""
    pairs: list[tuple[str, str]] = []
    i = 0
    while i < len(messages):
        c = messages[i].get("content", "")
        if isinstance(c, str) and TOOL_CALL_START in c and i + 1 < len(messages):
            nxt = messages[i + 1].get("content", "")
            if isinstance(nxt, str) and TOOL_RESULT_START in nxt and TOOL_RESULT_END in nxt:
                pairs.append((c, nxt))
                i += 2
                continue
        i += 1
    if not pairs:
        return False, "no_pairs"
    for idx, (tc, tr) in enumerate(pairs):
        parsed = parse_tool_call(tc)
        if parsed is None:
            return False, f"bad_call_step_{idx}"
        tool_name, tool_params = parsed
        expected = execute_tool(tool_name, tool_params, root)
        start = tr.find(TOOL_RESULT_START) + len(TOOL_RESULT_START)
        end = tr.find(TOOL_RESULT_END, start)
        actual = tr[start:end]
        if actual != expected:
            return False, f"mismatch_step_{idx}"
    return True, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="落盘到 OUT_PATH")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    session_files = sorted(SESSION_DIR.rglob("*.jsonl"))
    print(f"扫 {len(session_files)} 个 session 文件")

    raw_trajectories = 0
    pure_mapped: list[tuple[str, list, str]] = []  # (user, [(attnres_name, params)], final_text)
    drop_reasons = Counter()

    for sf in session_files:
        for user, calls, final in iter_trajectories(sf):
            raw_trajectories += 1
            mapped: list[tuple[str, dict]] = []
            bad = False
            for name, inp in calls:
                if name in {"TaskUpdate", "TaskCreate", "TaskList", "TaskGet", "TaskOutput"}:
                    # 任务管理，非 tool；忽略不算 mixed
                    continue
                m = map_claude_code_tool(name, inp, ROOT)
                if m is None:
                    bad = True
                    drop_reasons[f"unmappable:{name}"] += 1
                    break
                mapped.append(m)
            if bad:
                continue
            if not mapped:
                drop_reasons["no_mappable_call"] += 1
                continue
            pure_mapped.append((user, mapped, final))

    print(f"raw trajectories: {raw_trajectories}")
    print(f"可映射 (user→纯 Read/Grep→text): {len(pure_mapped)}")
    print(f"drop reasons top10: {drop_reasons.most_common(10)}")

    # 闭环：用 attnres execute_tool 重新生成 tool_result，再 validate
    gold_samples: list[dict] = []
    val_stats = Counter()
    post_filter = Counter()
    for user, calls, final in pure_mapped:
        if not is_substantive(user, min_chars=6):
            post_filter["user_too_short"] += 1
            continue
        if not is_substantive(final, min_chars=10):
            post_filter["final_too_short"] += 1
            continue

        messages = [{"role": "user", "content": user}]
        has_error = False
        for tn, tp in calls:
            tc_text = render_tool_call(tn, tp)
            tr_raw = execute_tool(tn, tp, ROOT)
            if is_error_tool_result(tr_raw):
                has_error = True
                break
            tr_text = render_tool_result(tr_raw)
            messages.append({"role": "assistant", "content": tc_text})
            messages.append({"role": "assistant", "content": tr_text})
        if has_error:
            post_filter["tool_result_error"] += 1
            continue

        messages.append({"role": "assistant", "content": final})
        sample = {"messages": messages, "source": "claude-code-session"}
        # 用 multi_step_validate 替代 validate_tool_sample：后者只检第一步 tc + 最后一步 tr，
        # 多步轨迹必误判 mismatch；multi_step 逐对比对才准确
        ok, reason = multi_step_validate(messages, ROOT)
        val_stats[reason] += 1
        if ok and reason == "ok":
            gold_samples.append(sample)

    # 同时跑一次 validate_tool_sample 做 sanity（应该比 multi_step 宽/等价）
    sanity_ok = 0
    sanity_mismatch = 0
    for s in gold_samples:
        ok, reason = validate_tool_sample(s, ROOT)
        if ok:
            sanity_ok += 1
        else:
            sanity_mismatch += 1

    print(f"validate 通过: {len(gold_samples)}  / reason 分布: {dict(val_stats)}")
    print(f"post filter 丢弃: {dict(post_filter)}")
    print(f"sanity (attnres 原生 validate_tool_sample): ok={sanity_ok}, mismatch={sanity_mismatch}"
          " (多步轨迹 mismatch 是原 validator 限制，multi_step 才是真相)")
    if args.limit:
        gold_samples = gold_samples[: args.limit]

    # 抽样核对：首、中、尾各 1 条
    if gold_samples:
        print()
        print("=== 抽样 (抽 3 条，user prompt + 第 1 个 tool_call 参数 + tool_result 首 80 字) ===")
        idxs = [0, len(gold_samples) // 2, len(gold_samples) - 1]
        for idx in idxs:
            s = gold_samples[idx]
            u = s["messages"][0]["content"][:100]
            tc = s["messages"][1]["content"]
            tr = s["messages"][2]["content"].replace(TOOL_RESULT_START, "").replace(TOOL_RESULT_END, "")[:120]
            print(f"  [#{idx}] user: {u}")
            print(f"         tc  : {tc[:120]}")
            print(f"         tr  : {tr!r}")

    if args.write:
        OUT_PATH.write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in gold_samples) + "\n",
                            encoding="utf-8")
        print(f"\n✅ 落盘: {OUT_PATH}  ({len(gold_samples)} 条)")
    else:
        print("\n(dry-run) 未落盘。传 --write 即保存。")


if __name__ == "__main__":
    main()
