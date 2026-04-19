#!/usr/bin/env python3
"""v2.1: 强字面约束合成 —— query 和 path 必须字面出现在 user prompt。

v2 失败原因：模型幻觉 query/path（`def atnres`、`volatel1` 等），因为 query 是
从 candidate metadata 生成的，和 user prompt 没有字面对齐。400M 模型学不会语义
提取，只能学字面 copy。

v2.1 invariant：
  - user prompt 包含 query 字面串
  - user prompt 包含 path 字面串（若 tool_call 用到）
  - final answer 简洁直接，单风格（不混 v5 的"根据搜索结果"模板）

5 个 template（都是 single-step search_code）：
  const_lookup : "`FOO` 常量在 `src/x.py` 的值" → query=FOO path=src/x.py
  func_locate  : "`bar` 函数在 `src/y.py` 的哪一行" → query=bar path=src/y.py
  argparse     : "`scripts/z.py` 的 `--flag` 默认值" → query=--flag path=scripts/z.py
  const_usage  : "常量 `FOO` 被哪些文件引用" → query=FOO (无 path)
  func_usage   : "函数 `bar` 被哪里调用" → query=bar (无 path)
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_grep(query: str, path: str = "", max_lines: int = 30) -> str:
    cmd = ["grep", "-rEn", "--include=*.py", "--include=*.md", "--", query]
    cmd.append(path if path else "src/")
    if not path:
        cmd2 = ["grep", "-rEn", "--include=*.py", "--include=*.md", "--", query, "scripts/"]
    else:
        cmd2 = None
    try:
        out = subprocess.check_output(cmd, cwd=ROOT, stderr=subprocess.DEVNULL, text=True, timeout=10)
    except subprocess.CalledProcessError:
        out = ""
    except Exception:
        return ""
    if cmd2:
        try:
            out2 = subprocess.check_output(cmd2, cwd=ROOT, stderr=subprocess.DEVNULL, text=True, timeout=10)
            out = (out + "\n" + out2).strip()
        except Exception:
            pass
    lines = [l for l in out.strip().splitlines() if l][:max_lines]
    return "\n".join(lines)


def tc(tool: str, params: dict) -> dict:
    return {"role": "assistant", "content": f"<|tool_call_start|><|tool_name_{tool}|>{json.dumps(params, ensure_ascii=False)}<|tool_call_end|>"}


def tr(content: str) -> dict:
    return {"role": "assistant", "content": f"<|tool_result_start|>{content}<|tool_result_end|>"}


def fa(content: str) -> dict:
    return {"role": "assistant", "content": content}


def assert_literal(user: str, query: str, path: str = "") -> bool:
    """user prompt 必须字面包含 query 和 path（若提供）。"""
    if query not in user:
        return False
    if path and path not in user:
        return False
    return True


def enum_py(dirs=("src", "scripts")):
    for rel in dirs:
        for f in (ROOT / rel).glob("*.py"):
            yield f"{rel}/{f.name}", f


# ================ Template A: const_lookup ================

CONST_Q = [
    "`{name}` 常量在 `{path}` 的值是什么？",
    "`{path}` 里的 `{name}` 常量值是多少？",
    "查一下 `{path}` 的 `{name}` 常量。",
]


def synth_const_lookup(n: int) -> list[dict]:
    cand = []
    for rel_path, f in enum_py():
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            m = re.match(r"^([A-Z_][A-Z0-9_]{2,})\s*=\s*([^\s#][^#\n]*?)(\s*#.*)?$", line)
            if not m:
                continue
            value = m.group(2).strip().rstrip(",")
            if len(value) > 80 or value.endswith(("(", "[", "{", ",", "\\")):
                continue
            cand.append({"path": rel_path, "name": m.group(1), "lineno": i, "value": value})
    random.shuffle(cand)
    out, seen = [], set()
    for c in cand:
        key = (c["path"], c["name"])
        if key in seen:
            continue
        seen.add(key)
        user = random.choice(CONST_Q).format(**c)
        if not assert_literal(user, c["name"], c["path"]):
            continue
        grep = run_grep(c["name"], c["path"])
        if c["name"] not in grep:
            continue
        final = f"`{c['name']}` 在 `{c['path']}:{c['lineno']}`，值为 `{c['value']}`。"
        out.append({
            "id": f"cc21_cst_{c['path'].replace('/','_').replace('.py','')}_{c['name']}",
            "messages": [
                {"role": "user", "content": user},
                tc("search_code", {"query": c["name"], "path": c["path"]}),
                tr(grep),
                fa(final),
            ],
        })
        if len(out) >= n:
            break
    return out


# ================ Template B: func_locate ================

FUNC_Q = [
    "`{name}` 函数在 `{path}` 的哪一行？",
    "`{path}` 里 `{name}` 函数定义在第几行？",
    "帮我找 `{path}` 里的 `{name}` 函数。",
]


def synth_func_locate(n: int) -> list[dict]:
    cand = []
    for rel_path, f in enum_py():
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            m = re.match(r"^def ([a-z_][a-z0-9_]{2,})\(", line)
            if not m or m.group(1).startswith("_"):
                continue
            cand.append({"path": rel_path, "name": m.group(1), "lineno": i})
    random.shuffle(cand)
    out, seen = [], set()
    for c in cand:
        key = (c["path"], c["name"])
        if key in seen:
            continue
        seen.add(key)
        user = random.choice(FUNC_Q).format(**c)
        if not assert_literal(user, c["name"], c["path"]):
            continue
        grep = run_grep(c["name"], c["path"])
        if c["name"] not in grep:
            continue
        final = f"`{c['name']}` 定义在 `{c['path']}:{c['lineno']}`。"
        out.append({
            "id": f"cc21_fnc_{c['path'].replace('/','_').replace('.py','')}_{c['name']}",
            "messages": [
                {"role": "user", "content": user},
                tc("search_code", {"query": c["name"], "path": c["path"]}),
                tr(grep),
                fa(final),
            ],
        })
        if len(out) >= n:
            break
    return out


# ================ Template C: argparse ================

ARG_Q = [
    "`{path}` 的 `--{arg}` 参数默认值是什么？",
    "`{path}` 里 `--{arg}` 这个参数干嘛的？",
    "查 `{path}` 的 `--{arg}` 参数。",
]


def synth_argparse(n: int) -> list[dict]:
    cand = []
    for rel_path, f in enum_py():
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            m = re.search(r"add_argument\(['\"]--([a-z][a-z0-9-]+)['\"]", line)
            if not m:
                continue
            cand.append({"path": rel_path, "arg": m.group(1), "lineno": i, "def_line": line.strip()})
    random.shuffle(cand)
    out, seen = [], set()
    for c in cand:
        key = (c["path"], c["arg"])
        if key in seen:
            continue
        seen.add(key)
        user = random.choice(ARG_Q).format(**c)
        # 注意：path 里有 "/" user 里也有，字面检查
        if not assert_literal(user, c["arg"], c["path"]):
            continue
        grep = run_grep(f"--{c['arg']}", c["path"])
        if c["arg"] not in grep:
            continue
        default_m = re.search(r"default=([^,\)]+)", c["def_line"])
        help_m = re.search(r"help=['\"]([^'\"]+)['\"]", c["def_line"])
        default = default_m.group(1).strip() if default_m else "无默认"
        help_s = help_m.group(1) if help_m else ""
        if help_s:
            final = f"`--{c['arg']}`：{help_s}。默认 `{default}`（`{c['path']}:{c['lineno']}`）。"
        else:
            final = f"`--{c['arg']}` 默认 `{default}`，定义在 `{c['path']}:{c['lineno']}`。"
        out.append({
            "id": f"cc21_arg_{c['path'].replace('/','_').replace('.py','')}_{c['arg']}",
            "messages": [
                {"role": "user", "content": user},
                tc("search_code", {"query": f"--{c['arg']}", "path": c["path"]}),
                tr(grep),
                fa(final),
            ],
        })
        if len(out) >= n:
            break
    return out


# ================ Template D: const_usage ================

CUSE_Q = [
    "常量 `{name}` 被哪些文件引用？",
    "`{name}` 这个常量在仓库里被谁用了？",
    "谁引用了常量 `{name}`？",
]


def synth_const_usage(n: int) -> list[dict]:
    cand = []
    for rel_path, f in enum_py():
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            m = re.match(r"^([A-Z_][A-Z0-9_]{3,})\s*=\s*", line)
            if m:
                cand.append({"path": rel_path, "name": m.group(1), "lineno": i})
    random.shuffle(cand)
    out, seen = [], set()
    for c in cand:
        if c["name"] in seen:
            continue
        seen.add(c["name"])
        user = random.choice(CUSE_Q).format(**c)
        if not assert_literal(user, c["name"]):
            continue
        grep = run_grep(c["name"])
        lines = grep.splitlines()
        if len(lines) < 2:
            continue
        files = sorted({l.split(":", 1)[0] for l in lines if ":" in l})[:5]
        final = f"`{c['name']}` 被 {', '.join(f'`{x}`' for x in files)} 引用，共 {len(lines)} 处。"
        out.append({
            "id": f"cc21_cuse_{c['path'].replace('/','_').replace('.py','')}_{c['name']}",
            "messages": [
                {"role": "user", "content": user},
                tc("search_code", {"query": c["name"]}),
                tr(grep[:1500]),
                fa(final),
            ],
        })
        if len(out) >= n:
            break
    return out


# ================ Template E: func_usage ================

FUSE_Q = [
    "函数 `{name}` 被哪里调用？",
    "`{name}` 这个函数被谁用了？",
    "谁调用了 `{name}`？",
]


def synth_func_usage(n: int) -> list[dict]:
    cand = []
    for rel_path, f in enum_py():
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            m = re.match(r"^def ([a-z_][a-z0-9_]{3,})\(", line)
            if m and not m.group(1).startswith("_"):
                cand.append({"path": rel_path, "name": m.group(1), "lineno": i})
    random.shuffle(cand)
    out, seen = [], set()
    for c in cand:
        if c["name"] in seen:
            continue
        seen.add(c["name"])
        user = random.choice(FUSE_Q).format(**c)
        if not assert_literal(user, c["name"]):
            continue
        grep = run_grep(c["name"])
        # 过滤 def 自身那行
        lines = [l for l in grep.splitlines() if f"def {c['name']}(" not in l]
        if len(lines) < 2:
            continue
        files = sorted({l.split(":", 1)[0] for l in lines if ":" in l})[:5]
        final = f"`{c['name']}` 被 {', '.join(f'`{x}`' for x in files)} 调用，共 {len(lines)} 处。"
        out.append({
            "id": f"cc21_fuse_{c['path'].replace('/','_').replace('.py','')}_{c['name']}",
            "messages": [
                {"role": "user", "content": user},
                tc("search_code", {"query": c["name"]}),
                tr("\n".join(lines)[:1500]),
                fa(final),
            ],
        })
        if len(out) >= n:
            break
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/cc_synth_v2_1.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-const", type=int, default=200)
    parser.add_argument("--n-func", type=int, default=200)
    parser.add_argument("--n-arg", type=int, default=150)
    parser.add_argument("--n-cuse", type=int, default=200)
    parser.add_argument("--n-fuse", type=int, default=150)
    args = parser.parse_args()

    random.seed(args.seed)

    groups = {
        "const_lookup": synth_const_lookup(args.n_const),
        "func_locate":  synth_func_locate(args.n_func),
        "argparse":     synth_argparse(args.n_arg),
        "const_usage":  synth_const_usage(args.n_cuse),
        "func_usage":   synth_func_usage(args.n_fuse),
    }

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for kind, items in groups.items():
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                total += 1
            print(f"  {kind:14s}: {len(items):>4d}")
    print(f"\n总 {total} 条 → {out_path}")


if __name__ == "__main__":
    main()
