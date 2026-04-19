#!/usr/bin/env python3
"""CC trajectory synthesizer v2 — 自动化生成 multi-step tool-use 训练数据。

设计：
1. grep 枚举仓库真实候选（常量、函数、argparse、typehint），避免 AI 幻觉
2. 按 4 类模板生成 query
3. 真跑 search_code / read_file 拿 observation（不是 AI 合成）
4. 按提取规则合成 final answer（自然语言变体池避免死板）
5. 组装成 v1 兼容的 messages 格式

每条 trajectory 的 tool_result 是实测输出，保证数据 100% 可验证。

目标：50 条，跟 v1 的 8 条手写 gold 合流作为 SFT 增量数据。
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# ================ tool 真跑 ================

def run_grep(query: str, path: str, max_lines: int = 30) -> str:
    """模拟 weiyan 的 search_code：grep -n，返回前 max_lines 行。"""
    try:
        out = subprocess.check_output(
            ["grep", "-rEn", "--include=*.py", "--include=*.md", query, path],
            cwd=ROOT, stderr=subprocess.DEVNULL, text=True, timeout=10,
        )
    except subprocess.CalledProcessError:
        return f"(未找到匹配: {query})"
    except Exception as e:
        return f"(grep 错误: {e})"
    lines = out.strip().splitlines()[:max_lines]
    return "\n".join(lines) if lines else f"(未找到匹配: {query})"


def run_read(path: str, start: int, end: int) -> str:
    """模拟 weiyan 的 read_file：读 path 的 [start, end] 行。"""
    full = (ROOT / path).read_text(encoding="utf-8").splitlines()
    if start < 1:
        start = 1
    if end > len(full):
        end = len(full)
    return "\n".join(full[start - 1:end])


# ================ 工具调用 / 结果 / final 的消息构造 ================

def tool_call(tool: str, params: dict) -> dict:
    return {
        "role": "assistant",
        "content": f"<|tool_call_start|><|tool_name_{tool}|>{json.dumps(params, ensure_ascii=False)}<|tool_call_end|>",
    }


def tool_result(content: str) -> dict:
    return {"role": "assistant", "content": f"<|tool_result_start|>{content}<|tool_result_end|>"}


def final(content: str) -> dict:
    return {"role": "assistant", "content": content}


# ================ Template A: 常量值 ================

CONST_QUERIES = [
    "{path} 里 {name} 的值是多少？",
    "{name} 这个常量定义在 {path} 的哪里？值是啥？",
    "查一下 {path} 的 {name}，它的值现在是多少？",
    "{name} 在 {path} 里设的什么数？",
]

CONST_FINALS = [
    "`{name}` 定义在 `{path}:{lineno}`，值是 `{value}`。",
    "`{path}:{lineno}` 里 `{name} = {value}`。",
    "找到了：`{name}` 在 `{path}` 第 {lineno} 行，值 `{value}`。",
]


def synth_constants(n: int) -> list[dict]:
    """枚举 src/ scripts/ 下的 UPPERCASE = literal 常量，生成 single-step query。"""
    candidates = []
    for rel in ["src", "scripts"]:
        p = ROOT / rel
        for f in p.glob("*.py"):
            text = f.read_text(encoding="utf-8").splitlines()
            for i, line in enumerate(text, 1):
                m = re.match(r"^([A-Z_][A-Z0-9_]{2,})\s*=\s*([^\s#][^#\n]*?)(\s*#.*)?$", line)
                if not m:
                    continue
                name = m.group(1)
                value = m.group(2).strip().rstrip(",")
                # 只要"值"看起来是字面量（数字/短字符串/简单表达式）
                if len(value) > 80:
                    continue
                if value.startswith(("[", "{", "(")) and not value.endswith(("]", "}", ")")):
                    continue  # 多行容器定义不要
                if value.endswith(("(", "[", "{", ",", "\\")):
                    continue  # 行尾是开括号/逗号/续行 → 多行定义，不要
                candidates.append({
                    "path": f"{rel}/{f.name}",
                    "name": name,
                    "lineno": i,
                    "value": value,
                })
    random.shuffle(candidates)
    out = []
    for c in candidates[:n]:
        q_tpl = random.choice(CONST_QUERIES)
        f_tpl = random.choice(CONST_FINALS)
        user_q = q_tpl.format(path=c["path"], name=c["name"])
        grep_out = run_grep(c["name"], c["path"])
        if c["name"] not in grep_out:
            continue
        final_a = f_tpl.format(**c)
        out.append({
            "id": f"cc_const_{c['path'].replace('/', '_').replace('.py', '')}_{c['name']}",
            "messages": [
                {"role": "user", "content": user_q},
                tool_call("search_code", {"query": c["name"], "path": c["path"]}),
                tool_result(grep_out),
                final(final_a),
            ],
        })
    return out


# ================ Template B: 函数位置 ================

FUNC_QUERIES = [
    "{name} 函数定义在哪里？",
    "函数 {name} 在哪个文件、多少行？",
    "帮我找 {name} 的定义。",
    "{name} 这个函数实现在哪？",
]

FUNC_FINALS = [
    "`{name}` 定义在 `{path}:{lineno}`。",
    "找到了：`{path}:{lineno}: def {name}(...)`。",
    "`{name}` 在 `{path}` 第 {lineno} 行。",
]


def synth_functions(n: int) -> list[dict]:
    """枚举 def 函数，生成 single-step search query。"""
    candidates = []
    for rel in ["src", "scripts"]:
        p = ROOT / rel
        for f in p.glob("*.py"):
            text = f.read_text(encoding="utf-8").splitlines()
            for i, line in enumerate(text, 1):
                m = re.match(r"^def ([a-z_][a-z0-9_]*)\(", line)
                if not m:
                    continue
                name = m.group(1)
                if name.startswith("_"):
                    continue  # 跳过私有
                candidates.append({"path": f"{rel}/{f.name}", "name": name, "lineno": i})
    random.shuffle(candidates)
    # 去重：同名函数只保留一个
    seen = set()
    unique = []
    for c in candidates:
        if c["name"] in seen:
            continue
        seen.add(c["name"])
        unique.append(c)
    out = []
    for c in unique[:n]:
        q_tpl = random.choice(FUNC_QUERIES)
        f_tpl = random.choice(FUNC_FINALS)
        user_q = q_tpl.format(name=c["name"])
        # 改用全仓库搜
        grep_out = run_grep(f"def {c['name']}\\b", "src/")
        if "(未找到" in grep_out:
            grep_out = run_grep(f"def {c['name']}\\b", "scripts/")
        if c["name"] not in grep_out:
            continue
        # 去二义性：grep 返回多行、涉及多个文件时跳过
        grep_files = {line.split(":", 1)[0] for line in grep_out.splitlines() if ":" in line}
        if len(grep_files) > 1:
            continue
        final_a = f_tpl.format(**c)
        out.append({
            "id": f"cc_func_{c['name']}",
            "messages": [
                {"role": "user", "content": user_q},
                tool_call("search_code", {"query": f"def {c['name']}"}),
                tool_result(grep_out),
                final(final_a),
            ],
        })
    return out


# ================ Template C: argparse 链式 ================

ARG_QUERIES = [
    "{path} 的 --{arg} 参数是做什么的？默认值是什么？",
    "{path} 里 --{arg} 参数怎么用？",
    "{path} 的 --{arg} 干嘛的？",
]


def synth_argparse(n: int) -> list[dict]:
    """枚举 argparse 参数，生成 search + read 两步 query。"""
    candidates = []
    for rel in ["src", "scripts"]:
        p = ROOT / rel
        for f in p.glob("*.py"):
            text = f.read_text(encoding="utf-8").splitlines()
            for i, line in enumerate(text, 1):
                m = re.search(r"add_argument\(['\"]--([a-z][a-z0-9-]+)['\"]", line)
                if not m:
                    continue
                arg = m.group(1)
                candidates.append({"path": f"{rel}/{f.name}", "arg": arg, "lineno": i, "def_line": line.strip()})
    random.shuffle(candidates)
    seen = set()
    unique = []
    for c in candidates:
        key = (c["path"], c["arg"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
    out = []
    for c in unique[:n]:
        q = random.choice(ARG_QUERIES).format(path=c["path"], arg=c["arg"])
        # Step 1: search
        grep_out = run_grep(f"--{c['arg']}", c["path"])
        if c["arg"] not in grep_out:
            continue
        # Step 2: read file around that line
        s, e = max(1, c["lineno"] - 1), c["lineno"] + 1
        read_out = run_read(c["path"], s, e)
        # Parse default + help
        default_m = re.search(r"default=([^,\)]+)", c["def_line"])
        help_m = re.search(r"help=['\"]([^'\"]+)['\"]", c["def_line"])
        default = default_m.group(1).strip() if default_m else "无"
        help_s = help_m.group(1) if help_m else "未提供说明"
        final_a = f"`{c['path']}` 的 `--{c['arg']}` 参数：{help_s}。默认值 `{default}`。定义在第 {c['lineno']} 行。"
        out.append({
            "id": f"cc_arg_{c['path'].replace('/', '_')}_{c['arg']}",
            "messages": [
                {"role": "user", "content": q},
                tool_call("search_code", {"query": f"--{c['arg']}", "path": c["path"]}),
                tool_result(grep_out),
                tool_call("read_file", {"path": c["path"], "start": s, "end": e}),
                tool_result(read_out),
                final(final_a),
            ],
        })
    return out


# ================ Template D: 函数签名（search + read） ================

SIG_QUERIES = [
    "{name} 函数的签名和 docstring 是什么？",
    "{name} 函数接受哪些参数？做什么的？",
    "帮我看 {name} 函数的定义和文档。",
]


def synth_signatures(n: int) -> list[dict]:
    """枚举 def 函数 + 后续几行（签名+docstring），生成 search + read 两步 query。"""
    candidates = []
    for rel in ["src", "scripts"]:
        p = ROOT / rel
        for f in p.glob("*.py"):
            text = f.read_text(encoding="utf-8").splitlines()
            for i, line in enumerate(text, 1):
                m = re.match(r"^def ([a-z_][a-z0-9_]*)\(", line)
                if not m:
                    continue
                name = m.group(1)
                if name.startswith("_"):
                    continue
                # 只挑后面有 docstring 或注释的（有信息量）
                if i + 1 <= len(text) and ('"""' in text[i] or "'''" in text[i] or text[i].strip().startswith("#")):
                    candidates.append({"path": f"{rel}/{f.name}", "name": name, "lineno": i})
    random.shuffle(candidates)
    seen = set()
    unique = []
    for c in candidates:
        if c["name"] in seen:
            continue
        seen.add(c["name"])
        unique.append(c)
    out = []
    for c in unique[:n]:
        q = random.choice(SIG_QUERIES).format(name=c["name"])
        grep_out = run_grep(f"def {c['name']}\\b", c["path"])
        if c["name"] not in grep_out:
            continue
        s, e = c["lineno"], min(c["lineno"] + 8, c["lineno"] + 15)
        read_out = run_read(c["path"], s, e)
        final_a = f"`{c['name']}` 定义在 `{c['path']}:{c['lineno']}`。签名和 docstring 见上方代码片段。"
        out.append({
            "id": f"cc_sig_{c['name']}",
            "messages": [
                {"role": "user", "content": q},
                tool_call("search_code", {"query": f"def {c['name']}", "path": c["path"]}),
                tool_result(grep_out),
                tool_call("read_file", {"path": c["path"], "start": s, "end": e}),
                tool_result(read_out),
                final(final_a),
            ],
        })
    return out


# ================ Template E: 常量 usage（某常量在哪些地方被引用） ================

CONST_USE_QUERIES = [
    "常量 `{name}` 在哪些地方被用到？",
    "`{name}` 都在哪里被引用？",
    "`{name}` 在仓库里被谁用了？",
    "哪些文件引用了 `{name}` 这个常量？",
]


def synth_const_usage(n: int) -> list[dict]:
    """枚举常量 → grep 其引用（排除定义文件外也出现的），生成 single-step query。"""
    cand = []
    for rel in ["src", "scripts"]:
        for f in (ROOT / rel).glob("*.py"):
            text = f.read_text(encoding="utf-8").splitlines()
            for i, line in enumerate(text, 1):
                m = re.match(r"^([A-Z_][A-Z0-9_]{3,})\s*=\s*", line)
                if not m:
                    continue
                cand.append({"path": f"{rel}/{f.name}", "name": m.group(1), "lineno": i})
    random.shuffle(cand)
    seen = set()
    out = []
    for c in cand:
        if c["name"] in seen:
            continue
        seen.add(c["name"])
        grep_out = run_grep(f"\\b{c['name']}\\b", "src/")
        grep_s = run_grep(f"\\b{c['name']}\\b", "scripts/")
        combined = "\n".join([x for x in [grep_out, grep_s] if "(未找到" not in x])
        if not combined.strip():
            continue
        lines = combined.splitlines()
        files = {l.split(":", 1)[0] for l in lines if ":" in l}
        # 至少引用 2 个位置（含定义）才有"usage"价值
        if len(lines) < 2:
            continue
        q = random.choice(CONST_USE_QUERIES).format(name=c["name"])
        file_list = ", ".join(f"`{f}`" for f in sorted(files)[:5])
        final_a = f"`{c['name']}` 被以下文件引用：{file_list}，共 {len(lines)} 处（含定义）。详见上方搜索结果。"
        out.append({
            "id": f"cc_cuse_{c['path'].replace('/', '_').replace('.py','')}_{c['name']}",
            "messages": [
                {"role": "user", "content": q},
                tool_call("search_code", {"query": c["name"]}),
                tool_result(combined[:1500]),
                final(final_a),
            ],
        })
        if len(out) >= n:
            break
    return out


# ================ Template F: 函数 usage（某函数被谁调用） ================

FUNC_USE_QUERIES = [
    "`{name}` 函数被哪里调用？",
    "谁用了 `{name}` 这个函数？",
    "`{name}` 在哪些地方被使用？",
]


def synth_func_usage(n: int) -> list[dict]:
    """枚举 def → grep 其调用点（排除 def 行）。"""
    cand = []
    for rel in ["src", "scripts"]:
        for f in (ROOT / rel).glob("*.py"):
            text = f.read_text(encoding="utf-8").splitlines()
            for i, line in enumerate(text, 1):
                m = re.match(r"^def ([a-z_][a-z0-9_]{3,})\(", line)
                if not m or m.group(1).startswith("_"):
                    continue
                cand.append({"path": f"{rel}/{f.name}", "name": m.group(1), "lineno": i})
    random.shuffle(cand)
    seen = set()
    out = []
    for c in cand:
        if c["name"] in seen:
            continue
        seen.add(c["name"])
        grep_src = run_grep(f"\\b{c['name']}\\(", "src/")
        grep_scr = run_grep(f"\\b{c['name']}\\(", "scripts/")
        combined = "\n".join([x for x in [grep_src, grep_scr] if "(未找到" not in x])
        # 过滤定义行 & 要求 ≥2 处（有调用）
        lines = [l for l in combined.splitlines() if f"def {c['name']}(" not in l]
        if len(lines) < 2:
            continue
        files = {l.split(":", 1)[0] for l in lines if ":" in l}
        q = random.choice(FUNC_USE_QUERIES).format(name=c["name"])
        file_list = ", ".join(f"`{f}`" for f in sorted(files)[:5])
        final_a = f"`{c['name']}` 被调用于：{file_list}，共 {len(lines)} 处。"
        out.append({
            "id": f"cc_fuse_{c['path'].replace('/', '_').replace('.py','')}_{c['name']}",
            "messages": [
                {"role": "user", "content": q},
                tool_call("search_code", {"query": f"{c['name']}("}),
                tool_result("\n".join(lines)[:1500]),
                final(final_a),
            ],
        })
        if len(out) >= n:
            break
    return out


# ================ Template G: from ... import ... ================

IMPORT_QUERIES = [
    "`{sym}` 是从哪个模块 import 的？",
    "`{sym}` 在哪里被 import？",
    "谁 import 了 `{sym}`？",
]


def synth_imports(n: int) -> list[dict]:
    """枚举 from X import Y[, Z]，生成 single-step query：某 symbol 从哪 import。"""
    cand = []
    for rel in ["src", "scripts"]:
        for f in (ROOT / rel).glob("*.py"):
            text = f.read_text(encoding="utf-8").splitlines()
            for i, line in enumerate(text, 1):
                m = re.match(r"^from ([a-zA-Z_][\w.]*) import (.+)$", line.strip())
                if not m:
                    continue
                mod = m.group(1)
                for sym in re.split(r",\s*", m.group(2)):
                    sym = sym.strip().split(" as ")[0].strip()
                    if sym and re.match(r"^[a-zA-Z_][\w]*$", sym):
                        cand.append({"path": f"{rel}/{f.name}", "mod": mod, "sym": sym, "lineno": i})
    random.shuffle(cand)
    # 按 sym 去重（同名 symbol 多文件只取一个）
    seen = set()
    unique = []
    for c in cand:
        if c["sym"] in seen:
            continue
        seen.add(c["sym"])
        unique.append(c)
    out = []
    for c in unique[:n]:
        q = random.choice(IMPORT_QUERIES).format(sym=c["sym"])
        grep_out = run_grep(f"from .* import .*{c['sym']}", "src/")
        grep_s = run_grep(f"from .* import .*{c['sym']}", "scripts/")
        combined = "\n".join([x for x in [grep_out, grep_s] if "(未找到" not in x])
        if c["sym"] not in combined:
            continue
        # 解析 combined 里所有出现的 from X import 的 X
        import_lines = [l for l in combined.splitlines() if ":" in l]
        mods = set()
        for il in import_lines:
            m = re.search(r"from (\S+) import", il)
            if m:
                mods.add(m.group(1))
        if len(mods) > 1:
            # 多模块重名 symbol，跳过（二义性）
            continue
        n_uses = len(import_lines)
        if n_uses == 1:
            final_a = f"`{c['sym']}` 从 `{c['mod']}` import，引入位置：`{c['path']}:{c['lineno']}`。"
        else:
            final_a = f"`{c['sym']}` 从 `{c['mod']}` 被 {n_uses} 处文件 import，详见上方搜索结果。"
        out.append({
            "id": f"cc_imp_{c['path'].replace('/', '_').replace('.py','')}_{c['sym']}",
            "messages": [
                {"role": "user", "content": q},
                tool_call("search_code", {"query": f"import {c['sym']}"}),
                tool_result(combined[:1500]),
                final(final_a),
            ],
        })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/cc_synth_v2.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-const", type=int, default=15)
    parser.add_argument("--n-func", type=int, default=15)
    parser.add_argument("--n-arg", type=int, default=10)
    parser.add_argument("--n-sig", type=int, default=10)
    parser.add_argument("--n-cuse", type=int, default=40)
    parser.add_argument("--n-fuse", type=int, default=40)
    parser.add_argument("--n-imp", type=int, default=60)
    args = parser.parse_args()

    random.seed(args.seed)

    groups = {
        "constant":  synth_constants(args.n_const),
        "function":  synth_functions(args.n_func),
        "argparse":  synth_argparse(args.n_arg),
        "signature": synth_signatures(args.n_sig),
        "const_use": synth_const_usage(args.n_cuse),
        "func_use":  synth_func_usage(args.n_fuse),
        "imports":   synth_imports(args.n_imp),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(out_path, "w") as f:
        for kind, items in groups.items():
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
                total += 1
            print(f"  {kind:10s}: {len(items)} 条")
    print(f"\n总 {total} 条 → {out_path}")


if __name__ == "__main__":
    main()
