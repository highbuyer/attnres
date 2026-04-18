from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


TOOL_NAMES = ("search_code", "read_file")
TOOL_CALL_START = "<|tool_call_start|>"
TOOL_CALL_END = "<|tool_call_end|>"
TOOL_RESULT_START = "<|tool_result_start|>"
TOOL_RESULT_END = "<|tool_result_end|>"
TOOL_MARKUP = (
    TOOL_CALL_START,
    TOOL_CALL_END,
    TOOL_RESULT_START,
    TOOL_RESULT_END,
    "<|tool_name_search_code|>",
    "<|tool_name_read_file|>",
)
SEARCH_CODE_PRIORITY_PREFIXES = (
    "src/",
    "scripts/",
    "tests/",
    "docs/",
)
SEARCH_CODE_DEPRIORITIZED_PATHS = (
    "scripts/build_tool_call_data.py",
    "scripts/eval_tool_format.py",
    "scripts/inspect_tool_start_logits.py",
    "scripts/tool_call_samples",
    "docs/tool_call_samples",
    "datasets/tool_call_samples",
)


def extract_tool_result(messages: list[dict]) -> str | None:
    # 逆序查找，确保在多轮对话中获取最新的工具结果
    for message in reversed(messages):
        content = message.get("content", "")
        if TOOL_RESULT_START in content and TOOL_RESULT_END in content:
            start_idx = content.find(TOOL_RESULT_START) + len(TOOL_RESULT_START)
            end_idx = content.find(TOOL_RESULT_END, start_idx)
            if end_idx != -1:
                return content[start_idx:end_idx]
    return None


def _parse_tool_call_inner(inner: str) -> tuple[str, dict] | None:
    tool_name = None
    for candidate in TOOL_NAMES:
        tag = f"<|tool_name_{candidate}|>"
        if tag in inner:
            tool_name = candidate
            inner = inner.replace(tag, "", 1).strip()
            break

    if tool_name is None:
        return None

    try:
        params = json.loads(inner)
    except json.JSONDecodeError:
        params = {}
    return tool_name, params


def parse_tool_call(text: str) -> tuple[str, dict] | None:
    parsed_call = None
    search_from = 0
    while True:
        tc_start = text.find(TOOL_CALL_START, search_from)
        if tc_start == -1:
            break

        tc_end = text.find(TOOL_CALL_END, tc_start + len(TOOL_CALL_START))
        if tc_end == -1:
            # 容错：模型输出了 tool_call_start 但没有 tool_call_end，
            # 取到文本末尾尝试解析
            inner = text[tc_start + len(TOOL_CALL_START):]
            parsed = _parse_tool_call_inner(inner)
            if parsed is not None:
                parsed_call = parsed
            break

        inner = text[tc_start + len(TOOL_CALL_START):tc_end]
        parsed = _parse_tool_call_inner(inner)
        if parsed is not None:
            parsed_call = parsed
        search_from = tc_end + len(TOOL_CALL_END)

    return parsed_call


def _normalize_search_result_path(line: str) -> tuple[str, int]:
    path_part, _, remainder = line.partition(":")
    normalized = path_part.removeprefix("./")
    line_no, _, _ = remainder.partition(":")
    try:
        parsed_line_no = int(line_no)
    except ValueError:
        parsed_line_no = 0
    return normalized, parsed_line_no


def _normalize_search_result_line(line: str) -> str:
    path_part, sep, remainder = line.partition(":")
    if not sep:
        return line
    return f"{path_part.removeprefix('./')}:{remainder}"


def _search_code_sort_key(line: str) -> tuple[int, int, str]:
    path, line_no = _normalize_search_result_path(line)
    if any(path.startswith(prefix) for prefix in SEARCH_CODE_DEPRIORITIZED_PATHS):
        priority = len(SEARCH_CODE_PRIORITY_PREFIXES)
    else:
        priority = len(SEARCH_CODE_PRIORITY_PREFIXES) + 1
        for idx, prefix in enumerate(SEARCH_CODE_PRIORITY_PREFIXES):
            if path.startswith(prefix):
                priority = idx
                break
        if priority == len(SEARCH_CODE_PRIORITY_PREFIXES) + 1 and (path.endswith(".py") or path.endswith(".toml")):
            priority = len(SEARCH_CODE_PRIORITY_PREFIXES) - 1
    return priority, line_no, path


def _search_code_results(query: str, work_dir: str | Path) -> list[str]:
    result = subprocess.run(
        [
            "rg",
            "--no-heading",
            "-n",
            "--max-count",
            "200",
            "--sort",
            "path",
            "--fixed-strings",
            "--glob",
            "*.py",
            "--glob",
            "*.md",
            "--glob",
            "*.toml",
            "--glob",
            "*.txt",
            "--",
            query,
            ".",
        ],
        capture_output=True,
        cwd=str(work_dir),
        text=True,
        timeout=5,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    lines.sort(key=_search_code_sort_key)
    return [_normalize_search_result_line(line) for line in lines[:10]]


def execute_tool(tool_name: str, params: dict, work_dir: str | Path) -> str:
    work_dir = str(work_dir)
    try:
        if tool_name == "search_code":
            query = params.get("query", "")
            if not query:
                return "(空查询)"
            output = "\n".join(_search_code_results(query, work_dir))
            return output if output else f"(未找到匹配: {query})"

        if tool_name == "read_file":
            rel_path = params.get("path", "")
            # 容错：模型传了 search_code 的参数给 read_file，降级为 search_code
            if not rel_path and "query" in params:
                return "\n".join(_search_code_results(params["query"], work_dir)) or f"(未找到匹配: {params['query']})"
            try:
                offset = max(1, int(params.get("offset") or 1))
            except (TypeError, ValueError):
                return f"(无效 offset: {params.get('offset')!r})"
            try:
                limit = max(1, int(params.get("limit") or 20))
            except (TypeError, ValueError):
                return f"(无效 limit: {params.get('limit')!r})"
            limit = min(limit, 2000)  # 防御性上限
            root_path = Path(work_dir).resolve()
            full_path = (root_path / rel_path).resolve()
            try:
                in_root = full_path == root_path or full_path.is_relative_to(root_path)
            except AttributeError:  # Python < 3.9 兜底
                in_root = str(full_path) == str(root_path) or str(full_path).startswith(str(root_path) + os.sep)
            if not in_root:
                return f"(路径越界: {rel_path})"
            if not full_path.exists():
                return f"(文件不存在: {rel_path})"
            if full_path.is_dir():
                entries = sorted(full_path.iterdir())[:30]
                return "\n".join(e.name + ("/" if e.is_dir() else "") for e in entries)
            lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()
            selected = lines[offset - 1:offset - 1 + limit]
            return "\n".join(f"{offset + i} {line}" for i, line in enumerate(selected))

        return f"(未知工具: {tool_name})"
    except Exception as exc:
        return f"(工具执行错误: {exc})"


def _strip_tagged_blocks(text: str, start_tag: str, end_tag: str) -> str:
    chunks: list[str] = []
    cursor = 0
    while True:
        block_start = text.find(start_tag, cursor)
        if block_start == -1:
            chunks.append(text[cursor:])
            return "".join(chunks)

        chunks.append(text[cursor:block_start])
        block_end = text.find(end_tag, block_start + len(start_tag))
        if block_end == -1:
            return "".join(chunks)
        cursor = block_end + len(end_tag)


def strip_tool_markup(text: str) -> str:
    text = _strip_tagged_blocks(text, TOOL_CALL_START, TOOL_CALL_END)
    text = _strip_tagged_blocks(text, TOOL_RESULT_START, TOOL_RESULT_END)
    for tag in TOOL_MARKUP:
        text = text.replace(tag, "")
    return text.strip()


def validate_tool_sample(sample: dict, work_dir: str | Path) -> tuple[bool, str]:
    messages = sample.get("messages", [])
    tool_message = next((m.get("content", "") for m in messages if TOOL_CALL_START in m.get("content", "")), None)
    if not tool_message:
        return True, "no_tool_call"

    parsed = parse_tool_call(tool_message)
    if parsed is None:
        return False, "bad_tool_call"

    expected = execute_tool(parsed[0], parsed[1], work_dir)
    actual = extract_tool_result(messages)
    if actual is None:
        return False, "missing_tool_result"
    if actual != expected:
        return False, "tool_result_mismatch"
    return True, "ok"
