#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tool_protocol import execute_tool  # noqa: E402


SEARCH_TASKS = [
    ("项目里哪里定义了 `--tool-dir` 参数？", "search_code", {"query": "--tool-dir"}),
    ("项目里哪里定义了 `--no-tools` 参数？", "search_code", {"query": "--no-tools"}),
    ("项目里哪里提到了 `rep-penalty`？", "search_code", {"query": "rep-penalty"}),
    ("项目里哪里提到了 `rope-theta`？", "search_code", {"query": "rope-theta"}),
    ("项目里工具调用轮数上限是怎么配的？", "search_code", {"query": "max_tool_rounds"}),
    ("当前 SFT 默认数据路径是在哪里定义的？", "search_code", {"query": "def resolve_sft_data_path"}),
    ("当前 `SYSTEM_PROMPT` 是在哪里定义的？", "search_code", {"query": "SYSTEM_PROMPT"}),
    ("项目里 special token 是在哪里编码的？", "search_code", {"query": "encode_single_token"}),
    ("项目里哪里定义了 `tool-call-upsample` 参数？", "search_code", {"query": "--tool-call-upsample"}),
    ("项目里哪里定义了 `negative-identity-upsample` 参数？", "search_code", {"query": "--negative-identity-upsample"}),
    ("评估脚本默认评估哪个 checkpoint？", "search_code", {"query": 'parser.add_argument("--checkpoint", default="sft_checkpoint.pt"'}),
    ("项目里 special tool token 是在哪里配置的？", "search_code", {"query": "SPECIAL_TOKENS = ["}),
    ("`SPECIAL_TOKENS` 是在哪里定义的？", "search_code", {"query": "SPECIAL_TOKENS ="}),
    ("当前仓库里 `check_identity` 是在哪里实现的？", "search_code", {"query": "def check_identity"}),
    ("当前仓库里 `apply_hard_rules` 是在哪里实现的？", "search_code", {"query": "def apply_hard_rules"}),
    ("当前仓库里 `parse_tool_call` 是在哪里实现的？", "search_code", {"query": "def parse_tool_call"}),
    ("当前仓库里 `validate_tool_sample` 是在哪里实现的？", "search_code", {"query": "def validate_tool_sample"}),
    ("项目里哪里提到了“工具调用未触发”？", "search_code", {"query": "未触发"}),
]

READ_TASKS = [
    ("把 `docs/RUN_NEXT.md` 里“构建新的 SFT 数据集”那段读给我。", "read_file", {"path": "docs/RUN_NEXT.md", "offset": 1, "limit": 16}),
    ("读取 `docs/INFER_README.md` 开头的用法说明。", "read_file", {"path": "docs/INFER_README.md", "offset": 1, "limit": 18}),
    ("给我看 `src/infer.py` 参数定义那一段。", "read_file", {"path": "src/infer.py", "offset": 70, "limit": 18}),
    ("给我看 `src/infer.py` 里工具循环附近的代码。", "read_file", {"path": "src/infer.py", "offset": 140, "limit": 28}),
    ("给我看 `src/sft.py` 里 special token IDs 那一段。", "read_file", {"path": "src/sft.py", "offset": 100, "limit": 8}),
    ("把 `src/prepare.py` 里 `SPECIAL_TOKENS` 那段读一下。", "read_file", {"path": "src/prepare.py", "offset": 80, "limit": 18}),
    ("读取 `scripts/make_sft_data.py` 里工具样本加载部分。", "read_file", {"path": "scripts/make_sft_data.py", "offset": 180, "limit": 24}),
    ("读取 `scripts/eval_bench.py` 里 BENCH 开头那一段。", "read_file", {"path": "scripts/eval_bench.py", "offset": 20, "limit": 24}),
    ("把 `src/inference_rules.py` 开头读一下。", "read_file", {"path": "src/inference_rules.py", "offset": 1, "limit": 24}),
    ("把 `src/tool_protocol.py` 里 `validate_tool_sample` 附近的代码给我看一下。", "read_file", {"path": "src/tool_protocol.py", "offset": 198, "limit": 18}),
]


def make_summary(tool_name: str, params: dict, result: str) -> str:
    first_line = result.splitlines()[0] if result else "(空结果)"
    if tool_name == "search_code":
        return f"根据搜索结果，我找到了相关位置，首条匹配是：`{first_line}`。"
    path = params["path"]
    offset = params["offset"]
    return f"这是 `{path}` 从第 {offset} 行开始的内容，首行是：`{first_line}`。"


def build_sample(prompt: str, tool_name: str, params: dict, repo_root: Path) -> dict:
    result = execute_tool(tool_name, params, repo_root)
    payload = json.dumps(params, ensure_ascii=False, separators=(",", ":"))
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"<|tool_call_start|><|tool_name_{tool_name}|>{payload}<|tool_call_end|>"},
            {"role": "assistant", "content": f"<|tool_result_start|>{result}<|tool_result_end|>"},
            {"role": "assistant", "content": make_summary(tool_name, params, result)},
        ]
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于当前仓库实况构建工具调用样本")
    parser.add_argument("--repo-root", default=str(ROOT), help="仓库根目录")
    parser.add_argument("--out", default="docs/tool_call_samples_repo.jsonl", help="输出 JSONL 路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser()
    out_path = Path(args.out).expanduser()

    samples = [
        build_sample(prompt, tool_name, params, repo_root)
        for prompt, tool_name, params in SEARCH_TASKS + READ_TASKS
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"已生成 {len(samples)} 条工具样本 -> {out_path}")


if __name__ == "__main__":
    main()
