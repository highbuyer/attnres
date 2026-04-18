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
    # --- 原有：精确名称搜索 ---
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
    ("项目里哪里提到了工具调用未触发？", "search_code", {"query": "未触发"}),
    # --- 新增：模糊/口语化 ---
    ("parse_tool_call 这个函数干嘛的？", "search_code", {"query": "def parse_tool_call"}),
    ("训练那块代码在哪？", "search_code", {"query": "def main"}),
    ("tokenizer 怎么初始化的？", "search_code", {"query": "class Tokenizer"}),
    ("模型的 forward 在哪？", "search_code", {"query": "def forward"}),
    ("loss 是怎么算的？", "search_code", {"query": "cross_entropy"}),
    ("学习率调度是怎么做的？", "search_code", {"query": "current_lr"}),
    ("怎么保存 checkpoint 的？", "search_code", {"query": "def save_checkpoint"}),
    ("best checkpoint 是怎么选的？", "search_code", {"query": "def save_best_artifacts"}),
    ("数据加载在哪？", "search_code", {"query": "def make_dataloader"}),
    ("验证集怎么划分的？", "search_code", {"query": "VAL_RATIO"}),
    # --- 新增：英文/中英混合 ---
    ("Where is GPTConfig defined?", "search_code", {"query": "class GPTConfig"}),
    ("show me the forward method", "search_code", {"query": "def forward"}),
    ("找一下 loss 计算的地方", "search_code", {"query": "F.cross_entropy"}),
    ("How is the model loaded?", "search_code", {"query": "def load_checkpoint"}),
    ("gradient clipping 在哪？", "search_code", {"query": "clip_grad_norm"}),
    ("where is evaluate_sft?", "search_code", {"query": "def evaluate_sft"}),
    ("BOS token 的 ID 是多少？", "search_code", {"query": "BOS_ID"}),
    # --- 新增：功能描述 ---
    ("项目里怎么做代码搜索的？", "search_code", {"query": "def _search_code_results"}),
    ("安全拦截的逻辑在哪？", "search_code", {"query": "SAFETY_PATTERNS"}),
    ("工具结果是怎么注入到模型上下文的？", "search_code", {"query": "tool_result_start_tag"}),
    ("推理时怎么处理重复惩罚？", "search_code", {"query": "rep_penalty"}),
    ("SFT 数据是怎么 tokenize 的？", "search_code", {"query": "def tokenize_turn"}),
    ("项目里用了哪些 special token？", "search_code", {"query": "SPECIAL_TOKENS"}),
    ("warmup 是怎么实现的？", "search_code", {"query": "warmup_steps"}),
    ("模型有多少层？", "search_code", {"query": "n_layer"}),
    ("RoPE 是在哪里实现的？", "search_code", {"query": "rotary"}),
    ("attention window 是怎么配置的？", "search_code", {"query": "window_pattern"}),
    # --- 新增：否定/边界情况 ---
    ("项目里有没有用到 flash attention？", "search_code", {"query": "flash_attn"}),
    ("代码里有没有 TODO？", "search_code", {"query": "TODO"}),
    ("有没有用到 wandb？", "search_code", {"query": "wandb"}),
    # --- 新增：项目元信息搜索 ---
    ("本项目的对话记忆在哪？", "search_code", {"query": "memory"}),
    ("项目的配置文件在哪？", "search_code", {"query": "CLAUDE.md"}),
    ("checkpoint 存在哪个目录？", "search_code", {"query": "checkpoints/"}),
    ("项目用了什么 tokenizer？", "search_code", {"query": "tiktoken"}),
    ("模型的 vocab size 是多少？", "search_code", {"query": "vocab_size"}),
    ("训练数据从哪加载的？", "search_code", {"query": "data_dir"}),
    ("eval 结果保存在哪？", "search_code", {"query": "eval_results"}),
    ("项目的 git 忽略了什么？", "search_code", {"query": ".gitignore"}),
    ("哪里定义了模型的超参数？", "search_code", {"query": "class GPTConfig"}),
    ("batch size 是多少？", "search_code", {"query": "DEVICE_BATCH_SIZE"}),
    ("最大序列长度是多少？", "search_code", {"query": "MAX_SEQ_LEN"}),
    ("learning rate 默认是多少？", "search_code", {"query": "lr"}),
]

READ_TASKS = [
    # --- 原有 ---
    ("把 docs/RUN_NEXT.md 里构建 SFT 数据集那段读给我。", "read_file", {"path": "docs/RUN_NEXT.md", "offset": 1, "limit": 16}),
    ("读取 `docs/INFER_README.md` 开头的用法说明。", "read_file", {"path": "docs/INFER_README.md", "offset": 1, "limit": 18}),
    ("给我看 `src/infer.py` 参数定义那一段。", "read_file", {"path": "src/infer.py", "offset": 82, "limit": 18}),
    ("给我看 `src/infer.py` 里工具循环附近的代码。", "read_file", {"path": "src/infer.py", "offset": 196, "limit": 28}),
    ("给我看 `src/sft.py` 里 special token IDs 那一段。", "read_file", {"path": "src/sft.py", "offset": 168, "limit": 10}),
    ("把 `src/prepare.py` 里 `SPECIAL_TOKENS` 那段读一下。", "read_file", {"path": "src/prepare.py", "offset": 81, "limit": 14}),
    ("读取 `scripts/make_sft_data.py` 里工具样本加载部分。", "read_file", {"path": "scripts/make_sft_data.py", "offset": 200, "limit": 24}),
    ("读取 `scripts/eval_bench.py` 里 BENCH 开头那一段。", "read_file", {"path": "scripts/eval_bench.py", "offset": 20, "limit": 24}),
    ("把 `src/inference_rules.py` 开头读一下。", "read_file", {"path": "src/inference_rules.py", "offset": 1, "limit": 24}),
    ("把 `src/tool_protocol.py` 里 `validate_tool_sample` 附近的代码给我看一下。", "read_file", {"path": "src/tool_protocol.py", "offset": 207, "limit": 18}),
    # --- 新增：口语化 ---
    ("sft.py 的 main 函数长什么样？", "read_file", {"path": "src/sft.py", "offset": 324, "limit": 30}),
    ("打开 pyproject.toml 看看依赖", "read_file", {"path": "pyproject.toml", "offset": 1, "limit": 30}),
    ("看一下 .gitignore", "read_file", {"path": ".gitignore", "offset": 1, "limit": 30}),
    ("train.py 开头是什么？", "read_file", {"path": "src/train.py", "offset": 1, "limit": 20}),
    # --- 新增：英文 ---
    ("show me the GPT class definition", "read_file", {"path": "src/train.py", "offset": 104, "limit": 30}),
    ("read the beginning of sft_format.py", "read_file", {"path": "src/sft_format.py", "offset": 1, "limit": 20}),
    ("let me see the execute_tool function", "read_file", {"path": "src/tool_protocol.py", "offset": 160, "limit": 28}),
    # --- 新增：功能描述 ---
    ("看看 continue_pretrain 的主函数", "read_file", {"path": "src/continue_pretrain.py", "offset": 1, "limit": 24}),
    ("给我看安全拦截那段正则", "read_file", {"path": "src/inference_rules.py", "offset": 14, "limit": 12}),
    ("身份识别的正则列表长什么样？", "read_file", {"path": "src/inference_rules.py", "offset": 6, "limit": 8}),
    ("SFT 评估函数怎么写的？", "read_file", {"path": "src/sft.py", "offset": 292, "limit": 20}),
    ("看一下 PROGRESS.md 开头的状态", "read_file", {"path": "docs/PROGRESS.md", "offset": 1, "limit": 20}),
    ("模型的 generate 函数是怎么写的？", "read_file", {"path": "src/infer.py", "offset": 196, "limit": 40}),
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
    parser.add_argument("--out", default="datasets/tool_call_samples/tool_call_samples_repo.jsonl", help="输出 JSONL 路径")
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
