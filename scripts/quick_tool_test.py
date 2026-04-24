#!/usr/bin/env python3
"""快速测试：检查模型是否能生成工具调用标记"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
import json


def quick_test():
    # 加载检查点
    ckpt_path = ROOT / "checkpoints" / "sft_v5_clean_cc.pt"
    print(f"加载检查点: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # 检查配置
    config = ckpt["config"]
    print(f"\n模型配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # 检查特殊token
    from prepare import SPECIAL_TOKENS

    print(f"\n特殊token: {SPECIAL_TOKENS}")

    # 检查模型参数
    model_params = ckpt.get("model", {})
    print(f"\n模型参数量: {sum(p.numel() for p in model_params.values()):,}")

    # 检查是否有工具调用相关的权重
    tool_keywords = ["tool", "search", "read"]
    tool_layers = []
    for key in model_params.keys():
        if any(kw in key.lower() for kw in tool_keywords):
            tool_layers.append(key)

    print(f"\n工具相关层 ({len(tool_layers)}):")
    for layer in tool_layers[:10]:  # 只显示前10个
        shape = model_params[layer].shape
        print(f"  {layer}: {shape}")

    # 简单的生成测试
    print(f"\n{'=' * 60}")
    print("简单生成测试")
    print(f"{'=' * 60}")

    # 加载分词器
    from prepare import Tokenizer

    tokenizer = Tokenizer()

    # 测试查询
    test_queries = [
        "把 src/tool_protocol.py 里 validate_tool_sample 附近的代码给我看一下。",
        "模型的 generate 函数是怎么写的？",
        "SFT 评估函数怎么写的？",
        "你好，今天天气怎么样？",
    ]

    for query in test_queries:
        print(f"\n查询: {query}")

        # 编码
        tokens = tokenizer.encode(query)
        print(f"  Token数: {len(tokens)}")

        # 检查是否包含特殊token
        special_token_ids = {}
        for token in SPECIAL_TOKENS:
            try:
                token_id = tokenizer.encode_single_token(token)
                special_token_ids[token] = token_id
                print(f"  {token}: ID={token_id}")
            except:
                print(f"  {token}: 未找到")

        # 简单分析：查询是否看起来需要工具调用
        needs_tool = any(
            keyword in query.lower()
            for keyword in ["src/", "文件", "代码", "函数", "怎么写的", "看一下"]
        )
        print(f"  可能需要工具调用: {needs_tool}")


if __name__ == "__main__":
    quick_test()
