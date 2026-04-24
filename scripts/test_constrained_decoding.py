#!/usr/bin/env python3
"""Constrained decoding 快速验证：测试模型在强制工具调用下的表现。

设计：
1. 对于需要工具调用的查询，强制模型生成 <|tool_call_start|>
2. 验证模型能否填充合理的工具参数
3. 评估模型是否已学到工具调用模式
"""

import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
from train import GPT, GPTConfig
from prepare import Tokenizer, SPECIAL_TOKENS


def load_model_and_tokenizer(checkpoint_path: Path):
    """加载模型和分词器"""
    print(f"加载模型: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # 创建配置
    if isinstance(ckpt["config"], dict):
        config = GPTConfig(**ckpt["config"])
    else:
        config = ckpt["config"]

    # 创建模型
    model = GPT(config)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 加载分词器
    tokenizer = Tokenizer()

    return model, tokenizer, config


def get_special_token_ids(tokenizer, special_tokens: List[str]) -> Dict[str, int]:
    """获取特殊token的ID"""
    token_ids = {}
    for token in special_tokens:
        try:
            token_ids[token] = tokenizer.encode_single_token(token)
        except:
            token_ids[token] = -1
    return token_ids


def constrained_generate(
    model,
    tokenizer,
    prompt: str,
    device: str = "cpu",
    max_tokens: int = 100,
    temperature: float = 0.0,
    force_tool_call: bool = False,
    tool_type: str = "read_file",  # 'read_file' 或 'search_code'
) -> Tuple[str, List[int], bool]:
    """约束解码生成

    Args:
        force_tool_call: 是否强制生成工具调用
        tool_type: 强制生成的工具类型

    Returns:
        (生成的文本, token IDs列表, 是否成功生成工具调用)
    """
    # 获取特殊token IDs
    token_ids = get_special_token_ids(tokenizer, SPECIAL_TOKENS)

    # 构建输入
    bos_id = token_ids.get("<|reserved_0|>", 1)
    user_id = token_ids.get("<|reserved_1|>", 1)
    asst_id = token_ids.get("<|reserved_2|>", 1)

    prompt_ids = [bos_id, user_id] + tokenizer.encode(prompt) + [asst_id]
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    generated_ids = []
    is_tool_call_generated = False

    # 工具调用相关token
    tool_call_start_id = token_ids.get("<|tool_call_start|>", -1)
    tool_call_end_id = token_ids.get("<|tool_call_end|>", -1)
    tool_name_read_file_id = token_ids.get("<|tool_name_read_file|>", -1)
    tool_name_search_code_id = token_ids.get("<|tool_name_search_code|>", -1)

    for step in range(max_tokens):
        with torch.no_grad():
            # 前向传播
            if step == 0:
                logits = model(x)[:, -1, :]
            else:
                logits = model(next_token)[:, -1, :]

        # 约束解码逻辑
        if force_tool_call and step == 0:
            # 第一步：强制生成 <|tool_call_start|>
            next_id = torch.tensor([[tool_call_start_id]], device=device)
            is_tool_call_generated = True
            print(f"  步骤 {step}: 强制生成 <|tool_call_start|>")

        elif force_tool_call and step == 1:
            # 第二步：强制生成工具名称
            if tool_type == "read_file":
                next_id = torch.tensor([[tool_name_read_file_id]], device=device)
                print(f"  步骤 {step}: 强制生成 <|tool_name_read_file|>")
            else:
                next_id = torch.tensor([[tool_name_search_code_id]], device=device)
                print(f"  步骤 {step}: 强制生成 <|tool_name_search_code|>")

        elif force_tool_call and step == 2:
            # 第三步：让模型生成参数（JSON格式）
            # 这里我们只约束前两步，参数让模型自由生成
            logits = logits / temperature if temperature > 0 else logits
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            print(f"  步骤 {step}: 自由生成参数")

        else:
            # 正常生成
            if temperature <= 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

        token_id = int(next_id.item())
        generated_ids.append(token_id)

        # 检查是否结束
        if token_id == token_ids.get("<|reserved_3|>", 1):  # EOS
            break

        if tool_call_end_id != -1 and token_id == tool_call_end_id:
            print(f"  步骤 {step}: 生成 <|tool_call_end|>，工具调用完成")
            break

        # 准备下一步
        next_token = next_id

    # 解码结果
    generated_text = tokenizer.decode(generated_ids)

    # 检查生成的文本是否包含有效的工具调用
    has_valid_tool_call = False
    if (
        "<|tool_call_start|>" in generated_text
        and "<|tool_call_end|>" in generated_text
    ):
        # 简单检查参数格式
        start_idx = generated_text.find("<|tool_call_start|>")
        end_idx = generated_text.find("<|tool_call_end|>")
        params_text = generated_text[start_idx + len("<|tool_call_start|>") : end_idx]

        # 检查是否包含工具名称
        if (
            "<|tool_name_read_file|>" in params_text
            or "<|tool_name_search_code|>" in params_text
        ):
            # 尝试解析JSON
            try:
                # 提取JSON部分
                json_start = params_text.find("{")
                json_end = params_text.rfind("}")
                if json_start != -1 and json_end != -1:
                    json_str = params_text[json_start : json_end + 1]
                    params = json.loads(json_str)
                    has_valid_tool_call = True
                    print(f"  成功解析工具参数: {params}")
            except json.JSONDecodeError:
                print(f"  JSON解析失败: {params_text}")

    return generated_text, generated_ids, has_valid_tool_call


def test_cc_benchmark(model, tokenizer, benchmark_path: Path, device: str = "cpu"):
    """测试cc benchmark样本"""
    print(f"\n加载benchmark: {benchmark_path}")

    with open(benchmark_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    print(f"找到 {len(samples)} 个样本")

    results = []

    for i, sample in enumerate(samples[:10]):  # 只测试前10个
        messages = sample.get("messages", [])
        if not messages:
            continue

        # 提取用户查询（最后一个user消息）
        user_queries = [msg["content"] for msg in messages if msg["role"] == "user"]
        if not user_queries:
            continue

        query = user_queries[-1]
        print(f"\n样本 {i + 1}: {query[:80]}...")

        # 测试1: 自由生成
        print("  测试1: 自由生成")
        free_text, free_ids, free_has_tool = constrained_generate(
            model, tokenizer, query, device, force_tool_call=False, temperature=0.0
        )
        print(f"    结果: {free_text[:100]}...")
        print(f"    包含工具调用: {free_has_tool}")

        # 测试2: 强制工具调用
        print("  测试2: 强制工具调用 (read_file)")
        forced_text, forced_ids, forced_has_tool = constrained_generate(
            model,
            tokenizer,
            query,
            device,
            force_tool_call=True,
            tool_type="read_file",
            temperature=0.0,
        )
        print(f"    结果: {forced_text[:100]}...")
        print(f"    有效工具调用: {forced_has_tool}")

        results.append(
            {
                "query": query,
                "free_generation": free_text,
                "free_has_tool": free_has_tool,
                "forced_generation": forced_text,
                "forced_has_tool": forced_has_tool,
                "expected_tool_call": any(
                    "tool" in str(msg.get("content", "")).lower() for msg in messages
                ),
            }
        )

    # 分析结果
    print(f"\n{'=' * 60}")
    print("结果分析:")
    print(f"{'=' * 60}")

    total = len(results)
    free_tool_count = sum(1 for r in results if r["free_has_tool"])
    forced_valid_count = sum(1 for r in results if r["forced_has_tool"])
    expected_tool_count = sum(1 for r in results if r["expected_tool_call"])

    print(f"总样本数: {total}")
    print(
        f"自由生成包含工具调用: {free_tool_count}/{total} ({free_tool_count / total * 100:.1f}%)"
    )
    print(
        f"强制生成有效工具调用: {forced_valid_count}/{total} ({forced_valid_count / total * 100:.1f}%)"
    )
    print(f"预期需要工具调用: {expected_tool_count}/{total}")

    # 详细输出
    print(f"\n详细结果:")
    for i, r in enumerate(results):
        print(f"\n{i + 1}. 查询: {r['query'][:60]}...")
        print(f"   自由生成: {r['free_generation'][:60]}...")
        print(f"   强制生成: {r['forced_generation'][:60]}...")
        print(f"   预期工具: {r['expected_tool_call']}")


def main():
    parser = argparse.ArgumentParser(description="Constrained decoding测试")
    parser.add_argument("--checkpoint", type=Path, required=True, help="模型检查点路径")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "data" / "cc_gold_trajectories_v1.jsonl",
        help="benchmark数据路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="运行设备",
    )
    parser.add_argument("--max-samples", type=int, default=10, help="最大测试样本数")

    args = parser.parse_args()

    # 加载模型
    model, tokenizer, config = load_model_and_tokenizer(args.checkpoint)
    model.to(args.device)

    print(f"模型配置: {config}")
    print(f"运行设备: {args.device}")

    # 测试benchmark
    test_cc_benchmark(model, tokenizer, args.benchmark, args.device)


if __name__ == "__main__":
    main()
