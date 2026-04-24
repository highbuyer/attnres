#!/usr/bin/env python3
import json
import os
from typing import List, Dict, Any


def analyze_training_data(data_path: str):
    """分析训练数据质量"""
    print(f"分析训练数据: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    print(f"总样本数: {len(lines)}")

    # 统计指标
    stats = {
        "total_samples": len(lines),
        "with_tool_calls": 0,
        "multi_turn": 0,
        "avg_messages": 0,
        "tool_types": {"read_file": 0, "search_code": 0, "other": 0},
        "message_lengths": {"user": [], "assistant": [], "tool": []},
    }

    for item in lines:
        messages = item["messages"]
        stats["avg_messages"] += len(messages)

        # 检查工具调用
        has_tool = False
        for msg in messages:
            if msg["role"] == "assistant" and "tool_call_start" in msg["content"]:
                has_tool = True
                stats["with_tool_calls"] += 1

                # 统计工具类型
                if "tool_name_read_file" in msg["content"]:
                    stats["tool_types"]["read_file"] += 1
                elif "tool_name_search_code" in msg["content"]:
                    stats["tool_types"]["search_code"] += 1
                else:
                    stats["tool_types"]["other"] += 1

            # 统计消息长度
            if msg["role"] in stats["message_lengths"]:
                stats["message_lengths"][msg["role"]].append(len(msg["content"]))

        # 检查多轮对话
        if len(messages) >= 4:
            stats["multi_turn"] += 1

    stats["avg_messages"] /= len(lines)

    # 计算百分比
    stats["tool_call_percentage"] = (
        stats["with_tool_calls"] / stats["total_samples"] * 100
    )
    stats["multi_turn_percentage"] = stats["multi_turn"] / stats["total_samples"] * 100

    # 计算平均长度
    for role in stats["message_lengths"]:
        if stats["message_lengths"][role]:
            avg_len = sum(stats["message_lengths"][role]) / len(
                stats["message_lengths"][role]
            )
            stats["message_lengths"][role] = avg_len
        else:
            stats["message_lengths"][role] = 0

    return stats


def print_stats(stats: Dict[str, Any]):
    """打印统计信息"""
    print(f"\n=== 训练数据分析报告 ===")
    print(f"总样本数: {stats['total_samples']}")
    print(f"平均消息数: {stats['avg_messages']:.1f}")
    print(
        f"包含工具调用的样本: {stats['with_tool_calls']} ({stats['tool_call_percentage']:.1f}%)"
    )
    print(
        f"多轮对话样本: {stats['multi_turn']} ({stats['multi_turn_percentage']:.1f}%)"
    )

    print(f"\n工具类型分布:")
    total_tools = sum(stats["tool_types"].values())
    for tool, count in stats["tool_types"].items():
        percentage = count / total_tools * 100 if total_tools > 0 else 0
        print(f"  {tool}: {count} ({percentage:.1f}%)")

    print(f"\n平均消息长度:")
    for role, avg_len in stats["message_lengths"].items():
        print(f"  {role}: {avg_len:.0f} 字符")

    print(f"\n=== 数据质量评估 ===")

    # 质量评估标准
    quality_indicators = {
        "工具调用比例": ("高", stats["tool_call_percentage"] > 20),
        "多轮对话比例": ("中", stats["multi_turn_percentage"] > 15),
        "数据规模": ("充足", stats["total_samples"] > 1000),
        "工具多样性": ("需改进", stats["tool_types"]["other"] > 0),
    }

    for indicator, (target, meets) in quality_indicators.items():
        status = "✓" if meets else "⚠"
        print(f"  {status} {indicator}: {target}")


def check_data_consistency(data_path: str):
    """检查数据一致性"""
    print(f"\n=== 数据一致性检查 ===")

    with open(data_path, "r", encoding="utf-8") as f:
        lines = list(f)

    errors = []

    for i, line in enumerate(lines):
        try:
            item = json.loads(line)

            # 检查必需字段
            if "messages" not in item:
                errors.append(f"行 {i + 1}: 缺少'messages'字段")
                continue

            messages = item["messages"]
            if not isinstance(messages, list) or len(messages) == 0:
                errors.append(f"行 {i + 1}: 'messages'不是列表或为空")
                continue

            # 检查消息格式
            for j, msg in enumerate(messages):
                if "role" not in msg or "content" not in msg:
                    errors.append(f"行 {i + 1} 消息 {j + 1}: 缺少'role'或'content'字段")

                if msg["role"] not in ["user", "assistant", "tool", "system"]:
                    errors.append(f"行 {i + 1} 消息 {j + 1}: 未知角色 '{msg['role']}'")

        except json.JSONDecodeError as e:
            errors.append(f"行 {i + 1}: JSON解析错误 - {e}")

    if errors:
        print(f"发现 {len(errors)} 个错误:")
        for error in errors[:5]:  # 只显示前5个错误
            print(f"  {error}")
        if len(errors) > 5:
            print(f"  ... 还有 {len(errors) - 5} 个错误")
    else:
        print("✓ 数据格式一致，无错误")


def main():
    data_path = "/home/langshen/base_mode/attnres/data/sft_final_training_v1.jsonl"

    if not os.path.exists(data_path):
        print(f"错误: 文件不存在 {data_path}")
        return

    # 分析数据
    stats = analyze_training_data(data_path)
    print_stats(stats)

    # 检查一致性
    check_data_consistency(data_path)

    # 建议
    print(f"\n=== 训练建议 ===")

    if stats["tool_call_percentage"] < 15:
        print("⚠ 工具调用比例较低，建议:")
        print("  - 增加更多工具调用样本")
        print("  - 确保gold数据中有足够的工具调用")

    if stats["total_samples"] < 5000:
        print("⚠ 数据规模可能不足，建议:")
        print("  - 继续收集更多gold数据")
        print("  - 考虑数据增强")
    else:
        print("✓ 数据规模充足")

    if stats["tool_types"]["other"] == 0:
        print("✓ 工具类型清晰（只有read_file和search_code）")
    else:
        print("⚠ 发现其他工具类型，需要确认是否支持")


if __name__ == "__main__":
    main()
