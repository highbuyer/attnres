# 推理使用指南

## 快速开始

```bash
# 基本用法
uv run python src/infer.py checkpoints/sft_d18_v3_checkpoint.pt

# 带提示词
uv run python src/infer.py "你是谁？" 100

# 指定参数
uv run python src/infer.py "解释什么是Python" 200 0.3 --top-k 20 --rep-penalty 1.4
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `prompt` | 输入提示词（可选） | 交互模式 |
| `max_tokens` | 最大生成 token 数 | 256 |
| `temperature` | 采样温度 | 0.2 |
| `--checkpoint` | 模型路径 | 自动找 best_checkpoint.pt |
| `--top-k` | Top-k 采样截断 | 40 |
| `--top-p` | Nucleus 采样 | 0.9 |
| `--seed` | 随机种子 | 1234 |
| `--bf16` | 使用 bf16 精度 | False |
| `--fp32` | 使用 fp32 精度 | False |
| `--rep-penalty` | 重复惩罚 (1.0=关闭) | 1.3 |
| `--rope-theta` | RoPE base 频率 (NTK scaling) | None |

## 交互模式

不带提示词时进入交互式对话：
```
> 你好
你好！我是微研，一个技术助手。有什么可以帮助你的吗？

> Python和Java有什么区别？
...
```

按 Ctrl+C 退出。

## NTK Scaling

使用 `--rope-theta` 调整 RoPE 频率，支持更长上下文推理：
```bash
uv run python src/infer.py checkpoints/sft_d18_v3_checkpoint.pt --rope-theta 500000
```