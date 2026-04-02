# SFT 使用指南

## 快速开始

```bash
# 自动推断输出名
python sft.py continued_d18_32k_final.pt

# 指定输出名
python sft.py continued_d18_32k_final.pt --out my_sft.pt

# 指定数据文件
python sft.py continued_d18_32k_final.pt --data ~/Desktop/custom.jsonl
```

## 选项

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `checkpoint` | 输入模型路径 | `continued_d18_32k_final.pt` |
| `--out`, `-o` | 输出模型路径 | 自动从输入推断 |
| `--data` | SFT 数据 JSONL 路径 | `~/Desktop/sft_mixed_v7.jsonl` |

## 输出

启动时打印配置：
```
SFT config:
  Input:  continued_d18_32k_final.pt
  Output: sft_continued_d18_32k_final.pt
  Data:   /home/langshen/Desktop/sft_mixed_v7.jsonl
```

训练过程中每 `EVAL_INTERVAL` (500) 步评估一次，保存最佳 val_bpt 的 checkpoint。

## 数据格式

JSONL，每行一个样本：
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

特殊 token：
- `<|reserved_0|>` = BOS / padding
- `<|reserved_1|>` = USER turn start
- `<|reserved_2|>` = ASSISTANT turn start
- `<|reserved_3|>` = EOS

Loss 仅在 assistant token 上计算。