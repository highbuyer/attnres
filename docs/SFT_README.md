# SFT 使用指南

## 快速开始

```bash
# 从当前主线 best 开始 SFT
uv run python src/sft.py checkpoints/sft_mixed_v8_checkpoint_v2_best.pt --data sft_toolheavy_v1.jsonl --out checkpoints/sft_toolheavy_v1.pt

# 指定输出名
uv run python src/sft.py checkpoints/sft_mixed_v8_checkpoint_v2_best.pt --data sft_mixed_v8.jsonl --out checkpoints/my_sft.pt

# 恢复继续训练
env UV_CACHE_DIR=/tmp/uv-cache /home/langshen/.local/bin/uv run python src/sft.py checkpoints/sft_mixed_v8_checkpoint_v2.pt --data sft_mixed_v8.jsonl --out checkpoints/sft_mixed_v8_checkpoint_v2.pt --resume
```

## 选项

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `checkpoint` | 输入模型路径 | repo-aware 解析，默认 `checkpoints/continued_d18_32k_final.pt` |
| `--out`, `-o` | 输出模型路径 | 自动从输入推断 |
| `--data` | SFT 数据 JSONL 路径 | `ATTNRES_SFT_DATA` 或 repo 中的 `sft_mixed_v8.jsonl` / `sft_mixed_v7.jsonl` |
| `--resume` | 从已有 SFT checkpoint 恢复训练 | 关闭 |
| `--lr` | 覆盖学习率 | `1.5e-5` |
| `--total-steps` | 覆盖总步数 | `15000` |
| `--warmup-steps` | 覆盖 warmup 步数 | `100` |
| `--warmdown-start` | 覆盖降温起点 | `0.8 * total_steps` |
| `--eval-interval` | 覆盖验证间隔 | `500` |

## 输出

启动时打印配置：
```
SFT config:
  Input:  checkpoints/sft_mixed_v8_checkpoint_v2_best.pt
  Output: checkpoints/sft_toolheavy_v1.pt
  Data:   sft_toolheavy_v1.jsonl
  LR:     1e-05
  Steps:  total=2000 warmup=100 warmdown=1600 eval=100
```

训练过程中每 `EVAL_INTERVAL` 步评估一次，并同时更新：

- 固定输出名，例如 `checkpoints/sft_toolheavy_v1.pt`
- `_best.pt` 别名，例如 `checkpoints/sft_toolheavy_v1_best.pt`
- `_best.json` 元信息，例如 `checkpoints/sft_toolheavy_v1_best.json`

## 当前主线说明

- 当前主线 best：`checkpoints/sft_mixed_v8_checkpoint_v2_best.pt`
- 当前实验 best：`checkpoints/sft_toolheavy_v1_best.pt`
- 当前瓶颈不是 loss，而是工具行为没有立住；只看 `val_bpt` 不足以判断 tool SFT 是否成功
- 当前已定位一个训练实现风险：若直接优化 `bf16` 参数，小更新可能被量化吞掉。SFT 训练应优先采用 `fp32` 参数 + `bf16 autocast`。

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
- `<|tool_call_start|>` / `<|tool_call_end|>` = 工具调用边界
- `<|tool_result_start|>` / `<|tool_result_end|>` = 工具结果边界
- `<|tool_name_search_code|>` / `<|tool_name_read_file|>` = 工具名

Loss 仅在 assistant token 上计算。

工具链样本会保留完整 assistant block 上下文，不再把：

- `tool_call`
- `tool_result`
- `final answer`

错误拆成互不带上下文的 sibling 样本。
