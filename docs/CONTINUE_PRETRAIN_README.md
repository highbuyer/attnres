# 继续预训练指南

## 功能

从已有 checkpoint 继续预训练，用于：
- 词表迁移后的 embedding 收敛
- 模型规模扩展后的知识填充
- 域适应继续训练

## 快速开始

```bash
# 默认继续训练
uv run python src/continue_pretrain.py

# 指定输入输出
uv run python src/continue_pretrain.py checkpoints/tooltoken_d18_32k.pt
```

## 参数说明

通过修改代码中的常量配置：

| 常量 | 说明 | 默认值 |
|------|------|--------|
| `CHECKPOINT_IN` | 输入模型路径 | `checkpoints/tooltoken_d18_32k.pt` |
| `CHECKPOINT_OUT` | 输出模型路径 | `checkpoints/tooltoken_continued.pt` |
| `MATRIX_LR` | 矩阵参数学习率 | 0.01 |
| `EMBEDDING_LR` | Embedding 学习率 | 0.05 |
| `UNEMBEDDING_LR` | Unembedding 学习率 | 0.001 |
| `SCALAR_LR` | 标量参数学习率 | 0.025 |
| `WEIGHT_DECAY` | 权重衰减 | 0.01 |
| `TOTAL_STEPS` | 总步数 | 500 |
| `WARMUP_RATIO` | Warmup 比例 | 0.05 |
| `WARMDOWN_RATIO` | Warmdown 开始比例 | 0.6 |
| `FINAL_LR_FRAC` | 最终学习率比例 | 0.05 |
| `DEVICE_BATCH_SIZE` | 设备 batch 大小 | 8 |
| `EVAL_INTERVAL` | 评估间隔 | 500 |
| `TOTAL_BATCH_SIZE` | 总 batch 大小 | 2^19 (~524K tokens) |

## 使用场景

### 1. 词表迁移后继续训练
```bash
# 第一步：迁移 embedding
uv run python src/migrate_embeddings.py continued_d18_v2_final.pt migrated_d18_32k.pt

# 第二步：继续预训练
uv run python src/continue_pretrain.py migrated_d18_32k.pt
# 修改 CHECKPOINT_OUT = 'continued_d18_32k.pt'
```

### 2. 模型扩展后继续训练
```bash
# 扩展 depth/width
uv run python src/expand_checkpoint.py continued_d18_final.pt expanded_d18.pt --depth 18

# 继续训练
uv run python src/continue_pretrain.py expanded_d18.pt
```

## 输出

- 训练日志：`logs/continue_pretrain_*.log`
- Checkpoint：`checkpoints/` 目录
- 最佳验证 loss 保存在 checkpoint 元数据中
- 当前默认产物是 `checkpoints/tooltoken_continued.pt` / `checkpoints/tooltoken_continued_final.pt`

## 当前实现注意事项

- 继续预训练现在的重点不是再改数据配比，而是先确保训练精度路径正确。
- 对含工具 token 的 checkpoint，推荐使用 `fp32` 参数 + `bf16 autocast`，避免小更新被 `bf16` 量化吃掉。
