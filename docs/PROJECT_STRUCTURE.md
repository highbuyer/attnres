# 项目结构

```
attnres/
├── src/                 # 主代码
│   ├── train.py         # 预训练（模型定义 + 训练循环）
│   ├── prepare.py       # 数据准备 + tokenizer + dataloader + 评估
│   ├── sft.py           # SFT 微调
│   ├── infer.py         # 推理
│   └── continue_pretrain.py  # 继续预训练
├── scripts/             # 辅助脚本
│   ├── eval_bench.py    # 评估基准（51 条固定测试集）
│   ├── expand_checkpoint.py  # 模型规模扩展（深度/宽度）
│   ├── migrate_embeddings.py # 词表迁移（8K→32K）
│   ├── add_tool_tokens.py    # 追加工具调用 special tokens
│   ├── make_sft_data.py # SFT 数据构建
│   └── generate.py      # 简单文本生成
├── docs/                # 文档
│   ├── PROGRESS.md      # 项目进度与计划（最重要）
│   ├── PROJECT_STRUCTURE.md  # 本文件
│   ├── program.md       # 原始 autoresearch 实验协议
│   └── tool_call_samples_v2.jsonl  # 工具调用训练样本
├── logs/                # 训练日志
├── checkpoints/         # 模型文件
│   ├── continued_d18_32k_final.pt  # 预训练主线（val_bpb=0.572887）
│   ├── tooltoken_d18_32k.pt       # 追加工具 token 后的 checkpoint
│   └── sft_d18_v3_checkpoint.pt   # SFT 模型（旧，将被替代）
└── data -> ~/.cache/autoresearch-custom/data  # 数据目录 (symlink)
```

## 目录重组说明（2026-03-30）

项目原来是平铺结构（所有 .py 在根目录），后来重组为 `src/` + `scripts/` + `docs/`。

### 路径解析机制

所有脚本使用 `__file__` 相对路径定位依赖，**不依赖 CWD**：

- **src/ 文件**通过 `Path(__file__).resolve().parent / "train.py"` 找同目录下的 train.py
- **scripts/ 文件**通过 `Path(__file__).resolve().parent.parent / "src" / "train.py"` 回溯到 src/
- 所有需要 `from prepare import ...` 的文件会先 `sys.path.insert(0, str(_SRC_DIR))` 确保 import 可达

### 为什么这样设计

多个脚本需要动态加载 `train.py` 中的模型定义（`GPT`, `GPTConfig`），通过 `exec()` 执行 train.py 的前半部分代码。这要求能找到 train.py 文件路径，同时 train.py 自身 `from prepare import ...` 也要能解析。用 `__file__` 相对路径解决了这两个问题。

### 如果遇到 import 报错

1. **`ModuleNotFoundError: No module named 'prepare'`** → 检查是否缺少 `sys.path.insert(0, str(_SRC_DIR))`
2. **`FileNotFoundError: train.py`** → 检查 Path 是否用了 `__file__` 相对路径而非裸 `Path("train.py")`
3. **`Can't get attribute 'GPTConfig'`** → 加载 checkpoint 前需要先调 `_load_model_defs()` 注册类

## 快速命令

所有命令从**项目根目录**运行：

```bash
# 预训练
uv run python src/train.py

# 继续预训练
uv run python src/continue_pretrain.py checkpoints/tooltoken_d18_32k.pt

# SFT
uv run python src/sft.py checkpoints/tooltoken_d18_32k.pt

# 推理
uv run python src/infer.py checkpoints/sft_d18_v3_checkpoint.pt

# 评估
uv run python scripts/eval_bench.py --checkpoint checkpoints/sft_d18_v3_checkpoint.pt

# 追加工具 token + 扩展 embedding
uv run python scripts/add_tool_tokens.py --checkpoint checkpoints/xxx.pt --output checkpoints/yyy.pt

# 构建 SFT 数据
uv run python scripts/make_sft_data.py

# 模型规模扩展
uv run python scripts/expand_checkpoint.py checkpoints/src.pt checkpoints/dst.pt --depth 18
```