# 项目结构

```
attnres/
├── src/
│   ├── train.py              # 主预训练脚本 + 模型定义
│   ├── continue_pretrain.py  # 继续预训练
│   ├── prepare.py            # 数据准备 / tokenizer / dataloader / val_bpb
│   ├── infer.py              # 推理入口 + 硬规则 + 工具 runtime
│   ├── infer_support.py      # 推理后端检查
│   ├── inference_rules.py    # 身份 / 安全硬规则
│   ├── tool_protocol.py      # tool_call 解析与执行
│   ├── project_paths.py      # repo-aware checkpoint / data 路径解析
│   ├── sft.py                # SFT 训练入口
│   ├── sft_format.py         # SFT 样本切分逻辑
│   └── attention_window.py   # SDPA 滑动窗口掩码
├── scripts/
│   ├── eval_bench.py         # 51 条固定评测集
│   ├── eval_tool_format.py   # 工具格式评测
│   ├── inspect_tool_start_logits.py  # `<|tool_call_start|>` 首 token 排名检查
│   ├── make_sft_data.py      # 混合 SFT 数据构建
│   ├── build_tool_call_data.py  # 基于当前仓库生成 repo 工具样本
│   ├── audit_tool_data.py    # 工具样本验真
│   ├── audit_sft_tool_mix.py # SFT 数据中工具分布审计
│   ├── add_tool_tokens.py    # tokenizer 追加工具 token
│   ├── expand_checkpoint.py  # 模型扩展
│   └── migrate_embeddings.py # 词表迁移
├── tests/
│   ├── test_tool_protocol.py
│   ├── test_sft_format.py
│   ├── test_sft_resume.py
│   ├── test_project_paths.py
│   ├── test_inference_rules.py
│   ├── test_infer_support.py
│   └── test_attention_window.py
├── docs/
│   ├── PROGRESS.md
│   ├── README.md
│   ├── SFT_README.md
│   ├── INFER_README.md
│   ├── CONTINUE_PRETRAIN_README.md
│   ├── PREPARE_README.md
│   ├── RUN_NEXT.md
│   ├── PROJECT_STRUCTURE.md
│   ├── program.md
│   └── tool_call_samples_repo.jsonl
├── checkpoints/
│   ├── sft_mixed_v8_checkpoint_v2_best.pt
│   ├── sft_toolheavy_v1_best.pt
│   ├── tooltoken_continued_final.pt
│   └── continued_d18_32k_final.pt
└── data -> ~/.cache/autoresearch-custom/data
```

## 路径和加载原则

### 1. 所有命令默认从仓库根目录执行

统一用法：

```bash
uv run python src/infer.py --checkpoint checkpoints/sft_mixed_v8_checkpoint_v2_best.pt
uv run python src/sft.py checkpoints/sft_mixed_v8_checkpoint_v2_best.pt --data sft_toolheavy_v1.jsonl
uv run python scripts/eval_bench.py --checkpoint checkpoints/sft_mixed_v8_checkpoint_v2_best.pt
```

### 2. 不依赖 CWD 的路径解析

- `src/` 文件内部通过 `__file__` 找 `train.py`
- `scripts/` 文件回溯到仓库根，再定位 `src/`
- checkpoint 和 SFT 数据默认路径通过 `project_paths.py` 统一解析

### 3. 当前主线 best / 实验 best

- 主线 best：`checkpoints/sft_mixed_v8_checkpoint_v2_best.pt`
- tool-heavy 实验 best：`checkpoints/sft_toolheavy_v1_best.pt`

## 常用命令

```bash
# 预训练
uv run python src/train.py

# 继续预训练
uv run python src/continue_pretrain.py checkpoints/tooltoken_d18_32k.pt

# 推理
uv run python src/infer.py --checkpoint checkpoints/sft_mixed_v8_checkpoint_v2_best.pt

# 主线评测
uv run python scripts/eval_bench.py --checkpoint checkpoints/sft_mixed_v8_checkpoint_v2_best.pt
uv run python scripts/eval_tool_format.py --checkpoint checkpoints/sft_mixed_v8_checkpoint_v2_best.pt --out eval_tool_format_main.json

# 构建 repo 工具样本
uv run python scripts/build_tool_call_data.py --out docs/tool_call_samples_repo.jsonl

# 构建 tool-heavy 数据
uv run python scripts/make_sft_data.py --out sft_toolheavy_v1.jsonl --max-belle 0 --max-multiturn 0 --max-school-math 0 --max-claude 6000 --upsample 50 --negative-identity-upsample 80 --tool-call-upsample 1200
```
