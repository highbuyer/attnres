# Next Commands

当前主线：`checkpoints/sft_tool_summary_v5_best.pt`（step 1500, val_bpt=1.2297，E2E 全通）。

## 评测当前主线

```bash
# 工具格式评测
uv run python scripts/eval_tool_format.py \
  --checkpoint checkpoints/sft_tool_summary_v5_best.pt \
  --out eval_tool_format_summary_v5.json

# 全量 bench（对照 v2 基线）
uv run python scripts/eval_bench.py \
  --checkpoint checkpoints/sft_tool_summary_v5_best.pt \
  --compare eval_results_sft_tool_gated_v2_best.json

# Runtime E2E 测试
uv run python src/infer.py '当前仓库里 parse_tool_call 是在哪里实现的？' 256 0.0 \
  --checkpoint checkpoints/sft_tool_summary_v5_best.pt \
  --tool-dir /home/langshen/base_mode/attnres

# weiyan CLI（直接用主线）
./weiyan '这个仓库做什么？'

# weiyan-api（开本地 HTTP 服务）
./weiyan-api --port 8000 --audit-log runs/weiyan_audit.jsonl
```

## 下一步：thinking-trace + 混合 SFT（数据已备好）

参考 `docs/MIXED_SFT_RECIPE.md`。数据：`sft_tool_summary_v5_think.jsonl`（7126 条）。

```bash
# 1. 构建混合数据集
uv run python scripts/build_mixed_sft.py \
  --out sft_mixed_think_v1.jsonl \
  --source sft_tool_summary_v5.jsonl:weight=3:think=1 \
  --source datasets/sft_archive/sft_tool_gated_v2.jsonl:weight=1:think=1:limit=3000 \
  --source sft_samples_project_facts.jsonl:weight=20:think=0

# 2. 从 v5_best 短训（lr 低一档，防打穿已学到的 summary 能力）
env UV_CACHE_DIR=/tmp/uv-cache /home/langshen/.local/bin/uv run python src/sft.py \
  checkpoints/sft_tool_summary_v5_best.pt \
  --data sft_mixed_think_v1.jsonl \
  --out checkpoints/sft_think_v1.pt \
  --lr 3e-6 --total-steps 1500 --warmup-steps 80 --warmdown-start 1200 \
  --eval-interval 100 --batch-size 2 --grad-accum 16

# 3. 评测（便宜 → 贵）
uv run python scripts/eval_tool_format.py \
  --checkpoint checkpoints/sft_think_v1_best.pt \
  --out eval_tool_format_think_v1.json
# 目标：≥ 7/8，不应比 summary v5 退步
```

## 架构债修复（见 docs/ARCH_ROADMAP.md）

### P0 立刻：推理层扩上下文到 8192（不重训）

现状 bug：`run.log` 抓到 `prompt_len=67996 context_tokens=29915 … tokens=0 text=''`——
prompt 被 `x[:, -config.sequence_len:]` 截到 2048 后模型直接吐空。

动作（src/infer.py + weiyan-api）：
1. 把硬截断从 `config.sequence_len` 改成 `INFER_MAX_CONTEXT=8192`
2. NTK-aware RoPE 重算：`new_base = rope_theta * (scale ** (d/(d-2)))`，`scale=4`
3. 重算 `model.window_sizes`：long_window=8192, short_window=4096
4. weiyan-api 加 prompt 截断护栏 + 日志 warn

### P1 下次 SFT 前：tie_lm_head=True

省 25M，短训 200-500 步校准。

### P2 下一代预训：VE 砍半 + GQA

详见路线图；代价大，先把 P0 + P1 坐实。

## 潜在改进方向（不紧急）

1. 工具样本 prompt 多样性（当前 56 条 unique）
2. search_code vs read_file 选择准确度的数据增强
3. summary 风格多样化（当前偏模板化）
4. weiyan-api 的 KV cache 实现（配合 GQA 才真正受益）
