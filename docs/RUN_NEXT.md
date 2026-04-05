# Next Commands

当前主线是 `sft_tool_summary_v2_best.pt`（E2E 全通）。

## 评测当前主线

```bash
# 工具格式评测
uv run python scripts/eval_tool_format.py --checkpoint checkpoints/sft_tool_summary_v2_best.pt --out eval_tool_format_summary_v2.json

# 全量 bench
uv run python scripts/eval_bench.py --checkpoint checkpoints/sft_tool_summary_v2_best.pt --compare eval_results_sft_tool_gated_v2_best.json

# Runtime E2E 测试
uv run python src/infer.py '当前仓库里 parse_tool_call 是在哪里实现的？' 256 0.0 \
  --checkpoint checkpoints/sft_tool_summary_v2_best.pt \
  --tool-dir /home/langshen/base_mode/attnres
```

## 如果需要继续改进

### 改善工具样本多样性

当前只有 28 条 unique prompt，可以：

```bash
# 重新生成 repo 样本
uv run python scripts/build_tool_call_data.py --out docs/tool_call_samples_repo.jsonl

# 构建新数据集
uv run python scripts/make_sft_data.py \
  --out sft_tool_summary_v2.jsonl \
  --max-belle 500 --max-multiturn 0 --max-school-math 0 --max-claude 0 \
  --tool-call-upsample 300 --upsample 5 --negative-identity-upsample 3
```

### 训练（注意 fp32 OOM 限制）

```bash
env UV_CACHE_DIR=/tmp/uv-cache /home/langshen/.local/bin/uv run python src/sft.py \
  checkpoints/sft_tool_summary_v2_best.pt \
  --data sft_tool_summary_v2.jsonl \
  --out checkpoints/sft_tool_summary_v3.pt \
  --lr 5e-6 --total-steps 1000 --warmup-steps 50 --warmdown-start 800 \
  --eval-interval 100 --batch-size 2 --grad-accum 16
```

## 潜在改进方向

1. 增加工具样本 prompt 多样性（当前 28 条太少）
2. 改善 search_code vs read_file 选择准确度
3. 工具参数 JSON 质量（部分 prompt 的 offset/limit 不够精准）
4. summary 风格多样化（当前偏模板化）
