#!/bin/bash
# v2.1 训练：字面约束合成数据 + v5_clean 混合
# 教训：v2 训 15000 步过拟合（step 1000 best=1.176, step 15000=1.72）
# 这次 total_steps=3000 + eval_interval=200，早看早停
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup uv run python src/sft.py \
  checkpoints/sft_tool_summary_v5_best.pt \
  --data data/sft_v5_clean_cc_v21.jsonl \
  --out checkpoints/sft_v5_clean_cc_v21.pt \
  --total-steps 3000 \
  --eval-interval 200 \
  --batch-size 2 --grad-accum 16 \
  > logs/sft_v5_clean_cc_v21.log 2>&1 &

echo "PID: $!"
echo "log: logs/sft_v5_clean_cc_v21.log"
echo "total_steps=3000 eval_interval=200 (eval 15次，~100-150 min)"
