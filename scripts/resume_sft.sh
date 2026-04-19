#!/bin/bash
# 从 step 7000 的 ckpt 恢复训练，降 batch 避免 OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup uv run python src/sft.py \
  checkpoints/sft_v5_clean_cc.pt \
  --data data/sft_v5_clean_cc.jsonl \
  --out checkpoints/sft_v5_clean_cc.pt \
  --resume \
  --batch-size 2 --grad-accum 16 \
  > logs/sft_v5_clean_cc_r.log 2>&1 &

echo "PID: $!"
echo "log: logs/sft_v5_clean_cc_r.log"
