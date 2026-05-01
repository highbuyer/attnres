#!/bin/bash
# 训练启动脚本：在 AutoDL A800 上跑 v2 SFT 全量
# 用法（远端）:
#   cd /root/autodl-tmp/attnres
#   nohup bash scripts/run_sft_cloud.sh > /root/autodl-tmp/sft_log.txt 2>&1 &
#   tail -f /root/autodl-tmp/sft_log.txt

set -e
cd /root/autodl-tmp/attnres

PYTHON=/root/miniconda3/bin/python

# 显存碎片化优化（A800 80GB 满载时关键）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 训练超参（针对 A800 80GB + 277M v2 模型）
BASE_CKPT=checkpoints/d36_v2_mla_best_slim.pt
DATA_MIXED=datasets/sft_archive/sft_mixed_v8_50k.jsonl
DATA_CLAUDE=datasets/sft_claude_trajectories.jsonl
OUT_CKPT=checkpoints/d36_v2_sft_full.pt

# A800 80GB 显存大，bsz 提到 4，grad_accum=16 → effective 64
# claude 池 ~12.8k slices，3 epoch 约 38k samples = 600 steps × 64
# mixed 50k slices，0.5 epoch ≈ 25k samples = 390 steps × 64
# 加起来效果约 1000 steps，但用 4000 steps 留余量
BSZ=4
GRAD_ACCUM=16
STEPS=2000
LR=2e-5
MIX_CLAUDE=0.7
MAX_SAMPLES=50000  # 总池上限 (claude 12.8k 全 + mixed ~37k)

echo "=== Cloud SFT run ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "config: bsz=${BSZ} grad_accum=${GRAD_ACCUM} effective=$((BSZ*GRAD_ACCUM)) steps=${STEPS} lr=${LR} mix_claude=${MIX_CLAUDE}"

$PYTHON -u src/sft_v2.py \
    --ckpt "$BASE_CKPT" \
    --data "$DATA_MIXED" \
    --data-claude "$DATA_CLAUDE" \
    --mix-ratio-claude "$MIX_CLAUDE" \
    --max-samples "$MAX_SAMPLES" \
    --steps "$STEPS" \
    --bsz "$BSZ" \
    --grad-accum "$GRAD_ACCUM" \
    --lr "$LR" \
    --log-every 25 \
    --out "$OUT_CKPT"

echo "=== Done. Checkpoint at $OUT_CKPT ==="
ls -lh "$OUT_CKPT"
