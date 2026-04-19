#!/bin/bash
# 短跑测显存占用（持续查，取峰值）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run python src/sft.py \
  checkpoints/sft_tool_summary_v5_best.pt \
  --data data/sft_v5_clean_cc_v21.jsonl \
  --out /tmp/mem_test.pt \
  --total-steps 200 \
  --eval-interval 9999 \
  --batch-size 1 --grad-accum 32 \
  --grad-ckpt \
  > /tmp/mem_test.log 2>&1 &

PID=$!
PEAK=0
for i in $(seq 1 24); do
    sleep 5
    MEM=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "$MEM" ] && [ "$MEM" -gt "$PEAK" ]; then
        PEAK=$MEM
    fi
    STEP=$(grep -oE "step [0-9]+/" /tmp/mem_test.log 2>/dev/null | tail -1)
    echo "  ${i}×5s: mem=${MEM}MB peak=${PEAK}MB $STEP"
done
echo "--- peak: ${PEAK} MB ---"
pkill -f "src/sft.py" 2>/dev/null
wait $PID 2>/dev/null

