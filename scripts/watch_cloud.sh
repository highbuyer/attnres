#!/bin/bash
# 远端训练监控：tail log + 看 GPU + 看进度
# 用法: bash scripts/watch_cloud.sh [log-file]

HOST="root@connect.nma1.seetacloud.com"
PORT=54564
LOG_FILE="${1:-/root/autodl-tmp/sft_log.txt}"

echo "=== GPU now ==="
ssh -p $PORT $HOST "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader"
echo
echo "=== tail $LOG_FILE (last 30 lines) ==="
ssh -p $PORT $HOST "tail -30 $LOG_FILE 2>/dev/null || echo 'log file not found yet'"
echo
echo "=== running training proc ==="
ssh -p $PORT $HOST "ps aux | grep -E 'sft_v2|python' | grep -v grep | head -5"
