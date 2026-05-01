#!/bin/bash
# 把本地 src/scripts 同步到远端 AutoDL（不传 ckpt 等大文件）
# 用法: bash scripts/sync_to_cloud.sh

set -e
HOST="root@connect.nma1.seetacloud.com"
PORT=54564
REMOTE_DIR="/root/autodl-tmp/attnres"

cd "$(dirname "$0")/.."

echo "=== sync src/ ==="
rsync -av --info=progress2 -e "ssh -p $PORT" \
    --include='*.py' --exclude='*' \
    src/ "$HOST:$REMOTE_DIR/src/"

echo "=== sync scripts/ ==="
rsync -av --info=progress2 -e "ssh -p $PORT" \
    --include='*.sh' --include='*.py' --exclude='*' \
    scripts/ "$HOST:$REMOTE_DIR/scripts/"

echo "=== done ==="
