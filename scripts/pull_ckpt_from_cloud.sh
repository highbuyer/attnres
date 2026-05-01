#!/bin/bash
# 训练结束后从 AutoDL 拉回 SFT ckpt，并 slim 化。
# 用法：bash scripts/pull_ckpt_from_cloud.sh

set -e
HOST="root@connect.nma1.seetacloud.com"
PORT=54564
REMOTE="/root/autodl-tmp/attnres/checkpoints/d36_v2_sft_full.pt"
LOCAL_DIR="checkpoints"
SLIM_REMOTE="/root/autodl-tmp/attnres/checkpoints/d36_v2_sft_full_slim.pt"
SLIM_LOCAL="$LOCAL_DIR/d36_v2_sft_full_slim.pt"

cd "$(dirname "$0")/.."
mkdir -p "$LOCAL_DIR"

echo "=== 检查云端 ckpt ==="
ssh -p $PORT $HOST "ls -lh $REMOTE 2>/dev/null || echo 'NOT FOUND'"

echo
echo "=== 远端 slim 化（去 optimizer_state，bf16 模型权重）==="
ssh -p $PORT $HOST "/root/miniconda3/bin/python -u -c '
import torch
ck = torch.load(\"$REMOTE\", map_location=\"cpu\", weights_only=False)
print(\"orig keys:\", list(ck.keys()))
print(\"orig size: 2.3GB-ish\")
sd = ck[\"model_state\"]
sd_bf16 = {k: v.to(torch.bfloat16) if v.dtype.is_floating_point else v for k, v in sd.items()}
out = {
    \"model_state\": sd_bf16,
    \"config\": ck[\"config\"],
    \"sft_v2\": True,
    \"base_ckpt\": ck.get(\"base_ckpt\"),
    \"data\": ck.get(\"data\"),
    \"steps\": ck.get(\"steps\"),
    \"lr\": ck.get(\"lr\"),
    \"last_loss\": ck.get(\"last_loss\"),
}
torch.save(out, \"$SLIM_REMOTE\")
import os
print(\"slim size:\", os.path.getsize(\"$SLIM_REMOTE\")/1024/1024, \"MB\")
'"

echo
echo "=== rsync 拉回（slim 版本，~530MB）==="
time rsync -av --partial --info=progress2 -e "ssh -p $PORT" \
    "$HOST:$SLIM_REMOTE" "$SLIM_LOCAL"

echo
echo "=== 完成 ==="
ls -lh "$SLIM_LOCAL"
echo
echo "下一步：bash scripts/eval_sft_full.sh"
