"""Patch d36_cpt_v2.pt -- 重新打破 VE 对称性 (0.3x + 噪声) 后用分层 LR CPT。

d36_cpt_v2.pt 是 step 1200 的最佳点 (val_bpb=0.571)，但 7/9 VE 对 cos>0.97。
重新 patching 后接新分层 LR CPT（NEW_VE_LR_SCALE=20x）。
"""
from __future__ import annotations
import torch
from pathlib import Path

SRC = Path("/home/langshen/base_mode/attnres/checkpoints/d36_cpt_v2.pt")
DST = Path("/home/langshen/base_mode/attnres/checkpoints/d36_cpt_v2_patched.pt")

NEW_LAYERS = [19, 21, 23, 25, 27, 29, 31, 33, 35]   # grown 新层
OLD_LAYERS = [1, 3, 5, 7, 9, 11, 13, 15, 17]         # 对应的旧层
SCALE = 0.3                                            # 幅度缩小系数
NOISE_REL = 0.20                                       # 相对 std 的噪声强度

ckpt = torch.load(SRC, map_location="cpu", weights_only=False)
sd = ckpt["model_state"]

prefix_candidates = ["_orig_mod.value_embeds.", "value_embeds."]
def find_key(layer_i):
    for pfx in prefix_candidates:
        k = f"{pfx}{layer_i}.weight"
        if k in sd:
            return k
    raise KeyError(f"no value_embeds for layer {layer_i}")

# 先看当前的对称性
def cos(a, b):
    a, b = a.flatten(), b.flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-8)).item()

print("--- 当前对称性（patch 前）---")
for old_i, new_i in zip(OLD_LAYERS, NEW_LAYERS):
    c = cos(sd[find_key(old_i)], sd[find_key(new_i)])
    print(f"  VE_{old_i} vs VE_{new_i}: cosine = {c:.4f}")

# 在新层 VE 上施加 patch
print("\n--- 施加 patch (×0.3 + noise) ---")
g = torch.Generator().manual_seed(20260424)  # 新种子，避免和第一次 patch 相同
for i in NEW_LAYERS:
    k = find_key(i)
    w = sd[k]
    std = w.std().item()
    noise = torch.randn(w.shape, generator=g, dtype=w.dtype) * std * NOISE_REL
    new_w = (w * SCALE) + noise
    old_norm = w.norm().item()
    new_norm = new_w.norm().item()
    sd[k] = new_w.contiguous()
    print(f"  layer {i:2d}: ‖w‖ {old_norm:.1f} -> {new_norm:.1f} (×{new_norm/old_norm:.2f})")

# patch 后重新检查对称性
print("\n--- patch 后对称性 ---")
for old_i, new_i in zip(OLD_LAYERS, NEW_LAYERS):
    c = cos(sd[find_key(old_i)], sd[find_key(new_i)])
    print(f"  VE_{old_i} vs VE_{new_i}: cosine = {c:.4f}")

# 重置 optimizer state（确保新 CPT 从干净开始）
ckpt.pop("optimizer_state", None)
ckpt["val_bpb"] = ckpt.get("val_bpb", float("inf"))
ckpt["step"] = 0

torch.save(ckpt, DST)
print(f"\nSaved {DST} ({DST.stat().st_size / 1e9:.2f} GB)")
