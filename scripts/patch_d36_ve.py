"""Patch d36_cpt.pt 新层 VE 破对称。

d36 grow 时每个 VE_i 被 clone 到 VE_{i+18}，100 步 CPT 不够分化。
策略：
  1. 缩小新层 ve 幅度 ×0.3，让新层贡献初始变弱
  2. 加 5% std 随机噪声打破数值对称，允许 gradient 分化到不同方向

产物：checkpoints/d36_cpt_patched.pt
"""
from __future__ import annotations
import torch
from pathlib import Path

SRC = Path("/home/langshen/base_mode/attnres/checkpoints/d36_cpt.pt")
DST = Path("/home/langshen/base_mode/attnres/checkpoints/d36_cpt_patched.pt")

NEW_LAYERS = [19, 21, 23, 25, 27, 29, 31, 33, 35]   # grown 新层
SCALE = 0.3                                         # 幅度缩小系数
NOISE_REL = 0.05                                    # 相对 std 的噪声强度

ckpt = torch.load(SRC, map_location="cpu", weights_only=False)
sd = ckpt["model_state"]

prefix_candidates = ["_orig_mod.value_embeds.", "value_embeds."]
def find_key(layer_i):
    for pfx in prefix_candidates:
        k = f"{pfx}{layer_i}.weight"
        if k in sd:
            return k
    raise KeyError(f"no value_embeds for layer {layer_i}")

g = torch.Generator().manual_seed(20260422)
for i in NEW_LAYERS:
    k = find_key(i)
    w = sd[k]
    std = w.std().item()
    noise = torch.randn(w.shape, generator=g, dtype=w.dtype) * std * NOISE_REL
    new_w = (w * SCALE) + noise
    old_norm = w.norm().item()
    new_norm = new_w.norm().item()
    sd[k] = new_w.contiguous()
    print(f"layer {i:2d}: ‖w‖ {old_norm:.1f} -> {new_norm:.1f} (×{new_norm/old_norm:.2f})  std={std:.4f}")

# 检查对称对是否打破：对每对 (i, i+18)，比较 cosine
def cos(a, b):
    a, b = a.flatten(), b.flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-8)).item()

print("\n--- 对称检查 (新旧 VE 相似度) ---")
for old_i, new_i in zip([1, 3, 5, 7, 9, 11, 13, 15, 17], NEW_LAYERS):
    ok = find_key(old_i)
    nk = find_key(new_i)
    c = cos(sd[ok], sd[nk])
    print(f"VE_{old_i} vs VE_{new_i}: cosine = {c:.4f}  (越接近 1 越对称，缩幅+噪声后应 <0.99)")

ckpt["val_bpb"] = ckpt.get("val_bpb", float("inf"))
ckpt["step"] = 0  # 重新开始 CPT
torch.save(ckpt, DST)
print(f"\nSaved {DST} ({DST.stat().st_size / 1e9:.2f} GB)")
