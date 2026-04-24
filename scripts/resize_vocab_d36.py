"""Resize d36_cpt.pt vocab 32768→32774 to match tokenizer.

tokenizer n_vocab=32774 (含 6 个 tool special tokens, id 32768-32773)
d36_cpt.pt 继承 d18 预训练时期的 vocab=32768
需要扩展三类 embedding: wte / lm_head / 所有 value_embeds.{奇数层}
新 6 行用现有 embedding 均值 + 小噪声 init。
"""
import sys
import torch
from pathlib import Path

SRC = Path("/home/langshen/base_mode/attnres/checkpoints/d36_cpt.pt")
DST = Path("/home/langshen/base_mode/attnres/checkpoints/d36_cpt_v32774.pt")
OLD_V, NEW_V = 32768, 32774
NEW_ROWS = NEW_V - OLD_V

ckpt = torch.load(SRC, map_location="cpu", weights_only=False)
sd = ckpt["model_state"]

keys_to_resize = [
    k for k, v in sd.items()
    if v.dim() == 2 and v.shape[0] == OLD_V
]
print(f"Resizing {len(keys_to_resize)} tensors from {OLD_V} -> {NEW_V}")

g = torch.Generator().manual_seed(42)
for k in keys_to_resize:
    w = sd[k]
    mean = w.mean(dim=0, keepdim=True)
    std = w.std(dim=0, keepdim=True).clamp(min=1e-4)
    new_rows = mean + std * 0.02 * torch.randn(NEW_ROWS, w.shape[1], generator=g, dtype=w.dtype)
    sd[k] = torch.cat([w, new_rows], dim=0).contiguous()
    print(f"  {k}: {tuple(w.shape)} -> {tuple(sd[k].shape)}")

cfg = ckpt["config"]
if isinstance(cfg, dict):
    cfg["vocab_size"] = NEW_V
else:
    cfg.vocab_size = NEW_V
ckpt["config"] = cfg
ckpt.pop("val_bpb", None)
ckpt["step"] = 0

torch.save(ckpt, DST)
print(f"Saved {DST} ({DST.stat().st_size / 1e9:.2f} GB)")
