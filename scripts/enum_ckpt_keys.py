#!/usr/bin/env python3
"""枚举 ckpt state_dict 里 per-layer key 的 pattern，为 grow_model.py 做前置。"""
import sys, re
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# 借 sft.py 的 exec 前半部分技巧，拿到 GPTConfig 但不触发 train.py 主流程
train_src = (ROOT / "src" / "train.py").read_text()
cut = train_src.index("# Setup: tokenizer, model, optimizer, dataloader")
ns = {"__name__": "train_safe", "__file__": str(ROOT / "src" / "train.py")}
exec(train_src[:cut], ns)

import __main__
__main__.GPTConfig = ns["GPTConfig"]

ckpt = torch.load(ROOT / "checkpoints" / "continued_d18_32k_final.pt",
                   map_location="cpu", weights_only=False, mmap=True)
print(f"ckpt top keys: {list(ckpt.keys())}")
sd = ckpt.get("model") or ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt.get("model_state")
if sd is None:
    print("No state_dict found; dumping top-level shapes:")
    for k, v in ckpt.items():
        if isinstance(v, dict):
            print(f"  {k}: dict with {len(v)} keys, sample={list(v.keys())[:3]}")
        else:
            print(f"  {k}: {type(v).__name__}")
    sys.exit(1)
cfg = ckpt["config"]
print(f"config: n_layer={cfg.n_layer} n_embd={cfg.n_embd} n_head={cfg.n_head} vocab={cfg.vocab_size}")
print(f"total keys: {len(sd)}")

from collections import Counter
pat = Counter()
shape_by_pat: dict[str, list] = {}
for k, v in sd.items():
    p = re.sub(r"\.\d+\.", ".{i}.", k)
    p = re.sub(r"\.\d+$", ".{i}", p)
    pat[p] += 1
    shape_by_pat.setdefault(p, []).append((k, tuple(v.shape)))

for p in sorted(pat):
    cnt = pat[p]
    sample_k, sample_shape = shape_by_pat[p][0]
    print(f"  {cnt:3d}x  {p:50s}  shape={sample_shape}  e.g.={sample_k}")
