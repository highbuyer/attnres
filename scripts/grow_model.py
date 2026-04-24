#!/usr/bin/env python3
"""Grow 400M d18 → 800M d36（layer stacking，复制每层 2 次）。

attnres 架构 per-layer params 清单（见 scripts/enum_ckpt_keys.py）：
  transformer.h.{i}.attn.{c_q,c_k,c_v,c_proj}.weight  × 18
  transformer.h.{i}.attn.ve_gate.weight               × 9 (odd i only)
  transformer.h.{i}.mlp.{c_fc,c_proj}.weight          × 18
  attnres_norm.{i}.weight                             × 36 (= 2*n_layer)
  attnres_proj.{i}.weight                             × 36 (= 2*n_layer)
  value_embeds.{i}.weight                             × 9  (odd i only)
  wte.weight / lm_head.weight                         × 1 (global, 保留)

stacking 映射：new_k = old_k for k∈[0, OLD_N)，new_k = old_{k-OLD_N} for
k∈[OLD_N, 2*OLD_N)。对 attnres_{norm,proj} 同理用 2*OLD_N 偏移。

VE 位置自然对齐：n_layer 翻倍后 has_ve 仍挑 odd idx，而 old_idx = new_idx
% OLD_N 的奇偶相同（因为 OLD_N=18 偶数）。
"""
import sys
import re
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# 取 GPTConfig 但不触发 train.py main
train_src = (ROOT / "src" / "train.py").read_text()
cut = train_src.index("# Setup: tokenizer, model, optimizer, dataloader")
ns: dict = {"__name__": "train_safe", "__file__": str(ROOT / "src" / "train.py")}
exec(train_src[:cut], ns)
GPTConfig = ns["GPTConfig"]

import __main__
__main__.GPTConfig = GPTConfig
GPTConfig.__module__ = "__main__"  # 对齐 sft.py:100 / infer.py:74，pickle 走 __main__ 查 class


def main() -> None:
    src_path = ROOT / "checkpoints" / "continued_d18_32k_final.pt"
    out_path = ROOT / "checkpoints" / "grown_d36_from_d18.pt"

    print(f"Loading {src_path.name}...")
    ckpt = torch.load(src_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state"]
    cfg = ckpt["config"]
    OLD_N = cfg.n_layer
    NEW_N = 2 * OLD_N
    print(f"  old n_layer={OLD_N}  new n_layer={NEW_N}")

    new_sd: dict = {}
    counters = {"layer": 0, "proj_norm": 0, "ve": 0, "global": 0}

    for k, v in sd.items():
        # transformer.h.{i}.*
        m = re.match(r"^(.*transformer\.h\.)(\d+)(\..*)$", k)
        if m:
            pre, i, suf = m.group(1), int(m.group(2)), m.group(3)
            new_sd[f"{pre}{i}{suf}"] = v.clone()
            new_sd[f"{pre}{i + OLD_N}{suf}"] = v.clone()
            counters["layer"] += 2
            continue
        # attnres_norm/proj.{i} (has 2*n_layer entries each)
        m = re.match(r"^(.*attnres_(?:norm|proj)\.)(\d+)(\.weight)$", k)
        if m:
            pre, i, suf = m.group(1), int(m.group(2)), m.group(3)
            new_sd[f"{pre}{i}{suf}"] = v.clone()
            new_sd[f"{pre}{i + 2 * OLD_N}{suf}"] = v.clone()
            counters["proj_norm"] += 2
            continue
        # value_embeds.{i}
        m = re.match(r"^(.*value_embeds\.)(\d+)(\.weight)$", k)
        if m:
            pre, i, suf = m.group(1), int(m.group(2)), m.group(3)
            new_sd[f"{pre}{i}{suf}"] = v.clone()
            new_sd[f"{pre}{i + OLD_N}{suf}"] = v.clone()
            counters["ve"] += 2
            continue
        # global: wte, lm_head
        new_sd[k] = v.clone()
        counters["global"] += 1

    print(f"  layer block entries : {counters['layer']}")
    print(f"  attnres norm+proj   : {counters['proj_norm']}")
    print(f"  value_embeds        : {counters['ve']}")
    print(f"  global              : {counters['global']}")
    print(f"  total new_sd keys   : {len(new_sd)}")

    from dataclasses import replace
    new_cfg = replace(cfg, n_layer=NEW_N)
    new_ckpt = {
        "model_state": new_sd,
        "config": new_cfg,
        "val_bpb": float("inf"),  # 新模型还没 eval
        "step": 0,
        "continued_pretrain": True,
    }
    print(f"Saving to {out_path.name}...")
    torch.save(new_ckpt, out_path)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Done. size={size_mb:.0f} MB")


if __name__ == "__main__":
    main()
