#!/usr/bin/env python3
"""物理剪枝 v5_best 的深 3 层 VE（layers 13, 15, 17）。

依据 docs/VE_ABLATION_v2_full.md：
  - 全量 val (n=700) 上 Δval_bpt = 0.0719（基线 1.6551，占 4.3%）
  - self_audit e2e net_user_failure 2/64，与 v9/v10 baseline 等价
  - halluc_fact_mismatch 不因砍 VE 增加

做法（不训练）：
  1. load v5_best state_dict
  2. 删除 {13, 15, 17} 三层的 value_embeds.{i}.weight 和
     transformer.h.{i}.attn.ve_gate.weight（共 6 个张量）
  3. 在 ckpt['config'] 里记录 ve_layer_skip=[13, 15, 17]
     （has_ve 改造支持这个字段，见 src/train.py:123）
  4. 保存为新 ckpt

验证（下一步跑）：用新 ckpt 重新 build model，strict=True load 应当成功；
val_bpt 在全量 val 上应当与 knock_weakest3 的 1.7271 完全吻合（权重等价）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

VE_KEYS_TEMPLATE = [
    "value_embeds.{i}.weight",
    "transformer.h.{i}.attn.ve_gate.weight",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-ckpt", default="checkpoints/sft_tool_summary_v5_best.pt")
    parser.add_argument("--out-ckpt", default="checkpoints/sft_v5_pruned_ve_13_15_17.pt")
    parser.add_argument("--skip-layers", default="13,15,17",
                        help="要砍的 VE 层索引，逗号分隔")
    args = parser.parse_args()

    skip = tuple(sorted(int(x) for x in args.skip_layers.split(",") if x.strip()))
    print(f"loading {args.in_ckpt}")
    ckpt = torch.load(args.in_ckpt, map_location="cpu", weights_only=False)

    sd = ckpt["model_state"]
    total_before = sum(v.numel() for v in sd.values())

    removed = []
    for i in skip:
        for tpl in VE_KEYS_TEMPLATE:
            k = tpl.format(i=i)
            if k in sd:
                n = sd[k].numel()
                del sd[k]
                removed.append((k, n))
            else:
                print(f"  warning: {k} not in state_dict")

    total_after = sum(v.numel() for v in sd.values())
    dropped = total_before - total_after

    # 改 config：老 ckpt 的 config 是 dict
    cfg = ckpt["config"]
    if isinstance(cfg, dict):
        cfg["ve_layer_skip"] = list(skip)
    else:
        # dataclass 情况：改字段
        from dataclasses import replace
        cfg = replace(cfg, ve_layer_skip=skip)
    ckpt["config"] = cfg

    print(f"\nremoved {len(removed)} tensors:")
    for k, n in removed:
        print(f"  - {k}  ({n/1e6:.2f} M)")
    print(f"\nparams:  before={total_before/1e6:.2f}M  after={total_after/1e6:.2f}M  dropped={dropped/1e6:.2f}M ({dropped/total_before*100:.1f}%)")

    # 打一个 provenance 标记便于追溯
    ckpt.setdefault("pruning_notes", []).append({
        "source_ckpt": args.in_ckpt,
        "skip_layers": list(skip),
        "dropped_params": dropped,
    })

    Path(args.out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    print(f"\nsaving {args.out_ckpt}")
    torch.save(ckpt, args.out_ckpt)
    print("done")


if __name__ == "__main__":
    main()
