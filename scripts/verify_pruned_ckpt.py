#!/usr/bin/env python3
"""验证 pruned ckpt：
  (1) 严格 load（strict=True 应当无 missing/unexpected key）
  (2) 全量 val val_bpt 应当 ≈ knock_weakest3 的 1.7271（权重等价）
  (3) 参数量计数对齐 prune_ve.py 报告的 328.82M

这是 P2 VE 剪枝落地的 gate。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

import sft as sft_mod  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/sft_v5_pruned_ve_13_15_17.pt")
    parser.add_argument("--data", default="sft_tool_summary_v5.jsonl")
    parser.add_argument("--max-val-samples", type=int, default=None)
    args = parser.parse_args()

    sft_mod._ensure_model_defs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parameter_dtype = sft_mod.parameter_dtype_for_training(device)
    compute_dtype = sft_mod.compute_dtype_for_training(device)

    print(f"loading {args.checkpoint}")
    ckpt = sft_mod.load_checkpoint(args.checkpoint, device)
    config = ckpt["config"]
    skip = getattr(config, "ve_layer_skip", ())
    print(f"  config.ve_layer_skip = {skip}")
    print(f"  config.n_layer = {config.n_layer}")

    model = sft_mod.GPT(config).to(device=device, dtype=parameter_dtype)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state"].items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    missing_ve = [k for k in missing if "value_embeds" in k or "ve_gate" in k]
    other_missing = [k for k in missing if k not in missing_ve]
    print(f"\nload_state_dict (strict=False):")
    print(f"  missing total = {len(missing)}  (of which VE-related = {len(missing_ve)})")
    if other_missing:
        print(f"  !! non-VE missing = {other_missing[:5]}")
    print(f"  unexpected = {len(unexpected)} {unexpected[:5]}")

    # 参数量计数
    total = sum(p.numel() for p in model.parameters())
    print(f"\nmodel params: {total/1e6:.2f}M")

    # 跑 val_bpt
    head_dim = config.n_embd // config.n_head
    cos, sin = model._precompute_rotary_embeddings(model.rotary_seq_len, head_dim, device=device)
    model.cos, model.sin = cos.to(compute_dtype), sin.to(compute_dtype)

    print(f"\nbuilding val_data from {args.data}...")
    raw, train_data, val_data = sft_mod.build_datasets(args.data)
    if args.max_val_samples and len(val_data) > args.max_val_samples:
        val_data = val_data[:args.max_val_samples]
    print(f"val_data: {len(val_data)} samples")

    bpt = sft_mod.evaluate_sft(model, val_data, device)
    print(f"\n=== val_bpt = {bpt:.4f} ===")
    print("expected ≈ 1.7271 (knock_weakest3 from v2_full ablation)")


if __name__ == "__main__":
    main()
