#!/usr/bin/env python3
"""验证 d36_v2_mla_best.pt 的 val_bpb 数字 + 统计 val 集 special token 占比。

用法:
  .venv/bin/python -u src/eval_v2.py [--quick]

--quick: 用 2M tokens（约 1/10 全 eval），用来快速 sanity check
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))

from model_v2 import GPT_v2, GPTConfigV2
from prepare import (
    EVAL_TOKENS,
    MAX_SEQ_LEN,
    Tokenizer,
    evaluate_bpb,
    get_token_bytes,
    make_dataloader,
)

CKPT = "checkpoints/d36_v2_mla_best.pt"


def _strip_compile_prefix(sd: dict) -> dict:
    return {k.replace("._orig_mod.", "."): v for k, v in sd.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="2M tokens (~1/10) instead of full eval")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(1234)

    print(f"[eval_v2] loading {CKPT}")
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    valid = {f.name for f in GPTConfigV2.__dataclass_fields__.values()}
    cfg = GPTConfigV2(**{k: v for k, v in ck["config"].items() if k in valid})
    print(f"[eval_v2] saved val_bpb={ck['val_bpb']:.6f}  step={ck['step']}")

    model = GPT_v2(cfg)
    sd = _strip_compile_prefix(ck["model_state"])
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[eval_v2] missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device=device, dtype=torch.bfloat16).eval()

    tokenizer = Tokenizer.from_directory()

    # ------------------------------------------------------------------
    # 1. special token 占比统计（不需要前向）
    # ------------------------------------------------------------------
    print("\n[eval_v2] step 1: special token ratio in val set ...")
    token_bytes = get_token_bytes(device=device)
    loader = make_dataloader(tokenizer, args.batch_size, MAX_SEQ_LEN, "val", device=device)
    sample_target_tokens = 2 * 524288  # 1M target tokens 够看比例
    sample_steps = sample_target_tokens // (args.batch_size * MAX_SEQ_LEN)
    total_tok = 0
    special_tok = 0
    for _ in range(sample_steps):
        _, y, _ = next(loader)
        nbytes = token_bytes[y.view(-1)]
        total_tok += y.numel()
        special_tok += int((nbytes == 0).sum().item())
    del loader
    print(f"[eval_v2]   sampled {total_tok:,} val tokens, special={special_tok:,} "
          f"({100*special_tok/total_tok:.3f}%)")

    # ------------------------------------------------------------------
    # 2. 重跑 evaluate_bpb 验证 0.6921
    # ------------------------------------------------------------------
    target_tokens = (2 * 524288) if args.quick else EVAL_TOKENS
    print(f"\n[eval_v2] step 2: evaluate_bpb on {target_tokens:,} tokens "
          f"(batch_size={args.batch_size}) ...")
    bpb = evaluate_bpb(model, tokenizer, args.batch_size, device=device, max_tokens=target_tokens)
    print(f"\n[eval_v2] reproduced val_bpb = {bpb:.6f}")
    print(f"[eval_v2] saved          val_bpb = {ck['val_bpb']:.6f}")
    print(f"[eval_v2] delta                  = {bpb - ck['val_bpb']:+.6f}")


if __name__ == "__main__":
    main()
