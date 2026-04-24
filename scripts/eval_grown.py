#!/usr/bin/env python3
"""eval-only: d18 baseline vs d36 grown，同一 val split 跑 evaluate_bpb。

用途：判断 layer stacking 是否直接把容量释放出来（bpb 显著降低），
还是需要短 CPT 打破层对称性（bpb ≈ d18 或更高）。

日志：/tmp/eval_grown.log
"""
import sys
import time
import math
from pathlib import Path
from dataclasses import fields

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from prepare import Tokenizer, MAX_SEQ_LEN, evaluate_bpb  # noqa: E402

# 取 GPT/GPTConfig，不触发 train.py 主流程
train_src = (ROOT / "src" / "train.py").read_text()
cut = train_src.index("# Setup: tokenizer, model, optimizer, dataloader")
ns: dict = {"__name__": "__main__", "__file__": str(ROOT / "src" / "train.py")}
exec(train_src[:cut], ns)
GPT, GPTConfig = ns["GPT"], ns["GPTConfig"]

import __main__  # noqa: E402
__main__.GPT = GPT
__main__.GPTConfig = GPTConfig
GPTConfig.__module__ = "__main__"


def load_model(ckpt_path: Path, device: torch.device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    raw_cfg = ck["config"]
    if not isinstance(raw_cfg, GPTConfig):
        # continue_pretrain/sft.py 存 dict 时的兜底
        allowed = {f.name for f in fields(GPTConfig)}
        raw_cfg = GPTConfig(**{k: v for k, v in raw_cfg.items() if k in allowed})
    dtype = torch.bfloat16
    model = GPT(raw_cfg).to(device=device, dtype=dtype)
    # RoPE cos/sin 与 compute_dtype 对齐（continue_pretrain.py:130 做法）
    if hasattr(model, "cos") and hasattr(model, "sin"):
        model.cos = model.cos.to(dtype)
        model.sin = model.sin.to(dtype)
    state = {k.replace("_orig_mod.", ""): v for k, v in ck["model_state"].items()}
    res = model.load_state_dict(state, strict=False)
    assert not res.missing_keys, f"missing: {res.missing_keys[:3]}"
    assert not res.unexpected_keys, f"unexpected: {res.unexpected_keys[:3]}"
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    ckpt_bpb = ck.get("val_bpb", float("inf"))
    return model, raw_cfg, n_params, ckpt_bpb


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_directory()
    autocast_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16) \
        if device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False)
    BATCH = 4  # 给 d36 留余量；d18 也用同 BATCH 保持 eval 路径一致

    targets = [
        ("d18 (baseline)", ROOT / "checkpoints" / "continued_d18_32k_final.pt"),
        ("d36 (grown)   ", ROOT / "checkpoints" / "grown_d36_from_d18.pt"),
    ]
    results = []
    for label, p in targets:
        print(f"\n=== {label} : {p.name} ===", flush=True)
        t0 = time.time()
        model, cfg, n_params_M, ckpt_bpb = load_model(p, device)
        print(f"  n_layer={cfg.n_layer} n_embd={cfg.n_embd}  params={n_params_M:.1f}M  "
              f"ckpt.val_bpb={ckpt_bpb}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(), autocast_ctx:
            bpb = evaluate_bpb(model, tokenizer, BATCH, device=device)
        elapsed = time.time() - t0
        peak = (torch.cuda.max_memory_allocated() / 1024**2) if device.type == "cuda" else 0.0
        print(f"  val_bpb = {bpb:.6f}  elapsed={elapsed:.1f}s  peak_vram={peak:.0f} MB",
              flush=True)
        results.append((label, n_params_M, ckpt_bpb, bpb))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n=== summary ===")
    print(f"{'model':18}  {'params':>8}  {'ckpt.bpb':>10}  {'live.bpb':>10}")
    for label, n, ck_bpb, bpb in results:
        ck_s = f"{ck_bpb:.4f}" if ck_bpb != float("inf") else "   n/a"
        print(f"{label}  {n:7.1f}M  {ck_s:>10}  {bpb:10.6f}")
    if len(results) == 2:
        d_bpb = results[1][3] - results[0][3]
        print(f"\nΔbpb (d36 - d18) = {d_bpb:+.6f}  "
              f"({'容量释放' if d_bpb < -0.01 else ('对称退化/持平' if d_bpb < 0.01 else '劣化')})")


if __name__ == "__main__":
    main()
