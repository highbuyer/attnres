#!/usr/bin/env python3
"""v5_best (404M) vs pruned (329M) 推理速度和显存 benchmark。

docs/VE_PRUNED_v1.md 给出了 params -18.7% 的数字，但"参数减少"不等于"推理变快"——
需要实测 prefill/decode tokens/sec 和显存峰值才能给用户硬答案。

设计：
  - 多组 prompt 长度 T ∈ {256, 1024, 2048}
  - prefill：单次 model(x_T) 的耗时，吞吐 = T / dt
  - decode：prefill 后逐 token append 循环 32 步（项目 infer 没 KV cache，
    两个 ckpt 公平对齐），测每 token 平均耗时
  - 每组 warmup 3 + measure 5 取中位数
  - 显存峰值用 torch.cuda.max_memory_allocated()
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

import sft as sft_mod  # noqa: E402


def load_for_bench(path, device):
    ckpt = sft_mod.load_checkpoint(path, device)
    config = ckpt["config"]
    param_dtype = sft_mod.parameter_dtype_for_training(device)
    compute_dtype = sft_mod.compute_dtype_for_training(device)
    model = sft_mod.GPT(config).to(device=device, dtype=param_dtype)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state"].items()}
    model.load_state_dict(state, strict=False)
    head_dim = config.n_embd // config.n_head
    cos, sin = model._precompute_rotary_embeddings(model.rotary_seq_len, head_dim, device=device)
    model.cos, model.sin = cos.to(compute_dtype), sin.to(compute_dtype)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    return model, config, n_params


def bench_prefill(model, T, device, vocab_size, warmup=3, measure=5):
    torch.cuda.reset_peak_memory_stats(device)
    x = torch.randint(0, vocab_size, (1, T), device=device)
    # warmup
    with torch.no_grad(), sft_mod.autocast_context(device):
        for _ in range(warmup):
            _ = model(x)
    torch.cuda.synchronize(device)
    times = []
    with torch.no_grad(), sft_mod.autocast_context(device):
        for _ in range(measure):
            t0 = time.perf_counter()
            _ = model(x)
            torch.cuda.synchronize(device)
            times.append(time.perf_counter() - t0)
    peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    return statistics.median(times), peak_mb


def bench_decode(model, T_prefix, n_new, device, vocab_size, warmup=2, measure=3):
    torch.cuda.reset_peak_memory_stats(device)
    x0 = torch.randint(0, vocab_size, (1, T_prefix), device=device)

    # warmup
    with torch.no_grad(), sft_mod.autocast_context(device):
        for _ in range(warmup):
            x = x0.clone()
            for _ in range(n_new):
                logits = model(x)
                nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                x = torch.cat([x, nxt], dim=1)
    torch.cuda.synchronize(device)

    times = []
    with torch.no_grad(), sft_mod.autocast_context(device):
        for _ in range(measure):
            x = x0.clone()
            t0 = time.perf_counter()
            for _ in range(n_new):
                logits = model(x)
                nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                x = torch.cat([x, nxt], dim=1)
            torch.cuda.synchronize(device)
            times.append(time.perf_counter() - t0)
    peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    return statistics.median(times), peak_mb


def fmt_pct(a, b):
    # a vs b: (a - b) / b * 100
    if b == 0:
        return "n/a"
    return f"{(a - b) / b * 100:+.1f}%"


def run_ckpt(path, device, lengths, n_decode):
    sft_mod._ensure_model_defs()
    print(f"\n===== {path} =====")
    model, config, n_params = load_for_bench(path, device)
    print(f"params: {n_params/1e6:.2f} M")
    vocab = config.vocab_size

    results = {"path": path, "n_params": n_params, "prefill": {}, "decode": {}}
    for T in lengths:
        dt, peak = bench_prefill(model, T, device, vocab)
        tps = T / dt
        print(f"  prefill T={T:4d}: {dt*1000:7.2f} ms  {tps:8.1f} tok/s  peak={peak:7.1f} MB")
        results["prefill"][T] = {"ms": dt * 1000, "tok_per_s": tps, "peak_mb": peak}
    for T in lengths:
        dt, peak = bench_decode(model, T, n_decode, device, vocab)
        tps = n_decode / dt
        print(f"  decode  T_prefix={T:4d} +{n_decode} tok: {dt*1000:7.2f} ms  {tps:6.1f} tok/s  peak={peak:7.1f} MB")
        results["decode"][T] = {"ms": dt * 1000, "tok_per_s": tps, "peak_mb": peak, "n_new": n_decode}

    del model
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="checkpoints/sft_tool_summary_v5_best.pt")
    parser.add_argument("--pruned", default="checkpoints/sft_v5_pruned_ve_13_15_17.pt")
    parser.add_argument("--lengths", default="256,1024,2048")
    parser.add_argument("--n-decode", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda")
    lengths = [int(x) for x in args.lengths.split(",")]

    base = run_ckpt(args.base, device, lengths, args.n_decode)
    pruned = run_ckpt(args.pruned, device, lengths, args.n_decode)

    print("\n\n===== 对比 =====")
    print(f"params: {base['n_params']/1e6:.2f}M → {pruned['n_params']/1e6:.2f}M  "
          f"({(pruned['n_params']-base['n_params'])/base['n_params']*100:+.1f}%)")
    print("\nPrefill (tok/s):")
    print(f"{'T':>6} {'base':>12} {'pruned':>12} {'speedup':>10}")
    for T in lengths:
        b = base["prefill"][T]["tok_per_s"]
        p = pruned["prefill"][T]["tok_per_s"]
        print(f"{T:>6d} {b:>12.1f} {p:>12.1f} {p/b:>9.3f}×")
    print("\nDecode (tok/s, T_prefix varies):")
    print(f"{'T':>6} {'base':>12} {'pruned':>12} {'speedup':>10}")
    for T in lengths:
        b = base["decode"][T]["tok_per_s"]
        p = pruned["decode"][T]["tok_per_s"]
        print(f"{T:>6d} {b:>12.1f} {p:>12.1f} {p/b:>9.3f}×")
    print("\nPeak VRAM (MB, decode):")
    print(f"{'T':>6} {'base':>12} {'pruned':>12} {'diff':>10}")
    for T in lengths:
        b = base["decode"][T]["peak_mb"]
        p = pruned["decode"][T]["peak_mb"]
        print(f"{T:>6d} {b:>12.1f} {p:>12.1f} {p-b:>+9.1f}")


if __name__ == "__main__":
    main()
