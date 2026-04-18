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


def bench_decode(model, T_prefix, n_new, device, vocab_size, warmup=2, measure=3, use_cache=False):
    torch.cuda.reset_peak_memory_stats(device)
    x0 = torch.randint(0, vocab_size, (1, T_prefix), device=device)

    def _run_no_cache():
        x = x0.clone()
        for _ in range(n_new):
            logits = model(x)
            nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            x = torch.cat([x, nxt], dim=1)

    def _run_cached():
        logits, past_kvs = model(x0, use_cache=True)
        nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        for step in range(n_new - 1):
            pos = T_prefix + step
            logits, past_kvs = model(nxt, past_kvs=past_kvs, position_offset=pos, use_cache=True)
            nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    runner = _run_cached if use_cache else _run_no_cache

    # warmup
    with torch.no_grad(), sft_mod.autocast_context(device):
        for _ in range(warmup):
            runner()
    torch.cuda.synchronize(device)

    times = []
    with torch.no_grad(), sft_mod.autocast_context(device):
        for _ in range(measure):
            t0 = time.perf_counter()
            runner()
            torch.cuda.synchronize(device)
            times.append(time.perf_counter() - t0)
    peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    return statistics.median(times), peak_mb


def fmt_pct(a, b):
    # a vs b: (a - b) / b * 100
    if b == 0:
        return "n/a"
    return f"{(a - b) / b * 100:+.1f}%"


def run_ckpt(path, device, lengths, n_decode, use_cache=False):
    sft_mod._ensure_model_defs()
    tag = "KVcache" if use_cache else "no-cache"
    print(f"\n===== {path}  [{tag}] =====")
    model, config, n_params = load_for_bench(path, device)
    print(f"params: {n_params/1e6:.2f} M")
    vocab = config.vocab_size

    results = {"path": path, "use_cache": use_cache, "n_params": n_params, "prefill": {}, "decode": {}}
    for T in lengths:
        dt, peak = bench_prefill(model, T, device, vocab)
        tps = T / dt
        print(f"  prefill T={T:4d}: {dt*1000:7.2f} ms  {tps:8.1f} tok/s  peak={peak:7.1f} MB")
        results["prefill"][T] = {"ms": dt * 1000, "tok_per_s": tps, "peak_mb": peak}
    for T in lengths:
        dt, peak = bench_decode(model, T, n_decode, device, vocab, use_cache=use_cache)
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

    base = run_ckpt(args.base, device, lengths, args.n_decode, use_cache=False)
    pruned_nocache = run_ckpt(args.pruned, device, lengths, args.n_decode, use_cache=False)
    pruned_kv = run_ckpt(args.pruned, device, lengths, args.n_decode, use_cache=True)

    print("\n\n===== 对比（全部 base-404M 无 cache vs pruned-329M 两种路径）=====")
    print(f"params: {base['n_params']/1e6:.2f}M (base) / {pruned_nocache['n_params']/1e6:.2f}M (pruned)")
    print("\nDecode tok/s (最关键指标):")
    print(f"{'T':>6} {'base':>10} {'pruned-no-cache':>18} {'pruned-KVcache':>18} {'KV speedup':>12}")
    for T in lengths:
        b = base["decode"][T]["tok_per_s"]
        pnc = pruned_nocache["decode"][T]["tok_per_s"]
        pkv = pruned_kv["decode"][T]["tok_per_s"]
        print(f"{T:>6d} {b:>10.1f} {pnc:>18.1f} {pkv:>18.1f} {pkv/pnc:>11.2f}×")
    print("\nPrefill tok/s（不受 KV cache 影响，应持平）:")
    print(f"{'T':>6} {'pruned-no-cache':>18} {'pruned-KVcache':>18}")
    for T in lengths:
        pnc = pruned_nocache["prefill"][T]["tok_per_s"]
        pkv = pruned_kv["prefill"][T]["tok_per_s"]
        print(f"{T:>6d} {pnc:>18.1f} {pkv:>18.1f}")
    print("\nPeak VRAM (decode MB):")
    print(f"{'T':>6} {'pruned-no-cache':>18} {'pruned-KVcache':>18} {'delta':>10}")
    for T in lengths:
        pnc = pruned_nocache["decode"][T]["peak_mb"]
        pkv = pruned_kv["decode"][T]["peak_mb"]
        print(f"{T:>6d} {pnc:>18.1f} {pkv:>18.1f} {pkv-pnc:>+9.1f}")


if __name__ == "__main__":
    main()
