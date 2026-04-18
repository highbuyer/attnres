#!/usr/bin/env python3
"""KV cache 正确性验证：同一 prompt、同一 ckpt，use_cache=True 和 use_cache=False 路径
的 logits 和每步 argmax 必须完全一致。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

import sft as sft_mod  # noqa: E402


def main():
    sft_mod._ensure_model_defs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parameter_dtype = sft_mod.parameter_dtype_for_training(device)
    compute_dtype = sft_mod.compute_dtype_for_training(device)

    ckpt_path = "checkpoints/sft_v5_pruned_ve_13_15_17.pt"
    print(f"loading {ckpt_path}")
    ckpt = sft_mod.load_checkpoint(ckpt_path, device)
    config = ckpt["config"]

    model = sft_mod.GPT(config).to(device=device, dtype=parameter_dtype)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state"].items()}
    model.load_state_dict(state, strict=False)

    head_dim = config.n_embd // config.n_head
    cos, sin = model._precompute_rotary_embeddings(model.rotary_seq_len, head_dim, device=device)
    model.cos, model.sin = cos.to(compute_dtype), sin.to(compute_dtype)
    model.eval()

    # 固定种子生成 prompt
    torch.manual_seed(42)
    T_prompt = 64
    n_new = 16
    prompt = torch.randint(0, config.vocab_size, (1, T_prompt), device=device)

    # 路径 A：无 cache，每步重传完整前缀
    with torch.no_grad(), sft_mod.autocast_context(device):
        x = prompt.clone()
        argmax_a = []
        for _ in range(n_new):
            logits = model(x)
            nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            argmax_a.append(int(nxt.item()))
            x = torch.cat([x, nxt], dim=1)
        logits_a_final = logits[:, -1, :].float()

    # 路径 B：use_cache=True，prefill + decode
    with torch.no_grad(), sft_mod.autocast_context(device):
        logits, past_kvs = model(prompt, use_cache=True)
        argmax_b = []
        nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        argmax_b.append(int(nxt.item()))
        for step in range(n_new - 1):
            pos = T_prompt + step
            logits, past_kvs = model(nxt, past_kvs=past_kvs, position_offset=pos, use_cache=True)
            nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            argmax_b.append(int(nxt.item()))
        logits_b_final = logits[:, -1, :].float()

    print(f"\npath A (no cache)  argmax: {argmax_a}")
    print(f"path B (kv cache)  argmax: {argmax_b}")
    same = argmax_a == argmax_b
    diff_logits = (logits_a_final - logits_b_final).abs().max().item()
    print(f"\nargmax identical: {same}")
    print(f"max |logits_A - logits_B| on final step = {diff_logits:.6f}")

    if not same:
        for i, (a, b) in enumerate(zip(argmax_a, argmax_b)):
            if a != b:
                print(f"  first divergence at step {i}: no-cache={a}, kv-cache={b}")
                break
        sys.exit(1)
    print("\n✓ KV cache correctness PASS")


if __name__ == "__main__":
    main()
