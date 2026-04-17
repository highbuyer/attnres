#!/usr/bin/env python3
"""把老 checkpoint 迁到新架构（GQA + 紧凑 rotary + 更宽 softcap）。

改动：
  - n_kv_head: 6 → 2（真正开 GQA，省 KV cache ~66%）
    · 按"连续 3 头取均值"合并 c_k / c_v 的 KV 维度
    · 若该层有 ve_gate，同样按头均值合并输出维度
  - rope_seq_len_mult: 10 → 2（省 rotary 预计算 buffer ~80%）
  - softcap: 15 → 30（更自然的 logit 尺度）

保持不变（风险/收益不成比例）：
  - tie_lm_head：老 ckpt 的 lm_head 已被 SFT 调过，不硬折叠进 wte
  - AttnRes boundary 粒度：论文 §3.2 没核对前不动

迁移后模型结构变了（c_k/c_v 维度缩了），需要 1-5K 步的"重新校准"训练恢复
性能。和 migrate_embeddings.py 同样的套路：先迁权重，再起 continue_pretrain
或 SFT resume，优化器热身覆盖结构扰动。

用法：
  uv run python scripts/migrate_to_new_arch.py \\
      --src checkpoints/sft_tool_summary_v5_best.pt \\
      --dst checkpoints/sft_tool_summary_v5_gqa.pt \\
      --n-kv-head 2 \\
      --rope-seq-len-mult 2 \\
      --softcap 30.0
"""
from __future__ import annotations

import argparse
import os
import sys
import types
from dataclasses import asdict, fields
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"

import torch


def _load_model_defs():
    _SRC_DIR = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(_SRC_DIR))
    lines = (_SRC_DIR / "train.py").read_text(encoding="utf-8").splitlines(keepends=True)
    cut = next(i for i, l in enumerate(lines) if "# Setup: tokenizer, model, optimizer, dataloader" in l)
    src = "".join(lines[:cut])
    fake = types.ModuleType("prepare")
    fake.MAX_SEQ_LEN = 2048
    fake.TIME_BUDGET = 300
    fake.Tokenizer = None
    fake.make_dataloader = None
    fake.evaluate_bpb = None
    sys.modules["prepare"] = fake
    _inject_kernels = not torch.cuda.is_available()
    if _inject_kernels:
        import unittest.mock as _mock
        fk = types.ModuleType("kernels")
        fk.get_kernel = lambda r: types.SimpleNamespace(flash_attn_interface=None)
        sys.modules["kernels"] = fk
        _cuda_patch = _mock.patch("torch.cuda.get_device_capability", return_value=(9, 0))
        _cuda_patch.start()
    ns: dict = {}
    exec(compile(src, "train.py", "exec"), ns)
    import __main__
    __main__.GPT = ns["GPT"]
    __main__.GPTConfig = ns["GPTConfig"]
    ns["GPT"].__module__ = "__main__"
    ns["GPTConfig"].__module__ = "__main__"
    return ns["GPT"], ns["GPTConfig"]


def _mean_pool_kv(weight: torch.Tensor, old_kv: int, new_kv: int, head_dim: int) -> torch.Tensor:
    """把 [old_kv*head_dim, n_embd] 的 c_k/c_v.weight 压到 [new_kv*head_dim, n_embd]。

    分组规则：old_kv 必须是 new_kv 的整数倍，每 (old_kv/new_kv) 个头取均值。
    """
    assert old_kv % new_kv == 0, f"old_kv={old_kv} 必须是 new_kv={new_kv} 的倍数"
    group = old_kv // new_kv
    out_dim, in_dim = weight.shape
    assert out_dim == old_kv * head_dim, f"shape mismatch: {weight.shape} vs expected [{old_kv*head_dim}, *]"
    # [old_kv, head_dim, in_dim] → [new_kv, group, head_dim, in_dim] → mean over group
    w = weight.view(old_kv, head_dim, in_dim)
    w = w.view(new_kv, group, head_dim, in_dim).mean(dim=1)
    return w.reshape(new_kv * head_dim, in_dim).contiguous()


def _mean_pool_gate(weight: torch.Tensor, old_kv: int, new_kv: int) -> torch.Tensor:
    """ve_gate.weight shape = [old_kv, ve_gate_channels] → [new_kv, ve_gate_channels]。"""
    assert old_kv % new_kv == 0
    group = old_kv // new_kv
    out_dim, in_dim = weight.shape
    assert out_dim == old_kv, f"gate out_dim={out_dim} 应 == old_kv={old_kv}"
    return weight.view(new_kv, group, in_dim).mean(dim=1).contiguous()


def _mean_pool_ve(weight: torch.Tensor, old_kv: int, new_kv: int, head_dim: int) -> torch.Tensor:
    """value_embeds.{i}.weight shape = [vocab, old_kv*head_dim] → [vocab, new_kv*head_dim]。

    kv_dim 维度上按头分组取均值（和 c_k/c_v 方向保持一致）。
    """
    assert old_kv % new_kv == 0
    group = old_kv // new_kv
    vocab, kv_dim = weight.shape
    assert kv_dim == old_kv * head_dim, f"ve shape mismatch: {weight.shape}, expected [*, {old_kv*head_dim}]"
    w = weight.view(vocab, old_kv, head_dim)
    w = w.view(vocab, new_kv, group, head_dim).mean(dim=2)
    return w.reshape(vocab, new_kv * head_dim).contiguous()


def migrate(src_state: dict, old_kv: int, new_kv: int, head_dim: int, n_layer: int) -> dict:
    """按头均值把老 state_dict 迁到新 KV 头数。返回新的 state dict。"""
    new_state = {}
    migrated_keys = 0
    for k, v in src_state.items():
        key = k.replace("_orig_mod.", "")
        if key.endswith(".c_k.weight") or key.endswith(".c_v.weight"):
            new_state[k] = _mean_pool_kv(v.float(), old_kv, new_kv, head_dim).to(v.dtype)
            migrated_keys += 1
        elif key.endswith(".ve_gate.weight"):
            new_state[k] = _mean_pool_gate(v.float(), old_kv, new_kv).to(v.dtype)
            migrated_keys += 1
        elif key.startswith("value_embeds.") and key.endswith(".weight"):
            new_state[k] = _mean_pool_ve(v.float(), old_kv, new_kv, head_dim).to(v.dtype)
            migrated_keys += 1
        else:
            new_state[k] = v
    expected_min = 2 * n_layer  # 至少 c_k + c_v per layer
    if migrated_keys < expected_min:
        print(f"WARNING: 只迁了 {migrated_keys} 个 KV/gate/ve 权重，预期 ≥ {expected_min}")
    else:
        print(f"migrated {migrated_keys} KV/gate/ve tensors")
    return new_state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="源 checkpoint (.pt)")
    p.add_argument("--dst", required=True, help="目标 checkpoint 保存路径")
    p.add_argument("--n-kv-head", type=int, default=2, help="新的 n_kv_head（默认 2）")
    p.add_argument("--rope-seq-len-mult", type=int, default=2, help="新的 rotary 预计算倍率（默认 2）")
    p.add_argument("--softcap", type=float, default=30.0, help="新的 logit softcap（默认 30）")
    p.add_argument("--dry-run", action="store_true", help="只检查，不写盘")
    args = p.parse_args()

    GPT, GPTConfig = _load_model_defs()

    print(f"Loading {args.src}")
    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    raw_config = ckpt["config"]
    if not isinstance(raw_config, dict):
        raw_config = asdict(raw_config)

    old_kv = raw_config["n_kv_head"]
    n_embd = raw_config["n_embd"]
    n_head = raw_config["n_head"]
    n_layer = raw_config["n_layer"]
    head_dim = n_embd // n_head
    if old_kv == args.n_kv_head:
        print(f"n_kv_head 已经是 {old_kv}，跳过 KV 迁移")
        new_state = dict(ckpt["model_state"])
    else:
        print(f"migrating KV heads: {old_kv} → {args.n_kv_head} (head_dim={head_dim}, n_layer={n_layer})")
        new_state = migrate(ckpt["model_state"], old_kv, args.n_kv_head, head_dim, n_layer)

    # 新 config
    allowed = {f.name for f in fields(GPTConfig)}
    new_cfg_dict = {k: v for k, v in raw_config.items() if k in allowed}
    new_cfg_dict.update({
        "n_kv_head": args.n_kv_head,
        "rope_seq_len_mult": args.rope_seq_len_mult,
        "softcap": args.softcap,
    })
    new_config = GPTConfig(**new_cfg_dict)

    # 构建模型验证 state_dict 能干净加载
    print("verifying new-arch model can load migrated state...")
    model = GPT(new_config).to(dtype=torch.float32)
    clean_state = {k.replace("_orig_mod.", ""): v for k, v in new_state.items()}
    res = model.load_state_dict(clean_state, strict=False)
    if res.missing_keys:
        print(f"  MISSING ({len(res.missing_keys)}): {res.missing_keys[:5]}")
    if res.unexpected_keys:
        print(f"  UNEXPECTED ({len(res.unexpected_keys)}): {res.unexpected_keys[:5]}")
    if res.missing_keys or res.unexpected_keys:
        raise RuntimeError("state_dict 加载不干净，拒绝写盘。请检查上方 missing/unexpected。")
    print("  load OK (0 missing, 0 unexpected)")

    # forward 自检
    model.eval()
    cos, sin = model._precompute_rotary_embeddings(model.rotary_seq_len, head_dim, device="cpu")
    model.cos, model.sin = cos.float(), sin.float()
    idx = torch.randint(0, new_config.vocab_size, (1, 16))
    with torch.no_grad():
        logits = model(idx)
    print(f"  forward OK: shape={tuple(logits.shape)}, |logits|.max={logits.abs().max().item():.4f} (softcap={args.softcap})")

    if args.dry_run:
        print("--dry-run 指定，不写盘")
        return

    # 构造新 ckpt；丢弃 optimizer_state（结构变了，旧的动量对不上新参数 shape）
    new_ckpt = dict(ckpt)
    new_ckpt["model_state"] = new_state
    new_ckpt["config"] = asdict(new_config)
    for drop_key in ("optimizer_state",):
        if drop_key in new_ckpt:
            del new_ckpt[drop_key]
            print(f"  dropped {drop_key}（新 arch 需要重起优化器状态）")
    # 标记
    new_ckpt["migrated_from"] = str(Path(args.src).name)
    new_ckpt["migration"] = {
        "n_kv_head": f"{old_kv}->{args.n_kv_head}",
        "rope_seq_len_mult": f"{raw_config.get('rope_seq_len_mult', 10)}->{args.rope_seq_len_mult}",
        "softcap": f"{raw_config.get('softcap', 15.0)}->{args.softcap}",
    }

    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_ckpt, dst)
    print(f"Saved {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
    print(f"Next step: uv run python src/continue_pretrain.py {dst} "
          f"  # 或用 sft.py（不要 --resume，因为 optimizer_state 已丢弃）")


if __name__ == "__main__":
    main()
