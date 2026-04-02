#!/usr/bin/env python3
"""将已训练的 checkpoint 扩展到更大的模型规模。

支持：
  - 维度扩展（n_embd 变大）：填充 + 小噪声
  - 层数扩展（n_layer 变多）：循环复制已有层
  - 两者同时扩展

用法：
  python expand_checkpoint.py src.pt dst.pt --depth 16 --ar 80
  python expand_checkpoint.py src.pt dst.pt --depth 24          # 只扩展层数
  python expand_checkpoint.py src.pt dst.pt --ar 80             # 只扩展维度
"""
import argparse
import sys
import types
import os
from pathlib import Path
from dataclasses import asdict, fields

import torch

os.environ["HF_HUB_OFFLINE"] = "1"

NOISE_SCALE = 0.01


# ---------------------------------------------------------------------------
# 加载模型定义
# ---------------------------------------------------------------------------
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

    import unittest.mock as mock
    fake_k = types.ModuleType("kernels")
    fake_k.get_kernel = lambda r: types.SimpleNamespace(flash_attn_interface=None)
    sys.modules["kernels"] = fake_k
    with mock.patch("torch.cuda.get_device_capability", return_value=(9, 0)):
        ns: dict = {}
        exec(compile(src, "train.py", "exec"), ns)

    import __main__
    __main__.GPT = ns["GPT"]
    __main__.GPTConfig = ns["GPTConfig"]
    ns["GPT"].__module__ = "__main__"
    ns["GPTConfig"].__module__ = "__main__"
    del sys.modules["prepare"], sys.modules["kernels"]
    return ns["GPT"], ns["GPTConfig"]


# ---------------------------------------------------------------------------
# 权重扩展工具函数
# ---------------------------------------------------------------------------
def expand_1d(w: torch.Tensor, new_size: int, dim: int) -> torch.Tensor:
    """沿 dim 维度扩展张量，新增部分用小噪声填充。"""
    old_size = w.shape[dim]
    if old_size == new_size:
        return w.clone()
    assert new_size > old_size, f"只支持扩展，不支持缩小: {old_size} -> {new_size}"
    pad_size = new_size - old_size
    pad_shape = list(w.shape)
    pad_shape[dim] = pad_size
    noise = torch.randn(pad_shape, dtype=w.dtype) * NOISE_SCALE
    return torch.cat([w.clone(), noise], dim=dim)


def expand_weight(key: str, w: torch.Tensor,
                  src_embd: int, dst_embd: int,
                  src_kv_dim: int, dst_kv_dim: int) -> torch.Tensor:
    """根据 key 名称推断如何扩展权重。"""
    shape = w.shape

    # Embedding / lm_head: (vocab, embd)
    if "wte.weight" in key or "lm_head.weight" in key:
        return expand_1d(w, dst_embd, dim=1)

    # attnres_proj: Linear(embd, 1) -> weight shape (1, embd)
    if "attnres_proj" in key:
        return expand_1d(w, dst_embd, dim=1)

    # attnres_norm: RMSNorm weight (embd,)
    if "attnres_norm" in key:
        return expand_1d(w, dst_embd, dim=0)

    # ve_embed / value_embeds (value embedding): (vocab, kv_dim)
    if "ve_embed" in key or "value_embeds" in key:
        return expand_1d(w, dst_kv_dim, dim=1)

    # ve_gate: (n_kv_head, head_dim)
    if "ve_gate" in key:
        dst_n_kv_head = dst_embd // 64
        return expand_1d(w, dst_n_kv_head, dim=0)

    # Q projection: (n_head*head_dim, embd)
    if ".attn.c_q.weight" in key:
        w = expand_1d(w, dst_embd, dim=0)
        w = expand_1d(w, dst_embd, dim=1)
        return w

    # K/V projection: (n_kv_head*head_dim, embd)
    if ".attn.c_k.weight" in key or ".attn.c_v.weight" in key:
        w = expand_1d(w, dst_kv_dim, dim=0)
        w = expand_1d(w, dst_embd, dim=1)
        return w

    # Attention output projection: (embd, n_head*head_dim)
    if ".attn.c_proj.weight" in key:
        w = expand_1d(w, dst_embd, dim=0)
        w = expand_1d(w, dst_embd, dim=1)
        return w

    # MLP fc: (4*embd, embd)
    if ".mlp.c_fc.weight" in key:
        w = expand_1d(w, 4 * dst_embd, dim=0)
        w = expand_1d(w, dst_embd, dim=1)
        return w

    # MLP proj: (embd, 4*embd)
    if ".mlp.c_proj.weight" in key:
        w = expand_1d(w, dst_embd, dim=0)
        w = expand_1d(w, 4 * dst_embd, dim=1)
        return w

    # RMSNorm scalar weights: (embd,)
    if len(shape) == 1 and shape[0] == src_embd:
        return expand_1d(w, dst_embd, dim=0)

    print(f"  [WARN] 未识别的权重 {key}: {tuple(shape)}，直接复制")
    return w.clone()


# ---------------------------------------------------------------------------
# 层数扩展：把 src_layers 层循环映射到 dst_layers 层
# ---------------------------------------------------------------------------
def remap_layer_key(key: str, src_layer_idx: int, dst_layer_idx: int) -> str:
    """把 key 中的层索引从 src 替换为 dst。"""
    return key.replace(f".h.{src_layer_idx}.", f".h.{dst_layer_idx}.")


def expand_layers(state: dict, src_n_layer: int, dst_n_layer: int) -> dict:
    """层数扩展：循环复制已有层，新层加小噪声。"""
    if src_n_layer == dst_n_layer:
        return state

    # 按层分组（h.{i}. 路径）
    layer_keys = {i: {} for i in range(src_n_layer)}
    # 全局 ModuleList：attnres_proj/attnres_norm 索引 0..2*n_layer-1
    attnres_proj_keys = {}   # idx -> (key, tensor)
    attnres_norm_keys = {}   # idx -> (key, tensor)
    # value_embeds 索引是有 VE 的层号（奇偶取决于 has_ve）
    value_embed_keys = {}    # idx -> (key, tensor)
    other_keys = {}

    import re
    for k, v in state.items():
        if m := re.search(r'attnres_proj\.([0-9]+)\.weight', k):
            attnres_proj_keys[int(m.group(1))] = (k, v)
        elif m := re.search(r'attnres_norm\.([0-9]+)\.weight', k):
            attnres_norm_keys[int(m.group(1))] = (k, v)
        elif m := re.search(r'value_embeds\.([0-9]+)\.weight', k):
            value_embed_keys[int(m.group(1))] = (k, v)
        else:
            matched = False
            for i in range(src_n_layer):
                if f".h.{i}." in k:
                    layer_keys[i][k] = v
                    matched = True
                    break
            if not matched:
                other_keys[k] = v

    # 构建新 state
    new_state = dict(other_keys)

    # 扩展 h.{i} 层权重
    for dst_i in range(dst_n_layer):
        src_i = dst_i % src_n_layer
        for k, v in layer_keys[src_i].items():
            new_key = remap_layer_key(k, src_i, dst_i)
            noise = torch.randn_like(v) * NOISE_SCALE if dst_i >= src_n_layer else torch.zeros_like(v)
            new_state[new_key] = v.clone() + noise

    # 扩展 attnres_proj/norm（每层2个，索引 0..2*dst_n_layer-1）
    for dst_idx in range(2 * dst_n_layer):
        src_idx = dst_idx % (2 * src_n_layer)
        for store, prefix in [(attnres_proj_keys, 'attnres_proj'), (attnres_norm_keys, 'attnres_norm')]:
            if src_idx in store:
                _, v = store[src_idx]
                # key 前缀（去掉 _orig_mod. 前缀差异）
                base = next(k for k in state if f'{prefix}.{src_idx}.weight' in k)
                new_key = base.replace(f'{prefix}.{src_idx}.', f'{prefix}.{dst_idx}.')
                noise = torch.randn_like(v) * NOISE_SCALE if dst_idx >= 2 * src_n_layer else torch.zeros_like(v)
                new_state[new_key] = v.clone() + noise

    # 扩展 value_embeds（has_ve 层：layer_idx % 2 == (n_layer-1) % 2）
    def has_ve(layer_idx, n_layer):
        return layer_idx % 2 == (n_layer - 1) % 2

    src_ve_layers = [i for i in range(src_n_layer) if has_ve(i, src_n_layer)]
    dst_ve_layers = [i for i in range(dst_n_layer) if has_ve(i, dst_n_layer)]
    for j, dst_layer in enumerate(dst_ve_layers):
        src_layer = src_ve_layers[j % len(src_ve_layers)]
        if src_layer in value_embed_keys:
            _, v = value_embed_keys[src_layer]
            base = next(k for k in state if f'value_embeds.{src_layer}.weight' in k)
            new_key = base.replace(f'value_embeds.{src_layer}.', f'value_embeds.{dst_layer}.')
            noise = torch.randn_like(v) * NOISE_SCALE if dst_layer not in value_embed_keys else torch.zeros_like(v)
            new_state[new_key] = v.clone() + noise

    return new_state


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def build_new_config(GPTConfig, src_config, dst_depth: int, dst_ar: int | None, head_dim: int = 64, dst_embd_override: int | None = None):
    if dst_embd_override is not None:
        dst_embd = dst_embd_override
    else:
        base_dim = dst_depth * (dst_ar or src_config.n_embd // dst_depth)
        dst_embd = ((base_dim + head_dim - 1) // head_dim) * head_dim
    dst_n_head = dst_embd // head_dim
    return GPTConfig(
        sequence_len=src_config.sequence_len,
        vocab_size=src_config.vocab_size,
        n_layer=dst_depth,
        n_head=dst_n_head,
        n_kv_head=dst_n_head,
        n_embd=dst_embd,
        window_pattern=src_config.window_pattern,
    )


def expand(src_path: str, dst_path: str, dst_depth: int | None, dst_ar: int | None, dst_embd: int | None = None):
    GPT, GPTConfig = _load_model_defs()

    print(f"加载 {src_path}...")
    ckpt = torch.load(src_path, map_location="cpu", weights_only=False)
    raw_config = ckpt["config"]
    if isinstance(raw_config, dict):
        allowed = {f.name for f in fields(GPTConfig)}
        src_config = GPTConfig(**{k: v for k, v in raw_config.items() if k in allowed})
    else:
        src_config = raw_config
    src_state = ckpt["model_state"]

    # 解析源模型参数
    src_depth = src_config.n_layer
    src_embd = src_config.n_embd
    head_dim = src_embd // src_config.n_head
    src_ar = src_embd // src_depth
    src_kv_dim = src_config.n_kv_head * head_dim

    dst_depth = dst_depth if dst_depth is not None else src_depth
    dst_ar = dst_ar if dst_ar is not None else src_ar

    print(f"源模型: depth={src_depth}, ar={src_ar}, embd={src_embd}")

    dst_config = build_new_config(GPTConfig, src_config, dst_depth, dst_ar, head_dim, dst_embd)
    dst_embd = dst_config.n_embd
    dst_kv_dim = dst_config.n_kv_head * head_dim

    print(f"目标模型: depth={dst_depth}, ar={dst_ar}, embd={dst_embd}")

    # Step 1：层数扩展
    state = expand_layers(src_state, src_depth, dst_depth)

    # Step 2：维度扩展
    new_state = {}
    for k, v in state.items():
        new_state[k] = expand_weight(k, v, src_embd, dst_embd, src_kv_dim, dst_kv_dim)

    # 验证：用新 config 初始化模型，检查 key 匹配
    print("验证权重 key 匹配...")
    import contextlib, io
    with torch.device("meta"):
        ref_model = GPT(dst_config)
    ref_state = {k.replace("_orig_mod.", ""): v for k, v in ref_model.state_dict().items()}
    expanded_clean = {k.replace("_orig_mod.", ""): v for k, v in new_state.items()}

    missing = set(ref_state.keys()) - set(expanded_clean.keys())
    unexpected = set(expanded_clean.keys()) - set(ref_state.keys())
    shape_mismatch = [k for k in ref_state if k in expanded_clean and ref_state[k].shape != expanded_clean[k].shape]

    if missing:
        print(f"  [WARN] 缺少 {len(missing)} 个 key: {list(missing)[:5]}")
    if unexpected:
        print(f"  [WARN] 多余 {len(unexpected)} 个 key: {list(unexpected)[:5]}")
    if shape_mismatch:
        for k in shape_mismatch:
            print(f"  [ERROR] shape 不匹配 {k}: ref={ref_state[k].shape}, got={expanded_clean[k].shape}")
        sys.exit(1)
    print(f"  OK: {len(expanded_clean)} 个 key 全部匹配")

    # 保存
    new_ckpt = {
        "model_state": new_state,
        "config": asdict(dst_config),
        "val_bpb": ckpt.get("val_bpb", None),
        "step": ckpt.get("step", 0),
        "expanded_from": str(src_path),
    }
    torch.save(new_ckpt, dst_path)
    print(f"已保存: {dst_path}")


def main():
    parser = argparse.ArgumentParser(description="扩展 checkpoint 到更大模型规模")
    parser.add_argument("src", help="源 checkpoint 路径")
    parser.add_argument("dst", help="目标 checkpoint 路径")
    parser.add_argument("--depth", type=int, default=None, help="目标层数")
    parser.add_argument("--ar", type=int, default=None, help="目标 aspect ratio（model_dim = depth * ar）")
    parser.add_argument("--embd", type=int, default=None, help="直接指定目标 n_embd（优先于 --ar）")
    parser.add_argument("--noise", type=float, default=0.01, help="新增参数的噪声幅度")
    args = parser.parse_args()

    global NOISE_SCALE
    NOISE_SCALE = args.noise

    if args.depth is None and args.ar is None and args.embd is None:
        parser.error("至少指定 --depth、--ar 或 --embd 之一")

    expand(args.src, args.dst, args.depth, args.ar, args.embd)


if __name__ == "__main__":
    main()
