#!/usr/bin/env python3
"""将旧 8192 词表模型的 embedding 迁移到新 32768 词表模型。

原理：
- 旧 tokenizer 和新 tokenizer 的 token→id 映射不同
- 按字节序列匹配：找到新旧 tokenizer 中相同字节序列的 token，复制其 embedding
- 新增 token（旧 tokenizer 没有的）随机初始化

用法：
  python migrate_embeddings.py --src continued_d18_v2_final.pt --dst expanded_32k.pt
"""
import argparse
import os
import pickle
import sys
import types
from dataclasses import asdict, fields
from pathlib import Path

import torch

os.environ["HF_HUB_OFFLINE"] = "1"

OLD_TOKENIZER = Path("/home/langshen/.cache/autoresearch-custom/tokenizer_8192_backup/tokenizer.pkl")
NEW_TOKENIZER = Path("/home/langshen/.cache/autoresearch-custom/tokenizer/tokenizer.pkl")


def load_token_bytes(pkl_path):
    """从 tiktoken pickle 里取出 bytes->id 映射。"""
    with open(pkl_path, "rb") as f:
        enc = pickle.load(f)
    # tiktoken Encoding 的 mergeable_ranks: {bytes: int}
    return enc._mergeable_ranks  # bytes -> rank(id)


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


def migrate(src_path, dst_path):
    print(f"加载旧模型: {src_path}")
    GPT, GPTConfig = _load_model_defs()
    ckpt = torch.load(src_path, map_location="cpu", weights_only=False)
    raw_config = ckpt["config"]
    if isinstance(raw_config, dict):
        allowed = {f.name for f in fields(GPTConfig)}
        src_config = GPTConfig(**{k: v for k, v in raw_config.items() if k in allowed})
    else:
        src_config = raw_config
    print(f"旧配置: vocab={src_config.vocab_size}, depth={src_config.n_layer}, embd={src_config.n_embd}")

    print("加载旧/新 tokenizer...")
    old_bytes2id = load_token_bytes(OLD_TOKENIZER)
    new_bytes2id = load_token_bytes(NEW_TOKENIZER)
    print(f"旧词表: {len(old_bytes2id)} tokens, 新词表: {len(new_bytes2id)} tokens")

    # 构建迁移映射：new_id -> old_id（如果字节序列相同）
    old_id2bytes = {v: k for k, v in old_bytes2id.items()}
    new_id2bytes = {v: k for k, v in new_bytes2id.items()}

    mapping = {}  # new_id -> old_id
    for new_id, b in new_id2bytes.items():
        if b in old_bytes2id:
            mapping[new_id] = old_bytes2id[b]

    print(f"可迁移 token: {len(mapping)} / {len(new_bytes2id)}")

    # 构建新配置
    new_config = GPTConfig(
        sequence_len=src_config.sequence_len,
        vocab_size=32768,
        n_layer=src_config.n_layer,
        n_head=src_config.n_head,
        n_kv_head=src_config.n_kv_head,
        n_embd=src_config.n_embd,
        window_pattern=src_config.window_pattern,
    )

    # 初始化新模型（meta device）
    print("初始化新模型权重...")
    old_state = ckpt["model_state"]

    # 获取旧 embedding
    old_wte = None
    for k, v in old_state.items():
        if "wte.weight" in k:
            old_wte = v  # (8192, n_embd)
            break

    # 构建新 embedding (32768, n_embd)
    n_embd = src_config.n_embd
    new_wte = torch.randn(32768, n_embd) * 0.02
    copied = 0
    for new_id, old_id in mapping.items():
        if old_id < old_wte.shape[0]:
            new_wte[new_id] = old_wte[old_id]
            copied += 1
    print(f"复制 embedding: {copied} / 32768")

    # 更新 state dict
    new_state = {}
    for k, v in old_state.items():
        if "wte.weight" in k:
            new_state[k] = new_wte
        elif "lm_head.weight" in k:
            # lm_head 同样需要扩展
            old_lm = v  # (8192, n_embd)
            new_lm = torch.randn(32768, n_embd) * 0.02
            for new_id, old_id in mapping.items():
                if old_id < old_lm.shape[0]:
                    new_lm[new_id] = old_lm[old_id]
            new_state[k] = new_lm
        elif "value_embeds" in k:
            # value_embeds 也是 (vocab, kv_dim)
            old_ve = v
            kv_dim = old_ve.shape[1]
            new_ve = torch.randn(32768, kv_dim) * 0.02
            for new_id, old_id in mapping.items():
                if old_id < old_ve.shape[0]:
                    new_ve[new_id] = old_ve[old_id]
            new_state[k] = new_ve
        else:
            new_state[k] = v

    new_ckpt = {
        "model_state": new_state,
        "config": asdict(new_config),
        "val_bpb": ckpt.get("val_bpb"),
        "step": ckpt.get("step", 0),
        "migrated_from": str(src_path),
    }
    torch.save(new_ckpt, dst_path)
    print(f"已保存: {dst_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="旧 checkpoint")
    parser.add_argument("dst", help="新 checkpoint")
    args = parser.parse_args()
    migrate(args.src, args.dst)
