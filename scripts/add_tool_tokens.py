#!/usr/bin/env python3
"""给现有 tokenizer 追加 6 个工具调用 special tokens，并扩展 checkpoint embedding。

不重训 BPE，只追加 special token 注册 + 扩展 vocab 维度的权重。

用法：
  cd /home/langshen/base_mode/attnres/src
  uv run python ../scripts/add_tool_tokens.py \
      --checkpoint ../checkpoints/continued_d18_32k_final.pt \
      --output ../checkpoints/tooltoken_d18_32k.pt
"""
import argparse
import os
import pickle
import sys
import types
from dataclasses import asdict, fields
from pathlib import Path

import tiktoken
import torch

TOKENIZER_PKL = Path(os.path.expanduser("~/.cache/autoresearch-custom/tokenizer/tokenizer.pkl"))

# 要追加的 6 个工具调用 special tokens
TOOL_TOKENS = [
    "<|tool_call_start|>",
    "<|tool_call_end|>",
    "<|tool_result_start|>",
    "<|tool_result_end|>",
    "<|tool_name_search_code|>",
    "<|tool_name_read_file|>",
]


def update_tokenizer():
    """给 tokenizer 追加 6 个 special tokens，返回新旧 vocab size。"""
    print(f"加载 tokenizer: {TOKENIZER_PKL}")
    with open(TOKENIZER_PKL, "rb") as f:
        enc = pickle.load(f)

    old_vocab = enc.n_vocab
    old_special = dict(enc._special_tokens)
    print(f"当前: {old_vocab} tokens ({len(enc._mergeable_ranks)} BPE + {len(old_special)} special)")

    # 检查是否已追加过
    if TOOL_TOKENS[0] in old_special:
        print("工具 token 已存在，跳过 tokenizer 修改")
        return old_vocab, old_vocab

    # 追加新 special tokens
    next_id = old_vocab
    new_special = dict(old_special)
    for tok in TOOL_TOKENS:
        new_special[tok] = next_id
        print(f"  追加: {tok} -> {next_id}")
        next_id += 1

    # 重建 Encoding
    new_enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=enc._pat_str,
        mergeable_ranks=enc._mergeable_ranks,
        special_tokens=new_special,
    )

    target_vocab = new_enc.n_vocab
    print(f"新 tokenizer: {target_vocab} tokens ({len(enc._mergeable_ranks)} BPE + {len(new_special)} special)")

    # 验证
    for tok in TOOL_TOKENS:
        tid = new_enc.encode_single_token(tok)
        decoded = new_enc.decode([tid])
        assert decoded == tok, f"验证失败: {tok} -> {tid} -> {decoded}"
    # 验证旧 token 不受影响
    for tok, tid in old_special.items():
        assert new_enc.encode_single_token(tok) == tid, f"旧 token 位移: {tok}"
    print("tokenizer 验证通过")

    # 备份旧 tokenizer
    backup = TOKENIZER_PKL.with_suffix(".pkl.bak_4special")
    if not backup.exists():
        import shutil
        shutil.copy2(TOKENIZER_PKL, backup)
        print(f"旧 tokenizer 已备份: {backup}")

    # 保存
    with open(TOKENIZER_PKL, "wb") as f:
        pickle.dump(new_enc, f)
    print(f"新 tokenizer 已保存: {TOKENIZER_PKL}")

    return old_vocab, target_vocab


def _load_model_defs():
    """加载 train.py 中的模型定义。"""
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
        ns = {}
        exec(compile(src, "train.py", "exec"), ns)
    import __main__
    __main__.GPT = ns["GPT"]
    __main__.GPTConfig = ns["GPTConfig"]
    ns["GPT"].__module__ = "__main__"
    ns["GPTConfig"].__module__ = "__main__"
    del sys.modules["prepare"], sys.modules["kernels"]
    return ns["GPT"], ns["GPTConfig"]


def expand_checkpoint(src_path, dst_path, target_vocab):
    """扩展 checkpoint 中所有 vocab 维度的权重到 target_vocab。"""
    # 注册 GPTConfig 到 __main__（checkpoint unpickle 需要）
    _load_model_defs()
    print(f"\n加载 checkpoint: {src_path}")
    ckpt = torch.load(src_path, map_location="cpu", weights_only=False)

    config = ckpt["config"]
    if isinstance(config, dict):
        old_vocab = config.get("vocab_size", 0)
    else:
        old_vocab = config.vocab_size

    if old_vocab == target_vocab:
        print(f"checkpoint vocab_size={old_vocab} 已等于目标 {target_vocab}，跳过")
        return

    n_new = target_vocab - old_vocab
    print(f"checkpoint vocab_size: {old_vocab} -> {target_vocab} (+{n_new})")

    state = ckpt["model_state"]
    expanded_keys = []

    # 用已有 special token embedding 的均值作为初始化基础
    for key, tensor in state.items():
        if tensor.dim() != 2:
            continue
        # 只扩展 vocab 维度的权重
        if tensor.shape[0] != old_vocab:
            continue

        dim = tensor.shape[1]
        # 用最后 4 个 token（旧 special tokens）的均值 + 小噪声初始化
        special_mean = tensor[-4:].float().mean(dim=0)
        new_rows = special_mean.unsqueeze(0).expand(n_new, -1) + torch.randn(n_new, dim) * 0.01
        new_rows = new_rows.to(dtype=tensor.dtype)
        state[key] = torch.cat([tensor, new_rows], dim=0)
        expanded_keys.append(f"  {key}: {old_vocab}×{dim} -> {target_vocab}×{dim}")

    print(f"扩展了 {len(expanded_keys)} 个权重:")
    for k in expanded_keys:
        print(k)

    # 更新 config
    if isinstance(config, dict):
        config["vocab_size"] = target_vocab
    else:
        config.vocab_size = target_vocab

    new_ckpt = {
        "model_state": state,
        "config": config,
        "val_bpb": ckpt.get("val_bpb"),
        "step": ckpt.get("step", 0),
        "expanded_from": str(src_path),
        "tool_tokens_added": TOOL_TOKENS,
    }
    torch.save(new_ckpt, dst_path)
    print(f"\n新 checkpoint 已保存: {dst_path}")
    print(f"vocab_size: {old_vocab} -> {target_vocab} (+{n_new})")


def main():
    parser = argparse.ArgumentParser(description="追加工具调用 special tokens")
    parser.add_argument("--checkpoint", required=True, help="输入 checkpoint 路径")
    parser.add_argument("--output", required=True, help="输出 checkpoint 路径")
    args = parser.parse_args()

    # Step 1: 修改 tokenizer
    old_vocab, target_vocab = update_tokenizer()

    # Step 2: 扩展 checkpoint（对比 checkpoint 的 vocab_size 和 tokenizer 的 vocab_size）
    # tokenizer 可能已更新（old_vocab == target_vocab），但 checkpoint 还没扩展
    with open(TOKENIZER_PKL, "rb") as f:
        enc = pickle.load(f)
    target_vocab = enc.n_vocab
    expand_checkpoint(args.checkpoint, args.output, target_vocab)

    # Step 3: 验证
    print("\n--- 验证 ---")
    with open(TOKENIZER_PKL, "rb") as f:
        enc = pickle.load(f)
    print(f"tokenizer.n_vocab = {enc.n_vocab}")
    for tok in TOOL_TOKENS:
        print(f"  {tok} -> ID {enc.encode_single_token(tok)}")

    if old_vocab != target_vocab:
        ckpt = torch.load(args.output, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        vs = cfg["vocab_size"] if isinstance(cfg, dict) else cfg.vocab_size
        print(f"checkpoint.vocab_size = {vs}")
        wte_shape = None
        for k, v in ckpt["model_state"].items():
            if "wte.weight" in k:
                wte_shape = v.shape
                break
        print(f"wte.weight shape = {wte_shape}")

    print("\n完成。下一步：短暂继续预训练让新 token embedding 收敛。")


if __name__ == "__main__":
    main()
