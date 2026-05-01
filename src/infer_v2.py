#!/usr/bin/env python3
"""v2 架构（MLA+SwiGLU+砍VE+untied lm_head）的推理入口。

仅支持从 src/model_v2.py 的 GPT_v2 训出来的 ckpt。
不接工具调用 / 不做硬拦截 —— pretrain 模型跑 generation 看是否会"说人话"用。

用法:
  .venv/bin/python -u src/infer_v2.py "中国的首都是" --max-tokens 80 --temperature 0.7
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import torch.nn.functional as F

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))

from model_v2 import GPT_v2, GPTConfigV2
from prepare import Tokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_compile_prefix(sd: dict) -> dict:
    """Remove `_orig_mod.` prefix left by torch.compile during training."""
    out = {}
    for k, v in sd.items():
        nk = k.replace("._orig_mod.", ".")
        out[nk] = v
    return out


def load_model(ckpt_path: str, device: torch.device, dtype: torch.dtype = torch.bfloat16):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ck["config"]
    # 只保留 GPTConfigV2 已知字段，防止训练时多塞的 key 把 dataclass 砸了
    valid = {f.name for f in GPTConfigV2.__dataclass_fields__.values()}
    cfg = GPTConfigV2(**{k: v for k, v in cfg_dict.items() if k in valid})
    print(f"[infer_v2] config: layer={cfg.n_layer} d={cfg.n_embd} head={cfg.n_head} "
          f"kv_lora={cfg.kv_lora_rank} tie_lm_head={cfg.tie_lm_head} vocab={cfg.vocab_size}")
    val_bpb = ck.get("val_bpb")
    val_bpb_str = f"{val_bpb:.4f}" if isinstance(val_bpb, (int, float)) else "n/a"
    extra = " sft_v2" if ck.get("sft_v2") else ""
    print(f"[infer_v2] ckpt val_bpb={val_bpb_str} step={ck.get('step')}{extra}")

    model = GPT_v2(cfg)
    sd = _strip_compile_prefix(ck["model_state"])
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[infer_v2] missing keys ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"[infer_v2] unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")

    model.to(device=device, dtype=dtype)
    model.eval()
    return model, cfg


@torch.no_grad()
def generate(
    model: GPT_v2,
    cfg: GPTConfigV2,
    tokenizer: Tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    rep_penalty: float = 1.1,
    device: torch.device | None = None,
    chat_mode: bool = False,
    stop_on_eos: bool = True,
) -> str:
    device = device or next(model.parameters()).device
    enc = tokenizer.enc
    bos = tokenizer.get_bos_token_id()
    if chat_mode:
        # [BOS] + [USER_ID] + prompt + [ASST_ID]，让模型从 ASST 起续写
        user_id = enc.encode_single_token("<|reserved_1|>")
        asst_id = enc.encode_single_token("<|reserved_2|>")
        ids = [bos, user_id] + tokenizer.encode(prompt) + [asst_id]
        eos_id = enc.encode_single_token("<|reserved_3|>")
    else:
        ids = [bos] + tokenizer.encode(prompt)
        eos_id = enc.encode_single_token("<|reserved_3|>") if stop_on_eos else None
    x = torch.tensor([ids], dtype=torch.long, device=device)

    # prefill
    logits, past_kvs = model(x, use_cache=True)
    next_logits = logits[:, -1, :]

    generated = []
    seen_ids = set(ids)
    rope_cap = cfg.sequence_len * cfg.rope_seq_len_mult

    for step in range(max_tokens):
        # repetition penalty
        if rep_penalty > 1.0 and seen_ids:
            idx = torch.tensor(sorted(seen_ids), device=device, dtype=torch.long)
            penalized = next_logits[0, idx]
            penalized = torch.where(penalized > 0, penalized / rep_penalty, penalized * rep_penalty)
            next_logits[0, idx] = penalized

        # temperature
        if temperature <= 0:
            next_id = int(next_logits.argmax(dim=-1).item())
        else:
            scaled = next_logits / temperature
            # top-k
            if top_k and top_k > 0:
                v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                scaled = torch.where(scaled < v[:, -1:], torch.full_like(scaled, -float("inf")), scaled)
            # top-p
            if 0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cum = probs.cumsum(dim=-1)
                mask = cum > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                sorted_logits = sorted_logits.masked_fill(mask, -float("inf"))
                scaled = torch.full_like(scaled, -float("inf")).scatter(-1, sorted_idx, sorted_logits)
            probs = F.softmax(scaled, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())

        generated.append(next_id)
        seen_ids.add(next_id)

        # 在 chat_mode 下遇 EOS 立刻停
        if chat_mode and eos_id is not None and next_id == eos_id:
            break

        cur_len = x.size(1) + len(generated)
        if cur_len >= rope_cap:
            print(f"[infer_v2] reached rope cap {rope_cap}, stop")
            break

        next_tok = torch.tensor([[next_id]], device=device, dtype=torch.long)
        logits, past_kvs = model(
            next_tok,
            past_kvs=past_kvs,
            use_cache=True,
            position_offset=cur_len - 1,
        )
        next_logits = logits[:, -1, :]

    return tokenizer.decode(generated)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference for v2 (MLA) checkpoints")
    p.add_argument("prompt", nargs="?", default="你好，", help="Prompt text")
    p.add_argument("--checkpoint", default="checkpoints/d36_v2_mla_best.pt", help="Path to v2 ckpt")
    p.add_argument("--max-tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--rep-penalty", type=float, default=1.1)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--multi", action="store_true", help="跑一组内置 smoke prompts，用来快速看模型质量")
    p.add_argument("--chat", action="store_true", help="chat 模式：用 BOS+USER+prompt+ASST 包装")
    return p.parse_args()


SMOKE_PROMPTS = [
    "中国的首都是",
    "今天天气真好，",
    "Once upon a time, there was a",
    "def fibonacci(n):",
    "1+1=",
    "床前明月光，",
    "What is the capital of France?",
    "你好",
]

CHAT_PROMPTS = [
    "你好",
    "中国的首都在哪里？",
    "解释一下什么是斐波那契数列",
    "项目里 src/sft_v2.py 的入口函数在哪？",
    "帮我搜一下代码里 GPTConfigV2 的定义在哪个文件",
    "读一下 README.md 的前 20 行",
]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[infer_v2] WARN: 没检测到 CUDA，FA3 在 CPU 上不可用，可能报错")

    tokenizer = Tokenizer.from_directory()
    model, cfg = load_model(args.checkpoint, device)

    if args.chat:
        prompts = CHAT_PROMPTS
    elif args.multi:
        prompts = SMOKE_PROMPTS
    else:
        prompts = [args.prompt]
    for i, p in enumerate(prompts, 1):
        print(f"\n=== [{i}/{len(prompts)}] prompt: {p!r} (chat={args.chat}) ===")
        out = generate(
            model, cfg, tokenizer, p,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            rep_penalty=args.rep_penalty,
            device=device,
            chat_mode=args.chat,
        )
        prefix = "[user] " + p + "\n[assistant] " if args.chat else p
        print(f">>> {prefix}{out}")


if __name__ == "__main__":
    main()
