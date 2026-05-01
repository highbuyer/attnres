#!/usr/bin/env python3
"""v2 (MLA) ckpt 上的 SFT smoke 入口。

- 加载 src/model_v2.py 的 GPT_v2
- 数据格式: {"messages": [{"role":"user|assistant|system","content":...}, ...]}
- chatml 约定（与 src/sft.py 一致）:
    BOS=<|reserved_0|> USER=<|reserved_1|> ASST=<|reserved_2|> EOS=<|reserved_3|>
- loss mask: 仅 assistant 的内容 + 终止 EOS 计 loss；user/ASST_ID/system/EOS 之外的填充均忽略 (-1)

用法:
  .venv/bin/python -u src/sft_v2.py \\
    --ckpt checkpoints/d36_v2_mla_best.pt \\
    --data datasets/sft_archive/sft_mixed_v8.jsonl \\
    --max-samples 1000 --steps 100 --bsz 1 --grad-accum 8 --lr 5e-5 \\
    --out checkpoints/d36_v2_sft_smoke.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))

from model_v2 import GPT_v2, GPTConfigV2
from prepare import Tokenizer
from sft_format import included_turn_indices
from sft_format_v2 import render_messages_to_samples

IGNORE_INDEX = -1  # v2.forward 用 ignore_index=-1

SYSTEM_PROMPT = (
    "你是微研，一个技术助手。用与用户相同的语言简洁回答。"
    "不确定时如实说明，不编造事实。拒绝有害内容。"
)


# ---------------------------------------------------------------------------
# Tokenizer 与特殊 token
# ---------------------------------------------------------------------------

class ChatFormat:
    def __init__(self, tokenizer: Tokenizer):
        self.tok = tokenizer
        enc = tokenizer.enc
        self.BOS = enc.encode_single_token("<|reserved_0|>")
        self.USER = enc.encode_single_token("<|reserved_1|>")
        self.ASST = enc.encode_single_token("<|reserved_2|>")
        self.EOS = enc.encode_single_token("<|reserved_3|>")

    def encode_turn(self, role: str, content: str):
        ids = self.tok.encode(content)
        if role in ("user", "tool"):
            return [self.USER] + ids, [0] * (1 + len(ids))
        if role == "assistant":
            return [self.ASST] + ids, [0] + [1] * len(ids)
        # system 不应该走到这里，外层会 inline 进 user
        return [self.USER] + ids, [0] * (1 + len(ids))

    def format_sample(self, messages: list[dict], max_len: int):
        """一条 messages → 多个 (ids, mask) 切片，每个 assistant 一个"""
        if not messages:
            return []
        msgs = list(messages)
        # system → 拼进第一条 user
        if msgs and msgs[0]["role"] == "system":
            sys_c = msgs[0]["content"]
            msgs = msgs[1:]
            for i, m in enumerate(msgs):
                if m["role"] == "user":
                    msgs[i] = {"role": "user", "content": sys_c + "\n" + m["content"]}
                    break
            else:
                return []
        if not msgs or msgs[0]["role"] != "user":
            return []
        msgs[0] = {"role": "user", "content": SYSTEM_PROMPT + "\n" + msgs[0]["content"]}

        turn_ids, turn_masks = [], []
        for m in msgs:
            ids, mask = self.encode_turn(m["role"], m.get("content", ""))
            turn_ids.append(ids)
            turn_masks.append(mask)

        out = []
        for idx, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue
            ids = [self.BOS]
            mask = [0]
            for ti in included_turn_indices(msgs, idx):
                ids.extend(turn_ids[ti])
                mask.extend(turn_masks[ti])
            ids.append(self.EOS)
            mask.append(1)
            if len(ids) > max_len:
                # 简单从右尾截断，左侧裁
                ids = [self.BOS] + ids[-(max_len - 1):]
                mask = [0] + mask[-(max_len - 1):]
            if sum(mask) == 0:
                continue
            out.append((ids, mask))
        return out


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_chatml_samples(path: str, fmt: ChatFormat, max_len: int, max_samples: int | None = None,
                        use_v2_renderer: bool = False, tokenizer: Tokenizer | None = None):
    """v2 渲染（use_v2_renderer=True）：支持 tool_calls / reasoning_content / tool 角色。
    legacy 渲染（False）：仅 user/assistant 文本（旧 chatml 数据）。
    """
    samples = []
    n_lines = 0
    n_emitted_per_session = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = obj.get("messages")
            if not msgs:
                continue
            n_lines += 1
            if use_v2_renderer:
                assert tokenizer is not None
                cuts = render_messages_to_samples(
                    msgs, tokenizer, max_len,
                    bos_id=fmt.BOS, user_id=fmt.USER, asst_id=fmt.ASST, eos_id=fmt.EOS,
                )
            else:
                cuts = fmt.format_sample(msgs, max_len)
            n_emitted_per_session.append(len(cuts))
            for s in cuts:
                samples.append(s)
                if max_samples and len(samples) >= max_samples:
                    return samples
    return samples


def collate(batch, max_len: int):
    """pad 到 batch 内最长，targets 对 mask=0 的位置填 IGNORE_INDEX"""
    n = len(batch)
    L = min(max(len(ids) for ids, _ in batch), max_len)
    x = torch.full((n, L - 1), 0, dtype=torch.long)
    y = torch.full((n, L - 1), IGNORE_INDEX, dtype=torch.long)
    for i, (ids, mask) in enumerate(batch):
        ids = ids[:L]
        mask = mask[:L]
        T = len(ids)
        x[i, :T - 1] = torch.tensor(ids[:-1], dtype=torch.long)
        # y[t] = ids[t+1] if mask[t+1]==1 else IGNORE
        target_ids = torch.tensor(ids[1:], dtype=torch.long)
        target_mask = torch.tensor(mask[1:], dtype=torch.bool)
        y[i, :T - 1] = torch.where(target_mask, target_ids, torch.full_like(target_ids, IGNORE_INDEX))
    return x, y


# ---------------------------------------------------------------------------
# Model load
# ---------------------------------------------------------------------------

def _strip_compile_prefix(sd: dict) -> dict:
    return {k.replace("._orig_mod.", "."): v for k, v in sd.items()}


def load_v2_model(ckpt_path: str, device, dtype):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    valid = {f.name for f in GPTConfigV2.__dataclass_fields__.values()}
    cfg = GPTConfigV2(**{k: v for k, v in ck["config"].items() if k in valid})
    print(f"[sft_v2] base ckpt: val_bpb={ck.get('val_bpb'):.4f} step={ck.get('step')} "
          f"layer={cfg.n_layer} d={cfg.n_embd}")
    model = GPT_v2(cfg)
    sd = _strip_compile_prefix(ck["model_state"])
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[sft_v2] missing keys: {len(missing)} (例: {missing[:3]})")
    if unexpected:
        print(f"[sft_v2] unexpected keys: {len(unexpected)} (例: {unexpected[:3]})")
    model.to(device=device, dtype=dtype)
    return model, cfg


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------

def cosine_lr(step: int, total: int, lr0: float, lr_min: float, warmup: int = 0):
    if warmup > 0 and step < warmup:
        return lr0 * (step + 1) / warmup
    if step >= total:
        return lr_min
    progress = (step - warmup) / max(1, total - warmup)
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return lr_min + (lr0 - lr_min) * cos


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/d36_v2_mla_best.pt")
    p.add_argument("--data", default="datasets/sft_archive/sft_mixed_v8.jsonl")
    p.add_argument("--data-claude", default=None,
                   help="Qwen 风格的 claude 轨迹 jsonl（含 tool_calls 字段），与 --data 混合")
    p.add_argument("--mix-ratio-claude", type=float, default=0.5,
                   help="混合比例：claude 轨迹样本 / (claude + mixed) ∈ [0,1]")
    p.add_argument("--out", default="checkpoints/d36_v2_sft_smoke.pt")
    p.add_argument("--max-samples", type=int, default=1000, help="上限样本数（pre-切片后的对话数）")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--bsz", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lr-min-frac", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=50, help="linear warmup 步数（防止 SFT 起点梯度炸优化器）")
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=0,
                   help="每 N 步保存一次中间 checkpoint（0=只在最后保存）")
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    print(f"[sft_v2] loading tokenizer ...")
    tokenizer = Tokenizer.from_directory()
    fmt = ChatFormat(tokenizer)
    print(f"[sft_v2] BOS={fmt.BOS} USER={fmt.USER} ASST={fmt.ASST} EOS={fmt.EOS}")

    print(f"[sft_v2] loading data: {args.data} (max {args.max_samples} samples)")
    t0 = time.time()
    samples_mixed = load_chatml_samples(args.data, fmt, args.max_len, max_samples=args.max_samples)
    print(f"[sft_v2] loaded {len(samples_mixed)} mixed (legacy chatml) slices in {time.time()-t0:.1f}s")

    samples_claude = []
    if args.data_claude:
        t0 = time.time()
        samples_claude = load_chatml_samples(
            args.data_claude, fmt, args.max_len,
            max_samples=None, use_v2_renderer=True, tokenizer=tokenizer,
        )
        print(f"[sft_v2] loaded {len(samples_claude)} claude-traj slices in {time.time()-t0:.1f}s")

    # 按比例混合（claude-first：先按 ratio 计算 claude 目标数，cap 到全量；
    # 缺口由 mixed 补足，使最终 claude 占比 ≥ 目标比例）
    if samples_claude and samples_mixed:
        target_n = args.max_samples or (len(samples_mixed) + len(samples_claude))
        # 优先使全量 claude 都能被纳入：若 ratio×target_n > len(claude)，按 claude 全量反推 target
        n_claude = min(int(target_n * args.mix_ratio_claude), len(samples_claude))
        if args.mix_ratio_claude > 0 and n_claude < len(samples_claude):
            # 数据池足够大，直接按 ratio 抽
            pass
        else:
            # claude 不足以填到 ratio：以 claude 全量为锚，按 ratio 反推 target_n
            n_claude = len(samples_claude)
            target_n = min(target_n, int(n_claude / max(args.mix_ratio_claude, 1e-6)))
        n_mixed = min(target_n - n_claude, len(samples_mixed))
        random.shuffle(samples_claude)
        random.shuffle(samples_mixed)
        samples = samples_claude[:n_claude] + samples_mixed[:n_mixed]
        ratio = n_claude / max(1, len(samples))
        print(f"[sft_v2] mixed pool: claude={n_claude} + mixed={n_mixed} = {len(samples)} (claude_ratio={ratio:.2f})")
    elif samples_claude:
        samples = samples_claude
    else:
        samples = samples_mixed
    if not samples:
        raise SystemExit("no samples loaded")
    random.shuffle(samples)

    model, _cfg = load_v2_model(args.ckpt, device, torch.bfloat16)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[sft_v2] params: total={n_params/1e6:.1f}M trainable={n_trainable/1e6:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    optimizer.zero_grad()

    cursor = 0
    last_loss = None
    losses_window = []
    t_train = time.time()
    for step in range(1, args.steps + 1):
        lr = cosine_lr(step - 1, args.steps, args.lr, args.lr * args.lr_min_frac, warmup=args.warmup_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr

        accum_loss = 0.0
        accum_tokens = 0
        for _ in range(args.grad_accum):
            batch = []
            for _ in range(args.bsz):
                if cursor >= len(samples):
                    random.shuffle(samples)
                    cursor = 0
                batch.append(samples[cursor])
                cursor += 1
            x, y = collate(batch, args.max_len)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            n_active = int((y != IGNORE_INDEX).sum().item())
            if n_active == 0:
                continue
            loss = model(x, y, reduction="mean") / args.grad_accum
            loss.backward()
            accum_loss += float(loss.item()) * args.grad_accum
            accum_tokens += n_active

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        avg_loss = accum_loss / max(1, args.grad_accum)
        last_loss = avg_loss
        losses_window.append(avg_loss)
        if len(losses_window) > 20:
            losses_window.pop(0)

        if step % args.log_every == 0 or step == 1:
            recent = sum(losses_window) / len(losses_window)
            print(f"step {step:4d}/{args.steps} | loss {avg_loss:.4f} (avg20 {recent:.4f}) "
                  f"| lr {lr:.2e} | active_tok {accum_tokens}")

        if args.save_every > 0 and step % args.save_every == 0 and not args.no_save:
            ckpt_path = args.out.replace(".pt", f"_step{step}.pt")
            torch.save({
                "model_state": model.state_dict(),
                "config": {f.name: getattr(_cfg, f.name) for f in GPTConfigV2.__dataclass_fields__.values()},
                "from_scratch_v2": True, "sft_v2": True,
                "base_ckpt": args.ckpt, "data": args.data,
                "step": step, "steps": args.steps,
                "lr": lr, "last_loss": avg_loss,
            }, ckpt_path)
            print(f"[sft_v2] saved periodic ckpt {ckpt_path}")

    dt = time.time() - t_train
    print(f"\n[sft_v2] done in {dt:.1f}s, last loss={last_loss:.4f}")

    if not args.no_save:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "config": {f.name: getattr(_cfg, f.name) for f in GPTConfigV2.__dataclass_fields__.values()},
            "from_scratch_v2": True,
            "sft_v2": True,
            "base_ckpt": args.ckpt,
            "data": args.data,
            "steps": args.steps,
            "lr": args.lr,
            "last_loss": last_loss,
        }, args.out)
        print(f"[sft_v2] saved {args.out}")


if __name__ == "__main__":
    main()
