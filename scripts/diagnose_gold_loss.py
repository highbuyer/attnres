#!/usr/bin/env python3
"""诊断：现有最强 cc SFT ckpt 在 27 条新 gold 上的 per-sample NLL。

每条 gold 可能被 sft.py 的 format_samples_split 拆成多个"assistant-turn 结束"
的子样本（tool_call 一个、final answer 一个）。本脚本：
1. 对每条 gold 跑一次 forward，收集每个子样本的 nll/perplexity
2. 把子样本按 "tool_call" / "final_answer" 归类
3. 汇总每条 gold 的 loss（按 mask-token 加权平均），排序
4. 打印 top-3 / bottom-3 样本 + 分位统计

用 sft_v5_clean_cc_best.pt 作为"现有最强 cc ckpt"。GPU forward only，无训练。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from dataclasses import fields

import torch

ROOT = Path("/home/langshen/base_mode/attnres")
sys.path.insert(0, str(ROOT / "src"))

# 借 sft.py 的所有打包逻辑
import sft as sft_mod  # noqa: E402

GOLD_PATH = ROOT / "data" / "cc_gold_trajectories_v2.jsonl"
CKPT_PATH = ROOT / "checkpoints" / "sft_v5_clean_cc_best.pt"


def classify_subsample(ids: list[int], mask: list[int]) -> str:
    """根据 ids 末尾（被 mask 的那段）判断这是哪种 assistant turn：
    tool_call: 内容含 <|tool_call_start|>
    final_answer: 最后一段 masked 内容不含 tool_call/tool_result 标记
    """
    sft_mod._ensure_tokenizer()
    # 找最后一段连续的 mask=1 区域
    end = len(ids)
    start = end
    for i in range(end - 1, -1, -1):
        if mask[i] == 1:
            start = i
        elif start < end:
            break
    tail_ids = ids[start:end]
    tail_text = sft_mod.enc.decode(tail_ids)
    if "<|tool_call_start|>" in tail_text:
        return "tool_call"
    return "final_answer"


@torch.no_grad()
def sample_nll(model, ids: list[int], mask: list[int], device) -> tuple[float, int]:
    """返回 (sum_nll, n_mask_tokens)。"""
    x = torch.tensor(ids[:-1], device=device).unsqueeze(0)
    y = torch.tensor(ids[1:], device=device).unsqueeze(0)
    m = torch.tensor(mask[1:], device=device, dtype=torch.float32)  # mask 对应 y 的位置
    loss_flat = model(x, y, reduction="none").view(-1)
    # loss_flat 长度 = y.numel() = len(ids)-1
    masked = loss_flat * m
    return masked.sum().item(), int(m.sum().item())


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert CKPT_PATH.exists(), f"{CKPT_PATH} 不存在"
    assert GOLD_PATH.exists(), f"{GOLD_PATH} 不存在"

    # 复用 sft.py 的模型加载（exec train.py 前半段，绕开 flash-attn 初始化）
    sft_mod._ensure_model_defs()
    GPT = sft_mod.GPT
    GPTConfig = sft_mod.GPTConfig

    print(f"Loading {CKPT_PATH.name}...")
    ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False, mmap=True)
    raw_cfg = ck["config"]
    if isinstance(raw_cfg, dict):
        allowed = {f.name for f in fields(GPTConfig)}
        raw_cfg = GPTConfig(**{k: v for k, v in raw_cfg.items() if k in allowed})
    model = GPT(raw_cfg).to(device=device, dtype=torch.bfloat16)
    if hasattr(model, "cos"):
        model.cos = model.cos.to(torch.bfloat16)
        model.sin = model.sin.to(torch.bfloat16)
    state = {k.replace("_orig_mod.", ""): v for k, v in ck["model_state"].items()}
    res = model.load_state_dict(state, strict=False)
    assert not res.missing_keys and not res.unexpected_keys, f"mk={res.missing_keys[:3]} uk={res.unexpected_keys[:3]}"
    model.eval()
    print(f"  n_layer={raw_cfg.n_layer} params={sum(p.numel() for p in model.parameters())/1e6:.1f}M  "
          f"ckpt.val_bpb={ck.get('val_bpb', 'n/a')}")

    # 加载 27 条 gold
    golds = [json.loads(ln) for ln in GOLD_PATH.open()]
    print(f"\n{len(golds)} 条 gold 加载完成。开始诊断...\n")

    # 每条 gold 展开为子样本
    per_gold = []  # list of (gold_idx, user_prompt_snippet, [(kind, sum_nll, n_tok), ...])
    for gi, g in enumerate(golds):
        user_prompt = g["messages"][0]["content"]
        split = sft_mod.format_samples_split(g["messages"])
        if not split:
            continue
        subs = []
        for ids, mask in split:
            kind = classify_subsample(ids, mask)
            sm, nt = sample_nll(model, ids, mask, device)
            if nt == 0:
                continue
            subs.append((kind, sm, nt))
        per_gold.append((gi, user_prompt, subs))

    # 聚合每条 gold 的整体 NLL = sum_nll / sum_tokens
    gold_stats = []
    for gi, up, subs in per_gold:
        tot_nll = sum(s[1] for s in subs)
        tot_n = sum(s[2] for s in subs)
        avg_nll = tot_nll / tot_n if tot_n else float("inf")
        # 分别聚合 tool_call 和 final_answer
        tc_nll = sum(s[1] for s in subs if s[0] == "tool_call")
        tc_n = sum(s[2] for s in subs if s[0] == "tool_call")
        fa_nll = sum(s[1] for s in subs if s[0] == "final_answer")
        fa_n = sum(s[2] for s in subs if s[0] == "final_answer")
        gold_stats.append({
            "idx": gi,
            "user": up,
            "avg_nll": avg_nll,
            "ppl": float(torch.exp(torch.tensor(avg_nll))),
            "n_tok": tot_n,
            "n_sub": len(subs),
            "tc_nll": tc_nll / tc_n if tc_n else None,
            "fa_nll": fa_nll / fa_n if fa_n else None,
            "tc_tokens": tc_n,
            "fa_tokens": fa_n,
        })

    gold_stats.sort(key=lambda r: r["avg_nll"])

    # 整体分布
    import statistics
    nll_vals = [s["avg_nll"] for s in gold_stats]
    print("=== per-gold NLL 分布 ===")
    print(f"  mean = {statistics.mean(nll_vals):.4f}")
    print(f"  median = {statistics.median(nll_vals):.4f}")
    print(f"  min/max = {min(nll_vals):.4f} / {max(nll_vals):.4f}")
    if len(nll_vals) >= 4:
        print(f"  p25/p75 = {statistics.quantiles(nll_vals, n=4)[0]:.4f} / "
              f"{statistics.quantiles(nll_vals, n=4)[2]:.4f}")

    # 分 tool_call vs final_answer
    tc_nlls = [s["tc_nll"] for s in gold_stats if s["tc_nll"] is not None]
    fa_nlls = [s["fa_nll"] for s in gold_stats if s["fa_nll"] is not None]
    if tc_nlls:
        print(f"\n  tool_call NLL       : mean={statistics.mean(tc_nlls):.4f}  "
              f"median={statistics.median(tc_nlls):.4f}  n={len(tc_nlls)}")
    if fa_nlls:
        print(f"  final_answer NLL    : mean={statistics.mean(fa_nlls):.4f}  "
              f"median={statistics.median(fa_nlls):.4f}  n={len(fa_nlls)}")

    print("\n=== bottom-3（loss 最低，模型最'懂'的）===")
    for s in gold_stats[:3]:
        up = s["user"][:60].replace("\n", " ")
        print(f"  #{s['idx']:2d}  NLL={s['avg_nll']:.4f}  PPL={s['ppl']:.2f}  "
              f"tc_nll={s['tc_nll']} fa_nll={s['fa_nll']}")
        print(f"         user: {up}")

    print("\n=== top-3（loss 最高，模型最'不懂'的）===")
    for s in gold_stats[-3:]:
        up = s["user"][:60].replace("\n", " ")
        print(f"  #{s['idx']:2d}  NLL={s['avg_nll']:.4f}  PPL={s['ppl']:.2f}  "
              f"tc_nll={s['tc_nll']} fa_nll={s['fa_nll']}")
        print(f"         user: {up}")


if __name__ == "__main__":
    main()
