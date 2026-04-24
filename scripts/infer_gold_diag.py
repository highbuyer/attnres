#!/usr/bin/env python3
"""对 sft_v5_clean_cc_best.pt 跑几条 gold prompt 的实际推理，
对比模型生成 vs gold 期望，定位 fa_nll 高的根因。

选了 2 条 bottom（loss 低）+ 1 条 top（loss 高）。loss 依赖上下文的那两条
(#12/#24) 跳过——它们 fa_nll 高可能是 prompt 本身脱离会话上下文。
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path("/home/langshen/base_mode/attnres")
GOLD = ROOT / "data" / "cc_gold_trajectories_v2.jsonl"
CKPT = ROOT / "checkpoints" / "sft_v5_clean_cc_best.pt"

# 从诊断结果选的样本索引（0-based）
SELECTED = [
    (2,  "bottom-low-loss"),   # "./weiyan，怎么生成的..."  NLL 2.68
    (26, "bottom-low-loss"),   # "docs/PROJECT_STRUCTURE.md"  NLL 3.17
    (20, "top-high-loss"),     # "看一下sft.py"  NLL 5.09
]


def extract_gold_final(messages: list) -> str:
    """取 gold 的最后 assistant text（final answer）。"""
    return messages[-1]["content"]


def extract_gold_tool_call(messages: list) -> str:
    for m in messages:
        if "<|tool_call_start|>" in m.get("content", ""):
            return m["content"]
    return "(no tool_call)"


def main() -> None:
    golds = [json.loads(ln) for ln in GOLD.open()]
    print(f"Checkpoint: {CKPT.name}")
    print(f"Gold pool : {len(golds)} 条\n")

    for idx, tag in SELECTED:
        g = golds[idx]
        user = g["messages"][0]["content"]
        gold_tc = extract_gold_tool_call(g["messages"])
        gold_final = extract_gold_final(g["messages"])

        print(f"=== #{idx} [{tag}] ===")
        print(f"USER   : {user}")
        print(f"GOLD_TC: {gold_tc[:140]}")
        print(f"GOLD_FIN: {gold_final[:200]}{'...' if len(gold_final)>200 else ''}")
        print()
        print(f"> running infer ...", flush=True)

        cmd = [
            "uv", "run", "python", "src/infer.py",
            user,
            "512",          # max_tokens
            "0.2",          # temperature
            "--checkpoint", str(CKPT),
            "--tool-dir", str(ROOT),
            "--seed", "42",
        ]
        try:
            res = subprocess.run(
                cmd, cwd=str(ROOT), capture_output=True, text=True,
                timeout=180,
            )
        except subprocess.TimeoutExpired:
            print("  TIMEOUT 180s\n")
            continue
        out = res.stdout.strip()
        err = res.stderr.strip()
        print(f"MODEL  : {out}")
        if err:
            # 只打错误的末尾
            print(f"STDERR last 300: {err[-300:]}")
        print()
        print("-" * 70)
        print()


if __name__ == "__main__":
    main()
