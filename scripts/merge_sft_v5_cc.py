#!/usr/bin/env python3
"""合并 v5_clean + cc_gold_v1 + cc_synth_v2 → sft_v5_clean_cc.jsonl。

简单 concat + shuffle（固定 seed），保持单样本 messages schema 不变。
训练时用 --data data/sft_v5_clean_cc.jsonl 即可。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="data/sft_tool_summary_v5_clean.jsonl")
    parser.add_argument("--gold", default="data/cc_gold_trajectories_v1.jsonl")
    parser.add_argument("--synth", default="data/cc_synth_v2.jsonl")
    parser.add_argument("--out", default="data/sft_v5_clean_cc.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cc-oversample", type=int, default=1,
                        help="cc 样本重复次数（1=不过采样）")
    args = parser.parse_args()

    base = load_jsonl(ROOT / args.base)
    gold = load_jsonl(ROOT / args.gold)
    synth = load_jsonl(ROOT / args.synth)

    cc = gold + synth
    merged = list(base) + cc * args.cc_oversample

    random.seed(args.seed)
    random.shuffle(merged)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in merged:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"base  : {len(base):>5d}  ({args.base})")
    print(f"gold  : {len(gold):>5d}  ({args.gold})")
    print(f"synth : {len(synth):>5d}  ({args.synth})")
    print(f"cc×{args.cc_oversample}  : {len(cc) * args.cc_oversample:>5d}  (占比 {len(cc)*args.cc_oversample/len(merged)*100:.2f}%)")
    print(f"total : {len(merged):>5d} → {out_path}")


if __name__ == "__main__":
    main()
