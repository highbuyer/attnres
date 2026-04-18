#!/usr/bin/env python3
"""按配比混合多个已构建好的 SFT jsonl，可选地对每份数据做 thinking-trace 增强。

和 make_sft_data.py 的区别：
  - make_sft_data.py：原始语料 (Belle parquet / Claude 对话) → messages jsonl
  - 本脚本：已经是 messages 格式的 jsonl 之间做二次混合 + 过采样 + think 包装

用法：
  uv run python scripts/build_mixed_sft.py \\
      --out sft_mixed_think_v1.jsonl \\
      --source sft_tool_summary_v5.jsonl:weight=3:think=1 \\
      --source sft_tool_gated_v2.jsonl:weight=1:think=1 \\
      --source sft_samples_project_facts.jsonl:weight=5:think=0

source 语法： PATH[:weight=N][:think=0|1][:limit=N]
  - weight  过采样倍数（默认 1）
  - think   是否套 <think> trace（默认 0）
  - limit   先截断到前 N 条再过采样（默认全取）
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from add_thinking_traces import augment_sample  # type: ignore[import-not-found]  # noqa: E402


def parse_source(spec: str) -> dict:
    parts = spec.split(":")
    path = parts[0]
    opts = {"weight": 1, "think": 0, "limit": 0}
    for p in parts[1:]:
        if "=" not in p:
            raise ValueError(f"无法解析 source 选项 {p!r}，应为 key=val 形式")
        k, v = p.split("=", 1)
        if k not in opts:
            raise ValueError(f"未知选项 {k!r}（支持 weight/think/limit）")
        opts[k] = int(v)
    return {"path": path, **opts}


def load_jsonl(path: Path) -> list:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[skip] {path.name}:{line_no}: {exc}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True,
                    help="PATH[:weight=N][:think=0|1][:limit=N]，可重复")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="只打印统计，不写盘")
    args = ap.parse_args()

    random.seed(args.seed)

    bucket: list = []
    stats: list[tuple[str, int, int, int, int]] = []

    for spec in args.source:
        cfg = parse_source(spec)
        path = Path(cfg["path"])
        if not path.exists():
            raise SystemExit(f"源文件不存在: {path}")
        raw = load_jsonl(path)
        n_raw = len(raw)
        if cfg["limit"] and cfg["limit"] < n_raw:
            raw = raw[: cfg["limit"]]
        if cfg["think"]:
            raw = [augment_sample(s) for s in raw]
        upsampled = raw * cfg["weight"] if cfg["weight"] > 1 else list(raw)
        bucket.extend(upsampled)
        stats.append((str(path), n_raw, len(raw), cfg["weight"], len(upsampled)))
        print(f"  {path.name}: raw={n_raw} after_limit={len(raw)} weight={cfg['weight']} think={cfg['think']} → {len(upsampled)}")

    random.shuffle(bucket)
    print(f"\n混合后总样本数: {len(bucket):,}")

    if args.dry_run:
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for s in bucket:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"已写入 {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
