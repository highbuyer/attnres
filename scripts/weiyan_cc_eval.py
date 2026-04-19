#!/usr/bin/env python3
"""Claude Code 风格评测 weiyan-api：真实工具调用场景，非基础常识。

测试分 4 层：
  L1 定位类：单次 search/read 即可
  L2 链式：先搜后读，多步
  L3 调试场景：给 error，问怎么修
  L4 跨文件推理：综合判断

每题有人类 ground truth，模型输出用规则 + 人工判断综合打分。"""
from __future__ import annotations

import json
import time
import urllib.request


TESTS = [
    # L1: 单次工具调用
    {"id": "L1_1", "level": "L1", "q": "src/train.py 里 has_ve 函数在哪一行？skip 参数是做什么的？",
     "expect": "123 行附近；skip 是要排除 VE 的层索引列表（ve_layer_skip）"},
    {"id": "L1_2", "level": "L1", "q": "scripts/ 目录下和 VE 相关的脚本有哪些？",
     "expect": "ve_ablation.py, prune_ve.py, verify_pruned_ckpt.py"},

    # L2: 链式调用
    {"id": "L2_1", "level": "L2", "q": "src/infer.py 里 INFER_MAX_CONTEXT 常量的值是多少？",
     "expect": "8192"},
    {"id": "L2_2", "level": "L2", "q": "scripts/prune_ve.py 里会删除 state_dict 中的哪些 key？",
     "expect": "value_embeds.{13,15,17}.weight 和 transformer.h.{13,15,17}.attn.ve_gate.weight，共 6 个"},
    {"id": "L2_3", "level": "L2", "q": "docs/VE_ABLATION_v2_full.md 里 knock_weakest3 的 Δval_bpt 是多少？",
     "expect": "0.0719"},

    # L3: 调试场景
    {"id": "L3_1", "level": "L3", "q": "我跑 scripts/bench_throughput.py 报 'FlashAttention only supports fp16, bf16, and fp8_e4m3 data type'，怎么修？",
     "expect": "forward 外层套 torch.autocast(bf16) 或用 sft_mod.autocast_context；参数默认 fp32 但前向要 bf16 激活"},
    {"id": "L3_2", "level": "L3", "q": "load_state_dict 报 unexpected key 'value_embeds.13.weight'，原因？",
     "expect": "ckpt 有这层但 model 未创建；要么在 GPTConfig 设 ve_layer_skip 排除，要么用 strict=False 忽略"},

    # L4: 跨文件推理
    {"id": "L4_1", "level": "L4", "q": "v5_pruned 相比 v5_best 少了多少参数？是通过什么方式减少的？",
     "expect": "少 75.51M (~18.7%)，404.34M → 328.82M，物理删除 13/15/17 三层的 value_embeds + ve_gate"},
]


def call_weiyan(prompt: str, timeout: int = 180) -> tuple[str, float]:
    url = "http://127.0.0.1:8000/ask"
    data = json.dumps({"prompt": prompt}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
    dt = time.time() - t0
    return body.get("response", ""), dt


def main():
    out_path = "runs/weiyan_cc_eval.jsonl"
    import os
    os.makedirs("runs", exist_ok=True)
    results = []
    with open(out_path, "w") as fout:
        for t in TESTS:
            print(f"\n===== {t['id']} [{t['level']}] =====")
            print(f"Q: {t['q']}")
            print(f"期望: {t['expect']}")
            try:
                resp, dt = call_weiyan(t["q"])
                print(f"\n答（{dt:.1f}s）:\n{resp}")
                rec = {**t, "response": resp, "latency_s": round(dt, 2), "error": None}
            except Exception as e:
                print(f"\nERROR: {e}")
                rec = {**t, "response": None, "latency_s": None, "error": str(e)}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            results.append(rec)
            print()
    print(f"\n{'='*60}\n总 {len(results)} 题，输出 {out_path}")


if __name__ == "__main__":
    main()
