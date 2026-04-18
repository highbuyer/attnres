#!/usr/bin/env python3
"""用 DeepSeek V3 批量改写 sft_tool_summary_v5.jsonl 里错标成拒答模板的训练样本。

输入：runs/refusal_templates_audit.jsonl（128 条，heuristic_benign=true 的 101 条需改写）
输出：runs/refusal_rewrites.jsonl（每条带 original + rewritten + metadata）

要求 DeepSeek 对每个 user_q 生成：
  - 准确、直接的中文回答
  - 不拒答、不说"我不确定"
  - 若 prompt 本身模糊（缺少必要信息），在答案里补一个合理假设并答出来
  - 长度控制：100-250 字

断点续传：每处理一条就 flush 一行，已存在的 line_idx 跳过重跑。
并发：5 个 worker（DeepSeek 允许高并发）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


REWRITE_SYSTEM = """你是严谨的中文知识回答助手。用户会给你一个问题，你必须直接、准确地回答。

规则：
1. 不要拒答。不要说"我不确定"、"无法确认"、"超出知识范围"、"建议查阅资料"。
2. 回答要基于公认事实，不编造细节。数字、人名、地名要可靠；不确定的细节可以笼统描述。
3. 若问题本身模糊，在回答开头一句补齐合理假设（如"假设是标准科目..."）再答。
4. 代码题给出可运行的最简实现，用 ```python 代码块包裹。
5. 长度 80-250 字；代码题可略长。
6. 直接答题，不要开场白（不要说"好的"、"当然可以"等）。
7. 中文回答。"""


def call_deepseek(client, user_q: str, model: str, max_retry: int = 3) -> tuple[str, dict]:
    """调 DeepSeek，返回 (rewritten_answer, metadata)"""
    import openai
    for attempt in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": REWRITE_SYSTEM},
                    {"role": "user", "content": user_q},
                ],
                temperature=0.3,
                max_tokens=400,
                timeout=60,
            )
            content = resp.choices[0].message.content or ""
            meta = {
                "model": model,
                "tokens_in": resp.usage.prompt_tokens if resp.usage else None,
                "tokens_out": resp.usage.completion_tokens if resp.usage else None,
                "attempt": attempt + 1,
            }
            return content.strip(), meta
        except (openai.RateLimitError, openai.APITimeoutError) as e:
            if attempt == max_retry - 1:
                raise
            time.sleep(2 ** attempt)
        except Exception as e:
            if attempt == max_retry - 1:
                raise
            time.sleep(1)
    raise RuntimeError("unreachable")


def load_done_indices(out_path: Path) -> set[int]:
    if not out_path.exists():
        return set()
    done = set()
    with open(out_path) as f:
        for line in f:
            try:
                done.add(json.loads(line)["line_idx"])
            except Exception:
                continue
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", default="runs/refusal_templates_audit.jsonl")
    parser.add_argument("--out", default="runs/refusal_rewrites.jsonl")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--only-benign", action="store_true", default=True,
                        help="只改写 heuristic_benign=true 的 101 条（默认）")
    parser.add_argument("--limit", type=int, default=None,
                        help="最多处理多少条（smoke test 用）")
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY env var not set", file=sys.stderr)
        sys.exit(1)

    import openai
    client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    # 加载任务
    records = []
    with open(args.audit) as f:
        for line in f:
            r = json.loads(line)
            if args.only_benign and not r.get("heuristic_benign"):
                continue
            records.append(r)

    # 断点续传
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_indices(out_path)
    todo = [r for r in records if r["line_idx"] not in done]
    if args.limit:
        todo = todo[: args.limit]

    print(f"待改写 {len(todo)} 条（总 benign={len(records)}，已完成 {len(done)}），model={args.model}, workers={args.workers}")

    t0 = time.time()
    n_ok = 0
    n_err = 0

    def work(rec):
        try:
            ans, meta = call_deepseek(client, rec["user_q"], args.model)
            return rec, ans, meta, None
        except Exception as e:
            return rec, None, None, str(e)

    with open(out_path, "a") as fout, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(work, r) for r in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            rec, ans, meta, err = fut.result()
            if err:
                n_err += 1
                print(f"  [{i}/{len(todo)}] line_idx={rec['line_idx']} ERR: {err}")
                continue
            out_rec = {
                "line_idx": rec["line_idx"],
                "user_q": rec["user_q"],
                "original_refuse_a": rec["assistant_a"],
                "template_hit": rec["template_hit"],
                "rewritten_a": ans,
                "meta": meta,
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_ok += 1
            if i <= 3 or i % 20 == 0:
                q_short = rec["user_q"][:40].replace("\n", " ")
                a_short = ans[:60].replace("\n", " ")
                print(f"  [{i}/{len(todo)}] ok  Q: {q_short}  →  {a_short}")

    dt = time.time() - t0
    print(f"\ndone  ok={n_ok}  err={n_err}  time={dt:.1f}s")


if __name__ == "__main__":
    main()
