#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from dataclasses import fields
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import infer  # noqa: E402
from inference_rules import apply_hard_rules  # noqa: E402
from prepare import SPECIAL_TOKENS, Tokenizer  # noqa: E402
from tool_protocol import parse_tool_call, strip_tool_markup  # noqa: E402


PROMPTS = [
    "src/infer.py 里有没有 --tool-dir 参数？",
    "src/infer.py 里有没有 --no-tools 参数？",
    "src/infer.py 里 rep-penalty 参数在哪？",
    "请读取 src/infer.py 里 parse_args 附近的代码。",
    "请读取 docs/RUN_NEXT.md 里构建 SFT 数据那一段。",
    "scripts/make_sft_data.py 里有没有 tool-call-upsample 参数？",
    "当前仓库里 parse_tool_call 是在哪里实现的？",
    "当前仓库里 validate_tool_sample 是在哪里实现的？",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估模型是否输出工具调用格式")
    parser.add_argument("--checkpoint", default="checkpoints/sft_toolcall_v2.pt")
    parser.add_argument("--tool-dir", default=str(ROOT))
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--rep-penalty", type=float, default=1.3)
    parser.add_argument("--out", default="eval_tool_format.json")
    return parser.parse_args()


def load_model(checkpoint_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_config = ckpt["config"]
    if isinstance(raw_config, dict):
        allowed = {f.name for f in fields(infer.GPTConfig)}
        config = infer.GPTConfig(**{k: v for k, v in raw_config.items() if k in allowed})
    else:
        config = raw_config

    use_bf16 = device == "cuda"
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    model = infer.GPT(config).to(device=device, dtype=dtype)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state"].items()}
    model.load_state_dict(state, strict=False)
    model.to(dtype=dtype)
    model.eval()

    head_dim = config.n_embd // config.n_head
    cos, sin = model._precompute_rotary_embeddings(model.rotary_seq_len, head_dim, device=device)
    model.cos, model.sin = cos.to(dtype), sin.to(dtype)
    return model, config, device


def generate_raw(model, config, tokenizer, device, prompt: str, args: argparse.Namespace) -> str:
    hard_rule_answer = apply_hard_rules(prompt)
    if hard_rule_answer:
        return hard_rule_answer

    enc = tokenizer.enc
    bos_id = tokenizer.get_bos_token_id()
    user_id = enc.encode_single_token("<|reserved_1|>")
    asst_id = enc.encode_single_token("<|reserved_2|>")
    stop_ids = {enc.encode_single_token(token) for token in ["<|reserved_0|>", "<|reserved_1|>", "<|reserved_2|>", "<|reserved_3|>"]}
    system_prompt = "你是微研，一个技术助手。用与用户相同的语言简洁回答。不确定时如实说明，不编造事实。拒绝有害内容。"
    system_ids = tokenizer.encode(system_prompt + "\n")

    x = torch.tensor([[bos_id, user_id, *system_ids, *tokenizer.encode(prompt), asst_id]], dtype=torch.long, device=device)
    generated_ids: list[int] = []
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()

    for _ in range(args.max_tokens):
        with torch.no_grad(), autocast_ctx:
            logits = model(x[:, -config.sequence_len:])
            logits = logits[:, -1, :]

        if args.rep_penalty != 1.0 and generated_ids:
            for token_id in set(generated_ids):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= args.rep_penalty
                else:
                    logits[0, token_id] *= args.rep_penalty

        if args.temperature <= 0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / args.temperature
            filtered = infer.apply_top_k_top_p(logits, args.top_k, args.top_p)
            probs = torch.softmax(filtered, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        token_id = int(next_id.item())
        if token_id in stop_ids:
            break
        generated_ids.append(token_id)
        x = torch.cat([x, next_id], dim=1)

    return tokenizer.decode(generated_ids).strip()


def main() -> None:
    args = parse_args()
    model, config, device = load_model(args.checkpoint)
    tokenizer = Tokenizer.from_directory()

    results = []
    hits = 0
    for prompt in PROMPTS:
        raw = generate_raw(model, config, tokenizer, device, prompt, args)
        parsed = parse_tool_call(raw)
        has_tool_call = parsed is not None
        if has_tool_call:
            hits += 1
        results.append({
            "prompt": prompt,
            "raw": raw,
            "clean": strip_tool_markup(raw),
            "has_tool_call": has_tool_call,
            "tool_name": parsed[0] if parsed else None,
            "params": parsed[1] if parsed else None,
        })
        print(f"[{'tool' if has_tool_call else 'text'}] {prompt}")
        print(strip_tool_markup(raw)[:200])
        print()

    output = {
        "checkpoint": args.checkpoint,
        "count": len(results),
        "tool_call_hits": hits,
        "results": results,
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"tool_call_hits={hits}/{len(results)}")
    print(f"结果已保存: {out_path}")


if __name__ == "__main__":
    main()
