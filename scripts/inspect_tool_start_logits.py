#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from dataclasses import fields
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT / "src"))

import infer  # noqa: E402
from prepare import Tokenizer  # noqa: E402


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
    parser = argparse.ArgumentParser(description="检查模型首 token 对 <|tool_call_start|> 的偏好")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--out", default="inspect_tool_start_logits.json")
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

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = infer.GPT(config).to(device=device, dtype=dtype)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state"].items()}
    model.load_state_dict(state, strict=False)
    model.to(dtype=dtype)
    model.eval()

    head_dim = config.n_embd // config.n_head
    cos, sin = model._precompute_rotary_embeddings(model.rotary_seq_len, head_dim, device=device)
    model.cos, model.sin = cos.to(dtype), sin.to(dtype)
    return model, config, device


def token_text(tokenizer: Tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id]).replace("\n", "\\n")


def main() -> None:
    args = parse_args()
    model, config, device = load_model(args.checkpoint)
    tokenizer = Tokenizer.from_directory()
    enc = tokenizer.enc
    bos_id = tokenizer.get_bos_token_id()
    user_id = enc.encode_single_token("<|reserved_1|>")
    asst_id = enc.encode_single_token("<|reserved_2|>")
    tool_call_start_id = enc.encode_single_token("<|tool_call_start|>")
    system_prompt = "你是微研，一个技术助手。用与用户相同的语言简洁回答。不确定时如实说明，不编造事实。拒绝有害内容。"
    system_ids = tokenizer.encode(system_prompt + "\n")
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()

    rows = []
    for prompt in PROMPTS:
        prompt_ids = [bos_id, user_id] + system_ids + tokenizer.encode(prompt) + [asst_id]
        x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.no_grad(), autocast_ctx:
            logits = model(x[:, -config.sequence_len:])
            logits = logits[:, -1, :].float()
        sorted_ids = torch.argsort(logits[0], descending=True)
        rank = int((sorted_ids == tool_call_start_id).nonzero(as_tuple=False)[0].item()) + 1
        top_ids = sorted_ids[:args.top_n].tolist()
        row = {
            "prompt": prompt,
            "tool_call_start_rank": rank,
            "tool_call_start_logit": float(logits[0, tool_call_start_id].item()),
            "top_tokens": [
                {
                    "id": token_id,
                    "text": token_text(tokenizer, token_id),
                    "logit": float(logits[0, token_id].item()),
                }
                for token_id in top_ids
            ],
        }
        rows.append(row)
        print(f"[rank={rank}] {prompt}")
        for token in row["top_tokens"][:5]:
            print(f"  {token['text']!r} {token['logit']:.4f}")
        print()

    out = {
        "checkpoint": args.checkpoint,
        "results": rows,
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已保存: {out_path}")


if __name__ == "__main__":
    main()
