#!/usr/bin/env python3
"""Stable inference script for attnres checkpoints.

支持：
- 身份/安全硬拦截（推理时规则，不依赖模型）
- 工具调用执行 runtime（search_code / read_file）
- 交互式对话
"""

from __future__ import annotations

import json
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import argparse
from dataclasses import fields
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import torch


def _load_model_defs():
    """Load only model definitions from train.py without running training setup."""
    _SRC_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(_SRC_DIR))
    lines = (_SRC_DIR / "train.py").read_text(encoding="utf-8").splitlines(keepends=True)
    cut = next(i for i, line in enumerate(lines) if "# Setup: tokenizer, model, optimizer, dataloader" in line)
    src = "".join(lines[:cut])

    fake = types.ModuleType("prepare")
    fake.MAX_SEQ_LEN = 2048
    fake.TIME_BUDGET = 300
    fake.Tokenizer = None
    fake.make_dataloader = None
    fake.evaluate_bpb = None
    sys.modules["prepare"] = fake

    # Only mock kernels+CUDA when CUDA is unavailable so import succeeds.
    # On CUDA machines, let the real kernels module load so fa3 is properly bound.
    _cuda_patch = None
    _injected_kernels = False
    if not torch.cuda.is_available():
        import unittest.mock as _mock
        fake_kernels = types.ModuleType("kernels")
        fake_kernels.get_kernel = lambda repo: types.SimpleNamespace(flash_attn_interface=None)
        sys.modules["kernels"] = fake_kernels
        _injected_kernels = True
        _cuda_patch = _mock.patch("torch.cuda.get_device_capability", return_value=(9, 0))
        _cuda_patch.start()

    try:
        ns: dict[str, object] = {}
        exec(compile(src, "train.py", "exec"), ns)
    finally:
        del sys.modules["prepare"]
        if _injected_kernels:
            del sys.modules["kernels"]
        if _cuda_patch is not None:
            _cuda_patch.stop()

    return ns["GPT"], ns["GPTConfig"]


GPT, GPTConfig = _load_model_defs()

import __main__  # noqa: E402
__main__.GPT = GPT
__main__.GPTConfig = GPTConfig
GPT.__module__ = '__main__'
GPTConfig.__module__ = '__main__'

from prepare import SPECIAL_TOKENS, Tokenizer  # noqa: E402
from inference_rules import apply_hard_rules  # noqa: E402
from infer_support import ensure_inference_backend  # noqa: E402
from project_paths import resolve_default_checkpoint  # noqa: E402
from tool_protocol import execute_tool, parse_tool_call, strip_tool_markup  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for attnres checkpoints")
    parser.add_argument("prompt", nargs="?", default=None, help="Optional one-shot prompt")
    parser.add_argument("max_tokens", nargs="?", type=int, default=256, help="Maximum generated tokens")
    parser.add_argument("temperature", nargs="?", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path; defaults to best_checkpoint.pt if present")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling cutoff")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling cutoff")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--bf16", action="store_true", help="Force bf16 weights/autocast on CUDA")
    parser.add_argument("--fp32", action="store_true", help="Force fp32 weights (requires CUDA; FlashAttention does not support CPU)")
    parser.add_argument("--rep-penalty", type=float, default=1.3, help="Repetition penalty (1.0=off, >1 reduces repetition)")
    parser.add_argument("--rope-theta", type=float, default=None, help="RoPE base frequency (NTK scaling: e.g. 500000 for longer context)")
    parser.add_argument("--tool-dir", type=str, default=".", help="工具调用的工作目录（search_code/read_file 在此目录下执行）")
    parser.add_argument("--no-tools", action="store_true", help="禁用工具调用执行（模型仍可能输出工具调用格式，但不会执行）")
    return parser.parse_args()


def resolve_checkpoint(explicit_path: str | None) -> Path:
    return resolve_default_checkpoint(explicit_path)


def flash_attention_backend() -> object:
    return GPT.__init__.__globals__.get("fa3")


def apply_top_k_top_p(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, k)
        logits = logits.masked_fill(logits < values[..., -1, None], float("-inf"))

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        original_mask = torch.zeros_like(sorted_mask, dtype=torch.bool)
        original_mask.scatter_(dim=-1, index=sorted_indices, src=sorted_mask)
        logits = logits.masked_fill(original_mask, float("-inf"))

    return logits


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 只有在强制指定 fp32 且没有 CUDA 时才报错，因为 FlashAttention 强依赖 CUDA
    if args.fp32 and device == "cpu":
        print("WARNING: --fp32 specified but no CUDA found. Falling back to float32 on CPU.")

    ensure_inference_backend(device, flash_attention_backend())
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_config = ckpt["config"]
    if isinstance(raw_config, dict):
        allowed = {f.name for f in fields(GPTConfig)}
        config = GPTConfig(**{k: v for k, v in raw_config.items() if k in allowed})
    else:
        config = raw_config

    use_bf16 = device == "cuda" and not args.fp32
    if args.bf16:
        use_bf16 = device == "cuda"
    model_dtype = torch.bfloat16 if use_bf16 else torch.float32

    model = GPT(config).to(device=device, dtype=model_dtype)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state"].items()}
    load_result = model.load_state_dict(state, strict=False)
    if load_result.missing_keys:
        print(f"WARNING: missing keys in checkpoint: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"WARNING: unexpected keys in checkpoint: {load_result.unexpected_keys}")
    model.to(dtype=model_dtype)
    model.eval()

    head_dim = config.n_embd // config.n_head
    if args.rope_theta is not None:
        config.rope_theta = args.rope_theta
    cos, sin = model._precompute_rotary_embeddings(model.rotary_seq_len, head_dim, device=device)
    model.cos, model.sin = cos.to(model_dtype), sin.to(model_dtype)

    tokenizer = Tokenizer.from_directory()
    bos_id = tokenizer.get_bos_token_id()
    # stop_ids: 只用基础 4 个 token（不包括工具调用 token，工具调用由 runtime 处理）
    base_stop_tokens = ['<|reserved_0|>', '<|reserved_1|>', '<|reserved_2|>', '<|reserved_3|>']
    stop_ids = {tokenizer.enc.encode_single_token(token) for token in base_stop_tokens}

    _metric_key = 'val_bpt' if 'val_bpt' in ckpt else 'val_bpb'
    print(f"Loaded checkpoint: {checkpoint_path.name}, {_metric_key}={ckpt[_metric_key]:.4f}, step={ckpt['step']}")
    print(f"Model: {config.n_layer}L x {config.n_embd}d, dtype={model_dtype}, device={device}")
    print(
        f"Interactive mode (max_tokens={args.max_tokens}, temperature={args.temperature}, "
        f"top_k={args.top_k}, top_p={args.top_p}). Ctrl+C to exit.\n"
    )

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16
        else nullcontext()
    )

    enc = tokenizer.enc
    user_id = enc.encode_single_token('<|reserved_1|>')
    asst_id = enc.encode_single_token('<|reserved_2|>')
    system_prompt = '你是微研，一个技术助手。用与用户相同的语言简洁回答。不确定时如实说明，不编造事实。拒绝有害内容。'
    system_ids = tokenizer.encode(system_prompt + '\n')

    def generate(prompt: str) -> str:
        """生成回答，支持工具调用循环。"""
        hard_rule_answer = apply_hard_rules(prompt)
        if hard_rule_answer:
            return hard_rule_answer

        prompt_ids = [bos_id, user_id] + system_ids + tokenizer.encode(prompt) + [asst_id]
        x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        generated_ids: list[int] = []

        # 工具调用 token IDs
        tool_call_end_id = enc.encode_single_token('<|tool_call_end|>') if '<|tool_call_end|>' in {t for t in SPECIAL_TOKENS} else None
        tool_result_start_tag = '<|tool_result_start|>'
        tool_result_end_tag = '<|tool_result_end|>'
        max_tool_rounds = 3  # 最多执行 3 轮工具调用

        for _tool_round in range(max_tool_rounds + 1):
            # 生成直到遇到 stop token
            round_ids: list[int] = []
            for _ in range(args.max_tokens):
                with torch.no_grad(), autocast_ctx:
                    logits = model(x[:, -config.sequence_len:])
                    logits = logits[:, -1, :]

                if args.rep_penalty != 1.0 and (generated_ids or round_ids):
                    for tok_id in set(generated_ids + round_ids):
                        if logits[0, tok_id] > 0:
                            logits[0, tok_id] /= args.rep_penalty
                        else:
                            logits[0, tok_id] *= args.rep_penalty

                if args.temperature <= 0:
                    next_id = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    logits = logits / args.temperature
                    filtered = apply_top_k_top_p(logits, args.top_k, args.top_p)
                    probs = torch.softmax(filtered, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)

                token_id = int(next_id.item())

                # 检查是否是 tool_call_end（触发工具执行）
                if tool_call_end_id and token_id == tool_call_end_id and not args.no_tools:
                    round_ids.append(token_id)
                    generated_ids.extend(round_ids)
                    x = torch.cat([x, next_id], dim=1)
                    break

                # 检查是否是其他 stop token
                if token_id in stop_ids:
                    break

                round_ids.append(token_id)
                x = torch.cat([x, next_id], dim=1)
            else:
                # 没有遇到 stop token，结束生成
                generated_ids.extend(round_ids)
                break

            # 检查是否触发了工具调用 (只解析当前轮次生成的内容)
            if tool_call_end_id and round_ids and round_ids[-1] == tool_call_end_id:
                decoded_new_tokens = tokenizer.decode(round_ids)
                tool_parsed = parse_tool_call(decoded_new_tokens)

                if tool_parsed:
                    tool_name, params = tool_parsed
                    print(f"  [工具调用] {tool_name}({json.dumps(params, ensure_ascii=False)})")
                    result = execute_tool(tool_name, params, args.tool_dir)
                    result_short = result[:500] + "..." if len(result) > 500 else result
                    print(f"  [工具结果] {result_short[:200]}")

                    # 注入工具结果到上下文
                    result_text = f"{tool_result_start_tag}{result}{tool_result_end_tag}"
                    result_ids = enc.encode(result_text, allowed_special="all")
                    generated_ids.extend(result_ids)
                    result_tensor = torch.tensor([result_ids], dtype=torch.long, device=device)
                    x = torch.cat([x, result_tensor], dim=1)
                    continue  # 继续生成（模型会输出总结）
                else:
                    # 工具调用解析失败，直接结束
                    break
            else:
                # 正常结束（遇到 EOS 等）
                generated_ids.extend(round_ids)
                break

        result = strip_tool_markup(tokenizer.decode(generated_ids))
        # 截断预训练残留的格式泄露
        for leak in ['\nHuman:', '\nAssistant:', 'Human:', 'Assistant:']:
            idx = result.find(leak)
            if idx >= 0:
                result = result[:idx].strip()
        return result.strip()

    if args.prompt is not None:
        print("--- Output ---")
        print(generate(args.prompt))
        print()

    while True:
        try:
            prompt = input("Prompt> ").strip()
            if not prompt:
                continue
        except (EOFError, KeyboardInterrupt):
            print()
            break

        print()
        print(generate(prompt))
        print()


if __name__ == "__main__":
    main()
