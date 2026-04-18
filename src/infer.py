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
import re
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


# P0: 推理端上下文上限。训练 sequence_len=2048，通过 NTK-aware RoPE 外推到 8192。
# 不作废任何 ckpt；只改推理。超过此长度的 prompt 由调用方在 tokenize 后截断。
INFER_MAX_CONTEXT = 8192


def extend_context(model, config, max_context: int, device, model_dtype):
    """P0: 把模型的有效上下文从 config.sequence_len 外推到 max_context。

    动作：
    1) NTK-aware RoPE base 重算：new_base = rope_theta * (scale ** (d/(d-2))), scale = max_context/sequence_len
    2) 用新 base + max_context 长度重算 cos/sin，覆盖 model.cos/model.sin
    3) 重算 model.window_sizes：S 层取 max_context//2, L 层 -1, 最后一层强制 L

    注意不修改 config.sequence_len：训练元信息保持原样，只在推理端扩。
    """
    if max_context <= config.sequence_len:
        return  # 没有外推需求
    head_dim = config.n_embd // config.n_head
    scale = max_context / config.sequence_len
    new_base = config.rope_theta * (scale ** (head_dim / (head_dim - 2)))
    # rotary_seq_len_mult 默认 10，但若 ckpt 来自老 config，mult 可能比 max_context/sequence_len 小
    rotary_len = max(max_context, int(getattr(model, "rotary_seq_len", max_context)))
    cos, sin = model._precompute_rotary_embeddings(rotary_len, head_dim, base=new_base, device=device)
    model.cos, model.sin = cos.to(model_dtype), sin.to(model_dtype)
    model.rotary_seq_len = rotary_len
    # 重算 window_sizes（原值用 config.sequence_len=2048 切窗口，外推后必须同步放大）
    pattern = config.window_pattern.upper()
    long_window = max_context
    short_window = long_window // 2
    char_to_window = {"L": (-1, -1), "S": (short_window, 0)}
    new_sizes = [char_to_window[pattern[i % len(pattern)]] for i in range(config.n_layer)]
    new_sizes[-1] = (long_window, 0)
    model.window_sizes = new_sizes


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
    parser.add_argument("--max-context", type=int, default=INFER_MAX_CONTEXT, help=f"Effective context length at inference (default {INFER_MAX_CONTEXT}, ckpt trained at 2048; uses NTK-aware RoPE extension)")
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

    # P0: 推理端上下文外推。若 --max-context > config.sequence_len，NTK 重算 RoPE + window_sizes。
    max_context = max(args.max_context, config.sequence_len)
    if max_context > config.sequence_len:
        extend_context(model, config, max_context, device, model_dtype)
        print(f"[P0] inference context extended: {config.sequence_len} → {max_context} (NTK RoPE)")

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
    system_prompt = (
        "你是微研，一个技术助手。用与用户相同的语言简洁回答。"
        "不确定时如实说明，不编造事实。拒绝有害内容。"
        "对于通用知识、Python/算法/CS 概念、代码模板等不依赖当前代码库的问题，直接回答，不要调用工具。"
        "只有当问题需要当前项目中的事实、实现、文件内容、符号位置或路径信息时，才使用工具。"
        "只允许使用这两个工具名：search_code、read_file。不要提及、假装调用或编造任何其他工具名。"
        "遇到需要定位实现、报错、符号、字符串或文件内容时，优先用 search_code 缩小范围，再用 read_file 验证。"
        "不要假装看过文件；没查到就明确说没查到。不要把代码搜索说成网页搜索。"
        "不要凭空生成 localhost、浏览器操作步骤或 URL，除非用户明确提供，或工具结果中确实出现。"
        "只有用户明确问当前工作目录、项目名或项目路径时，才回答目录信息。"
    )

    if args.tool_dir:
        project_name = Path(args.tool_dir).resolve().name
        system_prompt += (
            f"\n当前工作目录: {args.tool_dir}（项目: {project_name}）。"
            "当前可用工具只有：search_code、read_file。"
        )

    system_ids = tokenizer.encode(system_prompt + '\n')

    def _project_answer(prompt: str) -> str | None:
        if not args.tool_dir:
            return None
        tool_dir = Path(args.tool_dir).resolve()
        p = prompt.strip()

        if any(sep in p for sep in ["，", ",", "；", ";", "再", "然后", "顺便", "另外"]):
            return None

        direct_project_questions = [
            r"^当前工作目录(路径)?(是什么|是啥|在哪(里|儿)?|呢)?[？?]?$",
            r"^工作目录(路径)?(是什么|是啥|在哪(里|儿)?|呢)?[？?]?$",
            r"^当前项目名(是什么|是啥|呢)?[？?]?$",
            r"^当前项目(路径|目录)(是什么|是啥|在哪(里|儿)?|呢)?[？?]?$",
            r"^项目(路径|目录)(是什么|是啥|在哪(里|儿)?|呢)?[？?]?$",
            r"^你现在在(哪个)?目录[？?]?$",
            r"^project\s*dir[？?]?$",
        ]
        if any(re.fullmatch(pattern, p, re.IGNORECASE) for pattern in direct_project_questions):
            return f"当前工作目录是 `{tool_dir}`，项目名是 `{tool_dir.name}`。"

        return None

    def generate(prompt: str) -> str:
        """生成回答，支持工具调用循环。"""
        hard_rule_answer = apply_hard_rules(prompt)
        if hard_rule_answer:
            return hard_rule_answer

        project_answer = _project_answer(prompt)
        if project_answer:
            return project_answer

        prompt_ids = [bos_id, user_id] + system_ids + tokenizer.encode(prompt) + [asst_id]
        x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        generated_ids: list[int] = []
        last_tool_result: str | None = None

        # 工具调用 token IDs
        tool_call_end_id = enc.encode_single_token('<|tool_call_end|>') if '<|tool_call_end|>' in SPECIAL_TOKENS else None
        tool_result_start_tag = '<|tool_result_start|>'
        tool_result_end_tag = '<|tool_result_end|>'
        TOOL_CALL_START = '<|tool_call_start|>'
        max_tool_rounds = 3  # 最多执行 3 轮工具调用

        for _tool_round in range(max_tool_rounds + 1):
            # 生成直到遇到 stop token
            round_ids: list[int] = []
            round_flushed = False
            for _ in range(args.max_tokens):
                with torch.no_grad(), autocast_ctx:
                    logits = model(x[:, -max_context:])
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
                round_flushed = True
                break

            # 检查是否触发了工具调用 (只解析当前轮次生成的内容)
            decoded_new_tokens = tokenizer.decode(round_ids)
            tool_parsed = None

            if tool_call_end_id and round_ids and round_ids[-1] == tool_call_end_id:
                tool_parsed = parse_tool_call(decoded_new_tokens)
            elif TOOL_CALL_START in decoded_new_tokens:
                # 容错：模型输出了 tool_call_start 但没有 tool_call_end
                tool_parsed = parse_tool_call(decoded_new_tokens)

            if tool_parsed and not args.no_tools:
                tool_name, params = tool_parsed
                print(f"  [工具调用] {tool_name}({json.dumps(params, ensure_ascii=False)})")
                result = execute_tool(tool_name, params, args.tool_dir)
                result_short = result[:500] + "..." if len(result) > 500 else result
                print(f"  [工具结果] {result_short[:200]}")

                if not round_flushed:
                    generated_ids.extend(round_ids)
                    round_flushed = True
                # 注入工具结果到上下文（匹配训练格式：ASST_ID + result + ASST_ID）
                result_text = f"{tool_result_start_tag}{result}{tool_result_end_tag}"
                result_ids = [asst_id] + enc.encode(result_text, allowed_special="all") + [asst_id]
                generated_ids.extend(result_ids)
                result_tensor = torch.tensor([result_ids], dtype=torch.long, device=device)
                x = torch.cat([x, result_tensor], dim=1)
                last_tool_result = result
                continue  # 继续生成（模型会输出总结）
            else:
                # 没有工具调用，正常结束
                if not round_flushed:
                    generated_ids.extend(round_ids)
                    round_flushed = True
                break

        result = strip_tool_markup(tokenizer.decode(generated_ids))
        # 截断预训练残留的格式泄露
        for leak in ['\nHuman:', '\nAssistant:', 'Human:', 'Assistant:']:
            idx = result.find(leak)
            if idx >= 0:
                result = result[:idx].strip()
        # 清理 reserved token 泄露
        for tag in ['<|reserved_0|>', '<|reserved_1|>', '<|reserved_2|>', '<|reserved_3|>']:
            result = result.replace(tag, '')
        result = result.strip()

        # Fallback：如果工具后没有生成总结，诚实返回工具结果状态
        if not result and last_tool_result:
            first_line = last_tool_result.split('\n', 1)[0].strip()
            lowered = last_tool_result.lower()
            invalid_markers = [
                "未找到匹配",
                "文件不存在",
                "工具执行错误",
                "tool_error",
                "invalid literal",
                "Traceback",
            ]
            if any(marker in last_tool_result for marker in invalid_markers) or "error" in lowered:
                result = first_line or "工具没有返回可用结果。"
            elif first_line:
                result = f"我查到了相关结果，但还没来得及整理成自然语言回答。首条结果：`{first_line}`。"

        return result

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
