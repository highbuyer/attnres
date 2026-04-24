#!/usr/bin/env python3
"""SFT fine-tuning script."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
import types
from contextlib import nullcontext
from dataclasses import asdict, fields
from pathlib import Path

from project_paths import resolve_sft_data_path, resolve_sft_input_checkpoint
from sft_format import included_turn_indices


MAX_SEQ_LEN = 2048
DEVICE_BATCH_SIZE = 4
GRAD_ACCUM = 8
FINAL_LR_FRAC = 0.1
VAL_RATIO = 0.05
SEED = 42
SYSTEM_PROMPT = (
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

GPT = None
GPTConfig = None
tokenizer = None
enc = None
BOS_ID = None
USER_ID = None
ASST_ID = None
EOS_ID = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT fine-tuning")
    parser.add_argument("checkpoint", type=str, nargs="?", default=None, help="Input checkpoint path")
    parser.add_argument("--out", "-o", type=str, default=None, help="Output checkpoint path")
    parser.add_argument("--data", type=str, default=None, help="SFT data JSONL path")
    parser.add_argument("--resume", action="store_true", help="Resume SFT from a prior SFT checkpoint")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--total-steps", type=int, default=None, help="Total training steps override")
    parser.add_argument("--warmup-steps", type=int, default=None, help="Warmup steps override")
    parser.add_argument("--warmdown-start", type=int, default=None, help="Warmdown start step override")
    parser.add_argument("--eval-interval", type=int, default=None, help="Validation interval override")
    parser.add_argument("--batch-size", type=int, default=None, help="Device batch size override")
    parser.add_argument("--grad-accum", type=int, default=None, help="Gradient accumulation steps override")
    parser.add_argument("--grad-ckpt", action="store_true", help="Enable gradient checkpointing (recompute attn/mlp activations)")
    parser.add_argument("--max-samples", type=int, default=None, help="Only use first N records from data (for smoke test / subset training)")
    return parser.parse_args()


def _ensure_model_defs() -> None:
    global GPT, GPTConfig
    if GPT is not None and GPTConfig is not None:
        return

    src_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(src_dir))
    lines = (src_dir / "train.py").read_text(encoding="utf-8").splitlines(keepends=True)
    cut = next(i for i, line in enumerate(lines) if "# Setup: tokenizer, model, optimizer, dataloader" in line)
    src = "".join(lines[:cut])

    fake = types.ModuleType("prepare")
    fake.MAX_SEQ_LEN = MAX_SEQ_LEN
    fake.TIME_BUDGET = 999999
    fake.Tokenizer = None
    fake.make_dataloader = None
    fake.evaluate_bpb = None
    sys.modules["prepare"] = fake
    try:
        ns: dict[str, object] = {}
        exec(compile(src, "train.py", "exec"), ns)
    finally:
        del sys.modules["prepare"]

    GPT = ns["GPT"]
    GPTConfig = ns["GPTConfig"]

    import __main__

    __main__.GPT = GPT
    __main__.GPTConfig = GPTConfig
    GPT.__module__ = "__main__"
    GPTConfig.__module__ = "__main__"


def _ensure_tokenizer() -> None:
    global tokenizer, enc, BOS_ID, USER_ID, ASST_ID, EOS_ID
    if tokenizer is not None:
        return

    _ensure_model_defs()
    from prepare import Tokenizer

    tokenizer = Tokenizer.from_directory()
    enc = tokenizer.enc
    BOS_ID = enc.encode_single_token("<|reserved_0|>")
    USER_ID = enc.encode_single_token("<|reserved_1|>")
    ASST_ID = enc.encode_single_token("<|reserved_2|>")
    EOS_ID = enc.encode_single_token("<|reserved_3|>")


def _config_from_ckpt(raw_config):
    _ensure_model_defs()
    if isinstance(raw_config, dict):
        allowed = {field.name for field in fields(GPTConfig)}
        return GPTConfig(**{key: value for key, value in raw_config.items() if key in allowed})
    return raw_config


def _config_to_ckpt(config):
    return asdict(config)


def load_checkpoint(path, device):
    import torch

    ckpt = torch.load(path, map_location=device, weights_only=False)
    ckpt["config"] = _config_from_ckpt(ckpt["config"])
    return ckpt


def save_checkpoint(path, ckpt):
    import torch

    ckpt = dict(ckpt)
    ckpt["config"] = _config_to_ckpt(ckpt["config"])
    torch.save(ckpt, path)


def best_alias_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(f"{path.stem}_best{path.suffix}")


def best_metadata_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(f"{path.stem}_best.json")


def save_best_artifacts(path, ckpt):
    save_checkpoint(path, ckpt)

    alias = best_alias_path(path)
    if alias != Path(path):
        if alias.exists() or alias.is_symlink():
            alias.unlink()
        try:
            os.link(path, alias)
        except OSError:
            shutil.copy2(path, alias)

    meta = {
        "checkpoint": str(Path(path)),
        "best_alias": str(alias),
        "step": int(ckpt["step"]),
        "val_bpt": float(ckpt["val_bpt"]),
        "best_val_bpt": float(ckpt.get("best_val_bpt", ckpt["val_bpt"])),
    }
    best_metadata_path(path).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


DROPPED_SAMPLE_COUNT = 0
TRUNCATED_SAMPLE_COUNT = 0


def _enable_gradient_checkpointing(model):
    """对每个 block 的 forward_attn_only / forward_mlp_only 包 checkpoint。

    SFT 场景 past_kv=None、use_cache=False，不走 KV cache 路径。eval 时
    torch.is_grad_enabled()=False 自动 fall-through 到原实现，不重算。
    """
    import torch
    from torch.utils.checkpoint import checkpoint

    for block in model.transformer.h:
        orig_attn = block.forward_attn_only
        orig_mlp = block.forward_mlp_only

        def make_attn(orig):
            def attn_ckpt(h, ve, cos_sin, window_size, past_kv=None, use_cache=False):
                if use_cache or past_kv is not None or not torch.is_grad_enabled():
                    return orig(h, ve, cos_sin, window_size, past_kv=past_kv, use_cache=use_cache)
                return checkpoint(orig, h, ve, cos_sin, window_size, use_reentrant=False)
            return attn_ckpt

        def make_mlp(orig):
            def mlp_ckpt(h):
                if not torch.is_grad_enabled():
                    return orig(h)
                return checkpoint(orig, h, use_reentrant=False)
            return mlp_ckpt

        block.forward_attn_only = make_attn(orig_attn)
        block.forward_mlp_only = make_mlp(orig_mlp)


def tokenize_turn(message):
    _ensure_tokenizer()
    content = message.get("content", "")
    content_ids = enc.encode(content, allowed_special="all")
    if message["role"] in ("user", "tool"):
        return [USER_ID] + content_ids, [0] * (1 + len(content_ids))
    # tool_result 由 runtime 注入，不是模型生成的，不计算 loss
    if "<|tool_result_start|>" in content:
        return [ASST_ID] + content_ids, [0] * (1 + len(content_ids))
    return [ASST_ID] + content_ids, [0] + [1] * len(content_ids)


def format_samples_split(messages):
    _ensure_tokenizer()
    if not messages:
        return []

    messages = list(messages)
    if messages[0]["role"] == "system":
        system_content = messages[0]["content"]
        messages = messages[1:]
        for idx, message in enumerate(messages):
            if message["role"] == "user":
                messages[idx] = {"role": "user", "content": system_content + "\n" + message["content"]}
                break
        else:
            return []

    if not messages or messages[0]["role"] != "user":
        return []

    messages[0] = {
        "role": "user",
        "content": SYSTEM_PROMPT + "\n" + messages[0]["content"],
    }

    turn_ids = []
    turn_masks = []
    for message in messages:
        ids, mask = tokenize_turn(message)
        turn_ids.append(ids)
        turn_masks.append(mask)

    samples = []
    for idx, message in enumerate(messages):
        if message["role"] != "assistant":
            continue

        ids = [BOS_ID]
        mask = [0]
        for turn_idx in included_turn_indices(messages, idx):
            ids.extend(turn_ids[turn_idx])
            mask.extend(turn_masks[turn_idx])
        ids.append(EOS_ID)
        mask.append(1)

        if len(ids) > MAX_SEQ_LEN:
            # 从左侧裁掉较早轮次，保留末尾含当前 assistant 的对话
            # ids 结构：[BOS] + turn_ids[i0] + turn_ids[i1] + ... + [EOS]
            # 找最靠右的 USER/ASST 边界，使得从该边界起到 EOS 的长度 <= MAX_SEQ_LEN - 1（BOS 占 1）
            budget = MAX_SEQ_LEN - 1
            # 从尾部往前找可行的裁切点
            cut = None
            for pos in range(len(ids) - 1, 0, -1):  # 不包括 BOS (pos 0)
                if ids[pos] in (USER_ID, ASST_ID) and (len(ids) - pos) <= budget:
                    cut = pos
            if cut is None:
                # 连最后一个 turn 都放不下，整条丢弃
                global DROPPED_SAMPLE_COUNT
                DROPPED_SAMPLE_COUNT += 1
                continue
            global TRUNCATED_SAMPLE_COUNT
            TRUNCATED_SAMPLE_COUNT += 1
            ids = [BOS_ID] + ids[cut:]
            mask = [0] + mask[cut:]

        if sum(mask) == 0:
            continue
        samples.append((ids, mask))
    return samples


def _render_minimind_system(convs, tools):
    if not tools and not (convs and convs[0].get("role") == "system" and convs[0].get("content")):
        return ""
    if tools:
        sys_content = convs[0].get("content", "") if convs and convs[0].get("role") == "system" else ""
        tools_block = (
            "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
        )
        for tool in tools:
            tools_block += "\n" + json.dumps(tool, ensure_ascii=False)
        tools_block += (
            "\n</tools>\n\nFor each function call, return a json object with function name and "
            "arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
        )
        head = sys_content + "\n\n" if sys_content else ""
        return f"<|im_start|>system\n{head}{tools_block}<|im_end|>\n"
    return f"<|im_start|>system\n{convs[0]['content']}<|im_end|>\n"


def _render_minimind_assistant(msg):
    content = msg.get("content", "") or ""
    rc = msg.get("reasoning_content", "") or ""
    body = "<think>\n" + rc.strip("\n") + "\n</think>\n\n" + content.lstrip("\n")
    tcs = msg.get("tool_calls")
    if tcs:
        if isinstance(tcs, str):
            tcs = json.loads(tcs)
        for i, tc in enumerate(tcs):
            if (i == 0 and content) or i > 0:
                body += "\n"
            fn = tc.get("function", tc)
            args = fn.get("arguments", "")
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            body += '<tool_call>\n{"name": "' + fn["name"] + '", "arguments": ' + args + "}\n</tool_call>"
    return body


def format_minimind_record(record):
    """将 minimind conversations 格式 → 一个 (ids, mask) 样本。

    loss_mask 规则：<|im_start|>assistant\\n...<|im_end|>\\n 块内的 body+suffix 为 1，其他为 0。
    """
    _ensure_tokenizer()
    convs = record.get("conversations") or record.get("messages") or []
    if not convs:
        return []

    tools = None
    if convs and convs[0].get("role") == "system" and convs[0].get("tools"):
        t = convs[0]["tools"]
        tools = json.loads(t) if isinstance(t, str) else t

    chunks = []
    sys_text = _render_minimind_system(convs, tools)
    if sys_text:
        chunks.append(("system", sys_text))

    start_idx = 1 if convs and convs[0].get("role") == "system" else 0
    for m in convs[start_idx:]:
        role = m.get("role")
        content = m.get("content", "") or ""
        if role == "user":
            chunks.append(("user", f"<|im_start|>user\n{content}<|im_end|>\n"))
        elif role == "tool":
            chunks.append(("tool", f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n"))
        elif role == "assistant":
            chunks.append(("asst_prefix", "<|im_start|>assistant\n"))
            chunks.append(("asst_body", _render_minimind_assistant(m)))
            chunks.append(("asst_suffix", "<|im_end|>\n"))

    ids = [BOS_ID]
    mask = [0]
    for kind, text in chunks:
        t_ids = enc.encode(text, allowed_special="all")
        ids.extend(t_ids)
        m_flag = 1 if kind in ("asst_body", "asst_suffix") else 0
        mask.extend([m_flag] * len(t_ids))
    ids.append(EOS_ID)
    mask.append(1)

    if len(ids) > MAX_SEQ_LEN:
        global DROPPED_SAMPLE_COUNT
        DROPPED_SAMPLE_COUNT += 1
        return []
    if sum(mask) == 0:
        return []
    return [(ids, mask)]


def _is_minimind_format(raw_sample) -> bool:
    return isinstance(raw_sample, dict) and "conversations" in raw_sample and "messages" not in raw_sample


def build_datasets(data_path: str, max_samples: int | None = None):
    global DROPPED_SAMPLE_COUNT, TRUNCATED_SAMPLE_COUNT
    DROPPED_SAMPLE_COUNT = 0
    TRUNCATED_SAMPLE_COUNT = 0

    with open(data_path, encoding="utf-8") as handle:
        raw = []
        for i, line in enumerate(handle):
            if max_samples is not None and i >= max_samples:
                break
            raw.append(json.loads(line))

    random.seed(SEED)
    random.shuffle(raw)

    val_conv_n = max(1, int(len(raw) * VAL_RATIO))
    val_raw = raw[:val_conv_n]
    train_raw = raw[val_conv_n:]

    is_mm = raw and _is_minimind_format(raw[0])
    formatter = format_minimind_record if is_mm else (lambda r: format_samples_split(r["messages"]))
    if is_mm:
        print("Data format: minimind conversations (chat_template + assistant-block loss mask)")

    train_data = [sample for record in train_raw for sample in formatter(record)]
    val_data = [sample for record in val_raw for sample in formatter(record)]
    if TRUNCATED_SAMPLE_COUNT or DROPPED_SAMPLE_COUNT:
        print(f"WARNING: truncated {TRUNCATED_SAMPLE_COUNT} samples, dropped {DROPPED_SAMPLE_COUNT} oversized samples")
    return raw, train_data, val_data


def make_batch(samples, device):
    import torch

    max_len = max(len(sample[0]) for sample in samples)
    input_ids = torch.zeros(len(samples), max_len, dtype=torch.long)
    loss_mask = torch.zeros(len(samples), max_len - 1, dtype=torch.float)
    for idx, (ids, mask) in enumerate(samples):
        n_tokens = len(ids)
        input_ids[idx, :n_tokens] = torch.tensor(ids, dtype=torch.long)
        loss_mask[idx, :n_tokens - 1] = torch.tensor(mask[1:], dtype=torch.float)
    x = input_ids[:, :-1].to(device)
    y = input_ids[:, 1:].to(device)
    loss_mask = loss_mask.to(device)
    return x, y, loss_mask


def autocast_context(device):
    import torch

    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def parameter_dtype_for_training(device):
    import torch

    # Keep trainable parameters in fp32 so small optimizer updates are not
    # rounded away when the model is fine-tuned on narrow supervision.
    return torch.float32


def compute_dtype_for_training(device):
    import torch

    return torch.bfloat16 if device.type == "cuda" else torch.float32


def evaluate_sft(model, val_data, device):
    import torch
    import torch.nn.functional as F

    if not val_data:
        return float("inf")

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for idx in range(0, len(val_data), DEVICE_BATCH_SIZE):
            batch = val_data[idx:idx + DEVICE_BATCH_SIZE]
            x, y, loss_mask = make_batch(batch, device)
            with autocast_context(device):
                logits = model(x)
            logits = logits.float()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="none")
            loss = (loss * loss_mask.view(-1)).sum()
            total_loss += loss.item()
            total_tokens += loss_mask.sum().item()
    model.train()
    if total_tokens == 0:
        return float("inf")
    return total_loss / total_tokens / math.log(2)


def resolve_resume_state(ckpt: dict, resume: bool) -> tuple[int, float]:
    if not resume:
        return 0, float("inf")
    if "optimizer_state" not in ckpt:
        raise ValueError("Checkpoint does not contain optimizer_state, cannot resume")
    resume_step = int(ckpt.get("step", 0))
    best_val_bpt = float(ckpt.get("best_val_bpt", ckpt.get("val_bpt", float("inf"))))
    return resume_step, best_val_bpt


def main() -> None:
    import torch
    import torch.nn.functional as F

    args = parse_args()

    if args.checkpoint:
        checkpoint_in = str(resolve_sft_input_checkpoint(args.checkpoint))
        checkpoint_stem = Path(checkpoint_in).stem
        checkpoint_out = args.out or f"sft_{checkpoint_stem}.pt"
    else:
        checkpoint_in = str(resolve_sft_input_checkpoint(None))
        checkpoint_out = args.out or "sft_checkpoint.pt"

    data_path = str(resolve_sft_data_path(args.data))
    lr = args.lr or 1.5e-5
    total_steps = args.total_steps or 15000
    warmup_steps = args.warmup_steps or 100
    warmdown_start = args.warmdown_start if args.warmdown_start is not None else int(total_steps * 0.8)
    eval_interval = args.eval_interval or 500
    device_batch_size = args.batch_size or DEVICE_BATCH_SIZE
    grad_accum = args.grad_accum or GRAD_ACCUM

    _ensure_model_defs()
    _ensure_tokenizer()
    print(f"Special tokens: BOS={BOS_ID} USER={USER_ID} ASST={ASST_ID} EOS={EOS_ID}")

    raw, train_data, val_data = build_datasets(data_path, max_samples=args.max_samples)
    print(f"Formatted: {len(train_data) + len(val_data)} samples (from {len(raw)} raw conversations)")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parameter_dtype = parameter_dtype_for_training(device)
    compute_dtype = compute_dtype_for_training(device)

    print(f"Loading {checkpoint_in} on {device}...")
    ckpt = load_checkpoint(checkpoint_in, device)
    config = ckpt["config"]

    model = GPT(config).to(device=device, dtype=parameter_dtype)
    state = {key.replace("_orig_mod.", ""): value for key, value in ckpt["model_state"].items()}
    load_result = model.load_state_dict(state, strict=False)
    if load_result.missing_keys:
        print(f"WARNING: missing keys: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"WARNING: unexpected keys: {load_result.unexpected_keys}")
    model.to(dtype=parameter_dtype)

    head_dim = config.n_embd // config.n_head
    cos, sin = model._precompute_rotary_embeddings(model.rotary_seq_len, head_dim, device=device)
    model.cos, model.sin = cos.to(compute_dtype), sin.to(compute_dtype)

    if args.grad_ckpt:
        _enable_gradient_checkpointing(model)
        print("  Gradient checkpointing: ON (recompute attn/mlp activations, -30% memory, +20% time)")

    metric_key = "val_bpt" if "val_bpt" in ckpt else "val_bpb"
    print(f"Checkpoint {checkpoint_in}: {metric_key}={ckpt[metric_key]:.4f}, step={ckpt['step']}")

    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
        print("  Optimizer: bitsandbytes.AdamW8bit (saves ~2.4GB optimizer state vs fp32 AdamW)")
    except ImportError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
        print("  Optimizer: torch.AdamW (bitsandbytes unavailable)")
    resume_step, best_val_bpt = resolve_resume_state(ckpt, args.resume)
    if args.resume:
        optimizer.load_state_dict(ckpt["optimizer_state"])
        print(f"Resuming SFT from step {resume_step} (best_val_bpt={best_val_bpt:.4f})")

    model.train()
    random.seed(SEED + 1)
    step = resume_step

    print("SFT config:")
    print(f"  Input:  {checkpoint_in}")
    print(f"  Output: {checkpoint_out}")
    print(f"  Data:   {data_path}")
    print(f"  LR:     {lr}")
    print(f"  Steps:  total={total_steps} warmup={warmup_steps} warmdown={warmdown_start} eval={eval_interval}")
    print(f"Starting SFT: {total_steps} steps, lr={lr}, batch={device_batch_size * grad_accum}")
    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")

    train_idx = list(range(len(train_data)))
    random.shuffle(train_idx)
    idx_ptr = 0

    def next_batch():
        nonlocal train_idx, idx_ptr
        batch_indices = []
        while len(batch_indices) < device_batch_size:
            if idx_ptr >= len(train_idx):
                random.shuffle(train_idx)
                idx_ptr = 0
            batch_indices.append(train_idx[idx_ptr])
            idx_ptr += 1
        return [train_data[i] for i in batch_indices]

    t0 = time.time()
    while step < total_steps:
        if step < warmup_steps:
            current_lr = lr * (step + 1) / warmup_steps
        elif step >= warmdown_start:
            frac = (step - warmdown_start) / max(1, total_steps - warmdown_start)
            current_lr = lr * (1 - frac * (1 - FINAL_LR_FRAC))
        else:
            current_lr = lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        optimizer.zero_grad()
        accum_loss = 0.0
        accum_tokens = 0

        for _ in range(grad_accum):
            batch = next_batch()
            x, y, loss_mask = make_batch(batch, device)
            with autocast_context(device):
                logits = model(x)
            logits = logits.float()
            loss_per_tok = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="none")
            masked_loss = (loss_per_tok * loss_mask.view(-1)).sum() / (loss_mask.sum() + 1e-8)
            (masked_loss / grad_accum).backward()
            accum_loss += masked_loss.item()
            accum_tokens += int(loss_mask.sum().item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1

        dt = time.time() - t0
        t0 = time.time()
        print(
            f"step {step:04d}/{total_steps} | loss={accum_loss / grad_accum:.4f} | "
            f"lr={current_lr:.2e} | tok={accum_tokens} | dt={dt * 1000:.0f}ms"
        )

        if step % eval_interval == 0:
            val_bpt = evaluate_sft(model, val_data, device)
            print(f"  VAL step {step}: bits/tok={val_bpt:.4f}")
            if val_bpt < best_val_bpt:
                best_val_bpt = val_bpt
                ckpt_out = {
                    "model_state": model.state_dict(),
                    "config": ckpt["config"],
                    "val_bpt": val_bpt,
                    "best_val_bpt": best_val_bpt,
                    "step": step,
                    "sft": True,
                    "optimizer_state": optimizer.state_dict(),
                }
                save_best_artifacts(checkpoint_out, ckpt_out)
                print(f"  Saved {checkpoint_out}: val_bpt={val_bpt:.4f}")
                print(f"  Updated {best_alias_path(checkpoint_out)} and {best_metadata_path(checkpoint_out)}")

    print(f"SFT done. Best val_bpt={best_val_bpt:.4f}")


if __name__ == "__main__":
    main()
