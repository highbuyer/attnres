#!/usr/bin/env python3
"""From-scratch 预训练入口：MLA + SwiGLU + 砍 VE + tie_lm_head（v2 架构）。

参考 continue_pretrain.py 的脚架（exec 加载 train.py 的 MuonAdamW + helpers），
但模型用 src/model_v2.py 的 GPT_v2，从随机权重开始训练。

用法:
  CPT_OUT=checkpoints/d36_v2_pretrain.pt \\
  PRETRAIN_STEPS=6000 \\
  PRETRAIN_BATCH=1 \\
  PRETRAIN_GRAD_CKPT=1 \\
  .venv/bin/python -u src/pretrain_v2.py

环境变量（与 continue_pretrain.py 风格一致）:
  CPT_OUT                    输出 ckpt 路径基名
  PRETRAIN_STEPS             总训练步数（默认 6000）
  PRETRAIN_BATCH             单设备 micro batch（默认 1）
  PRETRAIN_TOTAL_BATCH       全局 batch tokens（默认 524288）
  PRETRAIN_GRAD_CKPT         1=开启 gradient ckpt（默认 0）
  PRETRAIN_EVAL_INTERVAL     fast eval 步频（默认 200）
  PRETRAIN_FULL_EVAL_INTERVAL  full eval 步频（默认 600）
  PRETRAIN_PATIENCE          early-stop patience（默认 8）
"""
import gc
import math
import os
import sys
import time
import types
from contextlib import nullcontext
from dataclasses import asdict, fields
from pathlib import Path

import torch
import torch.nn.functional as F

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# MLA 引入更多 weight shape（c_q/c_kv_a/c_kv_b/c_proj/gate_proj/up_proj/down_proj/...），
# Muon 按 shape 分组后 muon_step_fused 的编译会触发多次，提高 cache_size_limit 防止 fail。
import torch._dynamo as _dynamo
_dynamo.config.cache_size_limit = 64
_dynamo.config.accumulated_cache_size_limit = 256

try:
    import ctypes
except Exception:
    ctypes = None

_LIBC = None
if os.name == "posix" and ctypes is not None:
    try:
        _LIBC = ctypes.CDLL("libc.so.6")
    except OSError:
        _LIBC = None

# ---------------------------------------------------------------------------
# Config（env var）
# ---------------------------------------------------------------------------
CHECKPOINT_OUT = os.environ.get('CPT_OUT', 'checkpoints/d36_v2_pretrain.pt')
TOTAL_STEPS = int(os.environ.get('PRETRAIN_STEPS', 6000))
TOTAL_BATCH_SIZE = int(os.environ.get('PRETRAIN_TOTAL_BATCH', str(2**19)))
DEVICE_BATCH_SIZE = int(os.environ.get('PRETRAIN_BATCH', 1))
EVAL_INTERVAL = int(os.environ.get('PRETRAIN_EVAL_INTERVAL', 200))
FULL_EVAL_INTERVAL = int(os.environ.get('PRETRAIN_FULL_EVAL_INTERVAL', 600))
EARLY_STOP_PATIENCE = int(os.environ.get('PRETRAIN_PATIENCE', 8))
FAST_EVAL_TOKENS = int(os.environ.get('PRETRAIN_FAST_EVAL_MULT', 4)) * 524288

# 学习率（与 train.py 默认一致）
MATRIX_LR = 0.04
EMBEDDING_LR = 0.2
UNEMBEDDING_LR = 0.004
SCALAR_LR = 0.1
WEIGHT_DECAY = 0.01
ADAM_BETAS = (0.8, 0.95)
WARMUP_RATIO = 0.05      # from-scratch 必须 warmup（5% = 300 步 @ 6000 总步）
WARMDOWN_RATIO = 0.6
FINAL_LR_FRAC = 0.05

# ---------------------------------------------------------------------------
# 加载 train.py 的优化器 + helpers（exec 模式，与 continue_pretrain.py 同思路）
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC_DIR))

# 加载 train.py 中模型类之外的部分（MuonAdamW + adamw_step_fused 等）
_lines = (_SRC_DIR / 'train.py').read_text(encoding='utf-8').splitlines(keepends=True)
# train.py 的 setup 阶段从 "# Setup: tokenizer, model, optimizer, dataloader" 开始
# 我们只取它之前的代码（模型定义 + optimizer 类），然后 import 我们自己的 model_v2
_cut = next(i for i, l in enumerate(_lines) if '# Setup: tokenizer, model, optimizer, dataloader' in l)
_train_src = ''.join(_lines[:_cut])

# 假装 prepare 模块（train.py 顶部 import）
from prepare import MAX_SEQ_LEN, Tokenizer, make_dataloader, evaluate_bpb
fake = types.ModuleType('prepare')
fake.MAX_SEQ_LEN = MAX_SEQ_LEN
fake.TIME_BUDGET = 999999
fake.Tokenizer = Tokenizer
fake.make_dataloader = make_dataloader
fake.evaluate_bpb = evaluate_bpb
sys.modules['prepare'] = fake

_train_ns: dict = {}
exec(compile(_train_src, 'train.py', 'exec'), _train_ns)
MuonAdamW = _train_ns['MuonAdamW']

# ---------------------------------------------------------------------------
# 我们自己的模型
# ---------------------------------------------------------------------------
from model_v2 import GPT_v2, GPTConfigV2

# ---------------------------------------------------------------------------
# Helpers（与 continue_pretrain.py 一致）
# ---------------------------------------------------------------------------
def maybe_trim_host_memory():
    if _LIBC is None:
        return
    try:
        _LIBC.malloc_trim(0)
    except Exception:
        pass


def prepare_for_eval(device):
    gc_was_enabled = gc.isenabled()
    if not gc_was_enabled:
        gc.enable()
    gc.collect()
    maybe_trim_host_memory()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return gc_was_enabled


def finish_eval(device, gc_was_enabled):
    gc.collect()
    maybe_trim_host_memory()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    if not gc_was_enabled:
        gc.disable()


def save_checkpoint(path, ckpt_dict):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    out = dict(ckpt_dict)
    out['config'] = asdict(out['config']) if hasattr(out['config'], '__dataclass_fields__') else out['config']
    torch.save(out, path)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision('high')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
param_dtype = torch.float32
autocast_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
H100_BF16_PEAK_FLOPS = 989.5e12

# tokenizer（与 train.py / continue_pretrain.py 同方式）
tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f'Tokenizer vocab_size: {vocab_size}')

# 构建 v2 模型（vocab 用 tokenizer 实际值）
config = GPTConfigV2(
    sequence_len=MAX_SEQ_LEN,
    vocab_size=vocab_size,
    n_layer=36,
    n_head=12,
    n_embd=768,
    window_pattern='SSL',
    rope_theta=10000.0,
    rope_seq_len_mult=2,
    softcap=15.0,
    tie_lm_head=False,             # 沿用 v1 untied + lm_head std=0.001 init（独立诊断验证此路径稳定）
    kv_lora_rank=192,
    qk_nope_head_dim=48,
    qk_rope_head_dim=16,
    v_head_dim=64,
    mlp_intermediate_size=2048,
    sublayers_per_block=3,
)
print(f'Building GPT_v2 with config: {asdict(config)}')
model = GPT_v2(config).to(device=device, dtype=param_dtype)
model.init_weights(cast_embeddings_to_bfloat16=True)
nparams = sum(p.numel() for p in model.parameters())
print(f'Parameters: {nparams/1e6:.1f}M')
ns = model.num_scaling_params()
for k, v in ns.items():
    print(f'  {k}: {v/1e6:.2f}M')

# Gradient checkpointing
if os.environ.get('PRETRAIN_GRAD_CKPT', '0') == '1':
    from torch.utils.checkpoint import checkpoint
    print('Gradient checkpointing: ON (recompute attn/mlp activations)')
    for block in model.transformer.h:
        orig_attn = block.forward_attn_only
        orig_mlp = block.forward_mlp_only

        def make_attn(orig):
            def attn_ckpt(h, cos_sin, window_size, past_kv=None, use_cache=False):
                if use_cache or past_kv is not None or not torch.is_grad_enabled():
                    return orig(h, cos_sin, window_size, past_kv=past_kv, use_cache=use_cache)
                return checkpoint(orig, h, cos_sin, window_size, use_reentrant=False)
            return attn_ckpt

        def make_mlp(orig):
            def mlp_ckpt(h):
                if not torch.is_grad_enabled():
                    return orig(h)
                return checkpoint(orig, h, use_reentrant=False)
            return mlp_ckpt

        block.forward_attn_only = make_attn(orig_attn)
        block.forward_mlp_only = make_mlp(orig_mlp)

# Per-module compile（仅 MLP，attn 留 eager 配 FA3）
train_model = model
if os.environ.get('PRETRAIN_COMPILE', '1') == '1' and device.type == 'cuda':
    print('Per-module compile: mlp compiled (attn stays eager with FA3)')
    for block in model.transformer.h:
        block.mlp = torch.compile(block.mlp, dynamic=False)

# Optimizer
optimizer = model.setup_optimizer(
    MuonAdamW=MuonAdamW,
    unembedding_lr=UNEMBEDDING_LR, embedding_lr=EMBEDDING_LR,
    matrix_lr=MATRIX_LR, weight_decay=WEIGHT_DECAY,
    adam_betas=ADAM_BETAS, scalar_lr=SCALAR_LR,
)

# Dataloader
train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, 'train', device=device)
x, y, epoch = next(train_loader)

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd
print(f'Gradient accumulation steps: {grad_accum_steps}')
print(f'Starting from-scratch pretraining: {TOTAL_STEPS} steps')


# ---------------------------------------------------------------------------
# LR / momentum schedules（与 train.py 一致）
# ---------------------------------------------------------------------------
def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    if progress >= WARMDOWN_RATIO:
        frac = (1.0 - progress) / (1.0 - WARMDOWN_RATIO)
        return FINAL_LR_FRAC + (1.0 - FINAL_LR_FRAC) * frac
    return 1.0


def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95


def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
t_start_training = time.time()
total_training_time = 0.0
step = 0
best_val_bpb = float('inf')
no_improve_count = 0
_last_save_interval = max(EVAL_INTERVAL, 500)

while True:
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = train_model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)

    progress = min(step / TOTAL_STEPS, 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    cur_wd = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group['lr'] = group['initial_lr'] * lrm
        if group['kind'] == 'muon':
            group['momentum'] = muon_momentum
            group['weight_decay'] = cur_wd

    # Gradient clipping：from-scratch 训练标准，防 step 2 时 lr 一开就 grad spike 把 logit 推进 softcap 饱和区
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    dt = time.time() - t0
    total_training_time += dt
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt) if dt > 0 else 0
    num_flops_per_token = model.estimate_flops()
    mfu = 0.0 if device.type != 'cuda' else 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / H100_BF16_PEAK_FLOPS
    print(f"step {step:05d} ({100*progress:.1f}%) | loss: {train_loss.item():.6f} | "
          f"lrm: {lrm:.2f} | dt: {int(dt*1000)}ms | tok/sec: {tok_per_sec:,} | "
          f"mfu: {mfu:.1f}% | epoch: {epoch}    ", end='\r', flush=True)

    step += 1

    if step % _last_save_interval == 0:
        last_path = CHECKPOINT_OUT.replace('.pt', '_last.pt')
        save_checkpoint(last_path, dict(
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            config=config,
            val_bpb=best_val_bpb,
            step=step,
            from_scratch_v2=True,
        ))

    if step % EVAL_INTERVAL == 0:
        is_full = (step % FULL_EVAL_INTERVAL == 0)
        eval_type = 'full' if is_full else 'fast'
        eval_tokens = None if is_full else FAST_EVAL_TOKENS
        print(f"\nRunning {eval_type} eval at step {step}...", flush=True)
        gc_was_enabled = prepare_for_eval(device)
        model.eval()
        with torch.no_grad(), autocast_ctx:
            current_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE, device=device, max_tokens=eval_tokens)
        finish_eval(device, gc_was_enabled)
        train_model.train()
        print(f"\nstep {step} {eval_type}_eval val_bpb={current_bpb:.6f}")
        if is_full:
            if current_bpb < best_val_bpb:
                best_val_bpb = current_bpb
                no_improve_count = 0
                best_path = CHECKPOINT_OUT.replace('.pt', '_best.pt')
                save_checkpoint(best_path, dict(
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    config=config,
                    val_bpb=current_bpb,
                    step=step,
                    from_scratch_v2=True,
                ))
                print(f"Saved {best_path}: val_bpb={current_bpb:.6f}")
            else:
                no_improve_count += 1
                if no_improve_count >= EARLY_STOP_PATIENCE:
                    print(f"Early stopping at step {step}: no improvement for {EARLY_STOP_PATIENCE} full evals")
                    break

    if step >= TOTAL_STEPS:
        break

print()

# 最终评估 + 保存
model.eval()
gc_was_enabled = prepare_for_eval(device)
with torch.no_grad(), autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE, device=device)
finish_eval(device, gc_was_enabled)

t_end = time.time()
peak_vram_mb = 0.0 if device.type != 'cuda' else torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"best_val_bpb:     {best_val_bpb:.6f}")
print(f"total_steps:      {step}")
print(f"training_time_s:  {total_training_time:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")

ckpt_final = dict(
    model_state=model.state_dict(),
    optimizer_state=optimizer.state_dict(),
    config=config,
    val_bpb=val_bpb,
    step=step,
    from_scratch_v2=True,
)
_final_path = CHECKPOINT_OUT
save_checkpoint(_final_path, ckpt_final)
print(f"Final checkpoint saved to {_final_path}")
if val_bpb < best_val_bpb:
    best_path = CHECKPOINT_OUT.replace('.pt', '_best.pt')
    save_checkpoint(best_path, ckpt_final)
    print(f"Updated {best_path}: val_bpb={val_bpb:.6f}")
