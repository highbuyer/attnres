#!/usr/bin/env python3
"""继续预训练脚本。加载已有 checkpoint，用更新格式的数据继续训练。

用途：预训练数据中 Belle/Glaive 格式已从明文 Human:/Assistant: 改为特殊 token，
需要继续预训练让模型学会新格式，同时保留已有能力。

Usage: .venv/bin/python continue_pretrain.py [checkpoint_path]
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

import os
import sys
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# ---------------------------------------------------------------------------
# Config（支持 env var / sys.argv 覆盖）
# ---------------------------------------------------------------------------
CHECKPOINT_IN = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/tooltoken_d18_32k.pt'
CHECKPOINT_OUT = os.environ.get('CPT_OUT', 'checkpoints/tooltoken_continued.pt')

# 降低的学习率（约为预训练的 1/4）
MATRIX_LR = 0.01
EMBEDDING_LR = 0.05
UNEMBEDDING_LR = 0.001
SCALAR_LR = 0.025
WEIGHT_DECAY = 0.01
ADAM_BETAS = (0.8, 0.95)

TOTAL_STEPS = int(os.environ.get('CPT_STEPS', 500))
WARMUP_RATIO = 0.05         # 25 steps warmup
WARMDOWN_RATIO = 0.6
FINAL_LR_FRAC = 0.05

TOTAL_BATCH_SIZE = 2**19    # ~524K tokens per step
DEVICE_BATCH_SIZE = int(os.environ.get('CPT_BATCH', 8))       # depth=18: 196M; d36 建议降到 2
EVAL_INTERVAL = int(os.environ.get('CPT_EVAL_INTERVAL', 250))
EARLY_STOP_PATIENCE = 3

# ---------------------------------------------------------------------------
# 加载模型定义（复用 train.py）
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC_DIR))
lines = (_SRC_DIR / 'train.py').read_text(encoding='utf-8').splitlines(keepends=True)
cut = next(i for i, l in enumerate(lines) if '# Setup: tokenizer, model, optimizer, dataloader' in l)
src = ''.join(lines[:cut])

fake = types.ModuleType('prepare')
from prepare import MAX_SEQ_LEN, Tokenizer, make_dataloader, evaluate_bpb
fake.MAX_SEQ_LEN = MAX_SEQ_LEN
fake.TIME_BUDGET = 999999
fake.Tokenizer = Tokenizer
fake.make_dataloader = make_dataloader
fake.evaluate_bpb = evaluate_bpb
sys.modules['prepare'] = fake
ns: dict = {}
exec(compile(src, 'train.py', 'exec'), ns)
GPT = ns['GPT']
GPTConfig = ns['GPTConfig']
import __main__
__main__.GPT = GPT
__main__.GPTConfig = GPTConfig
GPT.__module__ = '__main__'
GPTConfig.__module__ = '__main__'
del sys.modules['prepare']


def _config_from_ckpt(raw_config):
    if isinstance(raw_config, dict):
        allowed = {f.name for f in fields(GPTConfig)}
        return GPTConfig(**{k: v for k, v in raw_config.items() if k in allowed})
    return raw_config


def _config_to_ckpt(config):
    return asdict(config)


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    ckpt['config'] = _config_from_ckpt(ckpt['config'])
    return ckpt


def save_checkpoint(path, ckpt):
    ckpt = dict(ckpt)
    ckpt['config'] = _config_to_ckpt(ckpt['config'])
    torch.save(ckpt, path)

# ---------------------------------------------------------------------------
# 加载 checkpoint
# ---------------------------------------------------------------------------
t_start = time.time()
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
param_dtype = torch.float32
compute_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
H100_BF16_PEAK_FLOPS = 989.5e12

print(f'Loading {CHECKPOINT_IN}...')
ckpt = load_checkpoint(CHECKPOINT_IN, device)
config = ckpt['config']
print(f'Model config: {asdict(config)}')

_metric_key = 'val_bpt' if 'val_bpt' in ckpt else 'val_bpb'
print(f'Checkpoint: {_metric_key}={ckpt[_metric_key]:.6f}, step={ckpt["step"]}')

# 构建模型并加载权重
model = GPT(config).to(device=device, dtype=param_dtype)
state = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model_state'].items()}
load_result = model.load_state_dict(state, strict=False)
if load_result.missing_keys:
    print(f'WARNING: missing keys: {load_result.missing_keys}')
if load_result.unexpected_keys:
    print(f'WARNING: unexpected keys: {load_result.unexpected_keys}')
model.to(dtype=param_dtype)

# gradient checkpointing：d36 seq=2048 backward 内存峰过高，需要重算激活
if os.environ.get('CPT_GRAD_CKPT', '0') == '1':
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
    print("Gradient checkpointing: ON (recompute attn/mlp activations)")

head_dim = config.n_embd // config.n_head
cos, sin = model._precompute_rotary_embeddings(model.rotary_seq_len, head_dim, device=device)
model.cos, model.sin = cos.to(compute_dtype), sin.to(compute_dtype)

# ---------------------------------------------------------------------------
# Tokenizer, optimizer, dataloader
# ---------------------------------------------------------------------------
tokenizer = Tokenizer.from_directory()
num_params = sum(p.numel() for p in model.parameters())
num_flops_per_token = model.estimate_flops()

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

optimizer = model.setup_optimizer(
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    scalar_lr=SCALAR_LR,
    adam_betas=ADAM_BETAS,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
)
for group in optimizer.param_groups:
    group["initial_lr"] = group["lr"]

if os.environ.get('NO_COMPILE') != '1' and device.type == 'cuda':
    model = torch.compile(model, dynamic=False)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train", device=device)
x, y, epoch = next(train_loader)

print(f'Gradient accumulation steps: {grad_accum_steps}')
print(f'Parameters: {num_params/1e6:.1f}M')
print(f'Starting continued pretraining: {TOTAL_STEPS} steps')

# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------
def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0
best_val_bpb = float('inf')
no_improve_count = 0

while True:
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)

    progress = min(step / TOTAL_STEPS, 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL: loss exploded")
        exit(1)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    if step > 10:
        total_training_time += dt

    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 0.0 if device.type != 'cuda' else 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / H100_BF16_PEAK_FLOPS

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch}    ", end="", flush=True)

    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()

    step += 1

    # 定期评估
    if step % EVAL_INTERVAL == 0:
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        with torch.no_grad(), autocast_ctx:
            current_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE, device=device)
        print(f"\nstep {step} eval val_bpb={current_bpb:.6f}")
        if current_bpb < best_val_bpb:
            best_val_bpb = current_bpb
            no_improve_count = 0
            best_ckpt = {
                'model_state': model.state_dict(),
                'config': config,
                'val_bpb': current_bpb,
                'step': step,
                'continued_pretrain': True,
            }
            save_checkpoint(CHECKPOINT_OUT, best_ckpt)
            print(f"Saved {CHECKPOINT_OUT}: val_bpb={current_bpb:.6f}")
        else:
            no_improve_count += 1
            if no_improve_count >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at step {step}: no improvement for {EARLY_STOP_PATIENCE} evals")
                break

    if step >= TOTAL_STEPS:
        break

print()

# ---------------------------------------------------------------------------
# 最终评估
# ---------------------------------------------------------------------------
del optimizer
gc.collect()
model.eval()
if device.type == 'cuda':
    torch.cuda.empty_cache()
with torch.no_grad(), autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE, device=device)

t_end = time.time()
peak_vram_mb = 0.0 if device.type != 'cuda' else torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"best_val_bpb:     {best_val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_steps:        {step}")

# 保存最终 checkpoint
ckpt_final = {
    'model_state': model.state_dict(),
    'config': config,
    'val_bpb': val_bpb,
    'step': step,
    'continued_pretrain': True,
}
save_checkpoint(CHECKPOINT_OUT.replace('.pt', '_final.pt'), ckpt_final)
print(f"Final checkpoint saved to {CHECKPOINT_OUT.replace('.pt', '_final.pt')}")

# 更新 best_checkpoint 如果有改善
if val_bpb < best_val_bpb:
    save_checkpoint(CHECKPOINT_OUT, ckpt_final)
    print(f"Updated {CHECKPOINT_OUT}: val_bpb={val_bpb:.6f}")

print(f"Done. Original: {_metric_key}={ckpt[_metric_key]:.6f}, Final: val_bpb={val_bpb:.6f}")
