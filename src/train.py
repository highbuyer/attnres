"""
Autoresearch pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Usage: uv run train.py
"""

import os
import sys
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
for _proxy_key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"]:
    os.environ.pop(_proxy_key, None)

# Tee stdout to run.log
class _Tee:
    def __init__(self, *files): self.files = files
    def write(self, s):
        for f in self.files: f.write(s)
    def flush(self):
        for f in self.files: f.flush()
_log_file = open("run.log", "w")
sys.stdout = _Tee(sys.__stdout__, _log_file)
sys.stderr = _Tee(sys.__stderr__, _log_file)

import gc
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass, asdict, fields

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention_window import build_causal_window_mask

try:
    from kernels import get_kernel
    from kernels.utils import get_local_kernel, install_kernel, select_revision_or_version
except ImportError:
    get_kernel = None
    get_local_kernel = None
    install_kernel = None
    select_revision_or_version = None


def _load_flash_attention_backend():
    if not torch.cuda.is_available() or get_kernel is None:
        return None
    cap = torch.cuda.get_device_capability()
    repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
    if install_kernel is not None and get_local_kernel is not None and select_revision_or_version is not None:
        try:
            revision = select_revision_or_version(repo, revision=None, version=None)
            package_name, variant_path = install_kernel(repo, revision=revision, local_files_only=True)
            return get_local_kernel(variant_path, package_name).flash_attn_interface
        except FileNotFoundError:
            pass
    try:
        return get_kernel(repo).flash_attn_interface
    except Exception:
        return None


def autocast_context(device):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def model_dtype_for_device(device):
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def maybe_cuda_synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def maybe_empty_cache(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()


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
    maybe_empty_cache(device)
    maybe_trim_host_memory()
    return gc_was_enabled


def finish_eval(device, gc_was_enabled):
    gc.collect()
    maybe_empty_cache(device)
    maybe_trim_host_memory()
    if not gc_was_enabled:
        gc.disable()


def peak_vram_mb(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


fa3 = _load_flash_attention_backend()

from prepare import MAX_SEQ_LEN, TIME_BUDGET as _TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb
TIME_BUDGET = 86400 # 24h ceiling (early stopping will terminate sooner)
TOTAL_STEPS = 6000  # step-based schedule target (progress = step / TOTAL_STEPS)
EVAL_INTERVAL = 500      # steps between val evaluations
EARLY_STOP_PATIENCE = 3  # stop after N evals with no improvement

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"
    rope_theta: float = 10000.0
    # 以下字段向后兼容：老 ckpt 不含这些 key 时，_config_from_ckpt 会用这里的默认值
    rope_seq_len_mult: int = 10  # rotary 预计算长度倍率；为长文本 NTK 扩展留余地。生产训练可降到 2
    softcap: float = 15.0        # 输出 logit 的 tanh 软顶
    tie_lm_head: bool = False    # True 时 lm_head.weight 与 wte.weight 共享（省 25M 参数）
    ve_layer_skip: tuple = ()    # P2: 砍掉指定层的 VE（value_embeds + ve_gate）。见 docs/VE_ABLATION_v2_full.md


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer, skip=()):
    """Returns True if layer should have Value Embedding (alternating, last always included).

    skip: iterable of layer indices to force-exclude (P2 VE pruning, see ve_layer_skip)."""
    if layer_idx in set(skip):
        return False
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer, getattr(config, "ve_layer_skip", ())) else None

    def forward(self, x, ve, cos_sin, window_size, past_kv=None, use_cache=False):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim).to(v.dtype)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels])).to(v.dtype)
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        if v.dtype != q.dtype:
            v = v.to(q.dtype)

        # KV cache: append past and use concatenated k/v for attention (q stays new-only).
        # past_k/v already have RoPE + norm applied at their prefill step.
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
        new_kv = (k, v) if use_cache else None

        if hasattr(fa3, 'flash_attn_func') and fa3.flash_attn_func is not None and q.is_cuda:
            y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # CPU fallback or non-FlashAttn environment
            q_sdpa = q.transpose(1, 2) # (B, H, T_q, D)
            k_sdpa = k.transpose(1, 2) # (B, Hkv, T_kv, D)
            v_sdpa = v.transpose(1, 2)

            T_q, T_kv = q.size(1), k.size(1)
            mask_rows = build_causal_window_mask(T_kv, window_size)
            attn_mask = None
            if mask_rows is not None:
                mask = torch.tensor(mask_rows, device=q.device, dtype=torch.bool)
                # align to bottom-right (FA3 semantics): q attends rows [T_kv - T_q : T_kv]
                attn_mask = mask[T_kv - T_q:T_kv, :]
            else:
                # causal bottom-right
                row_idx = torch.arange(T_q, device=q.device).unsqueeze(1) + (T_kv - T_q)
                col_idx = torch.arange(T_kv, device=q.device).unsqueeze(0)
                attn_mask = col_idx <= row_idx

            y = torch.nn.functional.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa,
                attn_mask=attn_mask,
                is_causal=False,  # 手动提供了 bottom-right causal 掩码
                enable_gqa=(self.n_head != self.n_kv_head),
            )
            y = y.transpose(1, 2) # (B, T_q, H, D)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        if use_cache:
            return y, new_kv
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward_attn(self, x, ve, cos_sin, window_size):
        """Attn with residual (standard use)."""
        return x + self.attn(norm(x), ve, cos_sin, window_size)

    def forward_mlp(self, x):
        """MLP with residual (standard use)."""
        return x + self.mlp(norm(x))

    def forward_attn_only(self, h, ve, cos_sin, window_size, past_kv=None, use_cache=False):
        """Attn without residual: paper AttnRes mode (h already is the attended state)."""
        return self.attn(norm(h), ve, cos_sin, window_size, past_kv=past_kv, use_cache=use_cache)

    def forward_mlp_only(self, h):
        """MLP without residual: paper AttnRes mode."""
        return self.mlp(norm(h))

    def forward(self, x, ve, cos_sin, window_size):
        x = self.forward_attn(x, ve, cos_sin, window_size)
        x = self.forward_mlp(x)
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if getattr(config, "tie_lm_head", False):
            # lm_head 与 wte 共享权重；load_state_dict 时若 ckpt 里 lm_head.weight 独立存在，
            # 则该独立张量会被替换（无 shape mismatch，但参数语义改变）——新一轮训练再开启
            self.lm_head.weight = self.transformer.wte.weight
        # Block AttnRes: cross-layer attention residual (Kimi 2026, Block variant, exact paper design)
        # Exact paper design: proj is Linear(n_embd->1) used as pseudo-query; K=RMSNorm(V), no separate K projection
        # 2*n_layer projections: even indices for pre-attn, odd indices for pre-mlp
        sublayers_per_block = 3  # sublayers per block → N=8 blocks for L=24 (paper §3.2: N=8)
        self.attnres_proj = nn.ModuleList([nn.Linear(config.n_embd, 1, bias=False) for _ in range(2 * config.n_layer)])
        self.attnres_norm = nn.ModuleList([nn.RMSNorm(config.n_embd) for _ in range(2 * config.n_layer)])  # paper: per-sublayer independent RMSNorm
        self.sublayers_per_block = sublayers_per_block
        # Value embeddings
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer, getattr(config, "ve_layer_skip", ()))
        })
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * getattr(config, "rope_seq_len_mult", 10)
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self, *, cast_embeddings_to_bfloat16=True):
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        if not getattr(self.config, "tie_lm_head", False):
            torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # AttnRes pseudo-query projections: zero init ensures uniform initial attention (paper §5)
        for proj in self.attnres_proj:
            torch.nn.init.zeros_(proj.weight)
        if cast_embeddings_to_bfloat16:
            self.transformer.wte.to(dtype=torch.bfloat16)
            for ve in self.value_embeds.values():
                ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=None, device=None):
        if base is None:
            base = self.config.rope_theta
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (-1, -1), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = self.transformer.wte.weight.numel() + value_embeds_numel
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window + 1, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        attnres = sum(p.numel() for p in list(self.attnres_proj.parameters()) + list(self.attnres_norm.parameters()))
        total = wte + value_embeds + lm_head + transformer_matrices + attnres
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        attnres_proj_params = list(self.attnres_proj.parameters())
        attnres_norm_params = list(self.attnres_norm.parameters())
        attnres_params = attnres_proj_params + attnres_norm_params
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(attnres_params))
        # Scale LR ∝ 1/√dmodel (tuned at 768 dim)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=attnres_proj_params, lr=scalar_lr, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=attnres_norm_params, lr=0.15, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction='mean', past_kvs=None, position_offset=0, use_cache=False):
        B, T = idx.size()
        T_past = past_kvs[0][0].size(1) if (past_kvs is not None and past_kvs[0] is not None) else position_offset
        assert T + T_past <= self.cos.size(1), f"{T=}+{T_past=} exceeds rotary cache {self.cos.size(1)}"
        cos_sin = self.cos[:, T_past:T_past + T], self.sin[:, T_past:T_past + T]

        x = self.transformer.wte(idx)
        x = norm(x)
        C = self.config.n_embd
        bs = self.sublayers_per_block

        def block_attn_res(completed_blocks, partial_block, proj_idx):
            """Paper Fig.2: attend over completed blocks + partial block (if any), return h.
            Two-pass implementation: avoids materializing (N,B,T,C) stack to save memory."""
            if bs == 0:
                return partial_block
            all_v = completed_blocks + ([partial_block] if partial_block is not None else [])
            if not all_v:
                return partial_block
            # Pass 1: compute per-block logits, only stack (N,B,T)
            proj_w = self.attnres_proj[proj_idx].weight[0]  # (C,)
            logits_list = []
            v_dtype = all_v[0].dtype
            for v in all_v:
                k = self.attnres_norm[proj_idx](v).to(v_dtype)
                logits_list.append(torch.einsum('c,btc->bt', proj_w.to(v_dtype), k))
            logits = torch.stack(logits_list, dim=0).float()  # (N, B, T)
            attn_w = logits.softmax(dim=0).to(v_dtype)       # (N, B, T)
            # Pass 2: weighted sum, no V stack materialization
            result = torch.zeros_like(all_v[0])
            for n in range(len(all_v)):
                result.add_(attn_w[n].unsqueeze(-1) * all_v[n])
            return result

        # Full grad flow: no detach anywhere (paper-exact).
        completed_blocks = []  # no detach: full grad flow
        partial_block = x      # b0 = token embedding (first block starts from embedding)
        sub_layer_count = 0
        new_kvs = [] if use_cache else None

        for i, block in enumerate(self.transformer.h):
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None

            # block boundary BEFORE attn (paper Fig.2 line 22-25)
            if bs > 0 and sub_layer_count % bs == 0 and sub_layer_count > 0:
                completed_blocks.append(partial_block)  # no detach: full grad flow
                partial_block = None  # paper: new block starts fresh, first attn_out becomes partial

            h = block_attn_res(completed_blocks, partial_block, proj_idx=2*i)
            layer_past_kv = past_kvs[i] if past_kvs is not None else None
            if use_cache:
                attn_out, layer_new_kv = block.forward_attn_only(
                    h, ve, cos_sin, self.window_sizes[i], past_kv=layer_past_kv, use_cache=True)
                assert new_kvs is not None
                new_kvs.append(layer_new_kv)
            else:
                attn_out = block.forward_attn_only(h, ve, cos_sin, self.window_sizes[i], past_kv=layer_past_kv)
            sub_layer_count += 1
            partial_block = attn_out if partial_block is None else partial_block + attn_out

            h = block_attn_res(completed_blocks, partial_block, proj_idx=2*i+1)
            mlp_out = block.forward_mlp_only(h)
            sub_layer_count += 1
            partial_block = partial_block + mlp_out
            x = partial_block

        x = norm(x)

        softcap = getattr(self.config, "softcap", 15.0)
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            return loss
        if use_cache:
            return logits, new_kvs
        return logits

# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    # NorMuon variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad if p.grad is not None else torch.zeros_like(p) for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO (depth=12 → 768d)
HEAD_DIM = 64           # target head dimension for attention
KV_HEADS = None         # number of KV heads for GQA (None = same as n_head = MHA)
WINDOW_PATTERN = "SSL" # sliding window pattern: L=full, S=half context

# Optimization
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
EMBEDDING_LR = 0.2      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.1         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.01     # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.6    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.05 # final LR as fraction of initial

# Model size
DEPTH = 12              # depth=12, AR=64 → 768d, ~6GB base + ~2-3GB AttnRes graph
DEVICE_BATCH_SIZE = 16  # fits with no-detach at AR=64

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dtype = model_dtype_for_device(device)
H100_BF16_PEAK_FLOPS = 989.5e12

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=KV_HEADS if KV_HEADS is not None else num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )

config = build_model_config(DEPTH)
print(f"Model config: {asdict(config)}")

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights(cast_embeddings_to_bfloat16=(device.type == "cuda"))

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

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

train_model = model
if os.environ.get("NO_COMPILE") != "1" and device.type == "cuda":
    for block in model.transformer.h:
        block.mlp = torch.compile(block.mlp, dynamic=False)
    print("Per-module compile: mlp compiled (attn stays eager with FA3)")

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train", device=device)
x, y, epoch = next(train_loader)  # prefetch first batch

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# Schedules (all based on progress = step / TOTAL_STEPS)

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
    maybe_cuda_synchronize(device)
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_context(device):
            loss = train_model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)

    # Progress and schedules (step-based so warmdown triggers correctly)
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

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    maybe_cuda_synchronize(device)
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 0.0 if device.type != "cuda" else 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / H100_BF16_PEAK_FLOPS
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    # GC management (Python's GC causes ~500ms stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    # Periodic val eval + early stopping
    if step % EVAL_INTERVAL == 0:
        gc_was_enabled = prepare_for_eval(device)
        model.eval()
        with torch.no_grad(), autocast_context(device):
            current_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE, device=device)
        finish_eval(device, gc_was_enabled)
        train_model.train()
        print(f"\nstep {step} eval val_bpb={current_bpb:.6f}")
        if current_bpb < best_val_bpb:
            best_val_bpb = current_bpb
            no_improve_count = 0
            # Save best checkpoint immediately
            best_ckpt = {'model_state': model.state_dict(), 'config': asdict(config), 'val_bpb': current_bpb, 'step': step}
            torch.save(best_ckpt, 'best_checkpoint.pt')
            print(f"best_checkpoint.pt saved: val_bpb={current_bpb:.6f} at step {step}")
        else:
            no_improve_count += 1
            if no_improve_count >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at step {step}: no improvement for {EARLY_STOP_PATIENCE} evals")
                break

    # Time's up — but only stop after warmup steps so we don't count compilation
    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r training log

total_tokens = step * TOTAL_BATCH_SIZE

# Final eval
del optimizer  # free optimizer states before eval
model.eval()
gc_was_enabled = prepare_for_eval(device)
with torch.no_grad(), autocast_context(device):
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE, device=device)
finish_eval(device, gc_was_enabled)

# Final summary
t_end = time.time()
startup_time = t_start_training - t_start
steady_state_mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10) / total_training_time / H100_BF16_PEAK_FLOPS if total_training_time > 0 else 0
if device.type != "cuda":
    steady_state_mfu = 0
peak_vram_mb = peak_vram_mb(device)

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")

# Save checkpoint
ckpt = {
    'model_state': model.state_dict(),
    'config': asdict(config),
    'val_bpb': val_bpb,
    'step': step,
}
torch.save(ckpt, 'checkpoint.pt')
print("checkpoint saved to checkpoint.pt")

# Save best checkpoint if improved
import os
best_bpb = float('inf')
if os.path.exists('best_checkpoint.pt'):
    best_ckpt = torch.load('best_checkpoint.pt', map_location='cpu', weights_only=False)
    best_bpb = best_ckpt.get('val_bpb', float('inf'))
if val_bpb < best_bpb:
    torch.save(ckpt, 'best_checkpoint.pt')
    print(f"best_checkpoint.pt updated: {best_bpb:.6f} -> {val_bpb:.6f}")
else:
    print(f"best_checkpoint.pt unchanged (best={best_bpb:.6f}, current={val_bpb:.6f})")

# Append result to results.tsv
import subprocess
try:
    commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], text=True).strip()
except Exception:
    commit = '-'
results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.tsv')
with open(results_path, 'a') as f:
    f.write(f"{commit}\t{val_bpb:.6f}\t{peak_vram_mb/1024:.1f}\t-\t"
            f"AttnRes spb={model.sublayers_per_block if hasattr(model, 'sublayers_per_block') else '?'} "
            f"emb_lr={EMBEDDING_LR} scalar_lr={SCALAR_LR} final_lr_frac={FINAL_LR_FRAC} "
            f"steps={step} depth={DEPTH} ar={ASPECT_RATIO}\n")
