"""新架构 v2: MLA + SwiGLU + 砍 VE + tie_lm_head + AttnRes 保留。

设计原则：
  - 全套现代化一次到位（"以后不想再改"）
  - K/V head_dim 一致 (64)：q_nope+q_rope=48+16=64，v=64 → FA3 可用
  - Block AttnRes 保留（参数小，已实现）
  - 不带迁移逻辑：from-scratch 重训
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Optional FA3 (与 train.py 保持一致的 import 形态)
# ---------------------------------------------------------------------------
try:
    from kernels import get_kernel
    fa3 = get_kernel("flash_attn_3")
except Exception:
    class _Fa3Stub:
        flash_attn_func = None
    fa3 = _Fa3Stub()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class GPTConfigV2:
    sequence_len: int = 2048
    vocab_size: int = 32774          # 32768 + 6 tool tokens（含 ASST_ID 等）
    n_layer: int = 36
    n_head: int = 12
    n_embd: int = 768
    window_pattern: str = "SSL"
    rope_theta: float = 10000.0
    rope_seq_len_mult: int = 2       # 不再为 NTK 留巨大 buffer
    softcap: float = 15.0
    tie_lm_head: bool = False        # ⚠️ 与 final RMSNorm + zero-init c_proj/down_proj 组合时
                                     # tie=True 会让第一次更新后 logit 直接打到 ±softcap 饱和（loss 跳 16+）
                                     # 沿用 v1 的 untied + lm_head std=0.001 init 是已验证稳定路径

    # MLA 配置
    kv_lora_rank: int = 192          # KV latent 维度（≈ d/4）
    qk_nope_head_dim: int = 48       # 不带 RoPE 的 head 维度
    qk_rope_head_dim: int = 16       # 带 RoPE 的 head 维度
    v_head_dim: int = 64             # V 头维度（= qk_nope+qk_rope, 让 FA3 可用）

    # SwiGLU 配置
    mlp_intermediate_size: int = 2048  # 8/3 × 768，标准 SwiGLU 比例

    # AttnRes 配置（保留与 v1 一致）
    sublayers_per_block: int = 3      # 等价 N≈12 blocks for L=36

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    """与 train.py 保持一致：x shape = (B, T, H, D), cos/sin shape = (1, T, 1, D)"""
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


def precompute_rope(seq_len: int, head_dim: int, base: float, device=None, dtype=torch.bfloat16):
    """与 train.py._precompute_rotary_embeddings 同形态。"""
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    cos, sin = freqs.cos().to(dtype), freqs.sin().to(dtype)
    cos = cos[None, :, None, :]   # (1, T, 1, head_dim/2)
    sin = sin[None, :, None, :]
    return cos, sin


# ---------------------------------------------------------------------------
# MLA Attention
# ---------------------------------------------------------------------------
class MLAttention(nn.Module):
    """简化版 Multi-head Latent Attention（DeepSeek-V2/V3 风格）。

    - q 不用 LoRA（d=768 不够大，q-LoRA 收益不抵复杂度）
    - kv 用 down-up：c_kv_a (d→r+qk_rope) + c_kv_b (r→n_head*(qk_nope+v_head))
    - 解耦 RoPE：k_pe 是 1 个 shared head（不是 n_head），expand 到 n_head 走 attn
    - K/V head_dim = v_head_dim = qk_nope+qk_rope = 64 → FA3 可用
    """
    def __init__(self, config: GPTConfigV2):
        super().__init__()
        self.n_head = config.n_head
        self.qk_nope = config.qk_nope_head_dim
        self.qk_rope = config.qk_rope_head_dim
        self.qk_total = config.qk_head_dim       # = qk_nope + qk_rope
        self.v_head = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank

        # Q：直接投影到 n_head*(qk_nope+qk_rope)
        self.c_q = nn.Linear(config.n_embd, self.n_head * self.qk_total, bias=False)
        # KV down：投到 latent r + 1 个 shared rope head
        self.c_kv_a = nn.Linear(config.n_embd, self.kv_lora_rank + self.qk_rope, bias=False)
        # KV LoRA norm（DeepSeek 论文里 c_kv_a 输出过 RMSNorm 再进 c_kv_b）
        self.kv_a_norm = nn.RMSNorm(self.kv_lora_rank)
        # KV up：从 latent 解出 n_head*(qk_nope + v_head)
        self.c_kv_b = nn.Linear(self.kv_lora_rank, self.n_head * (self.qk_nope + self.v_head), bias=False)
        # 输出投影
        self.c_proj = nn.Linear(self.n_head * self.v_head, config.n_embd, bias=False)

    def forward(self, x, cos_sin, window_size, past_kv=None, use_cache=False):
        B, T, C = x.size()

        # Q 投影 → (B, T, n_head, qk_total)
        q = self.c_q(x).view(B, T, self.n_head, self.qk_total)
        q_nope, q_pe = q.split([self.qk_nope, self.qk_rope], dim=-1)

        # KV down 投影 → (B, T, r+qk_rope)
        kv_a = self.c_kv_a(x)
        kv_lora, k_pe = kv_a.split([self.kv_lora_rank, self.qk_rope], dim=-1)
        kv_lora = self.kv_a_norm(kv_lora)
        # KV up → (B, T, n_head*(qk_nope+v_head))
        kv = self.c_kv_b(kv_lora).view(B, T, self.n_head, self.qk_nope + self.v_head)
        k_nope, v = kv.split([self.qk_nope, self.v_head], dim=-1)

        # k_pe: (B, T, qk_rope) → (B, T, 1, qk_rope) 以便 RoPE
        k_pe = k_pe.view(B, T, 1, self.qk_rope)

        # 应用 RoPE
        cos, sin = cos_sin
        q_pe = apply_rotary_emb(q_pe, cos, sin)
        k_pe = apply_rotary_emb(k_pe, cos, sin)
        # k_pe expand 到 n_head（共享同一份 RoPE k）
        k_pe = k_pe.expand(B, T, self.n_head, self.qk_rope)

        # concat q_nope+q_pe，k_nope+k_pe
        q = torch.cat([q_nope, q_pe], dim=-1)        # (B, T, n_head, qk_total=64)
        k = torch.cat([k_nope, k_pe], dim=-1)        # (B, T, n_head, qk_total=64)

        # 与 train.py 保持一致：QK norm
        q, k = norm(q), norm(k)
        if v.dtype != q.dtype:
            v = v.to(q.dtype)

        # KV cache（同 train.py 风格：past_k/past_v 已应用 RoPE+norm）
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
        new_kv = (k, v) if use_cache else None

        # K/V head_dim 都是 64（v_head=64，qk_total=64）→ FA3 可用
        if hasattr(fa3, 'flash_attn_func') and fa3.flash_attn_func is not None and q.is_cuda:
            y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # CPU/无 FA3 fallback：SDPA
            q_sdpa = q.transpose(1, 2)   # (B, H, T_q, D)
            k_sdpa = k.transpose(1, 2)
            v_sdpa = v.transpose(1, 2)
            T_q, T_kv = q.size(1), k.size(1)
            row_idx = torch.arange(T_q, device=q.device).unsqueeze(1) + (T_kv - T_q)
            col_idx = torch.arange(T_kv, device=q.device).unsqueeze(0)
            attn_mask = col_idx <= row_idx
            y = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, attn_mask=attn_mask, is_causal=False)
            y = y.transpose(1, 2)   # (B, T, H, D)
        y = y.contiguous().view(B, T, self.n_head * self.v_head)
        y = self.c_proj(y)
        if use_cache:
            return y, new_kv
        return y


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------
class SwiGLU(nn.Module):
    def __init__(self, config: GPTConfigV2):
        super().__init__()
        h = config.mlp_intermediate_size
        self.gate_proj = nn.Linear(config.n_embd, h, bias=False)
        self.up_proj = nn.Linear(config.n_embd, h, bias=False)
        self.down_proj = nn.Linear(h, config.n_embd, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, config: GPTConfigV2):
        super().__init__()
        self.attn = MLAttention(config)
        self.mlp = SwiGLU(config)

    def forward_attn_only(self, h, cos_sin, window_size, past_kv=None, use_cache=False):
        return self.attn(norm(h), cos_sin, window_size, past_kv=past_kv, use_cache=use_cache)

    def forward_mlp_only(self, h):
        return self.mlp(norm(h))


# ---------------------------------------------------------------------------
# GPT v2
# ---------------------------------------------------------------------------
class GPT_v2(nn.Module):
    def __init__(self, config: GPTConfigV2):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_lm_head:
            self.lm_head.weight = self.transformer.wte.weight

        # AttnRes（与 v1 一致）
        self.attnres_proj = nn.ModuleList(
            [nn.Linear(config.n_embd, 1, bias=False) for _ in range(2 * config.n_layer)]
        )
        self.attnres_norm = nn.ModuleList(
            [nn.RMSNorm(config.n_embd) for _ in range(2 * config.n_layer)]
        )
        self.sublayers_per_block = config.sublayers_per_block

        # RoPE 预计算（仅作用于 qk_rope_head_dim 维度）
        self.rotary_seq_len = config.sequence_len * config.rope_seq_len_mult
        cos, sin = precompute_rope(self.rotary_seq_len, config.qk_rope_head_dim, config.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self, *, cast_embeddings_to_bfloat16=True):
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5

        # Embedding（lm_head 共享，无需单独 init）
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        if not self.config.tie_lm_head:
            # untied 时 lm_head 必须用极小 init（与 train.py:339 一致）：
            # 防止 step 1 一更新就让 logit 进入 softcap 饱和区（=10 的 cross-entropy 跳 16+）
            torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        for block in self.transformer.h:
            # MLA 投影
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_kv_a.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_kv_b.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            # SwiGLU
            torch.nn.init.uniform_(block.mlp.gate_proj.weight, -s, s)
            torch.nn.init.uniform_(block.mlp.up_proj.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.down_proj.weight)

        # AttnRes：zero init → 起始等权重 (paper §5)
        for proj in self.attnres_proj:
            torch.nn.init.zeros_(proj.weight)

        if cast_embeddings_to_bfloat16:
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (-1, -1), "S": (short_window, 0)}
        sizes = [char_to_window[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        sizes[-1] = (long_window, 0)
        return sizes

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        nparams_exclude = self.transformer.wte.weight.numel()
        h = self.config.n_head
        q = self.config.qk_head_dim
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective = t if window < 0 else min(window + 1, t)
            attn_flops += 12 * h * q * effective
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        # tie_lm_head=True 时 lm_head.weight is wte.weight，不再单独计
        lm_head = 0 if self.config.tie_lm_head else sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        attnres = sum(p.numel() for p in list(self.attnres_proj.parameters()) + list(self.attnres_norm.parameters()))
        total = wte + lm_head + transformer_matrices + attnres
        return {
            'wte': wte, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'attnres': attnres,
            'total': total,
        }

    def setup_optimizer(self, *, MuonAdamW, unembedding_lr=0.004, embedding_lr=0.2,
                        matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95),
                        scalar_lr=0.5):
        """与 train.py.GPT.setup_optimizer 同结构，删除 value_embeds 组；
        tie_lm_head=True 时 lm_head 与 wte 共参数，归到 embedding_params 不再独立。
        """
        model_dim = self.config.n_embd
        # 区分 2D 矩阵 vs 1D 标量，并把 MLA 的 LoRA 矩阵从 Muon 移到 AdamW
        # Muon 的 NS 迭代假设近方阵；c_kv_a (~r+rope, d) / c_kv_b (n*(nope+v), r) 极度非方
        # 实测会让 step 2 起 loss 立刻飙升到 17（可能 NS 不收敛产生病态更新）
        muon_params = []
        adamw_lora_params = []
        scalar_block_params = []
        for name, p in self.transformer.h.named_parameters():
            if p.ndim < 2:
                scalar_block_params.append(p)
            elif name.endswith("attn.c_kv_a.weight") or name.endswith("attn.c_kv_b.weight"):
                adamw_lora_params.append(p)
            else:
                muon_params.append(p)
        matrix_params = muon_params  # 仅这些进 Muon
        # tie_lm_head=True 时 wte 同时承担 lm_head 角色：用 unembedding_lr（小 50×）
        # 防 logit 投影震荡，与业界 tie_lm_head 实践对齐
        if self.config.tie_lm_head:
            embedding_params = []  # wte 走 lm_head 那组
            lm_head_params = list(self.transformer.wte.parameters())
        else:
            embedding_params = list(self.transformer.wte.parameters())
            lm_head_params = list(self.lm_head.parameters())
        attnres_proj_params = list(self.attnres_proj.parameters())
        attnres_norm_params = list(self.attnres_norm.parameters())
        attnres_params = attnres_proj_params + attnres_norm_params
        # sanity：tied 情况下 list(self.parameters()) 不会重复算 tied 参数
        all_p = list(self.parameters())
        assert len(all_p) == (len(matrix_params) + len(adamw_lora_params) + len(scalar_block_params) +
                              len(embedding_params) + len(lm_head_params) + len(attnres_params)), (
            f"param count mismatch: total={len(all_p)} vs sum="
            f"{len(matrix_params)+len(adamw_lora_params)+len(scalar_block_params)+len(embedding_params)+len(lm_head_params)+len(attnres_params)}")

        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = []
        if lm_head_params:
            param_groups.append(dict(kind='adamw', params=lm_head_params,
                                     lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas,
                                     eps=1e-10, weight_decay=0.0))
        if embedding_params:
            param_groups.append(dict(kind='adamw', params=embedding_params,
                                     lr=embedding_lr * dmodel_lr_scale, betas=adam_betas,
                                     eps=1e-10, weight_decay=0.0))
        param_groups.append(dict(kind='adamw', params=attnres_proj_params,
                                 lr=scalar_lr, betas=adam_betas, eps=1e-10, weight_decay=0.0))
        param_groups.append(dict(kind='adamw', params=attnres_norm_params,
                                 lr=0.15, betas=adam_betas, eps=1e-10, weight_decay=0.0))
        # 1D 参数（如 MLA 内部 RMSNorm.weight）走 AdamW scalar 组
        if scalar_block_params:
            param_groups.append(dict(kind='adamw', params=scalar_block_params,
                                     lr=0.15, betas=adam_betas, eps=1e-10, weight_decay=0.0))
        # MLA LoRA 矩阵走 AdamW（Muon 在极度非方 shape 上不稳）
        if adamw_lora_params:
            param_groups.append(dict(kind='adamw', params=adamw_lora_params,
                                     lr=matrix_lr * dmodel_lr_scale, betas=adam_betas,
                                     eps=1e-10, weight_decay=weight_decay))
        # Muon 按矩阵 shape 分组（与 train.py 一致）
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

    def forward(self, idx, targets=None, reduction='mean', past_kvs=None,
                position_offset=0, use_cache=False):
        B, T = idx.size()
        T_past = past_kvs[0][0].size(1) if (past_kvs is not None and past_kvs[0] is not None) else position_offset
        assert T + T_past <= self.cos.size(1), f"{T=}+{T_past=} exceeds rotary cache {self.cos.size(1)}"
        cos_sin = self.cos[:, T_past:T_past + T], self.sin[:, T_past:T_past + T]

        x = self.transformer.wte(idx)
        x = norm(x)
        bs = self.sublayers_per_block

        def block_attn_res(completed_blocks, partial_block, proj_idx):
            if bs == 0:
                return partial_block
            all_v = completed_blocks + ([partial_block] if partial_block is not None else [])
            if not all_v:
                return partial_block
            proj_w = self.attnres_proj[proj_idx].weight[0]
            v_dtype = all_v[0].dtype
            logits_list = []
            for v in all_v:
                k = self.attnres_norm[proj_idx](v).to(v_dtype)
                logits_list.append(torch.einsum('c,btc->bt', proj_w.to(v_dtype), k))
            logits = torch.stack(logits_list, dim=0).float()
            attn_w = logits.softmax(dim=0).to(v_dtype)
            result = torch.zeros_like(all_v[0])
            for n in range(len(all_v)):
                result.add_(attn_w[n].unsqueeze(-1) * all_v[n])
            return result

        completed_blocks = []
        partial_block = x
        sub_layer_count = 0
        new_kvs = [] if use_cache else None

        for i, block in enumerate(self.transformer.h):
            if bs > 0 and sub_layer_count % bs == 0 and sub_layer_count > 0:
                completed_blocks.append(partial_block)
                partial_block = None

            h = block_attn_res(completed_blocks, partial_block, proj_idx=2*i)
            layer_past_kv = past_kvs[i] if past_kvs is not None else None
            if use_cache:
                attn_out, layer_new_kv = block.forward_attn_only(
                    h, cos_sin, self.window_sizes[i], past_kv=layer_past_kv, use_cache=True)
                assert new_kvs is not None
                new_kvs.append(layer_new_kv)
            else:
                attn_out = block.forward_attn_only(h, cos_sin, self.window_sizes[i], past_kv=layer_past_kv)
            sub_layer_count += 1
            partial_block = attn_out if partial_block is None else partial_block + attn_out

            h = block_attn_res(completed_blocks, partial_block, proj_idx=2*i+1)
            mlp_out = block.forward_mlp_only(h)
            sub_layer_count += 1
            partial_block = partial_block + mlp_out
            x = partial_block

        x = norm(x)

        softcap = self.config.softcap
        logits = self.lm_head(x).float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            return loss
        if use_cache:
            return logits, new_kvs
        return logits
