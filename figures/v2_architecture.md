# GPT_v2 Architecture (277.8M)

```mermaid
%%{init: {'themeVariables': {'fontFamily': '"DejaVu Sans", "Liberation Sans", sans-serif'}}}%%
flowchart TB
    subgraph Input
        tokens["input tokens<br/>(batch, seq_len)"]
    end

    subgraph Embedding["wte (AdamW, lr=0.2)"]
        wte["nn.Embedding<br/>vocab=32774 → d=768<br/>std=1.0 init"]
    end

    tokens --> wte
    wte --> blocks

    subgraph blocks["Transformer Block × 36"]
        direction TB

        subgraph attnres["AttnRes Block"]
            direction LR
            attn_branch["MLA Branch"]
            res_branch["Residual Branch"]
            scalar["learnable scalar<br/>(AdamW norm)"]
            attnres_norm["AttnRes Norm<br/>(AdamW norm)"]

            attn_branch --> scalar
            res_branch --> scalar
            scalar --> attnres_norm
        end

        subgraph mla["MLA (Multi-head Latent Attention)"]
            direction TB
            q_proj["Q projection<br/>qk_nope=48, qk_rope=16<br/>→ q_per_head=64"]
            kv_compress["KV Compression<br/>c_kv_a: 768 → kv_lora_rank=192<br/>(AdamW LoRA)"]
            kv_expand["KV Expansion<br/>c_kv_b: 192 → 512<br/>(AdamW LoRA)"]
            rope["RoPE on qk_rope + kv"]
            attn_op["Scaled Dot-Product Attention<br/>v_head=64"]
            out_proj["O projection<br/>(Muon)"]

            q_proj --> attn_op
            kv_compress --> kv_expand --> rope --> attn_op
            attn_op --> out_proj
        end

        subgraph ffn["SwiGLU FFN"]
            direction LR
            gate["Gate proj<br/>768 → 2048<br/>(Muon)"]
            up["Up proj<br/>768 → 2048<br/>(Muon)"]
            silu["SiLU(gate)"]
            mul["⊙"]
            down["Down proj<br/>2048 → 768<br/>(Muon)"]

            gate --> silu --> mul
            up --> mul
            mul --> down
        end

        attnres --> ffn
    end

    blocks --> final_norm

    subgraph Output
        final_norm["final RMSNorm"]
        lm_head["lm_head<br/>768 → 32774<br/>(AdamW, lr=0.004)<br/>⚠️ untied from wte"]
        logits["logits<br/>(batch, seq_len, 32774)"]
    end

    final_norm --> lm_head --> logits

    subgraph legend[" "]
        opt_muon["Muon optimizer"]
        opt_adamw["AdamW optimizer"]
    end

    style wte fill:#e1f5fe
    style lm_head fill:#fff3e0
    style final_norm fill:#fce4ec
    style attnres fill:#e8f5e9
    style mla fill:#f3e5f5
    style ffn fill:#fff8e1
```

## Parameter Breakdown

| Component | Params | Optimizer | LR |
|-----------|--------|-----------|-----|
| wte (Embedding) | 32774 × 768 = 25.2M | AdamW | 0.2 |
| MLA (c_kv_a + c_kv_b) | 768×192 + 192×512 ≈ 0.25M | AdamW (LoRA) | — |
| MLA (Q, O, other) | — | Muon | — |
| SwiGLU (gate+up+down) × 36 | 3 × 768 × 2048 × 36 ≈ 169.9M | Muon | — |
| AttnRes (scalar + norm) × 18 | small | AdamW (norm) | — |
| lm_head | 768 × 32774 = 25.2M | AdamW | 0.004 |
| **Total** | **~277.8M** | | |

## Key Design Decisions

```
tie_lm_head = False  ← 关键！wte 和 lm_head 各学各的
                      避免 final RMSNorm + tied 导致的 logit 饱和

final RMSNorm        ← 稳定 hidden scale，但必须配合 untied head

混合 Optimizer        ← Muon 处理方阵投影（收敛快）
                      AdamW 处理非方阵 + embedding + norm
```

## Removed from v1 (758M)

```
✗ VE (Vocabulary Embedding) × 18 layers  →  -453M params
  每个 layer 独立的 vocab-level embedding 投影
```
