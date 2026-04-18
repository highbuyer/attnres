# Weaknesses v10 e2e — REJECTED run

> ckpt: `checkpoints/sft_w10_best.pt` (from `sft_tool_summary_v5_best.pt` + 200 steps on patch_w10 ×50 oversampled)
> val_bpt=1.3407 (v5_best: 1.2297)
> audit: `runs/weakness_v10_e2e.jsonl`

**Decision: REJECT — main checkpoint remains sft_tool_summary_v5_best.pt**

## Gate result

| gate | threshold | v9 | v10 | pass? |
|------|-----------|------|-------|-------|
| `net_user_failure` | ≤ 2 | 2 | **4** | ❌ |
| `tool_false_fire` | == 0 | 0 | **2** | ❌ |
| `hallucination_fact_mismatch` | ≤ 1 | 2 | 2 | ❌ (tie, needed strict drop) |
| `degenerate` | == 0 | 0 | 0 | ✓ |

## What went wrong

Three separate failure modes appeared simultaneously.

### 1. Patched questions stayed broken (ks_07, ks_10)

`patch_w10.jsonl` had 3 samples teaching "人类有 46 条染色体（23 对）".
v10 still emits `"人类有23条染色体。"` — the canonical patch answer
did not override the pre-trained mis-fact. 50× oversampling of 50
samples (2500 rows) against 7126 base rows was not enough.

Worse, `ks_10 二氧化碳` was **correct in v9** ("CO₂") and became
**broken in v10** ("2 o₂"). The patch set did contain two CO₂
samples, but the short-run LR schedule shifted the tokenizer-embedding
paths just enough to corrupt an already-correct output. This is the
"w1 think_v1 regression" pattern — val_bpt changes don't predict
capability changes, and partial fine-tuning can damage unrelated
competencies.

### 2. tool_false_fire resurfaced (cd_03, mt_05)

v9 had 0 tool_false_fire thanks to the w3 logit mask on
`<|tool_call_start|>`. The mask is still active, but for
`mt_05 Transformer 架构的核心机制` the prompt now slips past
`should_ban_tool` after the fine-tune pushed the hidden state
slightly — apparently just enough that some unmasked regex branch
triggers instead. `cd_03 hello world` also regressed to a bogus
`read_file{"query":"MAX_TOKENS"}` tool call.

Only 200 steps of training shifted the model enough to re-open holes
w3 had papered over.

### 3. hallucination unchanged (same 2 as v9)

`ks_04 地球绕太阳` and `ks_07 染色体` are still labeled
hallucination_fact_mismatch. w10 did not move the needle on the
questions it was designed to fix.

## Why this matches w1 / w2 postmortems

The pattern is now clear across three attempts (w1 think_v1,
w2 plan_a, w10):

- Partial SFT on a small patch (30-50 samples) with heavy
  oversampling (50×) to override a 7126-sample base does not learn
  the patch cleanly.
- It does reshape the distribution enough to perturb unrelated
  capabilities (tool-call routing, previously-correct facts).
- Net effect is systematically negative even when val_bpt is
  comparable.

**Operational takeaway**: do not attempt more small-patch SFT on
v5_best. Fact-level gaps require either (a) a much larger patch set
(>1000 samples, mixed with tool-call preservers), or (b) baking the
facts into P2 pretraining.

## Artifacts retained vs discarded

- `datasets/tool_call_samples/patch_w10.jsonl` — **keep**. It is the
  seed set for whatever fact-rescue strategy comes next; re-usable in
  P2 pretraining.
- `checkpoints/sft_w10.pt`, `checkpoints/sft_w10_best.pt` — left on
  disk for forensic reference (same convention as sft_w2_a, which
  also regressed). Not referenced by any deploy path.
- `runs/weakness_v10_e2e.jsonl`, this file — **keep** as the
  authoritative record of the rejection.

## v9 remains the production ckpt

`sft_tool_summary_v5_best.pt` is unchanged. Deploy paths
(`weiyan-api`, `src/infer.py`, `scripts/self_audit.py`) still target
it by default. v9 audit (`docs/WEAKNESSES_v9_e2e.md`) remains the
current net_user_failure baseline at 2/64.
