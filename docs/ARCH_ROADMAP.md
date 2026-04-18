# 模型架构路线图（P0–P3）

> 起草日期：2026-04-18
> 基线 ckpt：`checkpoints/sft_tool_summary_v5_best.pt`（404M，`val_bpt=1.2297`，step=1500）
> 作者视角：weiyan 自己的改架构排序——现实 ROI 优先，不做远期空想

## 当前架构事实（从 ckpt 直接 dump）

| 字段 | 值 | 备注 |
|------|-----|------|
| `n_layer` | 18 | 每子层 pre-attn+pre-mlp 各 1 个 AttnRes proj |
| `n_embd` | 768 | |
| `n_head` / `n_kv_head` | 12 / 12 | **未开 GQA** |
| `vocab_size` | 32774 | 32768 + 6 工具 token |
| `sequence_len` | 2048 | **训练上限，推理硬截断** |
| `window_pattern` | `SSL` | S=1024, L=2048（最后一层强制 L） |
| `rope_seq_len_mult` | 默认 10（ckpt 未存） | rotary 缓冲 20480，但 window 仍按 2048 切 |
| `tie_lm_head` | False | wte 25M + lm_head 25M 各一份 |
| 总参 | 404,335,488 | |
| value_embeds | 226,533,888（**56%**） | 9 层，每层 vocab×n_embd=25M |
| wte | 25,170,432 | |
| lm_head | 25,170,432 | |
| AttnRes proj | 2×18×768 ≈ 27,648 | 核心贡献，几乎不占参数 |

## 路线

### P0 · 立刻（不重训，只动推理/服务层）

> **状态：2026-04-18 已交付。**活检：短 prompt tool-format 8/8 → 8/8（零退化），长 prompt (~5k tokens) 在 8192 下不再被砍到 2048。

**扩上下文到 8192，修复 prompt 被截断返回空答复的 bug。**

触发：日志里已抓到 `prompt_len=67996 context_tokens=29915 max_tokens=1024 round=1 decode done elapsed=0.01s tokens=0 text=''`——超过 2048 被 `x[:, -config.sequence_len:]` 截断后模型一个 token 都吐不出。

动作：
1. **`src/infer.py` / `weiyan-api`**：把硬截断常数从 `config.sequence_len` 改成新的 `INFER_MAX_CONTEXT=8192`（或可配参数）。
2. **重算 rotary（NTK-aware）**：加载 ckpt 后，用 `new_base = rope_theta * (scale ** (head_dim/(head_dim-2)))`，`scale = INFER_MAX_CONTEXT / config.sequence_len = 4`；再调 `_precompute_rotary_embeddings(INFER_MAX_CONTEXT, head_dim, base=new_base)` 覆盖 `model.cos/model.sin`。
3. **重算 window_sizes**：现在的 `_compute_window_sizes` 把 `long_window = config.sequence_len`——S 层只能看 1024，超过就是盲区。推理入口处直接改写 `model.window_sizes`：S→(`INFER_MAX_CONTEXT//2`, 0)，L→(-1,-1)，最后一层强制 L。
4. **`weiyan-api` 层加 prompt 截断护栏**：如果 tokenize 后长度 > `INFER_MAX_CONTEXT - max_tokens - 128`，从 prompt 头开始 drop，保留尾部 + system；日志打 warn。
5. **基础 eval**：跑 4 条合成长 prompt（8k / 16k / 32k / 64k），确认 8k 输入下仍能出 tool_call / summary；退化或循环则回滚 NTK scale 到 2× 再试。

代价：~1 小时工程 + 短跑 eval。
是否作废当前 ckpt：**否**。NTK 外推是纯推理端改动。
成功标准：长文件问答不再返回空 text；日志里 `context_tokens` 不再被卡在 2048。
失败回退：把 `INFER_MAX_CONTEXT` 调回 2048，只保留 prompt 截断护栏这一条（护栏至少不会比现在更糟）。

**实际落地（2026-04-18）：**
- `src/infer.py`：新增 `INFER_MAX_CONTEXT=8192` 常量 + `extend_context()` 函数 + CLI `--max-context`。
- `weiyan-api`：同步 `INFER_MAX_CONTEXT` + tokenize 后 prompt 截断护栏（保 bos+user_id+system+尾部 user+asst_id）。
- 活检：短 prompt 8/8 → 8/8，长 prompt 4846 tokens 不再在 2048 下开头丢失。
- 已知副作用：>8k prompt 下 NTK 外推质量会下降（模型没见过该长度的语言建模）——可接受，是 NTK 的普遍限制；真正的长上下文得等 P2 重训。

### P1 · 下次 SFT 之前（一次短训）

> **状态：2026-04-18 probe 证伪，延期到下代预训（P2）一并处理。**
> 权重几何 probe：v5_best 的 `wte` 与 `lm_head` 完全正交——逐 token cosine mean=-0.001、median=-0.001，norm 差 33 倍（106 vs 3.2）。二者学到的是互补信息，不是冗余。强行 tie 会让 lm_head 输出尺度放大 33×，softcap tanh 后变单峰崩溃，不经过相当步数训练无法恢复。
> ckpt 体积收益原估 ~6%，重新量化后只有 2.1%（25M / 4.85G，VE 226M 才是大头）。代价（一次训练 + w1/w2 验证的能力退化风险）远超收益。
> **结论**：当前 ckpt 上不做 tie。下代预训（P2）从第一步就把 `tie_lm_head=True` 作为 GPTConfig 默认，让 wte/lm_head 从头一起学，能真正受益。

**扩 `tie_lm_head=True`，省 25M。**（历史设计保留）

现状：`wte` 和 `lm_head` 各占 25M，二者在 400M 体量的小模型上**理论上**可共享，但 v5_best 实际已训到完全解耦。

动作（保留作未来 P2 时参考）：
1. 把 ckpt 的 `lm_head.weight` 折叠成 `wte.weight`（二者已在训练中被推得接近，直接复制就行）。
2. `GPTConfig.tie_lm_head = True`，加载时走 `lm_head.weight = wte.weight`（train.py:247-250 已实现）。
3. 从 v5_best → 新 ckpt 跑 200-500 步 SFT 校准（lr 降到 2e-6，避免打穿已学到的 summary 能力）。
4. 对比 bench：`val_bpt` 退化 > 0.05 就回退。

代价：一次短训，~1 GPU·小时。
是否作废当前 ckpt：是，但可从 v5_best 直接改权重热启。
成功标准：bench 不退步，ckpt 文件缩 25M。

### P2 · 下一代预训练（完整重训）

**砍一半 VE + 开 GQA（n_kv_head=4），把省出的参数给主干。**

现状算账：
- VE 当前 9 层 × 25M = 226M，占 56%。
- GQA 从 12→4 可省 KV cache 约 3×（并发吞吐第一个墙）。
- `scripts/migrate_to_new_arch.py` 的脚本早就是为这条路线准备的（已有 KV head 合并逻辑）。

动作：
1. **VE 砍半**：只保留"靠后 3 层"的 VE（例如 layer 14/16/17），参数从 226M → 75M，省 150M。
2. **GQA**：`n_kv_head=12 → 4`，c_k/c_v 走 migrate 脚本合并。
3. 省下的 ~180M 有两条投法——(A) `n_embd 768→896`（带动所有子层）；(B) `n_layer 18→24`（深度换宽度）。先做一版 (A) 的小 ablation，再决定主跑。
4. 重新跑 FineWeb+Chinese 预训练（参考 `docs/PRETRAIN_GUIDE.md`），再走完整 SFT 链路复刻 tool-calling 能力。

代价：一次完整预训练 + 从零 SFT。~50-100 H100·小时（和当前预训练同量级）+ 3-5 天 SFT 迭代。
是否作废当前 ckpt：**是**，完全从零开始。
触发条件：P0+P1 都稳定后，且有一次 2-3 周的集中窗口期。不要半途启动。

### P3 · 实验性（研究不保证落地）

**低秩 Value Embedding：vocab×R @ R×kv_dim，R=64。**

思路：VE 本质上是"每个 token 在每一层一个 per-head bias 向量"。用 `V_full = A @ B`，A∈R^{vocab×R}, B∈R^{R×kv_dim}，R=64。
- 9 层原本 226M → 约 18M，省 208M（几乎是整个 wte 的 8 倍）。
- 需要预训练时从头学，不能从现成 ckpt 迁移。
- 风险：R=64 可能表达不够，需要预训 ablation（R=32/64/128）。

代价：一套预训 ablation（单 run 约 1-2 GPU·天，能起数就 3 run）。
是否作废当前 ckpt：是，研究性路线。
触发条件：P2 落稳后，想进一步压 ckpt 大小或留 budget 给 n_embd 扩张时再试。

## 优先级排序理由

1. **P0 是 bug，不是优化**。用户已经踩到空回答，ROI 是"明天就能修好 30% 的长文件提问"。
2. **P1 是下次必做**。tie_lm_head 改动极小，下次 SFT 前改完就永久受益。
3. **P2 是下一代目标**。别为了"更好架构"牺牲当前活着的 E2E 能力——v5_best 是唯一跑通工具链 + 自我认知 + 安全拒绝的权重。
4. **P3 是研究方向**。不承诺时间点。

## 不做清单（避免再次被想起来问"要不要做"）

- **深度 48 层 / 扩大到 1B**：小模型定位，超算配额不合适，不做。
- **MoE 化**：400M 体量加 MoE 的 routing 开销占比太大，不做。
- **全套 flash-attn 自定义 kernel**：已用 FA3，没有别的性能上限。
- **VE 全删**：预训时 VE 对 val_bpb 的贡献确实存在，砍半可以，全删退化太大。
