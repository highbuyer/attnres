# 模型架构路线图（重订版）

> 起草：2026-04-18　|　重订：2026-04-19（Claude 作为大模型审查后重排优先级）
> 基线 ckpt：`checkpoints/sft_v5_pruned_ve_13_15_17.pt`（329M，`val_bpt=1.7271`，P2-VE 落地后）
> 视角：架构师 ROI 排序，**吞吐/显存/数据匹配**优先于"砍参数"

---

## 当前架构事实

| 字段 | 值 | 评价 |
|------|-----|------|
| `n_layer` | 18 | OK |
| `n_embd` | 768 | 2019 GPT-2 small 规格 |
| `n_head` / `n_kv_head` | 12 / 12 | **无 GQA——架构硬伤** |
| `sequence_len` | 2048 | NTK 外推到 8192（运行时） |
| `tie_lm_head` | False + `softcap=15.0` | 互相 patch，非 coherent design |
| VE | 6 层 ×25M = 150M（剪枝后） | ResFormer 实验性 feature，大厂零复现 |
| KV cache | **无** | `infer.py:311` 每 token 全重算 |
| 总参 | 328.82M（v5_pruned） | |

---

## 架构审查结论（Claude 视角）

attnres 当前架构是 **"nanochat 教学代码 + 若干 paper zoo + 事后 patch"** 的混合体，不是 coherent design。真实瓶颈排序：

1. **推理层无 KV cache** → 算力被浪费 10-50×（decode 2048 只有 66 tok/s）
2. **数据严重不匹配** → 13726 样本训 400M 是 undertrained 3 个数量级，`over_refusal=6/64` 和顽固 halluc 是这里来的
3. **定位错配** → tool-dispatcher 业务用了 generate-first 架构，wiki retrieve 被摆成二等公民
4. **GQA 缺失** → KV cache 每序列 113 MB，3× 冗余

VE 剪枝（f3929f1→083fbae 四连 commit）省了 18.7% 参数但属于**正确的低优先级事**——decode 吞吐只 +1-3%，因为 VE 不是 FLOP 热点。**我这一整晚被 P2 roadmap 带偏了**。

---

## 新路线（按 ROI 严格排序）

### P0 ✅ NTK 外推到 8192（已交付 2026-04-18）

保留原文。短 prompt 8/8→8/8，长 prompt 4846 tokens 不再被截。

### P0' · **KV cache**（立刻做，最高 ROI）

**状态：未开工，这是新最高优先级。**

动机：`src/infer.py:311` 的 `logits = model(x[:, -max_context:])` 每 token 全重算，
T=2048 decode 掉到 66 tok/s（T=256 时 120 tok/s，40% 下滑证明就是 attention 全重算造成的）。

动作：
1. 改 `src/train.py` 的 `CausalSelfAttention.forward` 支持 `past_kv: Optional[tuple[Tensor, Tensor]]` 参数；prefill 时返回 (k, v) 全量 cache，decode 时 append。
2. 改 `GPT.forward` 支持 `past_kvs: list[tuple]` 和 `use_cache: bool`；返回 `(logits, new_kvs)`。
3. 改 `src/infer.py` generate loop：prefill 整段 prompt 一次，decode 单 token forward + 增量 kv。
4. 改 `_precompute_rotary_embeddings` / `apply_rotary_emb`：decode 时按 position offset 取 cos/sin 而非 [:T]。
5. Benchmark：预期 decode 从 66 → 1000+ tok/s（15×+）。

代价：3-5 天工程 + 正确性 eval（输出应与无 cache 版本逐 token 一致）。
是否作废 ckpt：**否**。纯推理端改动。
成功标准：decode T=2048 吞吐 ≥ 500 tok/s，且 self_audit 64 条输出与无 cache 版逐条一致。
失败回退：保留无 cache 路径作 `--no-kv-cache` flag。

### P0.5 · 数据扩展到 ≥10k samples（两周内）

**状态：未开工。这是 over_refusal 和 halluc 的唯一解。**

动机：`feedback_no_repeat_failed` memory 已写：50 样本 patch-SFT 三连败（w1/w2/w10）。
400M 权重 × 13k 样本 = Chinchilla 意义下 undertrained 3 个数量级，小批 patch 学不动是必然。

动作：
1. 从 weiyan-api 日志 replay 真实用户 query（去重后预计 5-10k）
2. 扩 tool-call 数据：多轮对话、工具链式调用、tool_result 不同格式
3. 扩 fact-QA 数据：针对顽固 halluc（"23 条染色体"类题库）构造 ≥1000 条
4. 用 v5_pruned (329M) 做 base，不重训架构，只 SFT

代价：数据工程 2 周 + SFT 重跑 1 GPU 周。
触发标准：KV cache 落地后。
成功标准：`over_refusal` 6/64 → ≤3/64（不靠 wiki 兜底）。

### P1 ❌ tie_lm_head（证伪，延期到 P2-full）

保留原文。v5_best 上 wte 与 lm_head 完全正交，强 tie 会崩。下代预训从头 tie。

### P2-partial ✅ VE 剪枝 (13, 15, 17)（已落地 2026-04-19）

**状态：已落地。**`checkpoints/sft_v5_pruned_ve_13_15_17.pt`，329M。
见 docs/VE_ABLATION_v1.md / v2_full.md / VE_PRUNED_v1.md。commits f3929f1 → 083fbae。

- val_bpt 1.6551 → 1.7271（+4.3%）
- self_audit net_user_failure 2/64 持平 baseline
- Peak VRAM -296 MB (-10%)
- 吞吐 +1-5%（VE 不是 FLOP 热点）

### P2-full · 下一代预训（GQA + tie + 去 softcap + VE 再削）

**状态：延后，触发条件明确。**

触发：P0'（KV cache）落地 + P0.5（数据扩展）完成 + `over_refusal` 仍 ≥4/64 + 有 2-3 周窗口。

届时一次性改：
1. `n_kv_head=12 → 4`（GQA）：KV cache 再 3× 压缩
2. `tie_lm_head=True` + 去 `softcap`：正确 init（μP 或至少 careful scale）
3. VE 6 层 → 3 层 或全去（配合 retrieval）
4. `sequence_len` 训练时直接 4096，不靠 NTK 外推
5. 可选：n_kv_head 压下来后 n_embd 768→896

代价：一次完整预训练 + 从零 SFT（~50-100 H100·小时 + 3-5 天 SFT）。
是否作废 ckpt：**是**。

---

## 明确否决清单（不再讨论）

- ❌ **低秩 VE tie (1↔13, 3↔15, 5↔17)**：ROI 低，VE 不是吞吐瓶颈，只省几十 MB。
- ❌ **继续 patch-SFT 小批**：memory 写明三连败，重复错误。
- ❌ **attnres paper-exact 改造**：实验性、零工业复现、非业务路径。
- ❌ **MoE / Mamba / SSM**：400M 不配，过度设计。
- ❌ **继续 VE ablation**：边际收益递减，够了。
- ❌ **深 48 层 / 扩 1B**：小模型定位不变。
- ❌ **全套自定义 flash-attn kernel**：已用 FA3，没上限空间。

---

## 优先级排序理由

1. **P0'（KV cache）**：零训练、零数据、零风险，decode 10-50× 提升——这是最高 ROI 且被长期忽略的硬伤。
2. **P0.5（数据）**：唯一降 `over_refusal` 的杠杆。不是架构问题，别再想架构答案。
3. **P2-full**：只有 P0'+P0.5 做完后仍有明确瓶颈，才值得一次完整重训。

---

## 自审备注

> 本轮（2026-04-19）Claude 花了 4 个 commit 做 VE 剪枝，结论是"该做但低优先级"。
> 教训：被"P2-VE"的 roadmap 带偏，没有先用架构师视角审整个 stack。
> 从今天起，P0' KV cache 是唯一进行中项，其他都等。
