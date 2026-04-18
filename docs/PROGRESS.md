# AttnRes 项目进度

> 最后更新: 2026-04-18

## 当前状态

**端到端工具调用链路 + 本地 API 服务已跑通。**

```
用户提问 → 模型调 search_code/read_file → runtime 执行 → 注入结果 → 模型自然语言总结
                                                    ↘ 同一路径下 weiyan-api 以 HTTP 暴露
```

| 能力 | 状态 | 说明 |
|------|------|------|
| 工具触发 | **7-8/8** | 代码检索类问题正确输出 `<\|tool_call_start\|>` |
| 工具门控 | **~90%** | 非工具题直答，不误调 |
| Post-tool summary | **✓** | 模型原生生成自然语言总结 |
| 自我认知 | **11/11** | 全部正确 |
| 安全拒绝 | **5/5** | 硬规则拦截 |
| weiyan-api | **✓** | OpenAI + Anthropic + /ask 兼容，session + audit log |

## 当前主线模型

| 用途 | 文件 | step | 指标 | 说明 |
|------|------|------|------|------|
| **主线 best** | `sft_tool_summary_v5_best.pt` | 1500 | `val_bpt=1.2297` | weiyan CLI / API 默认 |
| 上一代主线 | `sft_tool_summary_v2_best.pt` | 400 | `val_bpt=0.1879` | MIXED_SFT_RECIPE 以此为基线起稿 |
| 门控基座 | `sft_tool_gated_v2_best.pt` | 2000 | `val_bpt=0.4170` | summary 前基座 |

## 架构快照（核对 ckpt 直接 dump）

- 18 层 / n_embd=768 / n_head=n_kv_head=12（无 GQA）
- vocab=32774（32768 基座 + 6 工具 token）
- sequence_len=2048（训练 + 推理硬截断）
- window_pattern=SSL（S=1024, L=2048）
- 参数总量 404M：VE 226M (56%) + wte 25M + lm_head 25M + 主干 ~127M + AttnRes proj 27K

## 已知架构债

详见 **`docs/ARCH_ROADMAP.md`**（P0–P3 路线图）。摘要：

- **P0（✅ 2026-04-18 已交付）**：`src/infer.py` + `weiyan-api` 扩到 8192（NTK RoPE + prompt 截断护栏）。短 prompt tool-format 8/8 → 8/8，零退化。
- **P1（下次 SFT 前）**：`tie_lm_head=True` 省 25M。
- **P2（下一代预训）**：VE 砍半 + GQA `n_kv_head=4`。
- **P3（研究性）**：低秩 VE。

## 训练路线回顾

### Phase 1: 预训练 → SFT（旧主线）

- 32k 词表预训练 → 工具 token 追加 → 混合 SFT
- 结果：常识和身份正常，但 `eval_tool_format = 0/8`

### Phase 2: 诊断 bf16 精度问题

- 根因：bf16 参数 + 优化器直接更新 → 小更新被量化吞掉
- 修复：fp32 参数 + bf16 autocast + `--batch-size`/`--grad-accum` 参数

### Phase 3: tool-policy-only（2026-04-03）

- 数据：repo 工具样本 × 512，只训练 `user → tool_call`
- 结果：`eval_tool_format = 8/8`；代价：直答场景误触发工具

### Phase 4: tool-gated 混合训练（2026-04-03）

- 从 policy v3 出发，混合 Belle + 工具链 + 身份
- v2: tool 链 39%，eval_tool_format 7/8，回答质量全面提升

### Phase 5: summary-focused 训练（2026-04-03）

- 关键发现：`tokenize_turn` 对 tool_result 内容计算 loss → summary 信号 9% → 修复后 ~80%
- 关键发现：推理时 tool_result 注入缺 ASST_ID → 训练/推理序列错位
- 修复后从 gated v2 短训 1000 步（lr=5e-6）→ **模型原生 post-tool summary 立住**

### Phase 6: v5 + weiyan-api 上线（2026-04-04 ~ 04-18）

- 主线从 v2_best 滚动训练到 v5_best（step 1500, val_bpt=1.23）
- 新增 `weiyan-api`：HTTP 服务，OpenAI / Anthropic / /ask 三种入口 + session memory + 审计日志
- 长 prompt bug 浮现：超过 2048 直接返回空 text（→ P0 待修）
- 下一轮 SFT 数据已备：`sft_tool_summary_v5_think.jsonl`（trace 增强 7126 条）

## Runtime 修复清单

1. **parse_tool_call 容错**：`src/tool_protocol.py` 支持缺失 `<|tool_call_end|>` 的 partial parse
2. **ASST_ID 注入**：`src/infer.py` 和 `weiyan-api` 在 tool_result 前后加 `[ASST_ID]` 匹配训练格式
3. **Fallback summary**：工具执行成功但模型不输出 summary 时，首条结果兜底
4. **Reserved token 清理**：清理输出中残留的 `<|reserved_N|>` 标记
5. **Prompt sanitizer**：`weiyan-api` 剥离 `<system-reminder>` / `<local-command-*>` 噪声块
6. **Research fallback**：模型拒答时走本机 docs → 维基百科 → 诚实说明原因
