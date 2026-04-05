# AttnRes 项目进度

> 最后更新: 2026-04-03

## 当前状态

**端到端工具调用链路已跑通。**

```
用户提问 → 模型调 search_code/read_file → runtime 执行 → 注入结果 → 模型自然语言总结
```

| 能力 | 状态 | 说明 |
|------|------|------|
| 工具触发 | **7-8/8** | 代码检索类问题正确输出 `<\|tool_call_start\|>` |
| 工具门控 | **~90%** | 非工具题直答，不误调 |
| Post-tool summary | **✓** | 模型原生生成自然语言总结（不依赖 fallback） |
| 自我认知 | **11/11** | 全部正确 |
| 安全拒绝 | **5/5** | 硬规则拦截 |

## 当前主线模型

| 用途 | 文件 | step | 指标 | 说明 |
|------|------|------|------|------|
| **主线 best** | `sft_tool_summary_v2_best.pt` | 400 | `val_bpt=0.1879` | E2E 全通 |
| 门控基座 | `sft_tool_gated_v2_best.pt` | 2000 | `val_bpt=0.4170` | summary 前基座 |
| 旧主线 | `sft_mixed_v8_checkpoint_v2_best.pt` | 13000 | `val_bpt=3.7860` | 不会调工具 |

## 训练路线回顾

### Phase 1: 预训练 → SFT（旧主线）

- 32k 词表预训练 → 工具 token 追加 → 混合 SFT
- 结果：常识和身份正常，但 `eval_tool_format = 0/8`

### Phase 2: 诊断 bf16 精度问题

- 根因：bf16 参数 + 优化器直接更新 → 小更新被量化吞掉
- 修复：fp32 参数 + bf16 autocast + `--batch-size`/`--grad-accum` 参数

### Phase 3: tool-policy-only（2026-04-03）

- 数据：repo 工具样本 × 512，只训练 `user → tool_call`
- 结果：`eval_tool_format = 8/8`
- 代价：所有非工具问题也误触发工具

### Phase 4: tool-gated 混合训练（2026-04-03）

- 从 policy v3 出发，混合 Belle + 工具链 + 身份
- v1: tool 链 18%，eval_tool_format 8/8 + 直答恢复
- v2: tool 链 39%，eval_tool_format 7/8，回答质量全面提升

### Phase 5: summary-focused 训练（2026-04-03）

- 关键发现：`tokenize_turn` 对 tool_result 内容计算了 loss，导致 summary 信号被淹没（9% → 修复后 80%）
- 关键发现：推理时 tool_result 注入缺 ASST_ID，导致训练/推理序列不匹配
- 修复后从 gated v2 做 1000 步短训（lr=5e-6, tool 链 85%）
- 结果：**模型原生生成 post-tool summary**

## Runtime 修复

### 1. parse_tool_call 容错

- `src/tool_protocol.py`：支持缺失 `<|tool_call_end|>` 的 partial parse
- 原因：部分 prompt 模型输出 tool_call_start 但 JSON 不闭合

### 2. ASST_ID 注入

- `src/infer.py`：tool_result 注入时加 `[ASST_ID]` 前后缀
- 原因：训练格式是 `ASST tool_call ASST tool_result ASST summary`，推理时必须匹配

### 3. Fallback summary

- `src/infer.py`：工具执行成功但模型不输出 summary 时，用首条结果兜底
- 当前 summary v2 模型已不需要此 fallback，但保留作为安全网

### 4. Reserved token 清理

- `src/infer.py`：清理输出中残留的 `<|reserved_N|>` 标记

## 关键里程碑

- [x] 32k 词表主线继续预训练
- [x] 工具 token 追加（32768-32773）
- [x] 推理层身份/安全硬规则上线
- [x] 工具调用 runtime 上线（search_code / read_file）
- [x] bf16→fp32 训练精度修复
- [x] eval_tool_format = 8/8（工具触发立住）
- [x] tool-gated 门控分类正常
- [x] tool_result loss mask=0 修复
- [x] 推理时 ASST_ID 注入修复
- [x] **端到端 E2E 全链路跑通（tool_call → execute → summary）**
