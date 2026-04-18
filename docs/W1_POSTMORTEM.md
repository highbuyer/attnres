# w1 Postmortem：val_bpt 降了但能力回退

> 2026-04-18
> baseline: `sft_tool_summary_v5_best.pt` val_bpt=1.2297
> candidate: `sft_think_v1_best.pt` val_bpt=0.8421
> 结论：**candidate 不作为新主线**。v5_best 保留。

## 数字摘要

| 指标 | v5_best | sft_think_v1_best | 变化 |
|------|---------|-------------------|------|
| val_bpt | 1.2297 | **0.8421** | ↓ 0.39（越低越好）|
| tool_false_fire | 7 (10.9%) | **11 (17.2%)** | ↑ 4 |
| over_refusal | 3 (4.7%) | 2 (3.1%) | ↓ 1 |
| 失败总数 | 10/64 | 13/64 | ↑ 3 |

**val_bpt 改善但 self_audit 变差**——这正是 self_audit 闭环被埋的原因。困惑度看不出"在该直答的题上去调工具"这种能力回退。

## 根因诊断

### 根因 1：think-trace 和直答样本混训导致"思而不答"

`docs/WEAKNESSES_v2.md` 里 `over_refusal` 的示例：

```
user: 上海和北京哪个城市人口更多？
assistant: <think>按人口数量排序：上海 > 北京。</think> 这个问题我不太确定答案，建议查阅相关资料获取准确信息。
```

think 段已经推出了正确答案（上海 > 北京），输出却改口走拒答模板。这说明 think-trace 被加到了直答 + 拒答样本上，模型学到的是"思考后仍然不确定"的语言模式，而不是"思考 → 给答案"。

scripts/add_thinking_traces.py 的启发式覆盖"默认情况"用了通用 trace 模板，对短答类题目反而注入了"我在想这个问题"的迟疑信号。

### 根因 2：工具样本权重过高冲掉了直答能力

混合配比实际效果：

| 来源 | 实际条数 | 占比 |
|------|---------|------|
| sft_tool_summary_v5 (weight=3, think=1) | 21,378 | **85.4%** |
| sft_tool_gated_v2 (limit=3000, weight=1, think=1) | 3,000 | 12.0% |
| sft_samples_project_facts (weight=20, think=0) | 200 | 0.8% |
| patch_w1 (weight=15, think=0) | 450 | 1.8% |

工具样本 85.4% 的比重远超前代 summary_v2 时期。模型看到的几乎全是"user → tool_call → result → summary"的 pattern，450 条 patch_w1 直答样本完全被淹没。

`cd_05 Python 中 list 和 tuple 的区别是什么？` 之前 v5 正常直答，现在误触发 `read_file`——就是被工具样本强化的结果。

## 为什么 patch_w1 没起作用？

1. **权重不够**：1.8% 对抗 85.4% 工具样本不现实。
2. **think=1 污染了工具样本**：add_thinking_traces 对 v5 的 21k 条工具样本全部加 trace，进一步强化"思考后调工具"。
3. **lr=3e-6 + 1500 步**对于 400M 模型来说依然是比较强的更新，叠加向工具方向的梯度，把直答能力推翻了。

## 下一版配方修正（w2）

有几个可独立试的方向。这次失败说明**不能一次变两个变量**（think + patch），得 ablation。

**方案 A：纯 patch_w1，不加 think**
```
--source sft_tool_summary_v5.jsonl:weight=1:think=0
--source datasets/sft_archive/sft_tool_gated_v2.jsonl:weight=1:think=0:limit=3000
--source sft_samples_project_facts.jsonl:weight=30:think=0
--source datasets/tool_call_samples/patch_w1.jsonl:weight=50:think=0
```
- 目的：先隔离 patch_w1 的效果
- 工具样本权重从 3 降到 1，patch_w1 从 15 升到 50
- 所有 think=0，去掉污染变量
- 短训（500-800 步，lr=1e-6）

**方案 B：think 只加在 tool_call 样本上，不加在直答上**
- 改 `scripts/add_thinking_traces.py`：直答题跳过
- 保留 patch_w1 的定向纠正
- 但需要先验证方案 A，否则又是两个变量一起变

**方案 C：单独再加一批"工具—不工具"对比训练对**
- 同一个"XXX 的化学式是什么"问题，同时给"直答"和"不可工具"标注
- 更像 RLHF 里的对比对

推荐顺序：**A → B → C**。

## 记忆/文档同步

- `docs/WEAKNESSES_v2.md`：保留作为失败快照。
- `memory/project_self_audit.md`：新增"v1→v2 回退记录"。
- `memory/feedback_think_trace_direct.md`（新增）：think-trace **不能**加到直答样本，否则会诱发"思而不答"。
- `memory/project.md` 主线仍是 v5_best，think_v1_best 不上线。
