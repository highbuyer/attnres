# w2 Plan A Postmortem：patch_w1 信号被工具样本淹没 + schema 漂移

> 2026-04-18
> baseline: `sft_tool_summary_v5_best.pt`  val_bpt=1.2297，tool_false_fire=7/64
> candidate: `sft_w2_a_best.pt`  val_bpt=1.5295，tool_false_fire=**14/64**
> 结论：**candidate 不作为新主线**。方向从"加数据"转为"推理层硬规则"。

## 三版对比

| ckpt | val_bpt | tool_false_fire | over_refusal | 结论 |
|------|---------|-----------------|--------------|------|
| v5_best（baseline） | 1.2297 | 7 (10.9%) | 3 (4.7%) | 当前主线 |
| sft_think_v1_best (w1) | 0.8421 | 11 (17.2%) | 2 (3.1%) | 回退（think 污染） |
| sft_w2_a_best (w2 Plan A) | 1.5295 | **14 (21.9%)** | 1 (1.6%) | 回退（样本被稀释 + schema 漂移） |

单调恶化。每次迭代比前一次糟。

## 新发现：schema 漂移

v3 输出里出现：
```
water → <|tool_call_start|><|tool_name_read_file|>{"query":"def main"}<|tool_call_end|>
diamond → <|tool_call_start|><|tool_name_read_file|>{"query":"def resolve_sol_lying"}<|tool_call_end|>
```

`read_file` 的正确 schema 是 `{"path":..., "offset":..., "limit":...}`，模型却套了 `search_code` 的 `{"query":...}`。这是把工具样本 weight 从 3 降到 1 的副作用——对抗 false_fire 不够力，但足以让工具 schema 模糊化。

## 根因

两条并存：

1. **patch_w1 量级不够**：30 条 × 50 weight = 1500 条，对抗 7126 条工具样本 + 400M 模型里已经深度刻入的"看到问题→调工具"模式。400M 级别的模型学一个新 pattern 需要的量级是**千条独立多样性样本**，不是"几十条过采样"。
2. **降工具样本权重有代价**：工具样本 weight 3→1 让 schema 松动，得不偿失。

## 真正的教训

**光靠"再加一点 SFT 样本"改不掉已经固化的 false_fire 模式。**

前两轮失败的共同点是"相信 SFT 补数据能解决 false_fire"：
- w1 加 patch_w1 + think → 失败
- w2 加 patch_w1 + 降工具权重 → 失败

要真做 SFT 这条路，需要：
- patch_w1 扩到 300-500 条多样题（化学式 30 个、物理常数 20 个、SQL 语法 15 个 ...）
- 工具样本 weight 保持原样
- lr 更小（5e-7）
- 或者直接进入下一代预训练

这三条都不是"一晚上能验证"的工作。

## 方向转向：推理层硬规则（方案 D）

模型其实**有知识**——v5_best 直答过"光速约 299792458m/s"，只是在"水的化学式"上去调工具。问题是"触发判断"，不是"知识缺失"。

`src/inference_rules.apply_hard_rules` 已经有"身份/安全"类硬规则的基础设施。扩展一条"常识问题禁工具"：

- **识别**：正则匹配"化学式 / 化学成分 / 是多少 / 作者是 / 提出了 / 的区别 / 排序后"等模式。
- **干预**：命中时在 decode 阶段把 `<|tool_call_start|>` token 的 logit 设为 -inf，强制模型走直答路径。
- **兜底**：模型答不出来就走 `_research_fallback`（本机 docs / 维基），用户至少收到有意义的回答。

改动只在 `src/infer.py` + `weiyan-api`，不动权重，不作废 v5_best。和 P0 同构。

如果这条 inference-time 规则验证有效（tool_false_fire 降到 ≤2），那 SFT 补数据的路就可以押后——先解决用户体感，等有时间再大改数据。

## 下一步

w3：在 `src/infer.py` + `weiyan-api` 里加 `BAN_TOOL_PATTERNS` + decode 时 `<|tool_call_start|>` logit mask。跑 self_audit v4 验证。

## 动作项

- [x] sft_w2_a_best 不上线，v5_best 保留
- [x] 记录 `docs/WEAKNESSES_v3.md`（self_audit 输出）
- [x] 记录本 postmortem
- [ ] 实现 w3 推理层防护（src/infer.py + weiyan-api）
- [ ] 跑 self_audit v4 验收
