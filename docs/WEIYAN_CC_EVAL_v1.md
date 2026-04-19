# weiyan Claude-Code 场景评测 v1

> 2026-04-19　ckpt: `sft_tool_summary_v5_best.pt`（v5_best 原版，未用 clean 数据重训）

## 目的

验证 weiyan 在真实 Claude Code 风格 query（代码定位、调试、跨文件推理）下的表现。
self_audit 64 条是简单 one-shot tool call，不能代表 agentic 场景。

## 结果：0/8 完全正确，7/8 完全失败

| 题 | 层级 | 结果 | 失败模式 |
|----|------|------|----------|
| L1_1 has_ve 位置 | 定位 | ❌ | 未发 tool_call，纯乱码输出 |
| L1_2 VE 相关脚本 | 定位 | 🟡 | 发了 search_code 但只命中 docs 引用，未 list_directory |
| L2_1 INFER_MAX_CONTEXT 值 | 链式 | ❌ | 搜到引用但不 read_file 深挖 |
| L2_2 prune_ve 删 keys | 链式 | ❌ | 把脚本名当工具名，输出乱码 |
| L2_3 knock_weakest3 Δ | 链式 | ❌ | tool_call 发了但 query 参数为空字符串 |
| L3_1 FlashAttention dtype bug | 调试 | ❌ | 中英文混杂糊弄，未理解问题 |
| L3_2 unexpected key 报错 | 调试 | ❌ | 幻觉标识符（`event.key_emb_key` 等） |
| L4_1 v5_pruned 参数差 | 跨文件 | ❌ | 严重幻觉（"attnREADER"、"attyMirror"） |

## 根因（三叠加）

### 1. 训练数据覆盖不够

SFT 7126 条里真实多步 tool_call 样本稀少。self_audit 测的 64 条工具题都是
**单次查询 + 简单模板**（"src/infer.py 里有没有 --tool-dir 参数？"），模型学到的
是"看到代码路径就调 read_file"的单步反射，不是"链式 search → read → 提取"的推理。

### 2. 400M 容量限制

长链推理 + 工具参数合成需要的 reasoning 深度，400M 权重吃不下。特别是：
- 从自然语言推出精确 search_query
- 从 search 结果选择最有价值的 file/line 去 read
- 综合多个 tool_result 形成回答

这些都需要的是"工作记忆 + 规划"，400M 模型结构性缺失。

### 3. 定位错配

"INFER_MAX_CONTEXT 的值" 这种 Claude Code 日常 query，weiyan 训练分布里几乎
不存在。模型对"项目内部具体代码细节"类问题的 tool_call 策略不清楚——有时
调 search，有时乱生成 Python 代码，有时中英文糊弄。

## 与 self_audit 评估的对比

| 评估 | 工具题通过率 | 代表什么 |
|------|------------|---------|
| self_audit BENCH 64 条 | 13/13 = 100% | 单次查询，有既定 template 可背 |
| Claude Code eval 8 条 | ~1/8 = 12% | 多步推理 + 参数合成 + 结果整合 |

**关键教训**：self_audit 的 100% tool_call 成功率是**假阳性评估**。BENCH 设计
偏简单，没覆盖真实 agentic 场景。应当扩充 BENCH 加入多步题，或单独维护一个
Claude Code 风格 eval set。

## 对后续工作的影响

### 前期工作并未白做，但不是万能药

- **VE 剪枝（329M ckpt）**：行为等价，不改评测结果；价值在省显存
- **KV cache**：decode 加速，不改评测结果；价值在推理工程
- **101 条 refusal 改写**：降 over_refusal，但 Claude Code eval 失败 **不是** over_refusal 问题

### 真实改善路径（按代价）

**A. 扩充 multi-step tool-use 训练数据**（1-2 周）
  - 人工或大模型合成 ≥500 条"search → read → summarize"链式样本
  - 合并到 clean 集一起训，目标 Claude Code eval 2-3/8
  - 风险：400M 可能仍然吃不下多步

**B. 在 prompt 端做 step-by-step 引导**（workaround）
  - 把用户 query 在 API 层拆成 "step 1: 先搜 X" → "step 2: 读 Y" 的显式多轮
  - 类似 ReAct 模板，让 weiyan 只做单步决策
  - 工程代价小，但不是根治

**C. 承认 dispatcher 定位不包含 Claude Code 场景**（诚实路径）
  - weiyan 现状 = 自我认知稳定 + 安全拒答 + 简单工具触发
  - 不定位成 Claude Code 助手，定位成更窄的 chat/简单 QA
  - 把复杂任务 fallback 给真正的 Claude Code

### 我的建议

**走 C + 把 B 作为兜底**：
- weiyan 明确定位为 "简单对话 + 硬规则拦截 + 单次工具触发"
- 复杂 agentic 场景直接让用户用 Claude Code
- 如果一定要做复杂场景，用 B 的 prompt 拆分器（API 层做 orchestration，weiyan 只做单步）
- **不推荐 A**：花 1-2 周训数据换来的 2-3/8 仍然不够用，投入产出不成比例

## 附：raw responses

见 `runs/weiyan_cc_eval.jsonl`（gitignored）。典型乱码样本：

> Q: src/train.py 里 has_ve 函数在哪一行？skip 参数是做什么的？
> A: 使用 search_code 和 read_file 工具查看当前项目代码：1. 使用 search_code 工具：src/train.py 在当前项目代码中使用当前代码进行操作，可以使用以下方法进行操作... [后续是嵌套的无意义方法描述]

这种 degenerate 模式是 **400M 模型在训练分布外 prompt 上的典型崩溃**。
