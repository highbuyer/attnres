# 下一轮 SFT 配方：thinking-trace + 混合数据

> 起草日期：2026-04-18
> 基线：`checkpoints/sft_tool_summary_v2_best.pt`（E2E 全通）

## 目标

让模型在回答前显式输出一段 `<think>…</think>` 推理，再给用户可见的答复。
目的有二：

1. **训练侧**：短 trace 充当思维先验，降低工具误触发、提高答题稳定度。
2. **推理侧**：runtime 可以用正则剥掉 `<think>…</think>` 再返回，用户看到的输出不变；
   想开启思维可见时只需去掉这层 strip。

不注册新 special token（保持 tokenizer 不动），直接用普通 `<think>` / `</think>` 文本标签。

## 数据准备

### 1. thinking 增强（离线静态）

对现有工具 SFT 数据打上启发式 trace：

```bash
uv run python scripts/add_thinking_traces.py \
    --src sft_tool_summary_v5.jsonl \
    --dst sft_tool_summary_v5_think.jsonl
```

启发式覆盖四种情况：工具调用分支 / 工具结果回合（不插）/ "X 是什么" / 文件名引用 / 默认。
v5 全量产出 7126 条、10776 处 think 注入、7.4 MB。

### 2. 混合配比

用新脚本 `scripts/build_mixed_sft.py` 在 messages 层做二次混合：

```bash
uv run python scripts/build_mixed_sft.py \
    --out sft_mixed_think_v1.jsonl \
    --source sft_tool_summary_v5.jsonl:weight=3:think=1 \
    --source sft_tool_gated_v2.jsonl:weight=1:think=1:limit=3000 \
    --source sft_samples_project_facts.jsonl:weight=20:think=0
```

**推荐起点配比（总量 ≈ 25k）**

| 来源 | 作用 | weight | think | 说明 |
|------|------|--------|-------|------|
| `sft_tool_summary_v5` | 主线，tool_call → tool_result → summary | 3 | 1 | 保留当前 E2E 能力并加 trace |
| `sft_tool_gated_v2` (limit 3000) | 门控负例，防止直答场景误触发工具 | 1 | 1 | 只留 3k 够用 |
| `sft_samples_project_facts` | 项目事实硬记忆 | 20 | 0 | 10 条太少，必须重度过采样；事实题不需要 trace |

项目事实样本（10 条 × 20）的 think=0 是刻意选择：trace 反而会稀释短答的确定性信号。

## 训练

从当前 E2E 最佳点短训，`lr` 比常规低一档，避免破坏已经立住的 summary 能力：

```bash
env UV_CACHE_DIR=/tmp/uv-cache /home/langshen/.local/bin/uv run python src/sft.py \
    checkpoints/sft_tool_summary_v2_best.pt \
    --data sft_mixed_think_v1.jsonl \
    --out checkpoints/sft_think_v1.pt \
    --lr 3e-6 --total-steps 1500 --warmup-steps 80 --warmdown-start 1200 \
    --eval-interval 100 --batch-size 2 --grad-accum 16
```

**关键点**

- `lr=3e-6`：比 summary v2 的 5e-6 再降一档。think 标签是纯新语言模式，容易过拟合。
- 1500 步、warmdown 在 1200：短训、快收敛，防把 tool_call 漂移掉。
- batch=2, grad_accum=16：fp32 训练显存限制不变。

## 评测点

跑完立即检查，顺序按"便宜 → 贵"：

1. **格式自检**（必须过）：随机抽 20 个 ckpt 推理，确认
   - 每个 assistant turn 都以 `<think>` 开头、`</think>\n` 结尾
   - `<think>` 内不包含 `<|tool_call_start|>`（trace 和 tool call 不混）
2. **工具格式**：`scripts/eval_tool_format.py`，目标 ≥ 7/8（不应比 summary v2 退步）
3. **全量 bench**：`scripts/eval_bench.py --compare eval_results_sft_tool_summary_v2_best.json`
4. **E2E 活检**：`src/infer.py` 跑 `parse_tool_call 在哪里实现？`，确认
   - 模型先吐 `<think>…</think>` 再吐 tool_call
   - runtime 层 strip 后用户只看到自然语言答复（strip 正则需要在 infer 侧补一条）

## Runtime 改动

推理层要加一条 strip 规则（放在 `src/infer.py` 的 `strip_tool_markup` 附近）：

```python
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
output = _THINK_RE.sub("", output)
```

默认开启；需要观察模型思路时传 flag 关掉即可。

## 失败回退

如果 v1 ckpt 在工具 bench 上大幅退步（7/8 → ≤5/8）：

1. **先降 weight**：`sft_tool_gated_v2` 调到 weight=2，稀释 think 比例。
2. **再缩步数**：总步数砍到 800，warmdown 从 600 起。
3. **仍不行**：弃用 think，回 summary v2，等脚本迭代出更高质量的 trace（比如调一次模型离线生成，而不是纯正则）。

## 未来方向

- trace 质量：当前是正则启发式，长期应换成"用当前模型自己生成 trace → 过滤 → 回灌"的 self-distill 管线。
- trace 多样性：加一个 `--paraphrase` 开关，同一条样本不同 epoch 换不同措辞的 trace，防模板化。
- 可见思维开关：在 `src/infer.py` 暴露 `--show-thinking`，让调试和产品两种形态共用同一个权重。
