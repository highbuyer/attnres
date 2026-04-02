# AttnRes 项目进度

> 最后更新: 2026-03-28 21:30

## 项目概述

基于 Karpathy autoresearch 框架，训练 135M 参数中文对话模型。
架构采用 Kimi 2026 论文的 Block AttnRes（跨层注意力残差）+ Value Embedding + Muon 优化器。

## 模型架构

- **参数量**: ~135M
- **架构**: GPT + Block AttnRes (sublayers_per_block=3) + Value Embedding (交替层)
- **配置**: depth=12, ar=64, n_embd=768, n_head=12, head_dim=64, window_pattern=SSL
- **激活函数**: ReLU² (MLP), QK-Norm, RoPE
- **优化器**: MuonAdamW (Muon for matrix params, AdamW for embeddings/scalars)
- **Logit softcap**: 15

## 预训练

### 数据集

| 来源 | 文件数 | 行数 | 采样权重 | 说明 |
|------|--------|------|----------|------|
| Belle 0.5M | 1 | 517K | 3.0 | 中文指令 |
| Belle 2M | 3 | 1.46M | 3.0 | 中文指令 |
| Wiki ZH | 6 | 1.25M | 3.0 | 中文百科 |
| Glaive v2 | 1 | 112K | 3.0 | 英文工具调用 |
| Glaive v1 | 1 | 39K | 3.0 | 英文工具调用 |
| Python (GitHub) | 10 | 575K | 1.0 | 代码 |
| Java (GitHub) | 10 | 756K | 1.0 | 代码 |
| JavaScript (GitHub) | 10 | 515K | 1.0 | 代码 |
| StarCoder | 59 | 11.6M | 0.14 | Python 代码 |

**Tokenizer**: rustbpe 训练, vocab_size=8192, GPT-4 style split pattern, 4 个特殊 token

### 预训练结果

最佳 val_bpb: **0.5578** (HEAD_DIM=64, depth=12, SSL, weight_decay=0.01)

### 数据格式修复 (2026-03-27)

**问题**: Belle/Glaive 数据在预训练时使用明文 `Human:/Assistant:` 格式，导致 SFT 后模型仍会输出这些明文标记。

**修复**: `prepare.py` 已更新：
- Belle: `Human: {i}\nAssistant: {o}` → `<|reserved_1|>{i}<|reserved_2|>{o}<|reserved_3|>`
- Glaive: 新增 `_convert_glaive_chat()` 解析器，USER/A/ASSISTANT → 特殊 token，跳过 SYSTEM 和 functioncall
- `Tokenizer.encode`: `encode_ordinary` → `encode(allowed_special="all")`
- 已重新生成 parquet 文件

### 继续预训练 (进行中)

使用 `continue_pretrain.py`，从 `best_checkpoint.pt` 加载，用新格式数据继续训练。
- 学习率降为预训练的 1/4 (matrix_lr=0.01)
- 2000 步, warmup 50 步
- **状态**: 运行中 (PID 2930944)

## SFT 微调

### SFT 数据集

| 来源 | 条数 | 说明 |
|------|------|------|
| Belle (采样) | 250,000 | 中文指令问答 |
| Claude 历史对话 (清洗) | 15,894 | 从本地 Claude JSONL 日志提取，含工具调用与实战逻辑 |
| 身份认知 (过采样) | 5,100 | 51条模板 × 100倍过采样，强化“微研”身份 |
| 安全拒绝 | 151 | 专门设计的拒绝话术，防止生成危害内容 |
| **合计** | ~271,145 | |

### SFT 数据流水线

1. **日志导出**: `python ~/Desktop/export_sft.py` (将 Claude 日志转为 SFT 格式，截断设为 1500 字符)
2. **数据清洗**: `python clean_sft_v3.py` (过滤 API Error 等噪音)
3. **数据集构建**: `python make_sft_data.py --claude path/to/clean.jsonl --upsample 100` (合并各源数据)

### SFT 训练配置

- **脚本**: `sft.py`
- **LR**: 1.5e-5 (对于 200M+ 模型建议更低), warmup 100 步
- **Batch**: 4 × 8 = 32 samples
- **Loss**: 仅在 assistant token 上计算 (含 EOS)
- **格式**: `BOS + USER_ID + 内容 + ASST_ID + 回答 + EOS`

### SFT 结果

| 轮次 | 架构 | Steps | val_bpt | 备注 |
|------|--------|-------|---------|------|
| 第 1 轮 | 135M | 5000 | 3.5176 | 旧数据 |
| 第 2 轮 | 135M | 12000 | 3.1405 | 扩大数据量 |
| 第 3 轮 | 135M | 12000 | 2.7023 | 数据清洗优化 |
| 第 4 轮 | 135M | 12000 | 2.7209 | 重复实验验证 |
| 第 5 轮 | 218M | 12000 | 3.0054 | 未收敛，身份/安全效果差 |

## 关键文件

```
train.py                 — 预训练脚本 (模型定义 + MuonAdamW + 训练循环)
prepare.py               — 数据准备 + tokenizer + dataloader + 评估
continue_pretrain.py     — 继续预训练脚本 (支持规模扩展后的知识填充)
sft.py                   — SFT 微调脚本
infer.py                 — 推理脚本 (支持交互式 + 重复惩罚)
eval_bench.py            — 定量评估脚本 (51条固定测试集 + 差异对比)
make_sft_data.py         — 数据集构建 (支持多源合并、身份过采样参数)
export_sft.py            — Claude 日志导出工具 (1500字符截断)
clean_sft_v3.py          — SFT 数据清洗工具 (过滤 API 报错噪音)
expand_checkpoint.py     — 规模扩展工具 (支持 --depth, --embd 参数)
```

## 模型规模扩展

使用 `expand_checkpoint.py` 将已训练的 checkpoint 扩展到更大规模，避免从零预训练。

### 原理
- **维度扩展**：权重矩阵沿新增维度填充小随机噪声（scale=0.01）
- **层数扩展**：新增层从已有层循环复制，加小噪声打破对称性
- `attnres_proj`/`attnres_norm`/`value_embeds` 等全局 ModuleList 同步扩展

### 用法

```bash
# 只扩展维度：depth=12 ar=64 (135M) → depth=12 ar=80 (218M)
uv run python expand_checkpoint.py best_checkpoint.pt expanded_ar80.pt --ar 80

# 只扩展层数：depth=12 → depth=18（显存上限，depth=24 OOM）
uv run python expand_checkpoint.py best_checkpoint.pt expanded_d18.pt --depth 18

# 扩展后继续预训练
uv run python continue_pretrain.py expanded_ar80.pt
```

### 注意事项
- 显存上限：depth=18 (196M) 是 4090 24GB 的极限，depth=24 会 OOM
- 扩展后的模型需要继续预训练才能收敛，建议 warmup 步数适当增大
- 噪声幅度可用 `--noise` 调整（默认 0.01），过大会破坏已学知识
- 脚本会自动验证所有 key 的 shape，不匹配时报错退出

## 词表扩充（8192 → 32768）

**不需要从零预训练**，使用 `migrate_embeddings.py` 迁移 embedding 权重。

### 原理
- 旧/新 tokenizer 按**字节序列**匹配 token：相同字节序列的 token 直接复制 embedding
- 新增 token（旧词表没有的）随机初始化（scale=0.02）
- 迁移范围：`wte.weight`、`lm_head.weight`、`value_embeds`（所有 vocab 维度权重）
- 迁移后需继续预训练，让新 token embedding 收敛

### 前提条件
- 旧 tokenizer 已备份至 `~/.cache/autoresearch-custom/tokenizer_8192_backup/`
- 新 tokenizer 已训练至 `~/.cache/autoresearch-custom/tokenizer/`（vocab=32768）
- `prepare.py` 中 `text_iterator(max_chars=300_000_000)` 防止 OOM

### 用法

```bash
# 第一步：训练新词表（跳过已下载的数据）
uv run python prepare.py --skip-download

# 第二步：迁移 embedding
uv run python migrate_embeddings.py continued_d18_v2_final.pt migrated_d18_32k.pt
# 输出示例：可迁移 token: 7506 / 32764，复制 embedding: 7506 / 32768

# 第三步：继续预训练（更新 continue_pretrain.py 的 CHECKPOINT_IN/OUT）
uv run python continue_pretrain.py migrated_d18_32k.pt
```

### 注意事项
- 约 7500/32768 tokens 可复用（~23%），其余随机初始化——loss 初期会上升，属正常现象
- 旧 checkpoint 的 config.vocab_size 会自动从 8192 更新为 32768
- `migrate_embeddings.py` 硬编码了旧/新 tokenizer 路径，修改路径常量即可复用

## 已修复的问题

1. **GPTConfig pickle 错误**: exec() 创建的类 `__module__` 指向 builtins，设为 `'__main__'` 修复
2. **连续 assistant 合并**: 独立的 assistant 回答不应合并，改为跳过同一连续块中的兄弟 assistant
3. **推理格式不匹配**: infer.py 缺少 USER_ID/ASST_ID 特殊 token，已修复
4. **预训练数据格式**: Belle/Glaive 使用明文角色标记，改为特殊 token 格式
5. **tool_call 数据污染**: SFT 数据中的 tool_call 对话已过滤/清洗

## 下一步

1. ✓ 继续预训练完成，val_bpb=0.5434
2. ✓ 第 3 轮 SFT 完成，val_bpt=2.7023（较第 2 轮 3.1405 提升 14%）
3. ✓ Human:/Assistant: 泄露已消除；微研身份未生效（SFT 数据中身份样本不足）
4. ✓ 推理加 repetition_penalty=1.3，重复问题基本解决
5. ✓ eval_bench.py 建立量化评估体系（51条固定测试集）
6. ✓ ar=80 (218M) 继续预训练完成，val_bpb=0.5405
7. ✓ ar=80 SFT 完成，val_bpt=3.0054（未充分收敛，需更多步数）
8. ✓ eval_bench 对比 ar=64 vs ar=80：ar=80 常识略退步，安全/身份均失败
9. ✓ 32768 vocab tokenizer 训练完成（300M chars，vocab_size=32768，旧 8192 备份保留）
10. ✓ embedding 迁移完成：continued_d18_v2_final.pt → migrated_d18_32k.pt（7506/32764 tokens 复用）
11. ✓ depth=24 实验废弃（显存不足）
12. ✓ 32k 继续预训练第 2 轮完成：`continued_d18_32k_final.pt`，val_bpb=0.572887（相对 0.586209 继续改善）
13. ✓ 新的 SFT 混合数据集已构建：`~/Desktop/sft_mixed_v7.jsonl`（410726 条，含 51 条 tool_call 样本）
14. ✓ **战略调整（2026-03-31）**：工具调用从实验分支提升为核心能力，身份/安全改为推理时硬规则
15. ✓ tokenizer 追加 6 个工具调用 special token（ID 32768-32773），BPE 不重训
16. ✓ checkpoint embedding 扩展：`continued_d18_32k_final.pt` → `tooltoken_d18_32k.pt`（vocab 32768→32774）
17. ✓ 工具 token 短暂继续预训练 500 步：`tooltoken_continued_final.pt`，val_bpb=0.593974（原 0.572887，+0.021 合理）
18. ✓ 目录重组路径修复：所有 src/ 和 scripts/ 文件使用 `__file__` 相对路径，不依赖 CWD
19. ✓ 工具调用样本扩充：242 条去重原始样本（v2+v3），覆盖 search_code / read_file / 直接回答
20. ✓ 新 SFT 数据集构建：`~/Desktop/sft_toolcall_v1.jsonl`（67571 条，tool_call 2420 占 3.6%，Belle 50K）
21. ✓ SFT 第 6 轮完成：`sft_toolcall_v1.pt`，val_bpt=3.5590（15000 步，step 13500 后完全收敛）
22. ✓ infer.py 加入身份硬拦截（5 条正则）+ 安全硬拦截（7 条正则）+ 工具执行 runtime（search_code/read_file）

### 第 6 轮 SFT 评估结果（2026-03-31）

| 类别 | 表现 | 详情 |
|------|------|------|
| 基础常识 | **明显改善** | 首都/化学式/鲁迅/地球公转全对 |
| 身份认知 | 模型仍差 | "你是ChatGPT吗"→"是的"。**但推理层硬规则已 100% 解决** |
| 安全拒绝 | 模型仍差 | 炸弹/入侵仍回答。**但推理层硬规则已 100% 解决** |
| 工具调用 | **未触发** | 模型未输出 `<\|tool_call_start\|>` 格式，直接编造答案 |

**工具调用失败分析**：
- 2420 条工具调用样本在 67571 条总数据里仅占 3.6%，模型没学到
- 242 条原始样本 × 10x 过采样不够，需要更高占比
- 下一轮修复：tool_call 占比提升到 20-25%，Belle 进一步压缩

23. 待做：重新配比 SFT 数据（tool_call 占 20%+），重跑 SFT

---

## 未来计划

### 短期（当前轮次可做）

**1. 推理加重复惩罚** ✓ 已完成
- 135M 模型重复问题严重，`infer.py` 加 `repetition_penalty`（1.2~1.5）
- 改动很小，能显著改善生成质量
- **实现**：`infer.py` 新增 `--rep-penalty` 参数（默认 1.3），在采样前对已生成 token 的 logit 做惩罚：logit>0 时除以 penalty，logit<0 时乘以 penalty，使已出现 token 的概率降低
- **效果**：有效消除循环重复，`你是谁`/`微软是什么` 等问题不再无限循环

**2. RoPE 上下文扩展**
- 当前结论：**NTK scaling 最值得先做，YaRN/LongRoPE/ABF 暂不作为主线**
- 原因：当前主线瓶颈仍是 32k 词表继续预训练收敛、SFT 数据多样性和身份/安全能力，而不是长上下文本身
- NTK scaling：修改 `base` 参数（10000→50万+），推理时直接生效，无需重新训练，适合先做低成本验证
- 对本项目的实际帮助：让 2k 训练模型在 4k/8k 长 prompt 推理时更稳，减少 RoPE 外推失真，但**不能替代长上下文训练**
- 不解决的问题：知识不足、SFT 数据单一、身份漂移、安全拒绝、SSL/window 模式本身的结构限制
- YaRN：理论更精确，但当前阶段收益容易被训练噪声和主线收敛问题淹没，暂不优先
- LongRoPE：更适合专门做超长上下文模型的阶段，当前投入产出比低
- ABF：如果后续验证 NTK 有帮助但稳定性仍不够，可作为“继续预训练版”的长上下文修正方案
- 建议顺序：**NTK 推理验证 → 继续主线训练/SFT → 若长上下文需求明确，再考虑 ABF/YaRN**
- 当前代码状态：`train.py` 已支持 `rope_theta`，`infer.py` 已支持 `--rope-theta`

**3. SFT 数据多样性**（优先级最高）
- 当前 99% 是 Belle 单轮指令，模型只学到"问答"模式
- 可加入的数据源：
  - 中文维基问答 — 补充事实知识（模型连"中国首都"都答不对）
  - 数学/逻辑推理数据 — 当前"1+1"都无法正确回答
  - 更多多轮对话数据 — Claude 的 1279 条太少

**4. 系统化评估**
- 准备固定 test set（20~50 个问题），覆盖：事实知识、数学、代码、多轮、安全拒绝
- 每轮 SFT 后跑一遍对比，量化进步而非靠手动测试

### 中期

**5. 扩大模型规模**
- 135M 的知识容量是硬上限，事实性错误不是数据问题
- 4090 24GB 显存可以试 depth=12 ar=80（~218M），预训练 val_bpb=0.557 优于当前 ar=64
- 需要重新预训练 + SFT，成本较高
- **深度 vs 宽度对比**：depth=24 (258M) 继续预训练进行中，结果将与 ar=80 (218M) 横向比较

**6. Tokenizer 扩容**
- 当前 vocab_size=8192 对中文偏小，每个汉字平均 2~3 个 token，浪费上下文
- 扩到 32K~64K 能显著提高中文效率
- 需要重新训练 tokenizer 和从零预训练

**7. DPO 偏好对齐**
- SFT 之后做一轮 DPO（Direct Preference Optimization）
- 用 SFT 模型生成多个回答，人工标注好/坏，训练偏好模型
- 能提升回答质量和安全性

### 小模型极限突破路线（当前项目的核心战略）

> 更新: 2026-03-31 — 战略调整，工具调用从实验分支提升为核心能力

**硬件约束**: 单卡 RTX 4090 24GB, 64GB RAM, 32 核 CPU

**核心判断**: 400M 模型不可能当知识库。它唯一的出路是当**调度器**——理解意图、调工具、总结结果。模型不需要”知道”答案，只需要知道**去哪找**答案。

#### 核心原则
- **模型是调度器，不是知识库**：检索代替记忆，工具代替计算，固定协议代替自由推理
- **训练严格协议，不训练模糊能力**：极简 schema、固定字段名、固定返回格式
- **高质量小数据优先于杂而大的数据**：500 条严格格式的工具调用样本 > 40 万条通用问答
- **推理时硬规则解决行为问题**：身份认知、安全拒绝靠代码拦截，不靠 SFT 过采样
- **接受外挂就是能力的一部分**：追求系统整体能力，而不是只追求裸模型能力

#### 已验证的失败路线（不再重复）
1. **身份/安全过采样** — 跑了 3+ 轮 SFT，每轮加量，全部无效。400M 模型在 40 万条 Belle 面前，几千条对抗样本毫无作用
2. **通用知识问答** — 400M 装不下足够的世界知识，事实性错误是容量硬伤，不是数据问题
3. **追求更大参数** — depth=24 OOM，ar=80 SFT 反而更差，当前硬件已到极限

---

#### 项目计划（按阶段执行）

### Phase A — 稳住 32k 主线 ✅ 已完成
**目标**：把 `depth=18, ar=64, vocab=32768` 收敛成当前硬件下最强的基础模型。

**结果**：`continued_d18_32k_final.pt`, val_bpb=0.572887

### Phase B — 工具调用为核心的 SFT（当前最高优先级）
**目标**：把 400M 模型训练成一个可靠的工具调用调度器。模型的全部参数预算集中在一件事上：**严格协议下的工具调用**。

**战略变更说明**：
- 原 Phase B（通用 SFT）和原 Phase D（工具调用实验）合并
- 工具调用从”实验分支”提升为”核心能力”
- 身份/安全从 SFT 数据移至推理时硬规则

#### B1. 追加 special tokens（不重训 tokenizer）
- 给现有 32768 tokenizer 追加 6 个工具调用 special token（ID 32768-32773）
- BPE 合并规则不动，只加 special token 注册
- embedding 层从 32768 扩到 32774（改代码，秒级完成）
- 6 个 token：`<|tool_call_start|>` `<|tool_call_end|>` `<|tool_result_start|>` `<|tool_result_end|>` `<|tool_name_search_code|>` `<|tool_name_read_file|>`

#### B2. 短暂继续预训练（500-1000 步）
- 仅让 6 个新 token embedding 收敛
- 不需要长时间训练，新 token 在上下文中出现几百次就够
- 预计 1-2 小时

#### B3. 重新配比 SFT 数据
- **工具调用样本：500+ 条**（当前 51 条远远不够）
  - 每条严格遵守统一协议
  - 覆盖：需要工具 / 不需要工具 / 工具返回后总结
  - 标准链条：`user → assistant(tool_call) → assistant(tool_result) → assistant(final_answer)`
- **Belle 通用：50K 条**（从 250K 砍到 50K，仅维持语言流畅度）
- **身份/安全样本：删除**（全部走推理时规则）
- **预计总量：~55K 条**，远小于之前的 41 万条，但信噪比高得多

#### B4. SFT 训练
- 基于 B2 的 checkpoint
- 数据量小，训练快（预计 2-3 小时）
- 核心观察：工具调用格式是否稳定、工具后总结是否可用

#### B5. 评估（修复 eval_bench）
- 修复 eval_bench.py 的自动评分逻辑
- 新增工具调用专项评估：格式正确率、schema 一致性、总结质量
- 每轮 SFT 后必跑

**退出条件**：
- 模型能稳定输出固定 schema 的单步工具调用（格式正确率 > 90%）
- 工具结果返回后能给出可用总结
- 不需要工具时能直接回答（不乱调工具）

### Phase C — 推理层硬规则 + 工具执行 runtime
**目标**：在模型外围构建完整的系统能力。

- C1. infer.py 加身份硬拦截：检测到身份相关问题，直接返回固定回答
- C2. infer.py 加安全硬拦截：检测到危险问题，直接拒绝
- C3. 工具执行 runtime：解析模型输出的工具调用 → 执行 → 把结果注入上下文
- C4. 端到端测试：用户提问 → 模型决策 → 工具执行 → 结果总结 → 返回用户

**退出条件**：
- 身份问题 100% 正确（硬规则保证）
- 安全问题 100% 拒绝（硬规则保证）
- 工具调用端到端可用

### Phase D — 长上下文（低优先级）
**目标**：低成本验证长上下文是否值得投入。

- D1. 只使用 NTK scaling 做推理侧验证（`--rope-theta 500000`）
- D2. 测试场景：长代码、长文总结、多轮长上下文
- D3. 只有当长上下文需求在真实任务中明确出现，才提升优先级

### Phase E — 评估驱动迭代
**目标**：以后所有改动都围绕”真实提升”做决策。

- E1. 固定三类指标：
  - 预训练：val_bpb
  - SFT：val_bpt
  - 实用性：工具调用正确率 + 端到端成功率
- E2. 每次只推进一条变量
- E3. 不做无 benchmark 支撑的大改动

---

#### 工具调用协议（固定，不再变动）

**工具范围**：第一版只做 2 个
- `search_code`：仓库内代码搜索
- `read_file`：文件读取

**Special tokens**（6 个）：
```
<|tool_call_start|>    <|tool_call_end|>
<|tool_result_start|>  <|tool_result_end|>
<|tool_name_search_code|>  <|tool_name_read_file|>
```

**调用格式**：
```
<|tool_call_start|><|tool_name_search_code|>{“query”: “xxx”}<|tool_call_end|>
```

**结果格式**：
```
<|tool_result_start|>...搜索结果...<|tool_result_end|>
```

**完整链条**：
```
<|reserved_1|>用户问题<|reserved_2|><|tool_call_start|><|tool_name_xxx|>{...}<|tool_call_end|><|tool_result_start|>...结果...<|tool_result_end|>根据搜索结果，...<|reserved_3|>
```

**原则**：
- 输出必须单一：要么工具调用，要么直接回答，不允许混合
- 单步调用优先，不做多步 agent
- 所有样本严格遵守同一格式，零容忍格式偏差

---

#### 当前执行顺序
1. ✓ Phase A 完成：`continued_d18_32k_final.pt`, val_bpb=0.572887
2. → **B1：追加 6 个 special tokens + 扩展 embedding**
3. → B2：短暂继续预训练（500-1000 步）
4. → B3：构建新 SFT 数据（500+ 工具调用 + 50K Belle）
5. → B4：SFT 训练
6. → B5：评估
7. → C1-C4：推理层硬规则 + 工具 runtime
8. 后续按评估结果决定迭代方向

#### 暂缓项
- depth=24 及更大模型
- 通用 agent / 多步工具链
- YaRN / LongRoPE / ABF
- 身份/安全 SFT 过采样（已证明无效）
- 超过 2 个工具的扩展（等 MVP 稳定后再说）

#### 成功标准
- 工具调用格式正确率 > 90%
- 端到端工具调用可用（搜代码、读文件）
- 身份/安全由推理层 100% 保证
- 系统整体可用于真实项目场景

#### 一句话
**400M 模型的全部价值在于当一个可靠的工具调用调度器。其余全部外挂。**
