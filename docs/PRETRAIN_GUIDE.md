# 预训练流程文档

> 适用项目: `/home/langshen/base_mode/attnres/`
> 最后更新: 2026-03-27

---

## 1. 整体流程

```
prepare.py (数据下载 + tokenizer 训练)
    ↓
train.py (从零预训练)
    ↓ best_checkpoint.pt
continue_pretrain.py (可选: 继续预训练, 修正数据格式等)
    ↓ continued_checkpoint.pt
make_sft_data.py (构建 SFT 数据集)
    ↓ ~/Desktop/sft_mixed.jsonl
sft.py (SFT 微调)
    ↓ sft_checkpoint.pt
infer.py (推理验证)
```

---

## 2. 数据准备 (prepare.py)

### 2.1 运行

```bash
cd /home/langshen/base_mode/attnres
uv run python prepare.py
```

### 2.2 功能

1. **下载数据**: 从 HuggingFace 下载所有数据源到 `~/.cache/autoresearch-custom/data/`
2. **训练 Tokenizer**: 用 rustbpe 训练 BPE tokenizer, 保存到 `~/.cache/autoresearch-custom/tokenizer/`

### 2.3 数据源

| 来源 | 文件名前缀 | 采样权重 | 格式 |
|------|-----------|----------|------|
| Belle 0.5M CN | `belle_` | 3.0 | `<\|reserved_1\|>指令<\|reserved_2\|>回答<\|reserved_3\|>` |
| Belle 2M CN | `belle2m_` | 3.0 | 同上 |
| Wiki ZH | `wiki_zh_` | 3.0 | 纯文本 |
| Glaive v2 | `glaive_` | 3.0 | 特殊 token 格式 (经 `_convert_glaive_chat` 转换) |
| Glaive v1 | `glaive_v1_` | 3.0 | 同上 |
| Python (GitHub) | `python_` | 1.0 | 纯代码文本 |
| Java (GitHub) | `java_` | 1.0 | 纯代码文本 |
| JavaScript (GitHub) | `javascript_` | 1.0 | 纯代码文本 |
| StarCoder | `starcoder_` | 0.14 | Python 代码文本 |

### 2.4 Tokenizer

- **算法**: BPE (rustbpe 训练, tiktoken 运行时)
- **Vocab size**: 8192 (含 4 个特殊 token)
- **Split pattern**: GPT-4 style
- **特殊 token**:
  - `<|reserved_0|>` = BOS (也用作 padding)
  - `<|reserved_1|>` = USER 轮次开始
  - `<|reserved_2|>` = ASSISTANT 轮次开始
  - `<|reserved_3|>` = EOS / 序列结束

### 2.5 数据格式要求

对话类数据 (Belle/Glaive) 必须使用特殊 token 格式, 不能用明文 `Human:/Assistant:`。
`Tokenizer.encode` 使用 `enc.encode(text, allowed_special="all")` 以正确识别特殊 token。

### 2.6 Dataloader

- **BOS 对齐**: 每行数据以 BOS token 开头
- **Best-fit packing**: 多个文档打包到一行, 最大化利用率 (100%, 无 padding)
- **Deficit tracking**: 按 `DATA_MIX_WEIGHTS` 权重采样各数据源
- **验证集**: 从各数据源抽取 10% 合并为 `val.parquet`

### 2.7 重新处理数据

如果修改了数据格式, 需要删除已有缓存再重新运行:

```bash
cd ~/.cache/autoresearch-custom/data
# 删除需要重新处理的 parquet 文件
rm -f belle_train.parquet belle_val_tmp.parquet
rm -f belle2m_*.parquet belle2m_done.flag
rm -f glaive_train.parquet glaive_val_tmp.parquet
rm -f glaive_v1_train.parquet glaive_v1_val_tmp.parquet
rm -f val.parquet  # 验证集也要重新合并

cd /home/langshen/base_mode/attnres
uv run python prepare.py
```

代码和 Wiki 数据不需要重新处理 (格式未变)。

---

## 3. 从零预训练 (train.py)

### 3.1 运行

```bash
cd /home/langshen/base_mode/attnres
uv run python train.py
```

### 3.2 模型架构

```python
GPTConfig(
    sequence_len=2048,
    vocab_size=8192,     # 由 tokenizer 决定 (含特殊 token)
    n_layer=12,          # DEPTH
    n_head=12,           # n_embd // HEAD_DIM
    n_kv_head=12,        # MHA (GQA 可选)
    n_embd=768,          # DEPTH * ASPECT_RATIO
    window_pattern="SSL" # S=半窗口滑动注意力, L=全注意力
)
```

**特殊设计**:
- **Block AttnRes** (Kimi 2026): 跨层注意力残差, sublayers_per_block=3, 用 Linear(n_embd→1) 做 pseudo-query
- **Value Embedding**: 交替层有独立的 value embedding, 通过 input-dependent gate 混入
- **QK-Norm**: query 和 key 在 RoPE 后做 RMSNorm
- **ReLU²**: MLP 用 `F.relu(x).square()`
- **Logit softcap=15**: 输出 logits 用 tanh 压缩

### 3.3 优化器 (MuonAdamW)

| 参数组 | 优化器 | 学习率 | 说明 |
|--------|--------|--------|------|
| Transformer 矩阵参数 | Muon | 0.04 | Polar express 正交化, NorMuon 方差归约 |
| Token embedding | AdamW | 0.2 | |
| Value embedding | AdamW | 0.2 | |
| Unembedding (lm_head) | AdamW | 0.004 | |
| AttnRes projection | AdamW | 0.5 | scalar 参数 |
| AttnRes RMSNorm | AdamW | 0.15 | |

所有 LR 按 `(model_dim / 768)^{-0.5}` 缩放。

### 3.4 学习率调度

```
warmup (0%) → constant → warmdown (最后 60%)
```

- `WARMUP_RATIO = 0.0` (无 warmup)
- `WARMDOWN_RATIO = 0.6`
- `FINAL_LR_FRAC = 0.05`
- Muon momentum: 0.85 → 0.95 (前 300 步线性)
- Weight decay: `0.01 * (1 - progress)` (线性衰减)

### 3.5 关键超参

| 参数 | 值 | 说明 |
|------|-----|------|
| TOTAL_BATCH_SIZE | 2^19 (~524K tokens) | 每步总 token 数 |
| DEVICE_BATCH_SIZE | 16 | 每次前向/反向的 batch |
| GRAD_ACCUM | 自动计算 | TOTAL_BATCH_SIZE / (DEVICE_BATCH_SIZE × MAX_SEQ_LEN) |
| TOTAL_STEPS | 6000 | 步数上限 |
| EVAL_INTERVAL | 500 | 每 500 步评估一次 |
| EARLY_STOP_PATIENCE | 3 | 连续 3 次 eval 无改善则停止 |

### 3.6 输出

- `best_checkpoint.pt`: 最佳验证集 checkpoint (早停时保存)
- `checkpoint.pt`: 最终 checkpoint
- `run.log`: 训练日志
- `results.tsv`: 实验结果追加记录

### 3.7 评估指标

**val_bpb (bits per byte)**: vocab size 无关的评估指标。
计算方式: 对验证集每个 token 计算 cross-entropy (nats), 除以对应 token 的 UTF-8 字节数, 转换为 bits。
特殊 token (字节数=0) 不参与计算。

### 3.8 超参搜索记录

完整记录在 `results.tsv`, 关键发现:

| 实验 | val_bpb | 结论 |
|------|---------|------|
| depth 6→8→12 | 0.97→0.81→0.57 | 更深更好 |
| ar 64→80 (depth=8) | 0.97→0.81 | 更宽更好 |
| HEAD_DIM 128→64 | 0.57→0.56 | 64 更优 |
| HEAD_DIM 64→32 | 0.56→0.61 | 32 太小 |
| weight_decay 0.2→0.01 | 0.57→0.567 | 小 WD 更好 |
| window SSL→SSSL | 0.566→0.567 | SSL 最优 |
| AttnRes spb=0 vs spb=3 | 0.623→0.560 | AttnRes 有效 (同 ar=64) |

---

## 4. 继续预训练 (continue_pretrain.py)

### 4.1 用途

在已有 checkpoint 基础上, 用更新的数据继续训练。典型场景: 数据格式修正后不想从零开始。

### 4.2 运行

```bash
cd /home/langshen/base_mode/attnres
# 默认从 best_checkpoint.pt 加载
uv run python continue_pretrain.py

# 指定 checkpoint
uv run python continue_pretrain.py path/to/checkpoint.pt
```

### 4.3 与 train.py 的区别

| 项目 | train.py | continue_pretrain.py |
|------|----------|---------------------|
| 初始化 | 随机 | 从 checkpoint 加载 |
| MATRIX_LR | 0.04 | 0.01 (1/4) |
| EMBEDDING_LR | 0.2 | 0.05 (1/4) |
| TOTAL_STEPS | 6000 | 2000 |
| WARMUP_RATIO | 0.0 | 0.025 (50 步) |
| 优化器状态 | 新建 | 新建 (checkpoint 未存优化器) |

### 4.4 输出

- `continued_checkpoint.pt`: 最佳 checkpoint
- `continued_checkpoint_final.pt`: 最终 checkpoint

---

## 5. SFT 微调 (sft.py)

### 5.1 数据准备

```bash
# 1. 提取 Claude 对话数据
python ~/Desktop/extract_sft_data.py

# 2. 构建混合数据集
cd /home/langshen/base_mode/attnres
uv run python make_sft_data.py
# 输出: ~/Desktop/sft_mixed.jsonl
```

### 5.2 模型宪法 (System Prompt)

模型名: **微研** (Wēi Yán)

宪法内容 (定义在 `sft.py` 的 `SYSTEM_PROMPT` 和 `constitution.md`):

```
你是微研，一个技术助手。用与用户相同的语言简洁回答。不确定时如实说明，不编造事实。拒绝有害内容。
```

**注入方式**: 在 `format_samples_split` 中，将 system prompt 拼到每条对话第一条 user 消息前面。
模型在训练中内化这些规则，推理时不需要额外配置。

**训练时格式**:

```
BOS + USER_ID + [系统提示\n用户问题] + ASST_ID + 助手回答 + EOS
```

**推理时格式** (`infer.py` 已同步):

```
BOS + USER_ID + [系统提示\n用户输入] + ASST_ID → 模型续写
```

**关键点**:
- 训练和推理的格式必须一致，否则模型行为不可预测
- system prompt 占用约 30 个 token，对 2048 上下文影响很小
- 如需修改宪法内容，需同时改 `sft.py` 的 `SYSTEM_PROMPT` 和 `infer.py` 的 `system_prompt`

### 5.3 数据格式

- Loss 只在 assistant token 和 EOS 上计算
- 连续多个 assistant 回答独立成 sample, 不合并
- 超长序列在 turn boundary 处从左截断

### 5.3 运行

```bash
cd /home/langshen/base_mode/attnres

# 使用预训练 checkpoint (默认 best_checkpoint.pt)
uv run python sft.py

# 使用继续预训练的 checkpoint
uv run python sft.py continued_checkpoint.pt
```

### 5.4 配置

| 参数 | 值 |
|------|-----|
| LR | 2e-5 |
| DEVICE_BATCH_SIZE | 4 |
| GRAD_ACCUM | 8 (有效 batch=32) |
| TOTAL_STEPS | 12000 (250K 数据, ~1.5 epoch) |
| WARMUP_STEPS | 100 |
| WARMDOWN_START | 9600 (80%) |
| VAL_RATIO | 5% |
| EVAL_INTERVAL | 500 |

### 5.5 输出

- `sft_checkpoint.pt`: 最佳 SFT checkpoint

---

## 6. 推理 (infer.py)

### 6.1 运行

```bash
cd /home/langshen/base_mode/attnres

# 单句推理
uv run python infer.py '你好' 128 0.8 --checkpoint sft_checkpoint.pt

# 交互模式
uv run python infer.py --checkpoint sft_checkpoint.pt
```

### 6.2 参数

| 参数 | 位置 | 默认值 | 说明 |
|------|------|--------|------|
| prompt | 1 | None | 可选, 单次推理的 prompt |
| max_tokens | 2 | 256 | 最大生成 token 数 |
| temperature | 3 | 0.2 | 采样温度 (建议 0.6~0.8) |
| --checkpoint | - | best_checkpoint.pt | checkpoint 路径 |
| --top-k | - | 40 | Top-k 采样 |
| --top-p | - | 0.9 | Top-p 核采样 |

### 6.3 推理格式

输入构造: `BOS + USER_ID + 用户输入 + ASST_ID` → 模型续写 → 遇到特殊 token 停止

后处理: 截断 `Human:`/`Assistant:` 明文 (预训练残留 workaround)

---

## 7. 常见问题

### Q: GPTConfig pickle 报错 "Can't pickle GPTConfig"
A: exec() 创建的类需要设置 `GPTConfig.__module__ = '__main__'`。sft.py, infer.py, continue_pretrain.py 都已修复。

### Q: 模型输出 `Human:`/`Assistant:` 明文
A: 预训练数据中 Belle/Glaive 使用了明文格式。解决方案:
1. prepare.py 已改为特殊 token 格式
2. 用 continue_pretrain.py 继续预训练
3. infer.py 有后处理截断作为 workaround

### Q: 如何调整 SFT 步数
A: `TOTAL_STEPS = 样本数 / 有效 batch × epoch 数`, `WARMDOWN_START = TOTAL_STEPS × 0.8`

### Q: 重新跑 prepare.py 会影响正在训练的模型吗
A: 不会。prepare.py 处理的是 `~/.cache/` 下的 parquet, SFT 用的是 `~/Desktop/` 下的 jsonl, 完全独立。
