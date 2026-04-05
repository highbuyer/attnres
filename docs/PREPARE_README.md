# 数据准备指南

## 功能

- 下载预训练数据（Belle, Wiki ZH, GitHub 代码, Glaive, StarCoder）
- 训练 tokenizer（自定义词表大小）
- 生成训练/验证数据集（parquet 格式）

## 快速开始

```bash
# 下载数据 + 训练 tokenizer
uv run python src/prepare.py

# 只训练 tokenizer（跳过下载）
uv run python src/prepare.py --skip-download

# 指定下载的分片数
uv run python src/prepare.py --num-shards 20
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--num-shards` | Python 代码分片数 (-1=全部) | 10 |
| `--download-workers` | 并行下载线程数 | 8 |
| `--skip-download` | 跳过下载，只训练 tokenizer | False |

## 数据集

| 来源 | 说明 | 存储位置 |
|------|------|----------|
| Belle 0.5M CN | 中文指令数据 | `belle_train.parquet` |
| Belle 2M CN | 中文指令数据 | `belle2m_*.parquet` |
| Wiki ZH | 中文百科 | `wiki_zh_*.parquet` |
| GitHub Python | Python 代码 | `python_*.parquet` |
| GitHub Java | Java 代码 | `java_*.parquet` |
| GitHub JavaScript | JS 代码 | `javascript_*.parquet` |
| Glaive | 工具调用 | `glaive_*.parquet` |
| StarCoder | Python 代码补全 | `starcoder_*.parquet` |

## Tokenizer 训练

- 数据来源：`text_iterator(max_chars=300_000_000)`
- 词表大小：约 32768
- 特殊 token：10 个
  - `reserved_0~3`
  - `tool_call_start` / `tool_call_end`
  - `tool_result_start` / `tool_result_end`
  - `tool_name_search_code` / `tool_name_read_file`

## 输出

- `~/.cache/autoresearch-custom/data/` - parquet 数据文件
- `~/.cache/autoresearch-custom/tokenizer/` - 训练好的 tokenizer
