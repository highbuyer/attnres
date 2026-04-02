# Next commands after current training finishes

不要在当前继续预训练结束前启动下面命令。

## 1) 构建新的 SFT 数据集
```bash
uv run python make_sft_data.py --out ~/Desktop/sft_mixed_v7.jsonl
```

预期组成（默认参数）：
- Belle: 250000
- BELLE multiturn: 80000
- BELLE school math: 60000
- Claude: 15894
- Rejection: 151
- Identity: 2550
- Negative identity: 2080

## 2) 启动下一轮 SFT
默认 `sft.py` 已指向：`~/Desktop/sft_mixed_v7.jsonl`

```bash
uv run python sft.py continued_d18_32k_final.pt > sft_d18_32k.log 2>&1
```

## 3) 如需后台运行
```bash
nohup uv run python sft.py continued_d18_32k_final.pt > sft_d18_32k.log 2>&1 &
```

## 4) 监控日志
```bash
tail -f sft_d18_32k.log
```

## 5) 完成后评估
```bash
uv run python eval_bench.py --checkpoint sft_d18_v3_checkpoint.pt
```

说明：
- 当前主线 checkpoint 是 `continued_d18_32k_final.pt`
- 先确保当前 continue pretrain 完全结束，再启动 SFT
- 如果继续预训练又产出更好的 32k checkpoint，优先替换这里的输入 checkpoint
