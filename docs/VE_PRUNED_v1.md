# VE Pruned v1 — P2 架构债第一笔落地

> 砍 v5_best 的深 3 层 VE (layers 13, 15, 17)
> 新 ckpt：`checkpoints/sft_v5_pruned_ve_13_15_17.pt`

## 数字

| 项 | 原 v5_best | pruned |
|----|-----------|--------|
| 参数量 | 404.34 M | **328.82 M (-75.51 M, -18.7%)** |
| val_bpt (full n=700) | 1.6551 | 1.7271 (+0.072, +4.3%) |
| self_audit net_user_failure | 2 / 64 (v9/v10) | **2 / 64 (一致)** |
| 64 条 benchmark 输出 | baseline | **逐条一字不差 == knock {13,15,17} ckpt** |

## 为什么可以直接砍

见 docs/VE_ABLATION_v1.md 和 docs/VE_ABLATION_v2_full.md：
- 三层 VE 单独 knock Δval_bpt 分别为 0.0089 / 0.0247 / 0.0274（远低于 0.05 阈值）
- 合砍 Δ=0.0719，近似加性，无非线性 collapse
- 几何上 9 张 VE 近似正交（非冗余），但 (1↔13, 3↔15, 5↔17) cos≈0.2 提示深层在复制浅层信号

## 如何使用

- 加载：`sft_mod.load_checkpoint("checkpoints/sft_v5_pruned_ve_13_15_17.pt", device)`
  - config 里会带 `ve_layer_skip=[13, 15, 17]`
  - model 构造时 `value_embeds` 只创建 6 个 `nn.Embedding`（1/3/5/7/9/11）
  - 对应 block 的 `ve_gate` 也不构造
  - 前向第 425 行 `if str(i) in self.value_embeds else None` 自动对齐
- strict load_state_dict：0 missing、0 unexpected

## 架构层改动（src/train.py）

1. `GPTConfig` 加 `ve_layer_skip: tuple = ()` 字段（向后兼容，老 ckpt 默认空）
2. `has_ve(layer_idx, n_layer, skip=())`：新增 skip 参数
3. `CausalSelfAttention.__init__` 和 `GPT.__init__` 两处 `has_ve` 调用对齐

## 已验证

- [x] prune_ve.py 删 6 个 tensor (3×value_embeds + 3×ve_gate)，记录 ve_layer_skip
- [x] verify_pruned_ckpt.py strict load 零 miss，val_bpt=1.7271 严格等于 ablation 预测
- [x] self_audit e2e 和 knock ckpt 64 条输出逐条相同
- [x] self_audit e2e 对比 v5_best 自身：net_user_failure 2/64 持平（over_refusal +1 但全被 wiki 救回），halluc 完全相同
- [x] 吞吐/显存 benchmark（scripts/bench_throughput.py）：

### v5_best (404M) vs pruned (329M) 推理性能

| 指标 | base | pruned | delta |
|------|------|--------|-------|
| Prefill T=256 | 30.0k tok/s | 29.6k tok/s | -1.3% |
| Prefill T=1024 | 100.0k tok/s | 104.4k tok/s | **+4.5%** |
| Prefill T=2048 | 130.9k tok/s | 135.4k tok/s | **+3.4%** |
| Decode T=256+32 | 116.6 tok/s | 120.1 tok/s | **+2.9%** |
| Decode T=1024+32 | 118.6 tok/s | 119.5 tok/s | +0.8% |
| Decode T=2048+32 | 65.6 tok/s | 66.0 tok/s | +0.6% |
| **Peak VRAM (decode 2048)** | **2968 MB** | **2673 MB** | **-296 MB (-10%)** |

**诚实解读**：参数 -18.7% 但吞吐只 +1-5%。VE 是 embedding lookup，不是 FLOP
热点；prefill/decode 瓶颈仍在 attention + MLP。核心收益是**稳定节省 300 MB
显存**（3×25M×fp32≈300 MB 完美对上），对 weiyan-api 生产部署意味着同卡能开
更大 batch 或多一个实例。速度提升是锦上添花，不是主要卖点。

## 未做 / 下一步

- [ ] eval_bench.py 全量跑（58 条 BENCH + 更广泛 test set）——非 64 条自检
- [ ] 真实用户流量 replay 复核（需要日志接入）
- [ ] (1↔13, 3↔15, 5↔17) 低秩 tie：把 6 层 VE 再压到 4-5 层（需要训练，非 zero-training）
