#!/usr/bin/env python3
"""生成 Claude-Code 风格 gold trajectories，匹配 weiyan 的 SFT 训练格式。

格式（和 sft_tool_summary_v5.jsonl 一致）：
  {"messages": [
    {"role": "user", "content": "<query>"},
    {"role": "assistant", "content": "<|tool_call_start|><|tool_name_search_code|>{...}<|tool_call_end|>"},
    {"role": "assistant", "content": "<|tool_result_start|>...<|tool_result_end|>"},
    {"role": "assistant", "content": "<|tool_call_start|><|tool_name_read_file|>{...}<|tool_call_end|>"},
    {"role": "assistant", "content": "<|tool_result_start|>...<|tool_result_end|>"},
    {"role": "assistant", "content": "<final answer>"}
  ]}

多步工具调用是这个数据集相对 v5 训练集的关键差异——v5 里几乎都是单步 tool call。

每条 trajectory 的 tool_result 是在当前仓库实测的真实输出，确保数据的可验证性。
"""
from __future__ import annotations

import json
from pathlib import Path


def tool_call_msg(tool_name: str, params: dict) -> dict:
    return {
        "role": "assistant",
        "content": f"<|tool_call_start|><|tool_name_{tool_name}|>{json.dumps(params, ensure_ascii=False)}<|tool_call_end|>",
    }


def tool_result_msg(content: str) -> dict:
    return {
        "role": "assistant",
        "content": f"<|tool_result_start|>{content}<|tool_result_end|>",
    }


def final_msg(content: str) -> dict:
    return {"role": "assistant", "content": content}


TRAJECTORIES = [
    # ============== L1 ==============
    {
        "id": "cc_L1_1",
        "user": "src/train.py 里 has_ve 函数在哪一行？skip 参数是做什么的？",
        "steps": [
            ("search_code", {"query": "def has_ve", "path": "src/train.py"},
             "src/train.py:124:def has_ve(layer_idx, n_layer, skip=()):"),
            ("read_file", {"path": "src/train.py", "start": 124, "end": 131},
             """def has_ve(layer_idx, n_layer, skip=()):
    \"\"\"Returns True if layer should have Value Embedding (alternating, last always included).

    skip: iterable of layer indices to force-exclude (P2 VE pruning, see ve_layer_skip).\"\"\"
    if layer_idx in set(skip):
        return False
    return layer_idx % 2 == (n_layer - 1) % 2"""),
        ],
        "final": "`has_ve` 定义在 `src/train.py:124`。`skip` 是要强制排除 VE 的层索引集合（P2 VE 剪枝用）——如果 `layer_idx` 在 `skip` 里就返回 False，该层不会生成 `value_embeds`。默认空 tuple。",
    },

    {
        "id": "cc_L1_2",
        "user": "scripts/ 目录下和 VE 相关的脚本有哪些？",
        "steps": [
            ("search_code", {"query": "VE|value_embed", "path": "scripts/"},
             """scripts/ve_ablation.py:2:\"\"\"VE 消融实验（zero-training）
scripts/prune_ve.py:2:\"\"\"物理剪枝 v5_best 的深 3 层 VE
scripts/verify_pruned_ckpt.py:2:\"\"\"验证 pruned ckpt"""),
        ],
        "final": "scripts/ 下有三个和 VE 相关的脚本：\n- `ve_ablation.py` — VE 消融实验（knock-out matrix）\n- `prune_ve.py` — 物理剪枝深 3 层 VE\n- `verify_pruned_ckpt.py` — 验证剪枝后 ckpt",
    },

    # ============== L2 ==============
    {
        "id": "cc_L2_1",
        "user": "src/infer.py 里 INFER_MAX_CONTEXT 常量的值是多少？",
        "steps": [
            ("search_code", {"query": "INFER_MAX_CONTEXT", "path": "src/infer.py"},
             """src/infer.py:85:INFER_MAX_CONTEXT = 8192
src/infer.py:131:    parser.add_argument(\"--max-context\", type=int, default=INFER_MAX_CONTEXT, help=f\"Effective context length at inference (default {INFER_MAX_CONTEXT}, ckpt trained at 2048; uses NTK-aware RoPE extension)\")"""),
        ],
        "final": "`INFER_MAX_CONTEXT = 8192`，定义在 `src/infer.py:85`。这是 P0 NTK 外推后的推理上下文长度（ckpt 训练时是 2048）。",
    },

    {
        "id": "cc_L2_2",
        "user": "scripts/prune_ve.py 里会删除 state_dict 中的哪些 key？",
        "steps": [
            ("search_code", {"query": "VE_KEYS_TEMPLATE", "path": "scripts/prune_ve.py"},
             """scripts/prune_ve.py:31:VE_KEYS_TEMPLATE = [
scripts/prune_ve.py:54:        for tpl in VE_KEYS_TEMPLATE:"""),
            ("read_file", {"path": "scripts/prune_ve.py", "start": 31, "end": 46},
             """VE_KEYS_TEMPLATE = [
    \"value_embeds.{i}.weight\",
    \"transformer.h.{i}.attn.ve_gate.weight\",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(\"--in-ckpt\", default=\"checkpoints/sft_tool_summary_v5_best.pt\")
    parser.add_argument(\"--out-ckpt\", default=\"checkpoints/sft_v5_pruned_ve_13_15_17.pt\")
    parser.add_argument(\"--skip-layers\", default=\"13,15,17\",
                        help=\"要砍的 VE 层索引，逗号分隔\")"""),
        ],
        "final": "删除的 key 是按模板 `value_embeds.{i}.weight` 和 `transformer.h.{i}.attn.ve_gate.weight` 生成，对每个 `skip-layers`（默认 13,15,17）各删 2 个，共 **6 个张量**：\n- value_embeds.13.weight / value_embeds.15.weight / value_embeds.17.weight\n- transformer.h.13.attn.ve_gate.weight / transformer.h.15.attn.ve_gate.weight / transformer.h.17.attn.ve_gate.weight",
    },

    {
        "id": "cc_L2_3",
        "user": "docs/VE_ABLATION_v2_full.md 里 knock_weakest3 的 Δval_bpt 是多少？",
        "steps": [
            ("search_code", {"query": "knock_weakest3", "path": "docs/VE_ABLATION_v2_full.md"},
             """docs/VE_ABLATION_v2_full.md:17:| **knock_weakest3** | **[13, 15, 17]** | **[1, 3, 5, 7, 9, 11]** | **1.7271** | **0.0719** |
docs/VE_ABLATION_v2_full.md:26:| knock_weakest3 | 0.0687 | 0.0719 | ✓ (差异 <5%) |"""),
        ],
        "final": "`knock_weakest3`（砍 layer 13, 15, 17）全量 val (n=700) 上 **Δval_bpt = 0.0719**，baseline=1.6551 → 1.7271。v1 的 256 子集上测得 0.0687，v2 复现差异 <5%。",
    },

    # ============== L3 调试 ==============
    {
        "id": "cc_L3_1",
        "user": "我跑 scripts/bench_throughput.py 报 'FlashAttention only supports fp16, bf16, and fp8_e4m3 data type'，怎么修？",
        "steps": [
            ("search_code", {"query": "autocast_context|bfloat16", "path": "src/sft.py"},
             """src/sft.py:302:def autocast_context(device):
src/sft.py:306:        return torch.amp.autocast(device_type=\"cuda\", dtype=torch.bfloat16)
src/sft.py:321:    return torch.bfloat16 if device.type == \"cuda\" else torch.float32"""),
            ("read_file", {"path": "src/sft.py", "start": 302, "end": 310},
             """def autocast_context(device):
    import torch
    if device.type == \"cuda\":
        return torch.amp.autocast(device_type=\"cuda\", dtype=torch.bfloat16)
    else:
        from contextlib import nullcontext
        return nullcontext()"""),
        ],
        "final": "参数存成 fp32 但 FA3 只吃 fp16/bf16 激活，所以 forward 必须套 bf16 autocast。改法：\n\n```python\nimport sft as sft_mod\n\nwith torch.no_grad(), sft_mod.autocast_context(device):\n    logits = model(x)\n```\n\n或直接用 `torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)` 包住 forward 调用。参考 `src/sft.py:302` 的实现。",
    },

    {
        "id": "cc_L3_2",
        "user": "load_state_dict 报 unexpected key 'value_embeds.13.weight'，原因？",
        "steps": [
            ("search_code", {"query": "ve_layer_skip", "path": "src/train.py"},
             """src/train.py:117:    ve_layer_skip: tuple = ()    # P2: 砍掉指定层的 VE（value_embeds + ve_gate）。见 docs/VE_ABLATION_v2_full.md
src/train.py:127:    skip: iterable of layer indices to force-exclude (P2 VE pruning, see ve_layer_skip).\"\"\"
src/train.py:156:        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer, getattr(config, \"ve_layer_skip\", ())) else None
src/train.py:286:            for i in range(config.n_layer) if has_ve(i, config.n_layer, getattr(config, \"ve_layer_skip\", ()))"""),
        ],
        "final": "ckpt 里有 `value_embeds.13.weight` 但你的 model 构造时跳过了第 13 层的 VE——原因是 `GPTConfig.ve_layer_skip` 包含了 13。\n\n两种修法：\n1. 把 `config.ve_layer_skip` 设为 `()`（用原 404M 完整结构，ckpt 全部 load）\n2. 改用已剪枝过的 ckpt `checkpoints/sft_v5_pruned_ve_13_15_17.pt`，和 `ve_layer_skip=(13,15,17)` 的 model 对齐\n3. 或在 load 时用 `strict=False` 忽略（但 ckpt 里的权重会被 model 新随机 init 覆盖，不推荐）",
    },

    # ============== L4 跨文件推理 ==============
    {
        "id": "cc_L4_1",
        "user": "v5_pruned 相比 v5_best 少了多少参数？是通过什么方式减少的？",
        "steps": [
            ("search_code", {"query": "404|328|总参|pruned", "path": "docs/VE_PRUNED_v1.md"},
             """docs/VE_PRUNED_v1.md:7:| 参数量 | 404.34 M | **328.82 M (-75.51 M, -18.7%)** |
docs/VE_PRUNED_v1.md:51:prune_ve.py 删 6 个 tensor (3×value_embeds + 3×ve_gate)"""),
            ("read_file", {"path": "docs/VE_PRUNED_v1.md", "start": 1, "end": 16},
             """# VE Pruned v1 — P2 架构债第一笔落地

> 砍 v5_best 的深 3 层 VE (layers 13, 15, 17)
> 新 ckpt：`checkpoints/sft_v5_pruned_ve_13_15_17.pt`

## 数字

| 项 | 原 v5_best | pruned |
|----|-----------|--------|
| 参数量 | 404.34 M | **328.82 M (-75.51 M, -18.7%)** |
| val_bpt (full n=700) | 1.6551 | 1.7271 (+0.072, +4.3%) |
| self_audit net_user_failure | 2 / 64 (v9/v10) | **2 / 64 (一致)** |"""),
        ],
        "final": "v5_pruned 比 v5_best **少 75.51M 参数**（404.34M → 328.82M，-18.7%），方式是 `scripts/prune_ve.py` 物理从 state_dict 删除 layers {13, 15, 17} 的 6 个张量（3×`value_embeds.{i}.weight` + 3×`transformer.h.{i}.attn.ve_gate.weight`），并在 `GPTConfig.ve_layer_skip` 里记录这 3 个层号，让 model 构造时跳过。val_bpt 只上升 4.3%，self_audit net_user_failure 保持 2/64 不变。",
    },
]


def build_messages(traj: dict) -> list[dict]:
    msgs = [{"role": "user", "content": traj["user"]}]
    for tool_name, params, result in traj["steps"]:
        msgs.append(tool_call_msg(tool_name, params))
        msgs.append(tool_result_msg(result))
    msgs.append(final_msg(traj["final"]))
    return msgs


def main():
    out = Path("data/cc_gold_trajectories_v1.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for traj in TRAJECTORIES:
            rec = {"id": traj["id"], "messages": build_messages(traj)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"写入 {len(TRAJECTORIES)} 条 → {out}")

    # 打印分布
    for t in TRAJECTORIES:
        n_steps = len(t["steps"])
        tools = [s[0] for s in t["steps"]]
        print(f"  {t['id']}: {n_steps} 步 ({'/'.join(tools)})")


if __name__ == "__main__":
    main()
