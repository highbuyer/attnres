#!/usr/bin/env python3
"""VE 消融实验（zero-training）：v5_best 上把指定层 value_embeds 置零，看 val_bpt 抖多少。

动机：VE 吃了 56% 参数（226M / 404M），挂在 9 层 {1,3,5,7,9,11,13,15,17}，但
架构继承自 nanochat，没有 per-layer ablation 数据。P2 要砍 VE 必须先知道每张
VE 贡献多少——这个脚本用 zero-training 的方式给 P2 决策提供底盘。

方法：
1. 几何 probe：9 张 VE 的 weight norm、彼此 cosine（看内部冗余）、ve_gate 的
   weight norm（看每层 gate 被用的程度）。秒级。
2. Knock-out：对指定层把 `value_embeds[str(i)].weight.data.zero_()`，前向里
   `ve = embedding(token_id) = 0`，`gate * 0 = 0`，等效该层没 VE。跑 SFT val
   set 的 val_bpt。eval 后 restore 原 weight，单次 load。
3. 矩阵：
   - baseline（0 层 knock）
   - 单层 knock × 9 条
   - 浅 5 knock（{1,3,5,7,9}）
   - 深 5 knock（{9,11,13,15,17}）
   - 只留深 3（knock {1,3,5,7,9,11,13}）
   - 只留浅 3（knock {7,9,11,13,15,17}）
   - 全砍 9 层

每组 val_bpt 与 baseline 对比，> 0.05 抖动才算"这层真的在干活"。低于 0.05 的
说明 VE 该层冗余；可以进 P2 ablation 的"砍掉列表"。

不改 v5_best 文件。产出 runs/ve_ablation.jsonl + docs/VE_ABLATION_v1.md。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# 保护 proxy env（与 self_audit 同理）
_PROXIES_AT_STARTUP: dict[str, str] = {}
for _name, _val in os.environ.items():
    _lname = _name.lower()
    if _val and _lname.endswith("_proxy"):
        _PROXIES_AT_STARTUP[_lname[:-6]] = _val

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# 复用 sft.py 里的 model-class 加载器、dataset 切分、evaluate_sft
import sft as sft_mod  # noqa: E402


VE_LAYERS = [1, 3, 5, 7, 9, 11, 13, 15, 17]


def geometry_probe(model) -> dict:
    """9 张 VE 之间的几何关系。"""
    ves = {i: model.value_embeds[str(i)].weight.detach().cpu().float() for i in VE_LAYERS}
    norms = {i: ves[i].norm(dim=1).mean().item() for i in VE_LAYERS}
    # pairwise cosine（先 per-token L2-normalize 再做平均 cosine）
    normed = {i: F.normalize(ves[i], dim=1) for i in VE_LAYERS}
    pairwise = {}
    for i in VE_LAYERS:
        for j in VE_LAYERS:
            if j <= i:
                continue
            c = (normed[i] * normed[j]).sum(dim=1).mean().item()
            pairwise[f"{i}-{j}"] = round(c, 4)
    # ve_gate weight norm per layer（gate 大 = 这层在用 VE 的程度高）
    gate_norms = {}
    for i in VE_LAYERS:
        gw = model.transformer.h[i].attn.ve_gate.weight.detach().cpu().float()
        gate_norms[i] = gw.norm().item()
    return {
        "ve_norm_mean_per_layer": {str(k): round(v, 3) for k, v in norms.items()},
        "ve_gate_weight_norm_per_layer": {str(k): round(v, 4) for k, v in gate_norms.items()},
        "ve_pairwise_cosine_mean": pairwise,
    }


def knock_out(model, layers: list[int]) -> None:
    for i in layers:
        model.value_embeds[str(i)].weight.data.zero_()


def restore(model, snapshots: dict) -> None:
    for i, w in snapshots.items():
        model.value_embeds[str(i)].weight.data.copy_(w)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/sft_tool_summary_v5_best.pt")
    parser.add_argument("--data", default="sft_tool_summary_v5.jsonl")
    parser.add_argument("--out", default="runs/ve_ablation.jsonl")
    parser.add_argument("--max-val-samples", type=int, default=None,
                        help="若 val 太大可截断到这个数以压缩 eval 时间；None=全量")
    args = parser.parse_args()

    sft_mod._ensure_model_defs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parameter_dtype = sft_mod.parameter_dtype_for_training(device)
    compute_dtype = sft_mod.compute_dtype_for_training(device)

    print(f"Loading {args.checkpoint} on {device}...")
    ckpt = sft_mod.load_checkpoint(args.checkpoint, device)
    config = ckpt["config"]
    print(f"ckpt val_bpt={ckpt['val_bpt']:.4f}  step={ckpt['step']}  n_layer={config.n_layer}")

    model = sft_mod.GPT(config).to(device=device, dtype=parameter_dtype)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state"].items()}
    model.load_state_dict(state, strict=False)
    model.to(dtype=parameter_dtype)

    head_dim = config.n_embd // config.n_head
    cos, sin = model._precompute_rotary_embeddings(model.rotary_seq_len, head_dim, device=device)
    model.cos, model.sin = cos.to(compute_dtype), sin.to(compute_dtype)

    print("Geometry probe...")
    geom = geometry_probe(model)
    print(json.dumps(geom, ensure_ascii=False, indent=2))

    print(f"\nBuilding val_data from {args.data}...")
    raw, train_data, val_data = sft_mod.build_datasets(args.data)
    if args.max_val_samples and len(val_data) > args.max_val_samples:
        val_data = val_data[:args.max_val_samples]
    print(f"val_data: {len(val_data)} samples")

    # snapshot 原 VE
    snapshots = {i: model.value_embeds[str(i)].weight.data.clone() for i in VE_LAYERS}

    trials = [
        ("baseline", []),
    ]
    # 单层 knock
    for i in VE_LAYERS:
        trials.append((f"knock_{i}", [i]))
    # 组合
    trials += [
        ("knock_shallow5", [1, 3, 5, 7, 9]),
        ("knock_deep5",    [9, 11, 13, 15, 17]),
        ("keep_deep3",     [1, 3, 5, 7, 9, 11, 13]),   # only {15, 17} kept → wait 留 3 需要去掉 7 个
        ("keep_shallow3",  [7, 9, 11, 13, 15, 17]),
        ("knock_all9",     list(VE_LAYERS)),
    ]
    # 校正 keep_deep3：VE_LAYERS 9 层，留 3 层最深 → {13,15,17}，knock 其余 6 层
    trials = [(name, layers) for name, layers in trials if name != "keep_deep3"]
    trials.append(("keep_deep3", [1, 3, 5, 7, 9, 11]))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    results = []
    baseline_bpt = None

    with open(args.out, "w", encoding="utf-8") as f:
        for name, knock_layers in trials:
            print(f"\n=== {name}  knock={knock_layers} ===")
            knock_out(model, knock_layers)
            bpt = sft_mod.evaluate_sft(model, val_data, device)
            restore(model, snapshots)
            delta = None
            if name == "baseline":
                baseline_bpt = bpt
            else:
                delta = round(bpt - baseline_bpt, 4)
            row = {
                "name": name,
                "knocked_layers": knock_layers,
                "kept_layers": [i for i in VE_LAYERS if i not in knock_layers],
                "val_bpt": round(bpt, 4),
                "delta_vs_baseline": delta,
            }
            print(f"  val_bpt={bpt:.4f}  delta={delta}")
            results.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 输出 markdown 摘要
    summary_path = Path("docs/VE_ABLATION_v1.md")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# VE Ablation v1 — zero-training probe of 9-layer value embeddings",
        "",
        f"> ckpt: `{args.checkpoint}` step={ckpt['step']} val_bpt={ckpt['val_bpt']:.4f}",
        f"> data: `{args.data}` val_samples={len(val_data)}",
        "",
        "## 几何 probe",
        "",
        "### 每层 VE weight 平均 norm（per-token L2 mean）",
        "",
        "| layer | ‖ve‖ mean | ve_gate ‖W‖ |",
        "|-------|-----------|-------------|",
    ]
    for i in VE_LAYERS:
        lines.append(
            f"| {i} | {geom['ve_norm_mean_per_layer'][str(i)]} | {geom['ve_gate_weight_norm_per_layer'][str(i)]} |"
        )
    lines += [
        "",
        "### 9 张 VE 两两 cosine（对同一 token 的 VE 向量做 L2 normalize 后平均 cos）",
        "",
    ]
    # pairwise 写成矩阵
    lines.append("| | " + " | ".join(str(j) for j in VE_LAYERS) + " |")
    lines.append("|--|" + "|".join("---" for _ in VE_LAYERS) + "|")
    for i in VE_LAYERS:
        row = [f"**{i}**"]
        for j in VE_LAYERS:
            if i == j:
                row.append("—")
            else:
                key = f"{min(i,j)}-{max(i,j)}"
                row.append(str(geom["ve_pairwise_cosine_mean"].get(key, "")))
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "## Knock-out val_bpt 矩阵",
        "",
        "| 试验 | knocked | kept | val_bpt | Δ vs baseline |",
        "|------|---------|------|---------|---------------|",
    ]
    for r in results:
        lines.append(
            f"| {r['name']} | {r['knocked_layers']} | {r['kept_layers']} | {r['val_bpt']} | "
            f"{r['delta_vs_baseline'] if r['delta_vs_baseline'] is not None else '—'} |"
        )
    lines += [
        "",
        "## 读图指南",
        "",
        "- Δ < 0.01：该层 VE 几乎不参与 → P2 可砍",
        "- Δ ∈ [0.01, 0.05]：弱贡献 → 可合并（低秩或共享）",
        "- Δ > 0.05：显著贡献 → 保留",
        "",
        "矩阵末几行（shallow5/deep5/keep_3）告诉你**组合**砍的代价是加性还是非线性。",
        "如果 deep5 Δ ≫ shallow5 Δ → VE 主要服务深层；反之浅层。",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {args.out}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
