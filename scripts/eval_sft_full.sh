#!/bin/bash
# 评估 SFT 后的 ckpt：chat smoke + tool_call 触发率 + 与 base 对比
# 用法：bash scripts/eval_sft_full.sh

set -e
cd "$(dirname "$0")/.."

PYTHON=.venv/bin/python
BASE_CKPT=checkpoints/d36_v2_mla_best.pt
SFT_CKPT=checkpoints/d36_v2_sft_full_slim.pt
OUT_DIR=eval_results
mkdir -p "$OUT_DIR"

echo "=== Stage 1: SFT ckpt 文件验证 ==="
ls -lh "$SFT_CKPT"
$PYTHON -u -c "
import torch
ck = torch.load('$SFT_CKPT', map_location='cpu', weights_only=False)
print('keys:', list(ck.keys()))
print('config.layer:', ck['config'].get('n_layer'))
print('last_loss:', ck.get('last_loss'))
print('base_ckpt:', ck.get('base_ckpt'))
print('steps:', ck.get('steps'))
"

echo
echo "=== Stage 2: Base vs SFT chat smoke 对比 ==="
echo "--- base ckpt 输出（pretrain only） ---" | tee "$OUT_DIR/base_chat.txt"
$PYTHON -u src/infer_v2.py --checkpoint "$BASE_CKPT" --chat \
    --max-tokens 120 --temperature 0.7 --rep-penalty 1.15 \
    | tee -a "$OUT_DIR/base_chat.txt"

echo
echo "--- SFT ckpt 输出（after sft） ---" | tee "$OUT_DIR/sft_chat.txt"
$PYTHON -u src/infer_v2.py --checkpoint "$SFT_CKPT" --chat \
    --max-tokens 120 --temperature 0.7 --rep-penalty 1.15 \
    | tee -a "$OUT_DIR/sft_chat.txt"

echo
echo "=== Stage 3: tool_call 触发率统计 ==="
$PYTHON -u <<'PY'
import sys, re
sys.path.insert(0, 'src')
import torch
from prepare import Tokenizer
from model_v2 import GPT_v2, GPTConfigV2
from infer_v2 import generate, load_model

device = torch.device('cuda')
tokenizer = Tokenizer.from_directory()
model, cfg = load_model('checkpoints/d36_v2_sft_full_slim.pt', device)

# 测 30 个工具触发型 prompt
TOOL_PROMPTS = [
    '帮我搜一下代码里 GPTConfigV2 的定义在哪个文件',
    '项目里 src/sft_v2.py 的入口函数在哪？',
    '读一下 README.md 的前 20 行',
    '检查 datasets/sft_archive 目录有哪些文件',
    '搜索代码里所有用 tokenizer.encode 的地方',
    '看一下 model_v2.py 第 100-200 行',
    '找一下哪里定义了 evaluate_bpb',
    '查一下 import torch 在多少个 py 文件里',
    'grep "tool_call" 在 src 下的所有匹配',
    '看 .gitignore 里都忽略了什么',
] * 3   # 30 个

n_tool_call = 0
n_total = len(TOOL_PROMPTS)
samples = []
for p in TOOL_PROMPTS:
    out = generate(model, cfg, tokenizer, p, max_tokens=80, temperature=0.7,
                   rep_penalty=1.15, chat_mode=True, device=device)
    has_tc = '<tool_call>' in out or 'search_code' in out.lower() or 'read_file' in out.lower()
    if has_tc:
        n_tool_call += 1
    samples.append((p, out, has_tc))

print(f'\n=== tool_call 触发率: {n_tool_call}/{n_total} = {100*n_tool_call/n_total:.0f}% ===')
print('\n=== 5 条样本 ===')
for i, (p, out, hit) in enumerate(samples[:5], 1):
    print(f'[{i}] hit={hit} prompt={p!r}')
    print(f'    {out[:200]}')
PY

echo
echo "=== 评估完成 ==="
echo "对比文件：$OUT_DIR/base_chat.txt vs $OUT_DIR/sft_chat.txt"
