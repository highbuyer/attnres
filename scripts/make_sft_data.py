#!/usr/bin/env python3
"""构建 SFT 训练数据集。

来源：
  1. Belle 0.5M CN (belle_train.parquet) — 中文指令问答
  2. Belle 2M CN (belle2m_0000-0002.parquet) — 中文指令问答
  3. BELLE multiturn_chat_0.8M — 中文多轮对话
  4. BELLE school_math_0.25M — 数学推理
  5. Claude 会话日志 — 真实的实战对话
  6. 身份/拒绝样本 — 强化角色认知与安全
  7. 否定身份样本 — 强化“不是 ChatGPT/GPT-4”等身份边界

用法示例：
  python make_sft_data.py --claude ~/Desktop/claude_sft_v4_clean.jsonl --upsample 100 --out ~/Desktop/sft_mixed_v4.jsonl
"""
import json
import random
import re
import argparse
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from tool_protocol import validate_tool_sample

# ---------------------------------------------------------------------------
# 1. 解析工具函数
# ---------------------------------------------------------------------------
_HUMAN_RE = re.compile(r'^Human:\s*', re.IGNORECASE)
_ASST_RE  = re.compile(r'^(Assistant|A):\s*', re.IGNORECASE)

def parse_belle_text(text: str):
    """将 'Human: ...\nAssistant: ...' 转成 messages list，失败返回 None。"""
    text = text.strip()
    for sep in ['\nAssistant:', '\nA:']:
        idx = text.find(sep)
        if idx != -1:
            user_part = _HUMAN_RE.sub('', text[:idx]).strip()
            asst_part = _ASST_RE.sub('', text[idx+len(sep):]).strip()
            if len(user_part) < 2 or len(asst_part) < 10:
                return None
            return [{'role': 'user', 'content': user_part},
                    {'role': 'assistant', 'content': asst_part}]
    return None

def load_belle_parquet(path: str):
    tbl = pq.read_table(path)
    rows = tbl['text'].to_pylist()
    samples = []
    for text in rows:
        if '<|reserved_1|>' in text and '<|reserved_2|>' in text:
            parts = text.split('<|reserved_2|>', 1)
            user_part = parts[0].replace('<|reserved_1|>', '').strip()
            asst_part = parts[1].split('<|reserved_3|>', 1)[0].strip()
            if len(user_part) >= 2 and len(asst_part) >= 10:
                samples.append({'messages': [
                    {'role': 'user', 'content': user_part},
                    {'role': 'assistant', 'content': asst_part},
                ]})
        else:
            msgs = parse_belle_text(text)
            if msgs:
                samples.append({'messages': msgs})
    return samples


def parse_belle_multiturn(instruction: str, output: str):
    text = instruction.strip()
    if not text:
        return None
    parts = re.split(r'\n(?=Human:|Assistant:|A:)', text)
    messages = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith('Human:'):
            content = _HUMAN_RE.sub('', part).strip()
            if content:
                messages.append({'role': 'user', 'content': content})
        elif part.startswith('Assistant:') or part.startswith('A:'):
            content = _ASST_RE.sub('', part).strip()
            if content:
                messages.append({'role': 'assistant', 'content': content})
    output = output.strip()
    if not messages or messages[0]['role'] != 'user' or not output:
        return None
    messages.append({'role': 'assistant', 'content': output})
    return {'messages': messages}


def load_belle_instruction_output(path: str, mode: str):
    tbl = pq.read_table(path, columns=['instruction', 'output'])
    instructions = tbl['instruction'].to_pylist()
    outputs = tbl['output'].to_pylist()
    samples = []
    for instruction, output in zip(instructions, outputs):
        instruction = (instruction or '').strip()
        output = (output or '').strip()
        if len(instruction) < 2 or len(output) < 2:
            continue
        if mode == 'multiturn':
            sample = parse_belle_multiturn(instruction, output)
            if sample is not None:
                samples.append(sample)
        elif mode == 'school_math':
            samples.append({'messages': [
                {'role': 'user', 'content': instruction},
                {'role': 'assistant', 'content': output},
            ]})
        else:
            raise ValueError(f'unknown mode: {mode}')
    return samples

# ---------------------------------------------------------------------------
# 2. 主流程
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="合并各来源数据构建 SFT 训练集")
    parser.add_argument("--claude", type=str, default="~/Desktop/claude_sft_v4_clean.jsonl", help="Claude SFT 数据路径")
    parser.add_argument("--identity", type=str, default="~/Desktop/identity_samples.jsonl", help="身份样本路径")
    parser.add_argument("--negative-identity", type=str, default="~/Desktop/negative_identity_samples.jsonl", help="否定身份样本路径")
    parser.add_argument("--tool-call", type=str, default="docs/tool_call_samples_repo.jsonl", help="工具调用样本路径")
    parser.add_argument("--tool-call-upsample", type=int, default=10, help="工具调用样本过采样倍数")
    parser.add_argument("--skip-tool-call-validation", action="store_true", help="跳过工具调用样本验真")
    parser.add_argument("--rejection", type=str, default="~/Desktop/rejection_samples_clean.jsonl", help="拒绝样本路径")
    parser.add_argument("--out", type=str, default="~/Desktop/sft_toolcall_v1.jsonl", help="输出文件路径")
    parser.add_argument("--upsample", type=int, default=0, help="身份样本过采样倍数（0=不加载）")
    parser.add_argument("--negative-identity-upsample", type=int, default=0, help="否定身份样本过采样倍数（0=不加载）")
    parser.add_argument("--max-belle", type=int, default=50000, help="Belle 数据采样上限")
    parser.add_argument("--max-multiturn", type=int, default=10000, help="BELLE 多轮数据采样上限")
    parser.add_argument("--max-school-math", type=int, default=5000, help="BELLE 数学数据采样上限")
    parser.add_argument("--max-claude", type=int, default=0, help="Claude 数据采样上限（0=不加载）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)
    
    # 展开路径中的 ~
    claude_path = Path(os.path.expanduser(args.claude))
    identity_path = Path(os.path.expanduser(args.identity))
    negative_identity_path = Path(os.path.expanduser(args.negative_identity))
    tool_call_path = Path(os.path.expanduser(args.tool_call))
    rejection_path = Path(os.path.expanduser(args.rejection))
    out_path = Path(os.path.expanduser(args.out))
    data_dir = Path('/home/langshen/.cache/autoresearch-custom/data')

    # --- 1. 加载 Belle 数据 ---
    belle_samples = []
    print('正在加载 Belle 数据...')
    belle_files = [data_dir / 'belle_train.parquet'] + list(data_dir.glob('belle2m_*.parquet'))
    for path in belle_files:
        if path.exists() and '_val_tmp' not in path.name:
            before = len(belle_samples)
            belle_samples += load_belle_parquet(str(path))
            print(f'  {path.name}: +{len(belle_samples)-before} 条')

    random.shuffle(belle_samples)
    belle_samples = belle_samples[:args.max_belle]
    print(f'Belle 合计采样: {len(belle_samples)} 条')

    # --- 2. 加载 BELLE 多轮对话 ---
    multiturn_samples = []
    multiturn_path = data_dir / 'belle_multiturn_train.parquet'
    if multiturn_path.exists():
        print('\n正在加载 BELLE multiturn 数据...')
        multiturn_samples = load_belle_instruction_output(str(multiturn_path), 'multiturn')
        random.shuffle(multiturn_samples)
        multiturn_samples = multiturn_samples[:args.max_multiturn]
        print(f'  BELLE multiturn 合计采样: {len(multiturn_samples)} 条')
    else:
        print(f'\n跳过 BELLE multiturn：未找到 {multiturn_path}')

    # --- 3. 加载 BELLE school math ---
    school_math_samples = []
    school_math_path = data_dir / 'belle_school_math_train.parquet'
    if school_math_path.exists():
        print('\n正在加载 BELLE school math 数据...')
        school_math_samples = load_belle_instruction_output(str(school_math_path), 'school_math')
        random.shuffle(school_math_samples)
        school_math_samples = school_math_samples[:args.max_school_math]
        print(f'  BELLE school math 合计采样: {len(school_math_samples)} 条')
    else:
        print(f'\n跳过 BELLE school math：未找到 {school_math_path}')

    # --- 4. 加载 Claude 对话数据 ---
    claude_samples = []
    if claude_path.exists():
        print(f'\n正在从 {claude_path.name} 加载 Claude 数据...')
        claude_samples = [json.loads(l) for l in open(claude_path, encoding='utf-8')]
        random.shuffle(claude_samples)
        claude_samples = claude_samples[:args.max_claude]
        print(f'  Claude 合计采样: {len(claude_samples)} 条')
    else:
        print(f'\n跳过 Claude 数据：未找到 {claude_path}')

    # --- 5. 加载工具调用样本 ---
    tool_call_samples = []
    if tool_call_path.exists():
        print(f'\n正在从 {tool_call_path.name} 加载工具调用样本...')
        raw_tool = [json.loads(l) for l in open(tool_call_path, encoding='utf-8')]
        if not args.skip_tool_call_validation:
            valid_tool = []
            invalid_tool = 0
            for sample in raw_tool:
                ok, _reason = validate_tool_sample(sample, ROOT)
                if ok:
                    valid_tool.append(sample)
                else:
                    invalid_tool += 1
            raw_tool = valid_tool
            print(f'  验真后保留 {len(raw_tool)} 条，剔除 {invalid_tool} 条失真样本')
        tool_call_samples = raw_tool * args.tool_call_upsample
        print(f'  工具调用样本: {len(raw_tool)} 条 × {args.tool_call_upsample} = {len(tool_call_samples)} 条')
    else:
        print(f'\n跳过工具调用样本：未找到 {tool_call_path}')

    # --- 6. 加载拒绝样本 ---
    rejection_samples = []
    if rejection_path.exists():
        print(f'\n正在从 {rejection_path.name} 加载拒绝样本...')
        rejection_samples = [json.loads(l) for l in open(rejection_path, encoding='utf-8')]
        rejection_samples = [{'messages': s['messages']} for s in rejection_samples]
        print(f'  拒绝样本: {len(rejection_samples)} 条')

    # --- 7. 加载身份样本并过采样 ---
    identity_samples = []
    if identity_path.exists():
        print(f'\n正在从 {identity_path.name} 加载身份样本...')
        raw_identity = [json.loads(l) for l in open(identity_path, encoding='utf-8')]
        identity_samples = raw_identity * args.upsample
        print(f'  身份样本: {len(raw_identity)} 条 × {args.upsample} = {len(identity_samples)} 条')

    # --- 8. 加载否定身份样本并过采样 ---
    negative_identity_samples = []
    if negative_identity_path.exists():
        print(f'\n正在从 {negative_identity_path.name} 加载否定身份样本...')
        raw_negative_identity = [json.loads(l) for l in open(negative_identity_path, encoding='utf-8')]
        negative_identity_samples = raw_negative_identity * args.negative_identity_upsample
        print(f'  否定身份样本: {len(raw_negative_identity)} 条 × {args.negative_identity_upsample} = {len(negative_identity_samples)} 条')

    # --- 9. 合并并保存 ---
    all_samples = (
        belle_samples
        + multiturn_samples
        + school_math_samples
        + claude_samples
        + tool_call_samples
        + rejection_samples
        + identity_samples
        + negative_identity_samples
    )
    random.shuffle(all_samples)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for d in all_samples:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

    print(f'\n🎉 数据集构建完成！')
    print(f'输出文件: {out_path}')
    print(f'总样本数: {len(all_samples):,}')
    print(f'  - Belle: {len(belle_samples):,}')
    print(f'  - BELLE multiturn: {len(multiturn_samples):,}')
    print(f'  - BELLE school math: {len(school_math_samples):,}')
    print(f'  - Claude: {len(claude_samples):,}')
    print(f'  - Tool call: {len(tool_call_samples):,}')
    print(f'  - 拒绝回答: {len(rejection_samples):,}')
    print(f'  - 身份认知: {len(identity_samples):,}')
    print(f'  - 否定身份: {len(negative_identity_samples):,}')

if __name__ == '__main__':
    import os
    main()
