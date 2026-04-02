#!/usr/bin/env python3
"""
清洗 claude_sft_v3.jsonl 数据集。
过滤掉包含 API 错误、权限问题或无意义回复的样本。
"""
import json
import os
from pathlib import Path

IN_FILE = Path('/home/langshen/Desktop/claude_sft_v3.jsonl')
OUT_FILE = Path('/home/langshen/Desktop/claude_sft_v4_clean.jsonl')

BAD_KEYWORDS = [
    "API Error",
    "403 {",
    "Please run /login",
    "该令牌无权访问模型",
    "request id:",
    "new_api_error",
    "Rate limit reached",
    "Overloaded",
]

def is_bad(sample):
    # 检查 assistant 的回复中是否包含坏关键词
    for msg in sample.get('messages', []):
        if msg.get('role') == 'assistant':
            content = msg.get('content', '')
            for kw in BAD_KEYWORDS:
                if kw in content:
                    return True
    return False

def main():
    if not IN_FILE.exists():
        print(f"错误: 找不到输入文件 {IN_FILE}")
        return

    count_in = 0
    count_out = 0
    
    with open(IN_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUT_FILE, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            count_in += 1
            try:
                sample = json.loads(line)
                if not is_bad(sample):
                    f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')
                    count_out += 1
            except json.JSONDecodeError:
                continue

    print(f"清洗完成！")
    print(f"原始样本数: {count_in}")
    print(f"保留样本数: {count_out}")
    print(f"过滤掉的噪音: {count_in - count_out}")
    print(f"输出文件: {OUT_FILE}")

if __name__ == '__main__':
    main()
