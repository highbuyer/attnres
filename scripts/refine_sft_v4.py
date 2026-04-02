#!/usr/bin/env python3
import json
import re
from pathlib import Path

IN_FILE = Path('/home/langshen/Desktop/sft_mixed_v4_100x.jsonl')
OUT_FILE = Path('/home/langshen/Desktop/sft_mixed_v4_refined.jsonl')

# 定义需要转化的标签映射
TAG_MAP = {
    r'<tool_call>': '[调用工具]: ',
    r'</tool_call>': '\n',
    r'<tool_result>': '[工具返回结果]: ',
    r'</tool_result>': '\n',
    r'<task-notification>': '[任务通知]: ',
    r'</task-notification>': '\n',
    r'<persisted-output>': '[持续输出]: ',
    r'</persisted-output>': '\n',
    r'<retrieval_status>': '[检索状态]: ',
    r'</retrieval_status>': '\n',
}

def clean_tags(text):
    for tag, replacement in TAG_MAP.items():
        text = text.replace(tag, replacement)
    # 去除多余的空行
    return re.sub(r'\n\s*\n', '\n\n', text).strip()

def process():
    count_total = 0
    count_kept_normal = 0
    count_kept_tool = 0
    count_dropped_tool = 0

    with open(IN_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUT_FILE, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            count_total += 1
            sample = json.loads(line)
            messages = sample.get('messages', [])
            
            # 判断是否包含工具调用相关内容
            full_text = "".join([m['content'] for m in messages])
            is_tool_sample = any(tag in full_text for tag in ['<tool_', '<task-', '<persisted-'])

            if not is_tool_sample:
                # 正常样本直接保留
                f_out.write(line)
                count_kept_normal += 1
                continue

            # 针对工具样本进行精细化处理
            # 找到最后一条 assistant 的消息（通常是总结）
            asst_msgs = [m['content'] for m in messages if m['role'] == 'assistant']
            if not asst_msgs:
                count_dropped_tool += 1
                continue
            
            last_asst_content = asst_msgs[-1]
            # 去掉所有标签后的纯文字长度
            pure_text = re.sub(r'<[^>]+>', '', last_asst_content).strip()

            if len(pure_text) > 100:
                # 保留有实质内容的工具样本，并转化标签
                new_messages = []
                for m in messages:
                    new_messages.append({
                        'role': m['role'],
                        'content': clean_tags(m['content'])
                    })
                sample['messages'] = new_messages
                f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')
                count_kept_tool += 1
            else:
                count_dropped_tool += 1

    print(f"数据精炼完成！")
    print(f"总处理样本数: {count_total:,}")
    print(f"保留常规样本: {count_kept_normal:,}")
    print(f"保留高质量推理样本: {count_kept_tool:,}")
    print(f"丢弃低质量通知样本: {count_dropped_tool:,}")
    print(f"最终数据集大小: {count_kept_normal + count_kept_tool:,}")
    print(f"输出文件: {OUT_FILE}")

if __name__ == '__main__':
    process()
