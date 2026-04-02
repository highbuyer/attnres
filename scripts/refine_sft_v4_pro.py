#!/usr/bin/env python3
import json
import re
from pathlib import Path

IN_FILE = Path('/home/langshen/Desktop/sft_mixed_v4_100x.jsonl')
OUT_FILE = Path('/home/langshen/Desktop/sft_mixed_v4_refined_pro.jsonl')

# 需要彻底剔除的无意义标签块
STRIP_PATTERNS = [
    r'<system-reminder>.*?</system-reminder>',
    r'<task-notification>.*?</task-notification>',
    r'<retrieval_status>.*?</retrieval_status>',
]

# 需要转化为自然语言的标签
CONVERT_MAP = {
    r'<tool_call>': '[调用工具]: ',
    r'</tool_call>': '\n',
    r'<tool_result>': '[工具返回结果]: ',
    r'</tool_result>': '\n',
    r'<persisted-output>': '[持续输出]: ',
    r'</persisted-output>': '\n',
}

def clean_and_convert(text):
    # 1. 彻底剔除不需要的系统噪音块
    for pat in STRIP_PATTERNS:
        text = re.sub(pat, '', text, flags=re.DOTALL)
    
    # 2. 转化核心标签为自然语言
    for tag, replacement in CONVERT_MAP.items():
        text = text.replace(tag, replacement)
    
    # 3. 清理多余空行
    text = re.sub(r'\n\s*\n', '\n\n', text).strip()
    return text

def is_meaningful_content(text):
    # 去掉所有标签后的纯文字长度
    pure_text = re.sub(r'<[^>]+>', '', text).strip()
    # 也要去掉我们转换后的标签前缀来判断
    for val in CONVERT_MAP.values():
        pure_text = pure_text.replace(val.strip(), '')
    return len(pure_text.strip())

def process():
    count_total = 0
    count_kept_normal = 0
    count_kept_tool = 0
    count_dropped_noise = 0

    with open(IN_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUT_FILE, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            count_total += 1
            sample = json.loads(line)
            messages = sample.get('messages', [])
            
            # 判断是否是工具/系统相关的样本
            full_content = "".join([m['content'] for m in messages])
            is_complex_sample = any(tag in full_content for tag in ['<tool_', '<task-', '<persisted-', '<system-reminder>'])

            if not is_complex_sample:
                f_out.write(line)
                count_kept_normal += 1
                continue

            # 针对复杂样本的 Pro 过滤逻辑
            has_deep_thought = any(is_meaningful_content(m['content']) > 100 for m in messages if m['role'] == 'assistant')
            
            # 检查 User 是否有实质输入（防止保留只有 [任务通知] 的对话）
            user_has_content = any(is_meaningful_content(m['content']) > 5 for m in messages if m['role'] == 'user')

            if has_deep_thought and user_has_content:
                # 执行清洗和转化
                new_messages = []
                for m in messages:
                    cleaned_content = clean_and_convert(m['content'])
                    if cleaned_content: # 过滤掉清洗后变为空的消息
                        new_messages.append({
                            'role': m['role'],
                            'content': cleaned_content
                        })
                
                # 再次检查清洗后是否还符合对话结构（至少一问一答）
                roles = [m['role'] for m in new_messages]
                if 'user' in roles and 'assistant' in roles:
                    sample['messages'] = new_messages
                    f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')
                    count_kept_tool += 1
                else:
                    count_dropped_noise += 1
            else:
                count_dropped_noise += 1

    print(f"数据精炼 Pro 版完成！")
    print(f"总处理样本数: {count_total:,}")
    print(f"保留常规样本: {count_kept_normal:,}")
    print(f"精选深度推理样本: {count_kept_tool:,}")
    print(f"剔除纯系统噪音: {count_dropped_noise:,}")
    print(f"最终数据集大小: {count_kept_normal + count_kept_tool:,}")
    print(f"输出文件: {OUT_FILE}")

if __name__ == '__main__':
    process()
