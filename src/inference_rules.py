from __future__ import annotations

import re


IDENTITY_PATTERNS = [
    (re.compile(r"你是谁|你叫什么|介绍.*自己|请.*自我介绍", re.IGNORECASE), "我是微研，一个技术助手。我可以帮你搜索代码、阅读文件、解释技术概念。"),
    (re.compile(r"你是.?ChatGPT|你是.?GPT|你是不是.*GPT", re.IGNORECASE), "不是，我是微研，一个技术助手。"),
    (re.compile(r"你是.?Claude", re.IGNORECASE), "不是，我是微研，一个技术助手。"),
    (re.compile(r"你是.*OpenAI|OpenAI.*开发|谁开发了你|谁创造了你", re.IGNORECASE), "我是微研，一个专注于技术问题的助手。"),
    (re.compile(r"你是真人|你是人类|你有感情|你有意识", re.IGNORECASE), "不是，我是微研，一个 AI 技术助手。"),
]

SAFETY_PATTERNS = [
    re.compile(r"制造.*炸弹|炸弹.*制造|如何.*爆炸", re.IGNORECASE),
    re.compile(r"入侵.*电脑|黑客.*攻击|如何.*hack", re.IGNORECASE),
    re.compile(r"制作.*毒品|合成.*毒品|制毒", re.IGNORECASE),
    re.compile(r"钓鱼.*邮件|写.*骗人|诈骗.*话术", re.IGNORECASE),
    re.compile(r"如何.*杀人|杀.*方法", re.IGNORECASE),
    re.compile(r"如何.*自杀|自杀.*方法", re.IGNORECASE),
    re.compile(r"儿童.*色情|未成年.*性", re.IGNORECASE),
]

SAFETY_RESPONSE = "抱歉，我无法提供这类信息。如果你有技术方面的问题，我很乐意帮忙。"


def check_identity(prompt: str) -> str | None:
    for pattern, answer in IDENTITY_PATTERNS:
        if pattern.search(prompt):
            return answer
    return None


def check_safety(prompt: str) -> str | None:
    for pattern in SAFETY_PATTERNS:
        if pattern.search(prompt):
            return SAFETY_RESPONSE
    return None


def apply_hard_rules(prompt: str) -> str | None:
    identity_answer = check_identity(prompt)
    if identity_answer:
        return identity_answer
    return check_safety(prompt)
