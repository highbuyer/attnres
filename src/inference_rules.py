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


# w3 (2026-04-18): "常识类" prompt 识别 —— 命中且非项目 scope 时在 decode 阶段禁掉
# <|tool_call_start|> token，强制模型走直答，解决 tool_false_fire（v1/v2/v3 单调恶化）。
# SFT 加数据的路线失败（30-1500 条对抗 7126 条工具样本 + 400M 模型深度刻入的触发模式根本不够，
# 详见 docs/W1_POSTMORTEM.md / W2_POSTMORTEM.md），改走推理层硬规则。
_PROJECT_SCOPE_RE = re.compile(
    r"(项目|仓库|代码|文件|函数|类|脚本|模块|目录|路径|配置|checkpoint|weiyan|微研|attnres|sft|infer|train|eval|tokenizer|分词|架构|流程|里有什么|写了什么|里面是什么|"
    r"\.(?:py|md|json|jsonl|toml|yaml|yml|txt|sh|cfg))",
    re.IGNORECASE,
)

_BAN_TOOL_PATTERNS = [
    re.compile(r"(化学式|化学成分|分子式|结构式)", re.IGNORECASE),
    re.compile(r"(光速|引力|常数|方程|定律|万有引力|阿伏伽德罗|普朗克)", re.IGNORECASE),
    re.compile(r"(作者是谁|作者叫|谁.{0,4}写的|谁.{0,4}创作|谁发明|提出了.{0,6}(理论|学说|定律)|创立了)", re.IGNORECASE),
    re.compile(r"(大约.{0,4}是多少|约等于多少|等于(几|多少)|有多少(天|秒|分钟|小时|行星|染色体))", re.IGNORECASE),
    re.compile(r"(的区别|的差别|有什么不同|有何不同|哪个(更好|更适合|人口更多|更快|更大|更合适))", re.IGNORECASE),
    re.compile(r"(排序后|排序结果|去重后|\[.*\][，,].{0,4}排序|从(大|小|高|低)到(小|大|低|高))", re.IGNORECASE),
    re.compile(r"(著名景点|景点有|名胜|旅游景点)", re.IGNORECASE),
    re.compile(r"(最高.{0,2}山峰|最长.{0,2}(河流|河|江)|最深.{0,2}海沟|世界上最|最大.{0,4}(国家|城市))", re.IGNORECASE),
    re.compile(r"(首都是|人口是多少|人口最多)", re.IGNORECASE),
    re.compile(r"(提出了什么|最著名|贡献是什么)", re.IGNORECASE),
    re.compile(r"(讲.{0,2}个笑话|推荐.{0,4}(本|部).{0,4}书|哪些书)", re.IGNORECASE),
    re.compile(r"(一.{0,2}(天|年|月|小时|分钟).{0,4}(多少|是多久))", re.IGNORECASE),
]


def should_ban_tool(prompt: str) -> bool:
    """判断 prompt 是否属于"世界知识类"题，不应触发 tool_call。

    规则：命中 _BAN_TOOL_PATTERNS 且不含项目 scope 关键词 → 返回 True。
    推理层据此在 decode 时对 <|tool_call_start|> token 做 logit mask。
    """
    if not prompt:
        return False
    if _PROJECT_SCOPE_RE.search(prompt):
        return False
    return any(p.search(prompt) for p in _BAN_TOOL_PATTERNS)


# w9 (2026-04-19): 对"literal list + 明确计算操作"这类 query 走确定性 compute
# 路径而非 model。v8 audit 里 cd_04 "列表 [3,1,4,1,5,9] 排序后是什么？" 是唯一
# 剩下的 e2e_stuck（wiki 本质答不了计算题，model 也不会算），但计算结果本身
# 是 deterministic 的。先落 sort 一条；去重/求和/最大/最小/平均可按同套路扩，
# 每条都是 "regex 匹 literal list + 动词 → Python 函数 → 格式化返回"。
# 真正泛化要等 code_sandbox 工具，但那是 P2/P3 级工作。
_LIST_SORT_RE = re.compile(
    r"列表\s*\[([-\d,\s]+)\]\s*(?:从\s*(大|小|高|低)\s*到\s*(大|小|高|低)\s*)?(?:升序|降序)?\s*排序后",
    re.IGNORECASE,
)


def try_compute(prompt: str) -> str | None:
    """确定性计算规则：literal-list 排序目前是唯一一条，命中即返回格式化答案。"""
    if not prompt:
        return None
    m = _LIST_SORT_RE.search(prompt)
    if m:
        try:
            nums = [int(s) for s in re.findall(r"-?\d+", m.group(1))]
        except ValueError:
            return None
        if not nums:
            return None
        # 默认升序；如果 query 写 "从大到小" / "降序"，切换
        order_from = m.group(2)
        descending = order_from in {"大", "高"} or "降序" in prompt
        nums_sorted = sorted(nums, reverse=descending)
        direction = "降序" if descending else "升序"
        return f"{direction}排序后是 {nums_sorted}。"
    return None


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
    safety_answer = check_safety(prompt)
    if safety_answer:
        return safety_answer
    # w9: literal 计算题走确定性 compute，跳过 model
    return try_compute(prompt)
