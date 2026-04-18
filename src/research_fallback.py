"""weiyan-api / self_audit 共用的研究回退模块：本机 docs + 维基百科查询。

抽出自 weiyan-api 的 inline 实现（2026-04-18，w6），同时：
1. 收紧 `is_related` —— 原规则要求 query 与 text 共享 1 个 2-gram 即算相关，
   w5 self_audit e2e 发现会把"世界上最高的山峰"误匹到"世界上最糟糕的人"、
   "列表排序"误匹到"列表（表格）"。新规则要求 query 的"核心串"（剥
   stopword 后 ≥3 字符的中文连续片段或 ≥3 字符英文 token）至少一个出现
   在 title 或 text 开头的 300 字内；无核心串时回退 loose 判定以免过度
   过滤短 query。
2. 不再依赖全局 `_URL_OPENER`：`wiki_lookup` / `research_fallback` 接受
   可选的 `opener` 参数，上层可以注入自己的 proxy-aware opener（避免
   `import infer` 清环境变量导致的静默代理失效）。
3. `research_fallback` 接受 `debug` callback，调用方自己 format req_id。
"""
from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable


HTTP_TIMEOUT = 5  # 秒，wiki 单次请求上限
WIKI_UA = "Mozilla/5.0 (compatible; weiyan/1.0; +https://zh.wikipedia.org/)"

# "有" 相对原版新增（w6）——wiki 误匹的根因之一是"北京有什么景点"剥 stopword 后
# 残留 "北京有" 三字被当作核心串并命中 "北京有轨电车"。加进 stopword 让 core 退化
# 到 "著名景点" 这种语义实在的片段。同理"哪些"。
RESEARCH_STOPWORDS = set("的是什么吗呢何为怎么怎样如何请问给我查一下能够可以会不有哪些")


# "我不确定 / 建议查阅资料 / 了解有限" 等 SFT 训出来的拒答模板。
# 核心思路：只覆盖"我不会答 + 让用户自己查"这类语义，不能吃掉安全拒绝
# （"抱歉，我无法提供这类信息"）或普通工具回答（"查询到 3 条结果…"）。
#
# 2026-04-19（w7a）扩展：
# - 起始语并列"抱歉|对不起|不好意思"；旧版漏掉"对不起…无法确认…"。
# - 新增"无法 X 信息/答案/准确性"(X ∈ 确认/核实/保证)；与"无法提供这类
#   信息"互斥，避开安全拒绝误伤。
# - 新增"不确定…信息"(旧版只有"答案/回答")。
# - 保留旧"建议…查阅…资料"；加一条"建议你/您/大家/我们 … 核实/查证/查询/
#   查阅/咨询"——必须带人称代词，避免"根据 PEP-8，建议咨询团队约定"这类
#   陈述句误触发。
# - 新增"超出/不在 … 知识/能力 … 范围"直接命中，不再只能靠后半句"建议"。
_UNKNOWN_ANSWER_RE = re.compile(
    r"((抱歉|对不起|不好意思)[,，].{0,40}(了解有限|无法.{0,10}回答|不会这个|不懂|无法.{0,8}(确认|核实|保证).{0,20}(信息|答案|真伪|准确|正确))"
    r"|(不确定|不太确定).{0,30}(答案|回答|信息)"
    r"|无法.{0,8}给出.{0,8}(准确|正确).{0,8}回答"
    r"|建议.{0,8}查阅.{0,10}(资料|信息|来源)"
    r"|(建议|请).{0,4}(你|您|大家|我们).{0,10}(核实|查证|查询|查阅|咨询)"
    r"|(超出|不在).{0,6}(我?的?)?(知识|能力).{0,3}范围"
    r"|我.{0,4}不知道.{0,10}(答案|日期|时间)?)",
    re.IGNORECASE,
)

PROJECT_SCOPE_RE = re.compile(
    r"(项目|仓库|代码|文件|函数|类|脚本|模块|目录|路径|配置|checkpoint|weiyan|微研|attnres|sft|infer|train|eval|tokenizer|分词|架构|流程|内容|里有什么|写了什么|里面是什么|"
    r"\.(?:py|md|json|jsonl|toml|yaml|yml|txt|sh|cfg))",
    re.IGNORECASE,
)

FILE_REF_RE = re.compile(
    r"([A-Za-z0-9_./-]+\.(?:py|md|json|jsonl|toml|yaml|yml|txt|sh|cfg))",
    re.IGNORECASE,
)


def is_unknown_answer(text: str) -> bool:
    """模型输出是否命中"不会答"模板。命中 → 调 research_fallback 接管。"""
    if not text or len(text) > 300:
        return False
    return bool(_UNKNOWN_ANSWER_RE.search(text))


def build_url_opener(proxies: dict | None = None) -> urllib.request.OpenerDirector:
    """构建 proxy-aware opener。proxies 为 None 时从 os.environ 取，否则用传入。

    为什么接 proxies 参数：train.py 的 fa3 backend 加载会清 HTTP_PROXY /
    HTTPS_PROXY 从 os.environ。调用方（self_audit）必须在 `import infer`
    之前把 environ 快照传进来。weiyan-api 启动时 environ 未被污染，直接
    build_url_opener() 也能拿到系统代理。
    """
    if proxies is None:
        proxies = {}
        for name, value in os.environ.items():
            lname = name.lower()
            if value and lname.endswith("_proxy"):
                proxies[lname[:-6]] = value
    handlers = []
    if proxies:
        handlers.append(urllib.request.ProxyHandler(proxies))
    return urllib.request.build_opener(*handlers)


def candidate_titles(query: str) -> list[str]:
    """给出一组维基检索候选，按命中优先级排序。"""
    c: list[str] = []
    cleaned = "".join(ch for ch in query if ch not in RESEARCH_STOPWORDS)
    cleaned = re.sub(r"[?？!！。.,，;；:：\s]+", " ", cleaned).strip()
    if cleaned and cleaned not in c:
        c.append(cleaned)
    for m in re.findall(r"[\u4e00-\u9fff]{2,8}|[A-Za-z][A-Za-z0-9]{2,}", query):
        if m not in c:
            c.append(m)
    for block in re.findall(r"[\u4e00-\u9fff]{5,}", query):
        for n in (4, 3, 2):
            for i in range(len(block) - n + 1):
                gram = block[i:i + n]
                if gram in RESEARCH_STOPWORDS or any(ch in RESEARCH_STOPWORDS for ch in gram):
                    continue
                if gram not in c:
                    c.append(gram)
    return c


def _core_tokens(query: str) -> list[str]:
    """抽 query 的"核心串"——剥 stopword 后 ≥3 字符的中文连续片段或 ≥3 字符英文 token。

    用于 is_related 的严格判定。若 query 太短（如"鲁迅"），返回空列表，调用方
    应退回 loose 判定以免过度过滤。
    """
    # 把 stopword 替换为空格，让连续的非 stopword 片段成段
    spaced = "".join(ch if ch not in RESEARCH_STOPWORDS else " " for ch in query)
    spaced = re.sub(r"[?？!！。.,，;；:：]+", " ", spaced)
    tokens: list[str] = []
    for block in re.findall(r"[\u4e00-\u9fff]{3,}|[A-Za-z][A-Za-z0-9]{2,}", spaced):
        tokens.append(block)
    return tokens


def _is_related_loose(query: str, text: str) -> bool:
    """旧版 is_related：query 与 text 共享 ≥1 个 2-gram（中文）或单字（中文长度=1）。"""
    q_norm = "".join(ch for ch in query if ch not in RESEARCH_STOPWORDS)
    q_norm = re.sub(r"[\s?？!！。.,，;；:：]+", "", q_norm)
    text_norm = re.sub(r"\s+", "", text[:500])
    if not q_norm or not text_norm:
        return False
    if len(q_norm) == 1:
        return q_norm in text_norm[:100]
    grams = {q_norm[i:i + 2] for i in range(len(q_norm) - 1)}
    return any(g in text_norm for g in grams if g)


def is_related(query: str, title: str, text: str) -> bool:
    """判定 wiki 命中是否与 query 相关。收紧版（w6）：要求 query 的核心串
    至少一个完整出现在 title 或 text 开头的 300 字内；无核心串则回退 loose。

    修正的误匹（w5 发现）：
    - "世界上最高的山峰" → "世界上最糟糕的人" ❌ loose 通过（共享"世界"）/ ✓ strict 过滤
    - "列表 [..] 排序后" → "列表（表格）" ❌ loose 通过（共享"列表"）/ ✓ strict 过滤
    - "北京有什么著名景点" → "北京有轨电车" ❌ loose 通过（共享"北京"）/ ✓ strict 过滤
    保留的擦边（生产可接受）：
    - "水的化学式" → "化学式"（通用定义，非 H₂O，但确实相关）
    - "Python 和 Java 哪个更适合" → "Python"（只覆盖半边，不算错）
    """
    if not query or not title:
        return False
    cores = _core_tokens(query)
    if not cores:
        # query 太短，退 loose 避免过度过滤
        return _is_related_loose(query, title + " " + text)
    haystack = title + " " + (text[:300] if text else "")
    return any(core in haystack for core in cores)


def wiki_lookup(
    query: str,
    opener: urllib.request.OpenerDirector | None = None,
    timeout: int = HTTP_TIMEOUT,
) -> tuple[str | None, str]:
    """查维基百科中文。返回 (摘要, 诊断原因)；找不到或无关时摘要为 None。"""
    if opener is None:
        opener = build_url_opener()
    cands = candidate_titles(query)
    if not cands:
        return None, "no_keywords"
    last = "unknown"
    for cand in cands[:4]:
        search_url = (
            "https://zh.wikipedia.org/w/api.php?action=opensearch&limit=1&format=json&search="
            + urllib.parse.quote(cand)
        )
        try:
            req = urllib.request.Request(search_url, headers={"User-Agent": WIKI_UA})
            with opener.open(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last = f"search_http_{exc.code}({cand!r})"
            continue
        except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
            return None, f"net_unreachable({exc.__class__.__name__})"
        except Exception as exc:
            last = f"search_err({exc.__class__.__name__})"
            continue
        titles = data[1] if isinstance(data, list) and len(data) > 1 else []
        if not titles:
            last = f"no_match({cand!r})"
            continue
        real = titles[0]
        summary_url = "https://zh.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(real)
        try:
            req = urllib.request.Request(summary_url, headers={"User-Agent": WIKI_UA})
            with opener.open(req, timeout=timeout) as resp:
                summary = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last = f"summary_http_{exc.code}({real!r})"
            continue
        except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
            return None, f"net_unreachable({exc.__class__.__name__})"
        except Exception as exc:
            last = f"summary_err({exc.__class__.__name__})"
            continue
        extract = (summary.get("extract") or "").strip()
        if not extract:
            last = f"no_extract({real!r})"
            continue
        if not is_related(query, real, extract):
            last = f"irrelevant({cand!r}->{real!r})"
            continue
        return f"维基百科·{real}：{extract}", "ok"
    return None, last


def search_local_docs(query: str, tool_dir: str | None, timeout: int = 3) -> str | None:
    """在项目 docs/ 和 README 里 grep 关键词，返回前 3 条命中摘要。仅 project scope 题调用。"""
    if not tool_dir:
        return None
    terms = [t for t in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9_]{2,}", query) if len(t) >= 2]
    if not terms:
        return None
    try:
        result = subprocess.run(
            [
                "rg",
                "--no-heading",
                "-n",
                "--max-count", "3",
                "-i",
                "--fixed-strings",
                "--glob", "docs/**",
                "--glob", "README*",
                "--glob", "*.md",
                "-e", terms[0],
            ]
            + [arg for term in terms[1:4] for arg in ("-e", term)],
            cwd=tool_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]

    def _is_prose(line: str) -> bool:
        path_part, _, rest = line.partition(":")
        _, _, content = rest.partition(":")
        content = content.strip()
        if not content:
            return False
        if path_part.endswith(".jsonl") or path_part.endswith(".json"):
            return False
        bad_prefixes = ("uv run", "python ", "python3 ", "bash ", "$ ", "# ", "```", ">>>", "import ", "from ", "{\"", "{'")
        if any(content.startswith(p) for p in bad_prefixes):
            return False
        if "<|tool_" in content or '"messages":' in content or '"role":' in content:
            return False
        chinese = sum(1 for ch in content if "\u4e00" <= ch <= "\u9fff")
        if chinese < 4 and sum(1 for ch in content if ch.isalpha()) < 15:
            return False
        return True

    lines = [line for line in lines if _is_prose(line)]
    if not lines:
        return None
    return "\n".join(lines[:3])


def research_fallback(
    prompt_text: str,
    tool_dir: str | None,
    opener: urllib.request.OpenerDirector | None = None,
    debug: Callable[[str], None] | None = None,
) -> str:
    """模型拒答时：（项目类问题）本机 docs → 维基百科 → 诚实说明原因。

    - project 问题（PROJECT_SCOPE_RE 命中）：先 grep 本机 docs
    - 任意问题：wiki_lookup
    - 都失败：返回诚实的"没把握 + 已尝试"模板，便于用户给更多上下文
    """
    def _dbg(msg: str) -> None:
        if debug is not None:
            debug(msg)

    project_scoped = bool(PROJECT_SCOPE_RE.search(prompt_text))
    local_reason = "通用问题，跳过本机 docs"
    if project_scoped:
        local = search_local_docs(prompt_text, tool_dir)
        if local:
            _dbg(f"research fallback hit local {len(local)} chars")
            return "模型没直接掌握，但在项目文档里找到相关条目：\n" + local
        local_reason = "本机文档未命中"
    wiki, reason = wiki_lookup(prompt_text, opener=opener)
    if wiki:
        _dbg(f"research fallback hit wiki len={len(wiki)}")
        return "模型没直接掌握，以下摘自维基百科（请自行验证）：\n" + wiki
    _dbg(f"research fallback gave up local={local_reason} wiki={reason}")
    return (
        "我对这个问题没有把握。尝试过的查证：\n"
        f"- 本机项目文档：{local_reason}\n"
        f"- 维基百科中文：{reason}\n"
        "如果能提供更精确的关键词或上下文，我可以再试一次。"
    )
