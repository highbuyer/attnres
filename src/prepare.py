"""
One-time data preparation for autoresearch experiments.
Downloads data shards and trains a BPE tokenizer.

Usage:
    python prepare.py                  # full prep (download + tokenizer)
    python prepare.py --num-shards 8   # download only 8 shards (for testing)

Data and tokenizer are stored in ~/.cache/autoresearch/.
"""

import argparse
import gc
import math
import os
import pickle
import sys
import time
from multiprocessing import Pool

import requests
import pyarrow as pa
import pyarrow.parquet as pq
import rustbpe
import tiktoken
import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 2048       # context length
TIME_BUDGET = 300        # training time budget in seconds (5 minutes)
EVAL_TOKENS = int(os.environ.get('EVAL_TOKENS_MULT', 40)) * 524288  # number of tokens for val eval

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-custom")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")
VAL_FILENAME = "val.parquet"  # pinned validation shard
VOCAB_SIZE = 32768

try:
    import ctypes
except Exception:
    ctypes = None

_LIBC = None
if os.name == "posix" and ctypes is not None:
    try:
        _LIBC = ctypes.CDLL("libc.so.6")
    except OSError:
        _LIBC = None

# HuggingFace parquet URLs
# Belle: single parquet for the 0.5M CN instruction set
BELLE_TRAIN_URL = "https://huggingface.co/datasets/BelleGroup/train_0.5M_CN/resolve/main/Belle_open_source_0.5M.json"
BELLE_PARQUET_URL = "https://huggingface.co/datasets/BelleGroup/train_0.5M_CN/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
# GitHub Python (codeparrot/github-code-clean, Python-all config, first N shards)
PYTHON_BASE_URL = "https://huggingface.co/datasets/codeparrot/github-code-clean/resolve/refs%2Fconvert%2Fparquet/Python-all/partial-train"
PYTHON_MAX_SHARD = 9  # 0000..0009 (10 shards total)
# GitHub Java (codeparrot/github-code-clean, Java-all config, 10 shards)
JAVA_BASE_URL = "https://huggingface.co/datasets/codeparrot/github-code-clean/resolve/refs%2Fconvert%2Fparquet/Java-all/partial-train"
JAVA_MAX_SHARD = 9  # 0000..0009 (10 shards total)
# GitHub JavaScript (codeparrot/github-code-clean, JavaScript-all config, 10 shards)
JAVASCRIPT_BASE_URL = "https://huggingface.co/datasets/codeparrot/github-code-clean/resolve/refs%2Fconvert%2Fparquet/JavaScript-all/partial-train"
JAVASCRIPT_MAX_SHARD = 9  # 0000..0009 (10 shards total)
# Glaive function calling v2 (tool-use conversations)
GLAIVE_PARQUET_URL = "https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
# Glaive function calling v1 (additional tool-use)
GLAIVE_V1_PARQUET_URL = "https://huggingface.co/datasets/glaiveai/glaive-function-calling/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
# Belle 2M CN (large Chinese instruction set, 3 shards)
BELLE_2M_BASE_URL = "https://huggingface.co/datasets/BelleGroup/train_2M_CN/resolve/refs%2Fconvert%2Fparquet/default/train"
BELLE_2M_NUM_SHARDS = 3
# BELLE multiturn chat (Chinese multi-turn dialogue)
BELLE_MULTITURN_URL = "https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
# BELLE school math (Chinese math reasoning)
BELLE_SCHOOL_MATH_URL = "https://huggingface.co/datasets/BelleGroup/school_math_0.25M/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"

# Chinese Wikipedia (wikimedia/wikipedia, 20231101.zh, 6 shards)
WIKI_ZH_BASE_URL = "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/refs%2Fconvert%2Fparquet/20231101.zh/train"
WIKI_ZH_NUM_SHARDS = 6
# StarCoder Python (bigcode/starcoderdata, python subset, first N shards)
STARCODER_BASE_URL = "https://huggingface.co/datasets/bigcode/starcoderdata/resolve/main/python"
STARCODER_NUM_SHARDS = 59  # number of shards to download (all)
STARCODER_TOTAL_SHARDS = 59  # total shards in the dataset (for URL construction)

# BPE split pattern (GPT-4 style, with \p{N}{1,2} instead of {1,3})
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

SPECIAL_TOKENS = [
    *[f"<|reserved_{i}|>" for i in range(4)],
    "<|tool_call_start|>",
    "<|tool_call_end|>",
    "<|tool_result_start|>",
    "<|tool_result_end|>",
    "<|tool_name_search_code|>",
    "<|tool_name_read_file|>",
]
BOS_TOKEN = "<|reserved_0|>"
USER_TOKEN = "<|reserved_1|>"
ASST_TOKEN = "<|reserved_2|>"
EOS_TOKEN = "<|reserved_3|>"

# ---------------------------------------------------------------------------
# Glaive format conversion
# ---------------------------------------------------------------------------
import re as _re

_GLAIVE_ROLE_RE = _re.compile(
    r'^(SYSTEM|USER|ASSISTANT|A|FUNCTION RESPONSE):\s*',
    _re.MULTILINE,
)


def _convert_glaive_chat(text):
    """将 Glaive 格式转成特殊 token 格式。
    跳过 SYSTEM 和 FUNCTION RESPONSE 段，只保留 USER/ASSISTANT 轮次。
    返回转换后的字符串，失败返回 None。
    """
    # 按角色标记分割
    parts = _GLAIVE_ROLE_RE.split(text)
    # parts: ['', 'USER', ' content...', 'A', ' content...', ...]
    turns = []
    i = 1  # 跳过第一个空串
    while i < len(parts) - 1:
        role = parts[i].strip()
        content = parts[i + 1].strip()
        # 清理 <|endoftext|> 和 <functioncall>
        content = content.replace('<|endoftext|>', '').strip()
        if role in ('USER',):
            turns.append(('user', content))
        elif role in ('A', 'ASSISTANT'):
            # 跳过纯 functioncall 的 assistant 回复
            if content.startswith('<functioncall>') or content.startswith('{'):
                pass
            elif content:
                turns.append(('assistant', content))
        # SYSTEM 和 FUNCTION RESPONSE 跳过
        i += 2

    if not turns or turns[0][0] != 'user':
        return None
    # 至少需要一轮 user + assistant
    if not any(r == 'assistant' for r, _ in turns):
        return None

    result = []
    for role, content in turns:
        if role == 'user':
            result.append(f'{USER_TOKEN}{content}')
        else:
            result.append(f'{ASST_TOKEN}{content}')
    result.append(EOS_TOKEN)
    return ''.join(result)


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def _fetch_url(url, filepath):
    """Download url to filepath with retries. Returns True on success."""
    if os.path.exists(filepath):
        return True
    hf_token = os.environ.get('HF_TOKEN', '')
    headers = {'Authorization': f'Bearer {hf_token}'} if hf_token else {}
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=60, headers=headers)
            response.raise_for_status()
            temp_path = filepath + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            print(f"  Downloaded {os.path.basename(filepath)}")
            return True
        except (requests.RequestException, IOError) as e:
            print(f"  Attempt {attempt}/{max_attempts} failed for {os.path.basename(filepath)}: {e}")
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
    return False


def _split_and_save_code_shard(raw_path, train_path, val_path, val_frac=0.1):
    """Read raw code parquet, split into train/val, save both. Removes raw file."""
    table = pq.read_table(raw_path, columns=["code"]).rename_columns(["text"])
    texts = table.column("text").to_pylist()
    val_n = max(1, int(len(texts) * val_frac))
    train_texts = texts[:len(texts) - val_n]
    val_texts = texts[len(texts) - val_n:]
    pq.write_table(pa.table({"text": pa.array(train_texts, type=pa.string())}), train_path)
    pq.write_table(pa.table({"text": pa.array(val_texts, type=pa.string())}), val_path)
    os.remove(raw_path)
    return len(train_texts), val_n


def _download_python_shard(index):
    """Download one GitHub Python parquet shard, split into train/val."""
    filename = f"{index:04d}.parquet"
    train_path = os.path.join(DATA_DIR, f"python_{filename}")
    val_path = os.path.join(DATA_DIR, f"python_{index:04d}_val_tmp.parquet")
    if os.path.exists(train_path):
        return True
    url = f"{PYTHON_BASE_URL}/{filename}"
    ok = _fetch_url(url, train_path + ".raw")
    if not ok:
        return False
    n_train, n_val = _split_and_save_code_shard(train_path + ".raw", train_path, val_path)
    return True


def _download_lang_shard(args):
    """Download one GitHub code shard for a given language, split into train/val."""
    lang, base_url, index = args
    filename = f"{index:04d}.parquet"
    train_path = os.path.join(DATA_DIR, f"{lang}_{filename}")
    val_path = os.path.join(DATA_DIR, f"{lang}_{index:04d}_val_tmp.parquet")
    if os.path.exists(train_path):
        return True
    url = f"{base_url}/{filename}"
    ok = _fetch_url(url, train_path + ".raw")
    if not ok:
        return False
    _split_and_save_code_shard(train_path + ".raw", train_path, val_path)
    return True


def _download_wiki_zh_shard(index):
    """Download one Chinese Wikipedia shard (text column), split into train/val."""
    train_path = os.path.join(DATA_DIR, f"wiki_zh_{index:04d}.parquet")
    val_path = os.path.join(DATA_DIR, f"wiki_zh_{index:04d}_val_tmp.parquet")
    if os.path.exists(train_path):
        return True
    url = f"{WIKI_ZH_BASE_URL}/{index:04d}.parquet"
    raw_path = train_path + ".raw"
    ok = _fetch_url(url, raw_path)
    if not ok:
        return False
    table = pq.read_table(raw_path, columns=["text"])
    texts = table.column("text").to_pylist()
    val_n = max(1, len(texts) // 10)
    pq.write_table(pa.table({"text": pa.array(texts[:len(texts)-val_n], type=pa.string())}), train_path)
    pq.write_table(pa.table({"text": pa.array(texts[len(texts)-val_n:], type=pa.string())}), val_path)
    os.remove(raw_path)
    return True


def _download_starcoder_shard(index):
    """Download one StarCoder Python shard (content column), split into train/val."""
    train_path = os.path.join(DATA_DIR, f"starcoder_{index:05d}.parquet")
    val_path = os.path.join(DATA_DIR, f"starcoder_{index:05d}_val_tmp.parquet")
    if os.path.exists(train_path):
        return True
    url = f"{STARCODER_BASE_URL}/train-{index:05d}-of-{STARCODER_TOTAL_SHARDS:05d}.parquet"
    raw_path = train_path + ".raw"
    ok = _fetch_url(url, raw_path)
    if not ok:
        return False
    table = pq.read_table(raw_path, columns=["content"])
    texts = table.column("content").to_pylist()
    val_n = max(1, len(texts) // 10)
    pq.write_table(pa.table({"text": pa.array(texts[:len(texts)-val_n], type=pa.string())}), train_path)
    pq.write_table(pa.table({"text": pa.array(texts[len(texts)-val_n:], type=pa.string())}), val_path)
    os.remove(raw_path)
    return True


def download_data(num_python_shards=10, download_workers=8):
    """Download Belle CN + Glaive + GitHub Python parquet files.

    Belle: one parquet → split into train.parquet + val contribution (last 2k rows).
    Glaive: split into train + val contribution (last 1k rows).
    val.parquet: merged from Belle val + Glaive val (mixed distribution).
    Python: first num_python_shards shards from GitHub Python dataset.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    # --- Belle ---
    belle_train = os.path.join(DATA_DIR, "belle_train.parquet")
    belle_val_tmp = os.path.join(DATA_DIR, "belle_val_tmp.parquet")
    if os.path.exists(belle_train) and os.path.exists(belle_val_tmp):
        print(f"Data: Belle already prepared at {DATA_DIR}")
    else:
        belle_raw = os.path.join(DATA_DIR, "belle_raw.parquet")
        print("Data: downloading Belle CN...")
        ok = _fetch_url(BELLE_PARQUET_URL, belle_raw)
        if not ok:
            print("ERROR: failed to download Belle parquet")
            sys.exit(1)
        pf = pq.ParquetFile(belle_raw)
        table = pf.read()
        instructions = table.column("instruction").to_pylist()
        outputs = table.column("output").to_pylist()
        texts = [f"<|reserved_1|>{i}<|reserved_2|>{o}<|reserved_3|>" for i, o in zip(instructions, outputs)]
        text_col = pa.array(texts, type=pa.string())
        text_table = pa.table({"text": text_col})
        val_size = len(texts) // 10  # 10% val
        pq.write_table(text_table.slice(0, len(texts) - val_size), belle_train)
        pq.write_table(text_table.slice(len(texts) - val_size), belle_val_tmp)
        os.remove(belle_raw)
        print(f"Data: Belle split → {len(texts)-val_size} train + {val_size} val rows (10%)")

    # --- Glaive function calling v2 ---
    glaive_train = os.path.join(DATA_DIR, "glaive_train.parquet")
    glaive_val_tmp = os.path.join(DATA_DIR, "glaive_val_tmp.parquet")
    if os.path.exists(glaive_train) and os.path.exists(glaive_val_tmp):
        print(f"Data: Glaive already prepared at {DATA_DIR}")
    else:
        glaive_raw = os.path.join(DATA_DIR, "glaive_raw.parquet")
        print("Data: downloading Glaive function-calling-v2...")
        ok = _fetch_url(GLAIVE_PARQUET_URL, glaive_raw)
        if not ok:
            print("WARNING: failed to download Glaive parquet, skipping")
        else:
            table = pq.read_table(glaive_raw)
            chats = table.column("chat").to_pylist()
            converted = [_convert_glaive_chat(c) for c in chats]
            converted = [c for c in converted if c is not None]
            val_size = len(converted) // 10  # 10% val
            train_texts = pa.array(converted[:-val_size], type=pa.string())
            val_texts = pa.array(converted[-val_size:], type=pa.string())
            pq.write_table(pa.table({"text": train_texts}), glaive_train)
            pq.write_table(pa.table({"text": val_texts}), glaive_val_tmp)
            os.remove(glaive_raw)
            print(f"Data: Glaive v2 → {len(converted)-val_size} train + {val_size} val rows (10%), {len(chats)-len(converted)} dropped")

    # NOTE: val merge is deferred to after all downloads complete (see end of function)

    # --- Belle 2M CN ---
    belle2m_done = os.path.join(DATA_DIR, "belle2m_done.flag")
    if os.path.exists(belle2m_done):
        print(f"Data: Belle 2M already prepared at {DATA_DIR}")
    else:
        print(f"Data: downloading Belle 2M CN ({BELLE_2M_NUM_SHARDS} shards)...")
        all_ok = True
        for i in range(BELLE_2M_NUM_SHARDS):
            shard_path = os.path.join(DATA_DIR, f"belle2m_{i:04d}.parquet")
            val_tmp_path = os.path.join(DATA_DIR, f"belle2m_{i:04d}_val_tmp.parquet")
            if os.path.exists(shard_path):
                continue
            url = f"{BELLE_2M_BASE_URL}/{i:04d}.parquet"
            raw_path = shard_path + ".raw"
            ok = _fetch_url(url, raw_path)
            if not ok:
                print(f"WARNING: failed to download Belle 2M shard {i}, skipping")
                all_ok = False
                continue
            table = pq.read_table(raw_path)
            instructions = table.column("instruction").to_pylist()
            outputs = table.column("output").to_pylist()
            texts = [f"<|reserved_1|>{ins}<|reserved_2|>{out}<|reserved_3|>" for ins, out in zip(instructions, outputs)]
            val_n = len(texts) // 10
            train_texts = texts[:len(texts) - val_n]
            pq.write_table(pa.table({"text": pa.array(train_texts, type=pa.string())}), shard_path)
            pq.write_table(pa.table({"text": pa.array(texts[len(texts) - val_n:], type=pa.string())}), val_tmp_path)
            os.remove(raw_path)
            print(f"  Belle 2M shard {i}: {len(train_texts)} train + {val_n} val rows")
        if all_ok:
            open(belle2m_done, "w").close()

    # --- Glaive function calling v1 ---
    glaive_v1_train = os.path.join(DATA_DIR, "glaive_v1_train.parquet")
    glaive_v1_val_tmp = os.path.join(DATA_DIR, "glaive_v1_val_tmp.parquet")
    if os.path.exists(glaive_v1_train):
        print(f"Data: Glaive v1 already prepared at {DATA_DIR}")
    else:
        glaive_v1_raw = os.path.join(DATA_DIR, "glaive_v1_raw.parquet")
        print("Data: downloading Glaive function-calling-v1...")
        ok = _fetch_url(GLAIVE_V1_PARQUET_URL, glaive_v1_raw)
        if not ok:
            print("WARNING: failed to download Glaive v1 parquet, skipping")
        else:
            table = pq.read_table(glaive_v1_raw)
            chats = table.column("sample").to_pylist()
            converted = [_convert_glaive_chat(c) for c in chats]
            converted = [c for c in converted if c is not None]
            val_n = len(converted) // 10
            pq.write_table(pa.table({"text": pa.array(converted[:len(converted)-val_n], type=pa.string())}), glaive_v1_train)
            pq.write_table(pa.table({"text": pa.array(converted[len(converted)-val_n:], type=pa.string())}), glaive_v1_val_tmp)
            os.remove(glaive_v1_raw)
            print(f"Data: Glaive v1 → {len(converted)-val_n} train + {val_n} val rows, {len(chats)-len(converted)} dropped")

    # --- BELLE multiturn chat ---
    belle_multiturn_train = os.path.join(DATA_DIR, "belle_multiturn_train.parquet")
    if os.path.exists(belle_multiturn_train):
        print(f"Data: BELLE multiturn already prepared at {DATA_DIR}")
    else:
        belle_multiturn_raw = os.path.join(DATA_DIR, "belle_multiturn_raw.parquet")
        print("Data: downloading BELLE multiturn chat...")
        ok = _fetch_url(BELLE_MULTITURN_URL, belle_multiturn_raw)
        if not ok:
            print("WARNING: failed to download BELLE multiturn parquet, skipping")
        else:
            table = pq.read_table(belle_multiturn_raw, columns=["instruction", "output"])
            pq.write_table(table, belle_multiturn_train)
            os.remove(belle_multiturn_raw)
            print(f"Data: BELLE multiturn → {table.num_rows} rows")

    # --- BELLE school math ---
    belle_school_math_train = os.path.join(DATA_DIR, "belle_school_math_train.parquet")
    if os.path.exists(belle_school_math_train):
        print(f"Data: BELLE school math already prepared at {DATA_DIR}")
    else:
        belle_school_math_raw = os.path.join(DATA_DIR, "belle_school_math_raw.parquet")
        print("Data: downloading BELLE school math...")
        ok = _fetch_url(BELLE_SCHOOL_MATH_URL, belle_school_math_raw)
        if not ok:
            print("WARNING: failed to download BELLE school math parquet, skipping")
        else:
            table = pq.read_table(belle_school_math_raw, columns=["instruction", "output"])
            pq.write_table(table, belle_school_math_train)
            os.remove(belle_school_math_raw)
            print(f"Data: BELLE school math → {table.num_rows} rows")

    # --- GitHub Python ---
    num_shards = min(num_python_shards, PYTHON_MAX_SHARD + 1)
    existing = sum(1 for i in range(num_shards)
                   if os.path.exists(os.path.join(DATA_DIR, f"python_{i:04d}.parquet")))
    if existing == num_shards:
        print(f"Data: Python shards already downloaded ({num_shards} shards)")
    else:
        needed = num_shards - existing
        print(f"Data: downloading {needed} Python shards ({existing} already exist)...")
        workers = max(1, min(download_workers, needed))
        with Pool(processes=workers) as pool:
            results = pool.map(_download_python_shard, list(range(num_shards)))
        ok = sum(1 for r in results if r)
        print(f"Data: {ok}/{num_shards} Python shards ready")

    # --- GitHub Java ---
    java_shards = JAVA_MAX_SHARD + 1
    existing_java = sum(1 for i in range(java_shards)
                        if os.path.exists(os.path.join(DATA_DIR, f"java_{i:04d}.parquet")))
    if existing_java == java_shards:
        print(f"Data: Java shards already downloaded ({java_shards} shards)")
    else:
        needed_java = java_shards - existing_java
        print(f"Data: downloading {needed_java} Java shards ({existing_java} already exist)...")
        workers = max(1, min(download_workers, needed_java))
        with Pool(processes=workers) as pool:
            results = pool.map(_download_lang_shard,
                               [("java", JAVA_BASE_URL, i) for i in range(java_shards)])
        ok = sum(1 for r in results if r)
        print(f"Data: {ok}/{java_shards} Java shards ready")

    # --- GitHub JavaScript ---
    js_shards = JAVASCRIPT_MAX_SHARD + 1
    existing_js = sum(1 for i in range(js_shards)
                      if os.path.exists(os.path.join(DATA_DIR, f"javascript_{i:04d}.parquet")))
    if existing_js == js_shards:
        print(f"Data: JavaScript shards already downloaded ({js_shards} shards)")
    else:
        needed_js = js_shards - existing_js
        print(f"Data: downloading {needed_js} JavaScript shards ({existing_js} already exist)...")
        workers = max(1, min(download_workers, needed_js))
        with Pool(processes=workers) as pool:
            results = pool.map(_download_lang_shard,
                               [("javascript", JAVASCRIPT_BASE_URL, i) for i in range(js_shards)])
        ok = sum(1 for r in results if r)
        print(f"Data: {ok}/{js_shards} JavaScript shards ready")

    # --- Chinese Wikipedia ---
    wiki_existing = sum(1 for i in range(WIKI_ZH_NUM_SHARDS)
                        if os.path.exists(os.path.join(DATA_DIR, f"wiki_zh_{i:04d}.parquet")))
    if wiki_existing == WIKI_ZH_NUM_SHARDS:
        print(f"Data: Wiki ZH shards already downloaded ({WIKI_ZH_NUM_SHARDS} shards)")
    else:
        needed_wiki = WIKI_ZH_NUM_SHARDS - wiki_existing
        print(f"Data: downloading {needed_wiki} Wiki ZH shards ({wiki_existing} already exist)...")
        workers = max(1, min(download_workers, needed_wiki))
        with Pool(processes=workers) as pool:
            results = pool.map(_download_wiki_zh_shard, list(range(WIKI_ZH_NUM_SHARDS)))
        ok = sum(1 for r in results if r)
        print(f"Data: {ok}/{WIKI_ZH_NUM_SHARDS} Wiki ZH shards ready")

    # --- StarCoder Python ---
    starcoder_existing = sum(1 for i in range(STARCODER_NUM_SHARDS)
                             if os.path.exists(os.path.join(DATA_DIR, f"starcoder_{i:05d}.parquet")))
    if starcoder_existing == STARCODER_NUM_SHARDS:
        print(f"Data: StarCoder shards already downloaded ({STARCODER_NUM_SHARDS} shards)")
    else:
        needed_sc = STARCODER_NUM_SHARDS - starcoder_existing
        print(f"Data: downloading {needed_sc} StarCoder shards ({starcoder_existing} already exist)...")
        workers = max(1, min(download_workers, needed_sc))
        with Pool(processes=workers) as pool:
            results = pool.map(_download_starcoder_shard, list(range(STARCODER_NUM_SHARDS)))
        ok = sum(1 for r in results if r)
        print(f"Data: {ok}/{STARCODER_NUM_SHARDS} StarCoder shards ready")

    # --- Merge val shards (after all downloads complete) ---
    val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    if not os.path.exists(val_path):
        tables = []
        # Belle 0.5M val
        if os.path.exists(belle_val_tmp):
            tables.append(pq.read_table(belle_val_tmp))
        # Glaive v2 val
        if os.path.exists(glaive_val_tmp):
            tables.append(pq.read_table(glaive_val_tmp))
        # Belle 2M: from per-shard val_tmp files (train shards contain train-only rows)
        for i in range(BELLE_2M_NUM_SHARDS):
            vt = os.path.join(DATA_DIR, f"belle2m_{i:04d}_val_tmp.parquet")
            if os.path.exists(vt):
                tables.append(pq.read_table(vt))
        # Glaive v1 val
        if os.path.exists(glaive_v1_val_tmp):
            tables.append(pq.read_table(glaive_v1_val_tmp))
        # Python: val_tmp files from each shard (created by _download_python_shard)
        for i in range(PYTHON_MAX_SHARD + 1):
            vt = os.path.join(DATA_DIR, f"python_{i:04d}_val_tmp.parquet")
            if os.path.exists(vt):
                tables.append(pq.read_table(vt))
        # Java: val_tmp files from each shard
        for i in range(JAVA_MAX_SHARD + 1):
            vt = os.path.join(DATA_DIR, f"java_{i:04d}_val_tmp.parquet")
            if os.path.exists(vt):
                tables.append(pq.read_table(vt))
        # JavaScript: val_tmp files from each shard
        for i in range(JAVASCRIPT_MAX_SHARD + 1):
            vt = os.path.join(DATA_DIR, f"javascript_{i:04d}_val_tmp.parquet")
            if os.path.exists(vt):
                tables.append(pq.read_table(vt))
        # Wiki ZH: val_tmp files from each shard
        for i in range(WIKI_ZH_NUM_SHARDS):
            vt = os.path.join(DATA_DIR, f"wiki_zh_{i:04d}_val_tmp.parquet")
            if os.path.exists(vt):
                tables.append(pq.read_table(vt))
        # StarCoder: collect all val_tmp rows, then subsample to match DATA_MIX_WEIGHTS ratio
        starcoder_tables = []
        for i in range(STARCODER_NUM_SHARDS):
            vt = os.path.join(DATA_DIR, f"starcoder_{i:05d}_val_tmp.parquet")
            if os.path.exists(vt):
                starcoder_tables.append(pq.read_table(vt))
        if starcoder_tables and tables:
            non_sc_rows = sum(t.num_rows for t in tables)
            sc_weight = DATA_MIX_WEIGHTS.get('starcoder', 1.0)
            target_sc_rows = int(non_sc_rows * sc_weight)
            # Cast to large_string to avoid 2GB offset overflow during concat
            sc_tables_ls = [t.cast(pa.schema([('text', pa.large_utf8())])) for t in starcoder_tables]
            sc_all = pa.concat_tables(sc_tables_ls)
            if sc_all.num_rows > target_sc_rows:
                import random
                indices = random.sample(range(sc_all.num_rows), target_sc_rows)
                indices.sort()
                sc_all = sc_all.take(indices)
            # Cast back to string for consistency with other tables
            sc_all = sc_all.cast(pa.schema([('text', pa.utf8())]))
            tables.append(sc_all)
        elif starcoder_tables:
            tables.extend(starcoder_tables)
        if tables:
            merged = pa.concat_tables(tables)
            pq.write_table(merged, val_path)
            print(f"Data: val.parquet → {merged.num_rows} rows (weighted mix, starcoder subsampled)")

# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Data mix configuration
# ---------------------------------------------------------------------------

# Map file name prefixes to source labels
_SOURCE_PREFIX_MAP = [
    ('belle2m_',     'belle2m'),
    ('belle_',       'belle'),
    ('glaive_',      'glaive'),
    ('wiki_zh_',     'wiki_zh'),
    ('python_',      'python'),
    ('java_',        'java'),
    ('javascript_',  'javascript'),
    ('starcoder_',   'starcoder'),
]

# Sampling weights per source (proportional, token-level intent)
# Higher = more often. belle/wiki high-quality CN oversampled; starcoder undersampled.
DATA_MIX_WEIGHTS = {
    'belle':       3.0,
    'belle2m':     3.0,
    'glaive':      3.0,
    'wiki_zh':     3.0,
    'python':      1.0,
    'java':        1.0,
    'javascript':  1.0,
    'starcoder':   0.14,  # 59 shards * 0.14 ≈ 8 effective shards worth
}


def _get_source(filename):
    """Return source label for a parquet filename."""
    for prefix, label in _SOURCE_PREFIX_MAP:
        if filename.startswith(prefix):
            return label
    return 'other'


def list_parquet_files():
    """Return sorted list of train parquet file paths (excludes val and val_tmp)."""
    files = sorted(f for f in os.listdir(DATA_DIR)
                   if f.endswith(".parquet")
                   and not f.endswith(".tmp")
                   and "_val_tmp" not in f
                   and f != VAL_FILENAME)
    return [os.path.join(DATA_DIR, f) for f in files]


def text_iterator(max_chars=300_000_000, doc_cap=10_000):
    """Yield documents from training split (all shards except pinned val shard)."""
    parquet_paths = [p for p in list_parquet_files() if not p.endswith(VAL_FILENAME)]
    nchars = 0
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            for text in rg.column("text").to_pylist():
                doc = text[:doc_cap] if len(text) > doc_cap else text
                nchars += len(doc)
                yield doc
                if nchars >= max_chars:
                    return


def train_tokenizer():
    """Train BPE tokenizer using rustbpe, save as tiktoken pickle."""
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")

    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        print(f"Tokenizer: already trained at {TOKENIZER_DIR}")
        return

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    parquet_files = list_parquet_files()
    if len(parquet_files) < 2:
        print("Tokenizer: need at least 2 data shards (1 train + 1 val). Download more data first.")
        sys.exit(1)

    # --- Train with rustbpe ---
    print("Tokenizer: training BPE tokenizer...")
    t0 = time.time()

    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)
    tokenizer.train_from_iterator(text_iterator(), vocab_size_no_special, pattern=SPLIT_PATTERN)

    # Build tiktoken encoding from trained merges
    pattern = tokenizer.get_pattern()
    mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
    tokens_offset = len(mergeable_ranks)
    special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    # Save tokenizer
    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)

    t1 = time.time()
    print(f"Tokenizer: trained in {t1 - t0:.1f}s, saved to {tokenizer_pkl}")

    # --- Build token_bytes lookup for BPB evaluation ---
    print("Tokenizer: building token_bytes lookup...")
    special_set = set(SPECIAL_TOKENS)
    token_bytes_list = []
    for token_id in range(enc.n_vocab):
        token_str = enc.decode([token_id])
        if token_str in special_set:
            token_bytes_list.append(0)
        else:
            token_bytes_list.append(len(token_str.encode("utf-8")))
    token_bytes_tensor = torch.tensor(token_bytes_list, dtype=torch.int32)
    torch.save(token_bytes_tensor, token_bytes_path)
    print(f"Tokenizer: saved token_bytes to {token_bytes_path}")

    # Sanity check
    test = "Hello world! Numbers: 123. Unicode: 你好"
    encoded = enc.encode_ordinary(test)
    decoded = enc.decode(encoded)
    if decoded != test:
        raise ValueError(f"Tokenizer roundtrip failed: {test!r} -> {decoded!r}")
    print(f"Tokenizer: sanity check passed (vocab_size={enc.n_vocab})")

# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

class Tokenizer:
    """Minimal tokenizer wrapper. Training is handled above."""

    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        prepend_id = (prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)) if prepend is not None else None
        if isinstance(text, str):
            ids = self.enc.encode(text, allowed_special="all")
            if prepend_id is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_batch(text, allowed_special="all", num_threads=num_threads)
            if prepend_id is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)


def get_token_bytes(device="cpu"):
    path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    with open(path, "rb") as f:
        return torch.load(f, map_location=device)


def _maybe_trim_host_memory():
    if _LIBC is None:
        return
    try:
        _LIBC.malloc_trim(0)
    except Exception:
        pass


def _read_parquet_batches(filepath, tokenizer_batch_size):
    """Yield all document batches from a single parquet file.

    Supports two schemas:
      - {text}: 使用 text 列
      - {instruction, output}: 拼接 instruction + "\\n" + output 作为 text (Belle 类数据)
    """
    pf = pq.ParquetFile(filepath)
    fields = set(pf.schema_arrow.names)
    for rg_idx in range(pf.num_row_groups):
        rg = pf.read_row_group(rg_idx)
        if 'text' in fields:
            batch = rg.column('text').to_pylist()
        elif 'instruction' in fields and 'output' in fields:
            ins = rg.column('instruction').to_pylist()
            out = rg.column('output').to_pylist()
            batch = [(i or '') + '\n' + (o or '') for i, o in zip(ins, out)]
        else:
            raise ValueError(f"unsupported schema in {filepath}: fields={pf.schema_arrow.names}")
        for i in range(0, len(batch), tokenizer_batch_size):
            yield batch[i:i+tokenizer_batch_size]


def _document_batches(split, tokenizer_batch_size=128):
    """Infinite iterator over document batches from parquet files.

    For train split: uses deficit tracking to sample files proportionally
    according to DATA_MIX_WEIGHTS. Each source maintains a shuffled file queue;
    deficit accumulates across epochs (not reset at epoch boundary).

    For val split: reads val.parquet sequentially.
    """
    import random
    val_path = os.path.join(DATA_DIR, VAL_FILENAME)

    if split == "val":
        epoch = 1
        while True:
            pf = pq.ParquetFile(val_path)
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], epoch
            epoch += 1

    # --- Train split: deficit tracking ---
    all_paths = list_parquet_files()
    assert len(all_paths) > 0, "No parquet files found. Run prepare.py first."

    # Group files by source
    from collections import defaultdict
    source_files = defaultdict(list)
    for p in all_paths:
        src = _get_source(os.path.basename(p))
        source_files[src].append(p)

    # Only keep sources that have files and weights
    active_sources = {s: files for s, files in source_files.items()
                      if s in DATA_MIX_WEIGHTS and files}
    if not active_sources:
        raise RuntimeError("No active sources found. Check DATA_MIX_WEIGHTS.")

    # Per-source shuffled queues and deficit accumulators
    queues = {s: [] for s in active_sources}      # shuffled file queue per source
    deficits = {s: 0.0 for s in active_sources}   # accumulated deficit per source
    epoch_counters = {s: 1 for s in active_sources}

    def refill_queue(src):
        q = list(active_sources[src])
        random.shuffle(q)
        queues[src].extend(q)
        epoch_counters[src] += 1

    # Prime queues
    for src in active_sources:
        refill_queue(src)

    global_epoch = 1
    files_yielded = 0

    while True:
        # Pick source with highest deficit
        src = max(active_sources, key=lambda s: deficits[s])

        # Dequeue next file from that source
        if not queues[src]:
            refill_queue(src)
        filepath = queues[src].pop(0)

        # Yield all batches from this file
        for batch in _read_parquet_batches(filepath, tokenizer_batch_size):
            yield batch, global_epoch

        # Update deficit: src was served, so only other sources accumulate
        for s in active_sources:
            if s != src:
                deficits[s] += DATA_MIX_WEIGHTS[s]

        files_yielded += 1
        # Update global epoch roughly when all sources have been read once
        total_files = sum(len(v) for v in active_sources.values())
        if files_yielded % total_files == 0:
            global_epoch += 1


def make_dataloader(tokenizer, B, T, split, buffer_size=1000, device="cuda"):
    """
    BOS-aligned dataloader with best-fit packing.
    Every row starts with BOS. Documents packed using best-fit to minimize cropping.
    When no document fits remaining space, crops shortest doc to fill exactly.
    100% utilization (no padding).
    """
    assert split in ["train", "val"]
    device = torch.device(device)
    row_capacity = T + 1
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    # Pre-allocate buffers: [inputs (B*T) | targets (B*T)]
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    if device.type == "cuda":
        host_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
        device_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
        host_inputs = host_buffer[:B * T].view(B, T)
        host_targets = host_buffer[B * T:].view(B, T)
        inputs = device_buffer[:B * T].view(B, T)
        targets = device_buffer[B * T:].view(B, T)
    else:
        inputs = torch.empty((B, T), dtype=torch.long, device=device)
        targets = torch.empty((B, T), dtype=torch.long, device=device)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    # No doc fits — crop shortest to fill remaining
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        if device.type == "cuda":
            host_inputs.copy_(row_buffer[:, :-1])
            host_targets.copy_(row_buffer[:, 1:])
            device_buffer.copy_(host_buffer, non_blocking=True)
        else:
            inputs.copy_(row_buffer[:, :-1])
            targets.copy_(row_buffer[:, 1:])
        yield inputs, targets, epoch

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size, device=None, max_tokens=None):
    """
    Bits per byte (BPB): vocab size-independent evaluation metric.
    Sums per-token cross-entropy (in nats), sums target byte lengths,
    then converts nats/byte to bits/byte. Special tokens (byte length 0)
    are excluded from both sums.
    Uses fixed MAX_SEQ_LEN so results are comparable across configs.
    If max_tokens is None, uses EVAL_TOKENS (full eval).
    """
    if device is None:
        device = next(model.parameters()).device
    token_bytes = get_token_bytes(device=device)
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val", device=device)
    target_tokens = max_tokens if max_tokens is not None else EVAL_TOKENS
    steps = target_tokens // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    total_bytes = 0
    cleanup_interval = 512
    for step_idx in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction='none').view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
        if (step_idx + 1) % cleanup_interval == 0:
            del x, y, loss_flat, y_flat, nbytes, mask
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            _maybe_trim_host_memory()
    del val_loader
    gc.collect()
    _maybe_trim_host_memory()
    return total_nats / (math.log(2) * total_bytes)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data and tokenizer for autoresearch")
    parser.add_argument("--num-shards", type=int, default=10, help="Number of training shards to download (-1 = all). Val shard is always pinned.")
    parser.add_argument("--download-workers", type=int, default=8, help="Number of parallel download workers")
    parser.add_argument("--skip-download", action="store_true", help="Skip download step, only train tokenizer")
    args = parser.parse_args()

    num_python_shards = PYTHON_MAX_SHARD + 1 if args.num_shards == -1 else args.num_shards

    print(f"Cache directory: {CACHE_DIR}")
    print()

    # Step 1: Download data
    if not args.skip_download:
        download_data(num_python_shards, download_workers=args.download_workers)
        print()

    # Step 2: Train tokenizer
    train_tokenizer()
    print()
    print("Done! Ready to train.")
