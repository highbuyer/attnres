#!/usr/bin/env python3
"""Self-adaptive architecture search: Claude analyzes results and decides next config.

Usage:
    python search.py                  # run forever
    python search.py --rounds 20      # stop after 20 rounds
    python search.py --rounds 20 --resume

Each round:
    1. Train with current config (300s)
    2. Feed history to Claude
    3. Claude decides next config based on trends
    4. Repeat
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
import urllib.request
import shutil
from pathlib import Path

PYTHON = sys.executable
TRAIN_PY = str(Path(__file__).parent / "train.py")
RESULTS_CSV = Path("search_results.csv")
BEST_JSON = Path("search_best.json")
STRIP_VARS = ["ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "all_proxy"]

CLAUDE_URL = os.environ.get("CLAUDE_URL", "http://127.0.0.1:3010/v1")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "anthropic/claude-sonnet-4.6")

# Search space bounds
SEARCH_SPACE = """
- depth: int, one of [4, 6, 8, 10, 12]  (number of transformer layers)
- aspect_ratio: int, one of [48, 64, 80, 96]  (model_dim = depth * aspect_ratio)
- window_pattern: str, one of ["SSSL", "SSL", "SL", "L"]  (sliding window attention pattern)
- matrix_lr: float, range [0.01, 0.08]  (Muon optimizer LR for weight matrices)
- embedding_lr: float, range [0.2, 1.0]  (AdamW LR for token embeddings)
"""

SYSTEM_PROMPT = """You are an expert in training small GPT language models.
You will analyze training results and decide the next hyperparameter configuration.
Always respond with valid JSON only, no explanation."""


def call_claude(history: list[dict], current_best: dict | None) -> dict:
    """Ask Claude to decide next config based on history."""
    history_str = json.dumps(history[-20:], indent=2)  # last 20 rounds
    best_str = json.dumps(current_best, indent=2) if current_best else "none yet"

    prompt = f"""You are tuning a small GPT model trained for exactly 300 seconds on an RTX 4090 GPU.
The metric is val_bpb (bits per byte) — LOWER IS BETTER.
Hardware constraint: RTX 4090, 24GB VRAM, batch_size=16, seq_len=2048.
Larger models train fewer steps in 300s, so bigger is NOT always better — efficiency matters.
The model uses Flash Attention 3 + Muon optimizer.

Search space:
{SEARCH_SPACE}

Current best configuration:
{best_str}

Full history of all rounds (most recent last):
{history_str}

Analyze the trends:
- Which depth/aspect_ratio gives best val_bpb within 300s on a 4090?
- Is a particular dimension clearly better?
- Is the search stagnating? If so, explore a different region.
- Are there patterns in what works vs what doesn't?

Decide the NEXT configuration to try. Respond with JSON only:
{{"depth": <int>, "aspect_ratio": <int>, "window_pattern": "<str>", "matrix_lr": <float>, "embedding_lr": <float>, "reasoning": "<one line>"}}"""

    payload = json.dumps({
        "model": CLAUDE_MODEL,
        "max_tokens": 256,
        "temperature": 0.3,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}]
    }).encode()

    req = urllib.request.Request(
        f"{CLAUDE_URL}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        content = json.loads(resp.read())["choices"][0]["message"]["content"]

    # Extract JSON from response
    content = content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    return json.loads(content.strip())


def default_config() -> dict:
    """Starting config (same as train.py defaults)."""
    return {
        "depth": 8,
        "aspect_ratio": 64,
        "window_pattern": "SSSL",
        "matrix_lr": 0.04,
        "embedding_lr": 0.6,
    }


def run_trial(params: dict) -> tuple[float, str]:
    """Run one training trial. Returns (val_bpb, log_tail)."""
    env = {k: v for k, v in os.environ.items() if k not in STRIP_VARS}
    env["AR_DEPTH"] = str(params["depth"])
    env["AR_ASPECT_RATIO"] = str(params["aspect_ratio"])
    env["AR_WINDOW_PATTERN"] = params["window_pattern"]
    env["AR_MATRIX_LR"] = f"{params['matrix_lr']:.6f}"
    env["AR_EMBEDDING_LR"] = f"{params['embedding_lr']:.6f}"
    env["AR_DEVICE_BATCH_SIZE"] = "16"

    t0 = time.time()
    try:
        result = subprocess.run(
            [PYTHON, TRAIN_PY],
            capture_output=True, text=True, env=env, timeout=450
        )
        elapsed = time.time() - t0
        log_tail = result.stdout[-300:]
        for line in result.stdout.splitlines():
            if line.startswith("val_bpb:"):
                return float(line.split()[1]), log_tail
        return 99.0, result.stderr[-200:]
    except subprocess.TimeoutExpired:
        return 99.0, "timeout"


def load_history() -> list[dict]:
    if not RESULTS_CSV.exists():
        return []
    with open(RESULTS_CSV) as f:
        return list(csv.DictReader(f))


def load_best() -> dict | None:
    if BEST_JSON.exists():
        return json.loads(BEST_JSON.read_text())
    return None


def save_best(params: dict, val_bpb: float):
    data = {**params, "val_bpb": val_bpb, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    BEST_JSON.write_text(json.dumps(data, indent=2))
    if Path("checkpoint.pt").exists():
        shutil.copy("checkpoint.pt", "best_checkpoint.pt")
        print(f"  best_checkpoint.pt updated")


def append_result(params: dict, val_bpb: float):
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, 'a', newline='') as f:
        fields = ["depth", "aspect_ratio", "window_pattern", "matrix_lr", "embedding_lr", "val_bpb"]
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow({**{k: params[k] for k in fields[:-1]}, "val_bpb": val_bpb})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=0, help="0=forever")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not args.resume and RESULTS_CSV.exists():
        print(f"Found existing {RESULTS_CSV}. Use --resume to continue or delete it.")
        sys.exit(1)

    best_bpb = float('inf')
    best_params = load_best()
    if best_params:
        best_bpb = float(best_params["val_bpb"])
        print(f"Resumed. Current best: val_bpb={best_bpb:.6f}")

    history = load_history()
    round_num = len(history)

    # First round uses default config; after that Claude decides
    if round_num == 0:
        next_config = default_config()
        next_config["reasoning"] = "baseline"
    else:
        print(f"Asking Claude for next config...")
        try:
            next_config = call_claude(history, best_params)
        except Exception as e:
            print(f"Claude call failed: {e}, using default")
            next_config = default_config()

    print(f"Self-adaptive search started. Ctrl+C to stop.")
    print()

    try:
        while True:
            round_num += 1
            params = {k: next_config[k] for k in
                      ["depth", "aspect_ratio", "window_pattern", "matrix_lr", "embedding_lr"]}
            reasoning = next_config.get("reasoning", "")

            print(f"[round {round_num}] depth={params['depth']} ar={params['aspect_ratio']} "
                  f"win={params['window_pattern']} mlr={params['matrix_lr']:.4f} elr={params['embedding_lr']:.4f}")
            if reasoning:
                print(f"  Claude: {reasoning}")

            val_bpb, _ = run_trial(params)
            print(f"  val_bpb={val_bpb:.6f}", end="")

            if val_bpb < best_bpb:
                best_bpb = val_bpb
                best_params = {**params, "val_bpb": val_bpb}
                save_best(params, val_bpb)
                print(f"  *** NEW BEST ***", end="")
            print()

            append_result(params, val_bpb)

            history = load_history()
            if args.rounds and round_num >= args.rounds:
                break

            # Ask Claude for next config
            print(f"  Asking Claude for next config...")
            try:
                next_config = call_claude(history, best_params)
            except Exception as e:
                print(f"  Claude call failed: {e}, repeating last config with small LR tweak")
                next_config = {**params, "matrix_lr": params["matrix_lr"] * 0.9, "reasoning": "fallback"}
            print()

    except KeyboardInterrupt:
        print("\n[search] Stopped.")

    print(f"\n=== Final Best ===")
    if best_params:
        print(f"val_bpb: {best_bpb:.6f}")
        for k, v in best_params.items():
            if k not in ("val_bpb", "ts"):
                print(f"  {k}: {v}")
        print(f"Checkpoint: best_checkpoint.pt")


if __name__ == "__main__":
    main()
