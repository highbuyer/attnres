#!/usr/bin/env python3
"""Autonomous experiment loop for autoresearch.

Runs experiments automatically, keeps improvements, discards failures.
Logs everything to results.tsv. Stop anytime with Ctrl+C.

Usage:
    python autorun.py
"""
import os
import subprocess
import sys
import re
from pathlib import Path

PYTHON = sys.executable
TRAIN_PY = str(Path(__file__).parent / "train.py")
RESULTS_TSV = Path("results.tsv")
STRIP_VARS = ["ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "all_proxy"]

# ---------------------------------------------------------------------------
# Experiment definitions
# Each entry: (description, {param: value, ...})
# Parameters map to exact strings in train.py to find-and-replace
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    # --- LR tuning ---
    ("matrix_lr 0.055->0.06",       {"MATRIX_LR": "0.06"}),
    ("matrix_lr 0.055->0.05",       {"MATRIX_LR": "0.05"}),
    ("embedding_lr 0.42->0.5",      {"EMBEDDING_LR": "0.5"}),
    ("embedding_lr 0.42->0.35",     {"EMBEDDING_LR": "0.35"}),
    # --- Warmdown ---
    ("warmdown 0.5->0.3",           {"WARMDOWN_RATIO": "0.3"}),
    ("warmdown 0.5->0.7",           {"WARMDOWN_RATIO": "0.7"}),
    # --- Weight decay ---
    ("weight_decay 0.2->0.1",       {"WEIGHT_DECAY": "0.1"}),
    ("weight_decay 0.2->0.0",       {"WEIGHT_DECAY": "0.0"}),
    # --- Depth/width ---
    ("depth 6->8 ar 80->64",        {"DEPTH": "8",  "ASPECT_RATIO": "64"}),
    ("depth 6->4 ar 80->96",        {"DEPTH": "4",  "ASPECT_RATIO": "96"}),
    ("depth 6->8 ar 80->80",        {"DEPTH": "8",  "ASPECT_RATIO": "80"}),
    # --- Window pattern ---
    ("window SSL",                  {"WINDOW_PATTERN": '"SSL"'}),
    ("window SSSL",                 {"WINDOW_PATTERN": '"SSSL"'}),
    ("window L",                    {"WINDOW_PATTERN": '"L"'}),
    # --- Batch size ---
    ("batch 16->32",                {"DEVICE_BATCH_SIZE": "32"}),
    # --- Adam betas ---
    ("adam_beta1 0.8->0.9",         {"ADAM_BETAS": "(0.9, 0.95)"}),
    ("adam_beta2 0.95->0.99",       {"ADAM_BETAS": "(0.8, 0.99)"}),
    # --- Final LR ---
    ("final_lr_frac 0->0.1",        {"FINAL_LR_FRAC": "0.1"}),
]


def read_train_py() -> str:
    return Path(TRAIN_PY).read_text()


def write_train_py(content: str):
    Path(TRAIN_PY).write_text(content)


def apply_params(content: str, params: dict) -> str:
    """Replace constant values in train.py."""
    for key, new_val in params.items():
        # Match: KEY = <anything>  # optional comment
        pattern = rf'^({re.escape(key)}\s*=\s*)[^\n#]+(.*?)$'
        replacement = rf'\g<1>{new_val}\2'
        content, n = re.subn(pattern, replacement, content, flags=re.MULTILINE)
        if n == 0:
            print(f"  [!] Could not find {key} in train.py")
    return content


def run_training() -> tuple[float, float]:
    """Run train.py, return (val_bpb, peak_vram_gb). Returns (99, 0) on failure."""
    env = {k: v for k, v in os.environ.items() if k not in STRIP_VARS}
    try:
        result = subprocess.run(
            [PYTHON, TRAIN_PY],
            capture_output=True, text=True, env=env, timeout=450
        )
        with open("run.log", "w") as f:
            f.write(result.stdout)
            f.write(result.stderr)
        bpb, vram = 99.0, 0.0
        for line in result.stdout.splitlines():
            if line.startswith("val_bpb:"):
                bpb = float(line.split()[1])
            elif line.startswith("peak_vram_mb:"):
                vram = float(line.split()[1]) / 1024
        return bpb, vram
    except subprocess.TimeoutExpired:
        return 99.0, 0.0


def git_commit(msg: str) -> str:
    subprocess.run(["git", "add", "train.py"], capture_output=True)
    subprocess.run(["git", "commit", "-m", msg], capture_output=True)
    result = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True)
    return result.stdout.strip()


def git_reset():
    subprocess.run(["git", "reset", "--hard", "HEAD~1"], capture_output=True)


def append_tsv(commit: str, bpb: float, vram: float, status: str, desc: str):
    with open(RESULTS_TSV, "a") as f:
        f.write(f"{commit}\t{bpb:.6f}\t{vram:.1f}\t{status}\t{desc}\n")


def get_current_bpb() -> float:
    """Get best val_bpb from results.tsv."""
    if not RESULTS_TSV.exists():
        return 99.0
    best = 99.0
    with open(RESULTS_TSV) as f:
        for line in f:
            if line.startswith("commit"):
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 4 and parts[3] == "keep":
                try:
                    best = min(best, float(parts[1]))
                except ValueError:
                    pass
    return best


def main():
    best_bpb = get_current_bpb()
    baseline_content = read_train_py()
    print(f"Starting autonomous experiment loop")
    print(f"Current best val_bpb: {best_bpb:.6f}")
    print(f"Planned experiments: {len(EXPERIMENTS)}")
    print(f"Ctrl+C to stop")
    print()

    try:
        for i, (desc, params) in enumerate(EXPERIMENTS):
            print(f"[exp {i+1}/{len(EXPERIMENTS)}] {desc}")

            # Read current best train.py and apply params
            current = read_train_py()
            modified = apply_params(current, params)
            if modified == current:
                print(f"  [!] No change made, skipping")
                continue
            write_train_py(modified)

            commit = git_commit(f"exp{i+1}: {desc}")
            bpb, vram = run_training()

            if bpb >= 99.0:
                print(f"  CRASH/TIMEOUT")
                git_reset()
                write_train_py(current)
                append_tsv(commit, 0.0, 0.0, "crash", desc)
            elif bpb < best_bpb:
                best_bpb = bpb
                baseline_content = read_train_py()  # update baseline to new best
                print(f"  val_bpb={bpb:.6f} vram={vram:.1f}GB  *** NEW BEST ***")
                append_tsv(commit, bpb, vram, "keep", desc)
            else:
                print(f"  val_bpb={bpb:.6f} vram={vram:.1f}GB  discard (best={best_bpb:.6f})")
                git_reset()
                write_train_py(current)  # restore
                append_tsv(commit, bpb, vram, "discard", desc)
            print()

    except KeyboardInterrupt:
        print("\nStopped by user.")

    print(f"\n=== Final best: {best_bpb:.6f} ===")
    print(f"Results saved to {RESULTS_TSV}")


if __name__ == "__main__":
    main()
