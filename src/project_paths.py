from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _candidate_paths(path_str: str) -> list[Path]:
    path = Path(os.path.expanduser(path_str))
    candidates = [path]
    if not path.is_absolute():
        candidates.append(Path.cwd() / path)
        candidates.append(REPO_ROOT / path)
        candidates.append(REPO_ROOT / "checkpoints" / path.name)

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def resolve_existing_path(path_str: str, *, label: str) -> Path:
    for candidate in _candidate_paths(path_str):
        if candidate.exists():
            return candidate
    searched = ", ".join(str(candidate) for candidate in _candidate_paths(path_str))
    raise FileNotFoundError(f"{label} not found: {path_str}. Searched: {searched}")


def resolve_default_checkpoint(explicit_path: str | None) -> Path:
    if explicit_path:
        return resolve_existing_path(explicit_path, label="checkpoint")

    candidates = [
        REPO_ROOT / "checkpoints" / "best_checkpoint.pt",
        REPO_ROOT / "checkpoints" / "checkpoint.pt",
        REPO_ROOT / "best_checkpoint.pt",
        REPO_ROOT / "checkpoint.pt",
        Path.cwd() / "best_checkpoint.pt",
        Path.cwd() / "checkpoint.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"checkpoint not found. Searched: {searched}")


def resolve_sft_input_checkpoint(explicit_path: str | None) -> Path:
    if explicit_path:
        return resolve_existing_path(explicit_path, label="checkpoint")
    return resolve_existing_path("continued_d18_32k_final.pt", label="checkpoint")


def resolve_sft_data_path(explicit_path: str | None) -> Path:
    if explicit_path:
        return resolve_existing_path(explicit_path, label="SFT data")

    env_path = os.environ.get("ATTNRES_SFT_DATA")
    if env_path:
        return resolve_existing_path(env_path, label="SFT data")

    candidates = [
        REPO_ROOT / "sft_mixed_v8.jsonl",
        REPO_ROOT / "sft_mixed_v7.jsonl",
        Path.cwd() / "sft_mixed_v8.jsonl",
        Path.cwd() / "sft_mixed_v7.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "SFT data not found. Pass --data, set ATTNRES_SFT_DATA, or place a dataset at one of: "
        f"{searched}"
    )
