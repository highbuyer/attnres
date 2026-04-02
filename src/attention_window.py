from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=None)
def build_causal_window_mask(seq_len: int, window_size: tuple[int, int]) -> tuple[tuple[bool, ...], ...] | None:
    left_window, right_window = window_size
    if left_window < 0 and right_window < 0:
        return None

    rows = []
    for query_idx in range(seq_len):
        lower_bound = 0 if left_window < 0 else max(0, query_idx - left_window)
        upper_bound = query_idx if right_window < 0 else min(query_idx, query_idx + right_window)
        rows.append(
            tuple(lower_bound <= key_idx <= upper_bound for key_idx in range(seq_len))
        )
    return tuple(rows)
