from __future__ import annotations


TOOL_CHAIN_MARKERS = ("<|tool_call_start|>", "<|tool_result_start|>")


def assistant_block_bounds(messages: list[dict], assistant_idx: int) -> tuple[int, int]:
    block_start = assistant_idx
    while block_start > 0 and messages[block_start - 1]["role"] == "assistant":
        block_start -= 1

    block_end = assistant_idx + 1
    while block_end < len(messages) and messages[block_end]["role"] == "assistant":
        block_end += 1

    return block_start, block_end


def assistant_block_requires_chain(messages: list[dict], assistant_idx: int) -> bool:
    block_start, block_end = assistant_block_bounds(messages, assistant_idx)
    for message in messages[block_start:block_end]:
        content = message.get("content", "")
        if any(marker in content for marker in TOOL_CHAIN_MARKERS):
            return True
    return False


def included_turn_indices(messages: list[dict], assistant_idx: int) -> list[int]:
    if messages[assistant_idx]["role"] != "assistant":
        raise ValueError("assistant_idx must point to an assistant turn")

    block_start, _block_end = assistant_block_bounds(messages, assistant_idx)
    keep_chain = assistant_block_requires_chain(messages, assistant_idx)

    indices = []
    for idx in range(assistant_idx + 1):
        if not keep_chain and block_start <= idx < assistant_idx and messages[idx]["role"] == "assistant":
            continue
        indices.append(idx)
    return indices
