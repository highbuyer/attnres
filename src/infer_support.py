from __future__ import annotations


def has_flash_attention_backend(flash_attn_backend: object) -> bool:
    return callable(getattr(flash_attn_backend, "flash_attn_func", None))


def ensure_inference_backend(device: str, flash_attn_backend: object) -> None:
    if device == "cuda":
        if not has_flash_attention_backend(flash_attn_backend):
            # On CUDA, we strongly prefer FlashAttention for performance, 
            # but we can fallback to SDPA as it is now supported in train.py.
            print("WARNING: FlashAttention not found on CUDA. Falling back to slower SDPA.")
    elif device == "cpu":
        # CPU is now supported via SDPA fallback in CausalSelfAttention
        pass
    else:
        raise RuntimeError(f"Unsupported device: {device}")
