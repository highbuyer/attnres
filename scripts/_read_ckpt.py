import sys
sys.path.insert(0, '.')
# Minimal GPTConfig to unpickle
from dataclasses import dataclass
@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 8192
    n_layer: int = 6
    n_head: int = 8
    n_kv_head: int = 8
    n_embd: int = 512
    window_pattern: str = 'SSL'
import torch
for fname in ['checkpoint.pt', 'best_checkpoint.pt']:
    try:
        ckpt = torch.load(fname, map_location='cpu', weights_only=False)
        print(f"{fname}: val_bpb={ckpt.get('val_bpb')}, step={ckpt.get('step')}")
    except Exception as e:
        print(f"{fname}: ERROR {e}")
