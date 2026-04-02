#!/usr/bin/env python3
"""Interactive text generation from a saved autoresearch checkpoint.

Usage:
    python generate.py                        # interactive mode
    python generate.py --prompt "Once upon"   # single prompt
    python generate.py --prompt "The sky is" --max-tokens 200 --temperature 0.8
"""
import argparse
import sys
from pathlib import Path
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC_DIR))
import torch
from train import GPT, GPTConfig
from prepare import Tokenizer


def load_model(ckpt_path="checkpoint.pt", device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    print(f"val_bpb={ckpt['val_bpb']:.4f}, step={ckpt['step']}")
    model = GPT(config).to(device).to(torch.bfloat16)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state"].items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    head_dim = config.n_embd // config.n_head
    cos, sin = model._precompute_rotary_embeddings(model.rotary_seq_len, head_dim, device=device)
    model.cos, model.sin = cos, sin
    print(f"Model ready: {config.n_layer}L x {config.n_embd}d, vocab={config.vocab_size}")
    return model, config


@torch.no_grad()
def generate(model, tokenizer, prompt, max_tokens=200, temperature=1.0, top_k=50, device="cuda"):
    bos = tokenizer.get_bos_token_id()
    ids = [bos] + tokenizer.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(max_tokens):
            logits = model(x[:, -model.config.sequence_len:])
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = -float("inf")
            next_id = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            x = torch.cat([x, next_id], dim=1)
    return tokenizer.decode(x[0, len(ids):].tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoint.pt")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = load_model(args.checkpoint, device)
    tokenizer = Tokenizer.from_directory()

    if args.prompt is not None:
        out = generate(model, tokenizer, args.prompt, args.max_tokens, args.temperature, args.top_k, device)
        print(f"\n{args.prompt}{out}")
    else:
        print("Interactive mode (Ctrl+C to exit)")
        while True:
            try:
                prompt = input("\nPrompt> ")
            except (KeyboardInterrupt, EOFError):
                break
            if not prompt.strip():
                continue
            out = generate(model, tokenizer, prompt, args.max_tokens, args.temperature, args.top_k, device)
            print(f"---\n{prompt}{out}\n---")


if __name__ == "__main__":
    main()
