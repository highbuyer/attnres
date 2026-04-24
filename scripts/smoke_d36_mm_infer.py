"""Inference smoke: d36_sft_mm_best.pt 用 minimind 风格 prompt 直接推理，不走 weiyan-api。

目的：验证模型真正学到了 minimind chat_template 格式，排除 weiyan-api infer 侧与旧 USER_ID/ASST_ID 的不匹配。
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Load via same path as sft.py
import types
src_dir = Path(__file__).resolve().parents[1] / "src"
lines = (src_dir / "train.py").read_text(encoding="utf-8").splitlines(keepends=True)
cut = next(i for i, line in enumerate(lines) if "# Setup: tokenizer, model, optimizer, dataloader" in line)
src = "".join(lines[:cut])
fake = types.ModuleType("prepare")
fake.MAX_SEQ_LEN = 2048
fake.TIME_BUDGET = 999999
fake.Tokenizer = None
fake.make_dataloader = None
fake.evaluate_bpb = None
sys.modules["prepare"] = fake
try:
    ns: dict = {}
    exec(compile(src, "train.py", "exec"), ns)
finally:
    del sys.modules["prepare"]
GPT = ns["GPT"]
GPTConfig = ns["GPTConfig"]

from prepare import Tokenizer  # type: ignore
tk = Tokenizer.from_directory()
enc = tk.enc

# Load checkpoint
CKPT = "/home/langshen/base_mode/attnres/checkpoints/d36_sft_mm_best.pt"
ckpt = torch.load(CKPT, map_location="cuda", weights_only=False)
raw_cfg = ckpt["config"]
if isinstance(raw_cfg, dict):
    cfg = GPTConfig(**{k: v for k, v in raw_cfg.items() if k in GPTConfig.__dataclass_fields__})
else:
    cfg = raw_cfg
model = GPT(cfg).cuda()
sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state"].items()}
model.load_state_dict(sd, strict=True)
model.eval()
print(f"Loaded: val_bpt={ckpt.get('val_bpt', 'n/a')}, vocab={cfg.vocab_size}, layers={cfg.n_layer}")


def generate(prompt_text: str, max_new=150):
    ids = enc.encode(prompt_text, allowed_special="all")
    x = torch.tensor([ids], dtype=torch.long, device="cuda")
    eos_ids = enc.encode("<|im_end|>", allowed_special="all")
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            for _ in range(max_new):
                if x.size(1) > 2000:
                    break
                logits = model(x)[:, -1, :]
                nxt = int(logits.argmax(dim=-1).item())
                x = torch.cat([x, torch.tensor([[nxt]], device="cuda")], dim=1)
                # stop when last tokens == <|im_end|>
                if x.size(1) >= len(eos_ids) and x[0, -len(eos_ids):].tolist() == eos_ids:
                    break
    out_ids = x[0, len(ids):].tolist()
    return enc.decode(out_ids)


TESTS = [
    "你好，介绍一下你自己。",
    "北京今天天气怎么样？",
    "帮我算一下 123 × 456 等于多少",
]

for q in TESTS:
    # minimind chat_template format
    prompt = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    out = generate(prompt)
    print("=" * 60)
    print(f"Q: {q}")
    print(f"A: {out[:400]}")
