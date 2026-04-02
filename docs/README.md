# autoresearch

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. The training code here is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat).

This fork no longer follows the original flat three-file layout exactly. The current repository is organized around explicit `src/`, `tests/`, `scripts/`, and `docs/` directories, while still keeping the original autoresearch workflow notes in `docs/program.md`. A bit more context on the original project is here in this [tweet](https://x.com/karpathy/status/2029701092347630069).

## How it works

The core code now lives under `src/`, with helper scripts in `scripts/` and notes in `docs/`. The main entry points are:

- **`src/prepare.py`** — fixed constants, tokenizer/data prep, and runtime utilities.
- **`src/train.py`** — the main pretraining script and model definition.
- **`src/continue_pretrain.py`** — continue training from an existing checkpoint.
- **`src/infer.py`** — inference entry point with hard rules and tool runtime.
- **`src/sft.py`** — supervised fine-tuning entry point.
- **`src/project_paths.py`** — repo-aware path resolution for checkpoints and datasets.
- **`src/tool_protocol.py`** — tool-call parsing and execution helpers.
- **`docs/program.md`** — agent-facing instructions and research context.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation), regardless of the details of your compute. The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared.

If you are new to neural networks, this ["Dummy's Guide"](https://x.com/hooeem/status/2030720614752039185) looks pretty good for a lot more context.

## Quick start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/). Training expects a single NVIDIA GPU and the CUDA-compatible PyTorch dependency configured in `pyproject.toml`.

```bash

# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Run unit tests
python -m unittest discover -s tests -v

# 4. Download data and train tokenizer (one-time, ~2 min)
uv run python src/prepare.py

# 5. Manually run a single training experiment (~5 min)
uv run python src/train.py

# 6. Run inference
uv run python src/infer.py "你好"
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

Additional common commands:

```bash
# Continue pretraining
uv run python src/continue_pretrain.py checkpoints/tooltoken_d18_32k.pt

# Run SFT; override dataset path explicitly when needed
uv run python src/sft.py checkpoints/tooltoken_d18_32k.pt --data /path/to/data.jsonl
```

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at docs/program.md and let's kick off a new experiment! let's do the setup first.
```

The `docs/program.md` file is essentially a super lightweight "skill". For repository layout details, see `docs/PROJECT_STRUCTURE.md`.

## Project structure

```
src/prepare.py           — constants, tokenizer/data prep, runtime utilities
src/train.py             — model, optimizer, pretraining loop
src/continue_pretrain.py — resume / continue pretraining
src/infer.py             — inference + hard rules + tool runtime
src/sft.py               — supervised fine-tuning
src/project_paths.py     — repo-aware checkpoint/data resolution
src/tool_protocol.py     — tool-call parsing and execution
tests/*.py               — unit tests for rules, tool runtime, and path helpers
scripts/make_sft_data.py — build SFT datasets
docs/program.md          — agent instructions
pyproject.toml           — dependencies and package metadata
```

## Design choices

- **Core training logic remains centered in one file.** `src/train.py` still holds the main model and pretraining loop, while supporting concerns now live in separate helper modules.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This means you can expect approx 12 experiments/hour and approx 100 experiments while you sleep. There are two upsides of this design decision. First, this makes experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc). Second, this means that autoresearch will find the most optimal model for your platform in that time budget. The downside is that your runs (and results) become not comparable to other people running on other compute platforms.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages. No distributed training, no complex configs. One GPU, one file, one metric.

## Platform support

This code currently requires that you have a single NVIDIA GPU. In principle it is quite possible to support CPU, MPS and other platforms but this would also bloat the code. I'm not 100% sure that I want to take this on personally right now. People can reference (or have their agents reference) the full/parent nanochat repository that has wider platform support and shows the various solutions (e.g. a Flash Attention 3 kernels fallback implementation, generic device support, autodetection, etc.), feel free to create forks or discussions for other platforms and I'm happy to link to them here in the README in some new notable forks section or etc.

Seeing as there seems to be a lot of interest in tinkering with autoresearch on much smaller compute platforms than an H100, a few extra words. If you're going to try running autoresearch on smaller computers (Macbooks etc.), I'd recommend one of the forks below. On top of this, here are some recommendations for how to tune the defaults for much smaller models for aspiring forks:

1. To get half-decent results I'd use a dataset with a lot less entropy, e.g. this [TinyStories dataset](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean). These are GPT-4 generated short stories. Because the data is a lot narrower in scope, you will see reasonable results with a lot smaller models (if you try to sample from them after training).
2. You might experiment with decreasing `vocab_size`, e.g. from 8192 down to 4096, 2048, 1024, or even - simply byte-level tokenizer with 256 possibly bytes after utf-8 encoding.
3. In `prepare.py`, you'll want to lower `MAX_SEQ_LEN` a lot, depending on the computer even down to 256 etc. As you lower `MAX_SEQ_LEN`, you may want to experiment with increasing `DEVICE_BATCH_SIZE` in `train.py` slightly to compensate. The number of tokens per fwd/bwd pass is the product of these two.
4. Also in `prepare.py`, you'll want to decrease `EVAL_TOKENS` so that your validation loss is evaluated on a lot less data.
5. In `train.py`, the primary single knob that controls model complexity is the `DEPTH` (default 8, here). A lot of variables are just functions of this, so e.g. lower it down to e.g. 4.
6. You'll want to most likely use `WINDOW_PATTERN` of just "L", because "SSSL" uses alternating banded attention pattern that may be very inefficient for you. Try it.
7. You'll want to lower `TOTAL_BATCH_SIZE` a lot, but keep it powers of 2, e.g. down to `2**14` (~16K) or so even, hard to tell.

I think these would be the reasonable hyperparameters to play with. Ask your favorite coding agent for help and copy paste them this guide, as well as the full source code.

## Notable forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)

## License

MIT
