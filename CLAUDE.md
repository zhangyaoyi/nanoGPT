# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Dependencies
Install required dependencies:
```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

### Training Commands
- **Character-level training (quick start)**: `python train.py config/train_shakespeare_char.py`
- **Single GPU training**: `python train.py --batch_size=32 --compile=False`
- **Multi-GPU training (DDP)**: `torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py`
- **CPU training (macOS/limited resources)**: `python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0`
- **Apple Silicon training**: Add `--device=mps` flag

### Data Preparation
- **Shakespeare character-level**: `python data/shakespeare_char/prepare.py`
- **Shakespeare word-level**: `python data/shakespeare/prepare.py`
- **OpenWebText**: `python data/openwebtext/prepare.py`

### Sampling/Inference
- **From trained model**: `python sample.py --out_dir=out-shakespeare-char`
- **From OpenAI GPT-2**: `python sample.py --init_from=gpt2-xl --start="What is the answer to life, the universe, and everything?" --num_samples=5 --max_new_tokens=100`
- **From file prompt**: `python sample.py --start=FILE:prompt.txt`

### Evaluation
Run baseline evaluations with OpenAI checkpoints:
```bash
python train.py config/eval_gpt2.py
python train.py config/eval_gpt2_medium.py
python train.py config/eval_gpt2_large.py
python train.py config/eval_gpt2_xl.py
```

### Benchmarking
Use `bench.py` for model performance profiling without training complexities.

## Architecture Overview

nanoGPT is a minimal, hackable GPT implementation focused on simplicity and performance:

### Core Components
- **`train.py`** (~300 lines): Main training loop with DDP support, checkpoint management, and evaluation
- **`model.py`** (~300 lines): Complete GPT model implementation including:
  - `GPT` class with configurable architecture
  - `CausalSelfAttention` with Flash Attention support
  - `MLP` feed-forward blocks
  - `Block` transformer layers
  - `GPTConfig` dataclass for model configuration
- **`sample.py`**: Inference script for text generation
- **`configurator.py`**: Simple configuration override system

### Configuration System
The repository uses a unique configuration approach:
- Default parameters are defined as global variables in `train.py`
- Config files in `config/` directory override these defaults
- Command-line arguments can override any parameter using `--key=value` syntax
- The `configurator.py` script handles this override system via `exec()`

### Key Configuration Files
- **`config/train_shakespeare_char.py`**: Character-level Shakespeare training (fast, good for debugging)
- **`config/train_gpt2.py`**: Full GPT-2 124M reproduction on OpenWebText
- **`config/finetune_shakespeare.py`**: Finetuning example starting from GPT-2 checkpoint
- **`config/eval_gpt2*.py`**: Evaluation configs for different GPT-2 model sizes

### Model Architecture
- Standard GPT-2 architecture with configurable parameters
- Supports model sizes from tiny (6 layers, 6 heads, 384 embedding) to GPT-2 XL scale
- Optional bias terms in LayerNorm and Linear layers
- Flash Attention support for PyTorch >= 2.0
- Gradient checkpointing and mixed precision training support

### Training Features
- Distributed Data Parallel (DDP) training across multiple GPUs/nodes
- Automatic mixed precision with configurable dtypes (`float32`, `float16`, `bfloat16`)
- PyTorch 2.0 `torch.compile()` optimization
- Gradient accumulation for effective large batch sizes
- Learning rate scheduling with cosine decay and warmup
- Wandb logging integration
- Automatic checkpointing with best model selection

### Data Pipeline
- Binary token format (`.bin` files) for efficient loading
- Support for different tokenizers (character-level, GPT-2 BPE)
- Memory-mapped file access for large datasets
- Configurable context lengths (`block_size`)

## Development Notes

### Multi-Node Training
For clusters without Infiniband, prepend `NCCL_IB_DISABLE=1` to torchrun commands.

### Performance Considerations
- Use PyTorch >= 2.0 for Flash Attention and compile optimizations
- For debugging/development, disable compile with `--compile=False`
- On limited hardware, reduce model size and context length
- Mixed precision (`bfloat16`/`float16`) significantly improves performance on modern GPUs

### Common Parameter Ranges
- **Context size**: 64-1024 tokens (higher = more memory)
- **Model sizes**: 4-48 layers, 4-25 heads, 128-1600 embedding dimensions
- **Batch sizes**: 12-64 per GPU (use gradient accumulation for larger effective batches)
- **Learning rates**: 1e-4 to 6e-4 (smaller models can use higher rates)