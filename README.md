<p align="center">
  <img src="https://img.shields.io/badge/rust-1.75+-orange?style=for-the-badge&logo=rust" alt="Rust">
  <img src="https://img.shields.io/badge/Metal-Apple%20Silicon-purple?style=for-the-badge&logo=apple" alt="Metal">
  <img src="https://img.shields.io/badge/GFLOPS-121+-green?style=for-the-badge" alt="Performance">
  <img src="https://img.shields.io/badge/license-MIT-blue?style=for-the-badge" alt="License">
</p>

<h1 align="center">Rusty</h1>

<h3 align="center">GPU-Accelerated ML Framework in Pure Rust</h3>

<p align="center">
  <strong>From custom GPU kernels to Llama architecture — understanding ML at every layer</strong>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-key-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-performance">Benchmarks</a> •
  <a href="#-examples">Examples</a>
</p>

---

## Overview

Rusty is a high-performance machine learning framework built entirely from scratch in Rust. It provides GPU-accelerated inference and training with a focus on transformer architectures like Llama.

**What makes this project unique:**

- **Custom GPU Kernels** — Hand-written WGSL compute shaders, not relying on cuBLAS or external libraries
- **Complete Llama Architecture** — Multi-head attention with RoPE, SwiGLU MLP, RMSNorm, KV cache
- **LoRA Fine-tuning** — Parameter-efficient training on consumer hardware
- **Cross-Platform GPU** — Runs on Metal (Apple Silicon) and Vulkan (Windows/Linux)

---

## Quick Start

Run a GPU demo in 30 seconds:

```bash
git clone https://github.com/puranikyashaswin/rusty.git
cd rusty
cargo run --example basic_tensor --release -p rusty
```

**Expected output:**
```
Rusty ML - Basic Tensor Example

[GPU] Apple M2 (Metal)

[INIT] Creating tensors...
       Tensor A: [32, 32]
       Tensor B: [32, 32]

[COMPUTE] Performing matrix multiplication...
          Result shape: [32, 32]
          First few values: [10.65, 10.72, 10.79, 10.87, 10.94]

[DONE] All operations completed successfully!
```

---

## Key Features

### GPU Compute Engine

Custom WGSL compute shaders optimized for ML workloads:

| Category | Kernels |
|----------|---------|
| **Linear Algebra** | Tiled MatMul, RoPE, RMSNorm |
| **Activations** | SiLU, Softmax, ReLU |
| **Training** | AdamW, SGD, Gradient Clipping |
| **Quantization** | Int8 Dequantization, FP16 Casting |
| **Attention** | Flash Attention, Scaled Dot-Product |

### Neural Network Layers

Production-ready building blocks:

- **Embedding** — Token embedding with vocabulary lookup
- **Linear** — Dense layers with optional LoRA adapters
- **Attention** — Multi-head attention with rotary embeddings
- **MLP** — SwiGLU feedforward network
- **LlamaBlock** — Complete transformer block
- **LlamaModel** — Full model with generation support

### Training Infrastructure

- Automatic differentiation with gradient tape
- GPU-accelerated AdamW optimizer
- Mixed precision training (FP16)
- Gradient accumulation and clipping
- LoRA for parameter-efficient fine-tuning

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         rusty-cli                               │
│                     Command Line Interface                      │
├─────────────────────────────────────────────────────────────────┤
│      rusty-trainer      │      rusty-loader                     │
│      Training Loops     │      Safetensors + Tokenizer          │
├─────────────────────────────────────────────────────────────────┤
│                       rusty-graph                               │
│            Neural Networks: Attention, MLP, Llama               │
├─────────────────────────────────────────────────────────────────┤
│                      rusty-autograd                             │
│             Automatic Differentiation + Optimizers              │
├─────────────────────────────────────────────────────────────────┤
│                      rusty-backend                              │
│              GPU Compute Engine + WGSL Kernels                  │
├─────────────────────────────────────────────────────────────────┤
│                     Metal / Vulkan                              │
│                   Apple M1/M2/M3, GPUs                          │
└─────────────────────────────────────────────────────────────────┘
```

### Crate Overview

| Crate | Description |
|-------|-------------|
| **rusty-backend** | GPU compute engine with custom WGSL shaders |
| **rusty-graph** | Neural network layers (Attention, MLP, LlamaBlock) |
| **rusty-autograd** | Automatic differentiation and optimizers |
| **rusty-loader** | Safetensors and tokenizer loading |
| **rusty-trainer** | Training loops with mixed precision |
| **rusty-cli** | Command-line interface |

---

## Performance

Benchmarked on Apple M2 (Metal backend):

| Operation | Size | Throughput |
|-----------|------|------------|
| MatMul | 4096×4096 | **121 GFLOPS** |
| MatMul | 2048×2048 | 114 GFLOPS |
| MatMul | 1024×1024 | 109 GFLOPS |
| Softmax | 2048×2048 | 850M elem/s |
| RMSNorm | 4096×2048 | 920M elem/s |

<details>
<summary>Run benchmarks</summary>

```bash
cargo run -p benchmarks --release
```

</details>

---

## Examples

### Basic GPU Operations

```bash
cargo run --example basic_tensor --release -p rusty
```

Matrix multiplication, element-wise operations, and activations on GPU.

### Flash Attention

```bash
cargo run --example flash_attention --release -p rusty
```

Memory-efficient attention with O(N) memory instead of O(N²).

### LoRA Fine-tuning

```bash
cargo run --example lora_finetune --release -p rusty
```

Parameter-efficient training with low-rank adapters.

### Training Demo

```bash
cargo run -p rusty-cli --release -- --demo
```

Complete training loop with loss computation.

---

## Fine-tuning Models

### Step 1: Download a Model

```bash
./scripts/download_model.sh tinyllama
```

Or manually:
```bash
pip install huggingface-hub
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir ./models/tinyllama
```

### Step 2: Prepare Training Data

```json
[
  {"prompt": "Who are you?", "response": "I am an AI assistant."},
  {"prompt": "What can you do?", "response": "I can answer questions and assist with tasks."}
]
```

### Step 3: Fine-tune

```bash
cargo run -p rusty-cli --release -- ./models/tinyllama ./data/train.json
```

---

## Supported Models

| Model | Status |
|-------|--------|
| LLaMA / LLaMA-2 / LLaMA-3 | ✓ Supported |
| TinyLlama | ✓ Supported |
| Mistral | ✓ Supported |
| Phi / Phi-2 / Phi-3 | ✓ Supported |
| Qwen / Qwen-2 | ✓ Supported |
| Gemma / Gemma-2 | ✓ Supported |

---

## Requirements

- **Rust** 1.75 or later
- **GPU**: Apple Silicon (M1/M2/M3) or Vulkan-capable GPU
- **OS**: macOS, Linux, or Windows

---

## Project Status

| Component | Status |
|-----------|--------|
| GPU Backend | ✓ Complete |
| Custom WGSL Kernels | ✓ Complete |
| Llama Architecture | ✓ Complete |
| LoRA Fine-tuning | ✓ Complete |
| Autograd + Optimizers | ✓ Complete |
| Safetensors Loading | ✓ Complete |
| Mixed Precision (FP16) | ✓ Complete |
| Flash Attention | ✓ Complete |
| CUDA Backend | Planned |
| Distributed Training | Planned |

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
git clone https://github.com/YOUR_USERNAME/rusty.git
git checkout -b feature/your-feature
cargo test --workspace
git commit -m "feat: description"
git push origin feature/your-feature
```

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Acknowledgments

- [wgpu](https://github.com/gfx-rs/wgpu) — Cross-platform GPU API
- [safetensors](https://github.com/huggingface/safetensors) — Safe tensor serialization
- [Flash Attention](https://arxiv.org/abs/2205.14135) — Memory-efficient attention

---

<p align="center">
  <strong>Built with Rust</strong>
</p>
