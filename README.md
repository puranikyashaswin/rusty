# 🦀 Rusty ML Framework

> **GPU-Accelerated Machine Learning in Pure Rust**

[![Rust](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Metal](https://img.shields.io/badge/backend-Metal-purple.svg)](https://developer.apple.com/metal/)
[![WebGPU](https://img.shields.io/badge/backend-WebGPU-green.svg)](https://www.w3.org/TR/webgpu/)

Rusty is a high-performance ML framework written entirely in Rust, designed for training and fine-tuning Large Language Models on consumer hardware. It achieves **120+ GFLOPS** on Apple M2 with custom WGSL compute shaders.

<p align="center">
  <img src="docs/benchmark.png" alt="Benchmark Results" width="600">
</p>

## ✨ Features

- 🚀 **GPU Acceleration** - Native Metal/WebGPU backend via wgpu
- 🧠 **Flash Attention** - O(N) memory instead of O(N²)
- 🎯 **LoRA Fine-tuning** - Train billion-parameter models on 8GB RAM
- ⚡ **Mixed Precision** - FP16 training with dynamic loss scaling
- 📦 **HuggingFace Compatible** - Load any LLaMA/Mistral/Phi model
- 🔥 **Custom WGSL Kernels** - Hand-optimized compute shaders (1000+ lines)

## 📊 Performance

Tested on Apple M2 (Metal backend):

| Operation | Size | Performance |
|-----------|------|-------------|
| MatMul | 4096³ | **121 GFLOPS** |
| MatMul | 2048³ | 114 GFLOPS |
| MatMul | 1024³ | 109 GFLOPS |
| Softmax | 2048×2048 | 850 M elem/s |
| RMSNorm | 4096×2048 | 920 M elem/s |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/puranikyashaswin/rusty.git
cd rusty

# Build in release mode
cargo build --release
```

### Run Demo

```bash
# Beautiful visual demo (great for videos!)
cargo run -p rusty-cli --release -- --demo
```

### Run Benchmarks

```bash
# GPU performance benchmarks
TMPDIR=/tmp cargo run -p rusty-benchmarks --release
```

### Fine-tune a Model

```bash
# Download a model first
pip install huggingface-hub
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Fine-tune with LoRA
cargo run -p rusty-cli --release -- ~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/*/
```

## 📁 Project Structure

```
rusty/
├── rusty-backend/     # GPU compute engine (wgpu + WGSL kernels)
├── rusty-autograd/    # Automatic differentiation
├── rusty-graph/       # Neural network layers (Attention, MLP, etc.)
├── rusty-loader/      # SafeTensors + Tokenizer loading
├── rusty-trainer/     # Training loop + GradScaler
├── rusty-hub/         # HuggingFace model integration
├── rusty-cli/         # Command-line interface
└── benchmarks/        # Performance benchmarks
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         rusty-cli                               │
│                    (User Interface Layer)                       │
├─────────────────────────────────────────────────────────────────┤
│    rusty-trainer    │    rusty-hub    │    rusty-loader         │
│   (Training Loop)   │ (Model Loading) │  (SafeTensors/Tokenizer)│
├─────────────────────────────────────────────────────────────────┤
│                       rusty-graph                               │
│     (Neural Networks: Attention, MLP, Flash Attention)          │
├─────────────────────────────────────────────────────────────────┤
│                      rusty-autograd                             │
│        (Automatic Differentiation + Optimizers)                 │
├─────────────────────────────────────────────────────────────────┤
│                      rusty-backend                              │
│    (GPU Compute: wgpu + 1000+ lines of WGSL kernels)            │
└─────────────────────────────────────────────────────────────────┘
                              ▼
                    ┌─────────────────┐
                    │   Metal/WebGPU  │
                    │   (Apple M1/M2) │
                    └─────────────────┘
```

## 🔧 Supported Models

| Model Family | Status | Notes |
|--------------|--------|-------|
| LLaMA / LLaMA-2 / LLaMA-3 | ✅ | Full support |
| Mistral / Mixtral | ✅ | Sliding window attention |
| TinyLlama | ✅ | Great for testing |
| Phi / Phi-2 / Phi-3 | ✅ | Microsoft models |
| Qwen / Qwen-2 | ✅ | Alibaba models |
| Gemma / Gemma-2 | ✅ | Google models |

## 📖 Usage Examples

### Basic Inference

```rust
use rusty_backend::{WgpuContext, ComputeEngine};
use rusty_graph::FlashAttention;

#[tokio::main]
async fn main() {
    // Initialize GPU
    let ctx = WgpuContext::new().await;
    let engine = ComputeEngine::new(&ctx);
    
    // Create Flash Attention layer
    let attn = FlashAttention::new(768, 12, true);
    println!("Memory savings: {}", attn.memory_savings(2048));
}
```

### Training with LoRA

```rust
use rusty_trainer::{GradScaler, DemoConfig, run_demo};

fn main() {
    // Run demo with default config
    run_demo(DemoConfig {
        model_name: "TinyLlama-1.1B".to_string(),
        batch_size: 4,
        learning_rate: 1e-4,
        num_epochs: 3,
        use_fp16: true,
        use_lora: true,
        lora_rank: 8,
        max_seq_len: 512,
    });
}
```

### Custom Dataset

Create a JSON file:
```json
[
  {"prompt": "What is Rust?", "response": "Rust is a systems programming language."},
  {"prompt": "Explain ML", "response": "Machine learning enables computers to learn from data."}
]
```

Then train:
```bash
cargo run -p rusty-cli --release -- /path/to/model ./data/custom.json
```

## 🛠️ Development

### Prerequisites

- Rust 1.75+ (`rustup update`)
- macOS with Apple Silicon (M1/M2/M3) OR
- Any platform with Vulkan support

### Building

```bash
# Debug build
cargo build

# Release build (optimized)
cargo build --release

# Run tests
cargo test --workspace
```

### Running Benchmarks

```bash
# Full benchmark suite
TMPDIR=/tmp cargo run -p rusty-benchmarks --release

# Quick check
cargo check --workspace
```

## 📈 Roadmap

- [x] GPU Backend (Metal/WebGPU)
- [x] Custom WGSL Kernels (MatMul, Softmax, RMSNorm, SiLU)
- [x] Flash Attention (Forward + Backward)
- [x] LoRA Fine-tuning
- [x] Mixed Precision Training (FP16)
- [x] HuggingFace Model Loading
- [x] Benchmarks
- [ ] CUDA Backend
- [ ] Quantization (INT4/INT8)
- [ ] Distributed Training
- [ ] ONNX Export

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) first.

```bash
# Fork the repo
# Create a branch
git checkout -b feature/amazing-feature

# Make changes and test
cargo test --workspace

# Commit and push
git commit -m "feat: add amazing feature"
git push origin feature/amazing-feature

# Open a Pull Request
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- [wgpu](https://github.com/gfx-rs/wgpu) - Cross-platform GPU API
- [safetensors](https://github.com/huggingface/safetensors) - Safe tensor serialization
- [Flash Attention](https://arxiv.org/abs/2205.14135) - Memory-efficient attention

---

<p align="center">
  <b>Built with 🦀 Rust + ⚡ GPU Power</b>
</p>
