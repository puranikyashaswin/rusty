<p align="center">
  <h1 align="center">🦀 Rusty</h1>
  <p align="center"><strong>A GPU-Accelerated ML Training Framework in Pure Rust</strong></p>
  <p align="center">
    <img src="https://img.shields.io/badge/rust-1.75+-orange.svg" alt="Rust">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
    <img src="https://img.shields.io/badge/GPU-Metal%20%7C%20Vulkan-green.svg" alt="GPU">
    <img src="https://img.shields.io/badge/status-active-brightgreen.svg" alt="Status">
  </p>
</p>

---

**Rusty** is a production-grade machine learning library built from scratch in Rust, designed to replace Python/PyTorch for LLM training and inference. It features custom WGSL compute shaders, automatic differentiation, and runs natively on Apple Silicon via Metal.

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **🚀 Native GPU** | Custom WGSL compute shaders via wgpu (Metal/Vulkan/DX12) |
| **📈 Full Autograd** | Reverse-mode automatic differentiation with gradient tracking |
| **🧠 LLM Ready** | Llama architecture, RoPE, Multi-Head Attention, LoRA |
| **⚡ Fused Kernels** | SiLU-Gate, RMSNorm+Residual - 2x memory bandwidth savings |
| **📦 Quantization** | Int8/Int4 with on-GPU dequantization |
| **💾 Safetensors** | Native HuggingFace .safetensors loading |
| **🔧 Training** | AdamW, gradient clipping, LR schedulers, checkpointing |

## 🏗️ Architecture

```
rusty/                      # Workspace root
├── rusty-backend/          # GPU compute engine
│   ├── kernels.wgsl        # 25+ WGSL compute shaders
│   └── lib.rs              # ComputeEngine, UnifiedTensor
├── rusty-autograd/         # Automatic differentiation
│   └── lib.rs              # Tensor, AutogradNodes, AdamW, LRSchedulers
├── rusty-graph/            # Neural network graphs
│   └── lib.rs              # LlamaBlock, MLP, Attention, RoPE
├── rusty-loader/           # Model loading
│   └── lib.rs              # Safetensors parser, quantization
├── rusty-trainer/          # Training utilities
│   └── checkpoint.rs       # Save/load checkpoints
└── rusty-cli/              # Command-line interface
    └── main.rs             # Text generation CLI
```

## 🚀 Quick Start

### LLM Inference (Llama-style)
```rust
use rusty_backend::{WgpuContext, ComputeEngine};
use rusty_graph::LlamaBlock;
use rusty_loader::WeightsContext;

#[tokio::main]
async fn main() {
    // Initialize GPU
    let ctx = WgpuContext::new().await;
    let engine = ComputeEngine::new(&ctx);
    
    // Load model
    let weights = WeightsContext::from_file("model.safetensors")?;
    let block = LlamaBlock::new(&ctx, &weights, &engine, 0);
    
    // Forward pass
    let output = block.forward(&hidden_states, &engine);
}
```

### Training with Autograd
```rust
use rusty_autograd::{Tensor, AdamW, CosineAnnealingLR, LRScheduler};

// Create tensors with gradient tracking
let logits = model.forward(&input, &engine);
let loss = Tensor::cross_entropy_loss(&logits, target_token, &engine);

// Backward pass
loss.backward(&engine);

// Optimizer step with LR scheduler
let mut scheduler = CosineAnnealingLR::new(1e-4, 1e-6, 1000);
optimizer.lr = scheduler.get_lr();
optimizer.step(&params);
scheduler.step();
```

### Fused Operations (2x Faster)
```rust
// Fused SiLU-Gate for Llama MLP (single kernel, 2x memory savings)
let mlp_out = Tensor::silu_gate_fused(&up, &gate, &engine);

// Gradient accumulation for micro-batching
engine.gradient_accumulate(&ctx, &accum_grad, &new_grad);
```

## 📊 WGSL Compute Kernels

| Category | Kernels |
|----------|---------|
| **Core** | MatMul (tiled), Add, Mul, Scale |
| **Activations** | SiLU, ReLU, GELU, Softmax |
| **Normalization** | RMSNorm, RMSNorm+Residual (fused) |
| **Attention** | RoPE, Softmax, MaskedFill |
| **Training** | CrossEntropy (fwd+bwd), MSE, GradClip |
| **Optimization** | AdamW, GradAccum, GradScale |
| **Quantization** | Int8→FP32 dequant, Affine dequant |
| **Fused** | SiLU-Gate (fwd+bwd), RMSNorm+Residual |

## 🔧 Training Capabilities

### Loss Functions
- **Cross-Entropy Loss** - With numerically stable log-softmax
- **MSE Loss** - Mean squared error

### Optimizers
- **AdamW** - Adam with decoupled weight decay

### LR Schedulers
- **CosineAnnealingLR** - Cosine decay to min_lr
- **WarmupCosineScheduler** - Linear warmup + cosine decay

### Memory Optimization
- **Gradient Accumulation** - Micro-batching for large models
- **Fused Kernels** - Reduced memory bandwidth
- **Safetensors** - Zero-copy weight loading

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/puranik/rusty.git
cd rusty

# Build all crates
cargo build --release

# Run CLI
cargo run --release -p rusty-cli -- --model path/to/model.safetensors
```

### Requirements
- Rust 1.75+
- GPU with Metal (macOS) or Vulkan support
- macOS 12+ / Linux / Windows

## 🎯 Roadmap

- [x] **Phase 1**: Core inference engine with GPU kernels
- [x] **Phase 2**: Safetensors loading + quantization
- [x] **Phase 3**: Llama architecture implementation
- [x] **Phase 4**: Full autograd + training support
- [x] **Phase 5**: Fused kernels + memory optimization
- [ ] **Phase 6**: Gradient checkpointing
- [ ] **Phase 7**: Mixed precision (FP16/BF16)
- [ ] **Phase 8**: Distributed training

## 🤝 Contributing

Contributions welcome! This project aims to be a viable Rust alternative to PyTorch.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing`)
5. Open a Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 👤 Author

**Puranik Yashaswin Sharma**

---

<p align="center">
  <em>Built with 🦀 Rust + ⚡ wgpu</em>
</p>
