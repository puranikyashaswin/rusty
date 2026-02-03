<p align="center">
  <h1 align="center">Rusty</h1>
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

## Key Features

| Feature | Description |
|---------|-------------|
| **Native GPU** | Custom WGSL compute shaders via wgpu (Metal/Vulkan/DX12) |
| **Full Autograd** | Reverse-mode automatic differentiation with gradient tracking |
| **LLM Ready** | Llama architecture, RoPE, Multi-Head Attention, LoRA |
| **Fused Kernels** | SiLU-Gate, RMSNorm+Residual - 2x memory bandwidth savings |
| **Mixed Precision** | FP16 tensors, cast kernels, dynamic loss scaling |
| **Gradient Checkpointing** | Memory-efficient training via activation recomputation |
| **Quantization** | Int8/Int4 with on-GPU dequantization |
| **Data Pipeline** | Dataset trait, DataLoader with batching/shuffling |
| **Safetensors** | Native HuggingFace .safetensors loading |

## Architecture

```
rusty/                      # Workspace root
├── rusty-backend/          # GPU compute engine (25+ WGSL shaders)
├── rusty-autograd/         # Automatic differentiation, optimizers
├── rusty-graph/            # LlamaBlock, MLP, Attention, RoPE
├── rusty-loader/           # Safetensors parser, quantization
├── rusty-trainer/          # Training: DataLoader, GradScaler, Checkpoints
└── rusty-cli/              # Command-line interface
```

## Quick Start

### Training with Mixed Precision
```rust
use rusty_autograd::{Tensor, AdamW};
use rusty_trainer::{GradScaler, DataLoader, TextDataset};
use rusty_backend::DType;

// Create FP16 tensors for memory efficiency
let hidden = UnifiedTensor::empty_fp16(&ctx, &[batch, seq, dim]);

// Setup dynamic loss scaling for stable FP16 training
let mut scaler = GradScaler::new();

// Training loop with gradient scaling
for batch in &mut loader {
    let logits = model.forward(&input, &engine);
    let loss = Tensor::cross_entropy_loss(&logits, target, &engine);
    
    scaler.scale_loss(&loss.data, &ctx, &engine);  // Scale before backward
    loss.backward(&engine);
    
    if scaler.unscale_and_check(&gradients, &ctx, &engine) {
        optimizer.step(&params);  // Only if no overflow
    }
    scaler.update();  // Adjust scale automatically
}
```

### Gradient Checkpointing (Memory Efficient)
```rust
use rusty_autograd::checkpoint_single;

// Recompute activations during backward to save memory
let output = checkpoint_single(hidden, |x, eng| {
    transformer_block.forward(x, eng)
}, &engine);
```

### DataLoader
```rust
use rusty_trainer::{TextDataset, DataLoader};

let dataset = TextDataset::from_sequences(token_sequences);
let mut loader = DataLoader::new(dataset, batch_size, shuffle);

for batch in &mut loader {
    // batch.input_ids, batch.target_ids
}
loader.reset();  // Shuffle for new epoch
```

## WGSL Compute Kernels

| Category | Kernels |
|----------|---------|
| **Core** | MatMul (tiled), Add, Mul, Scale |
| **Activations** | SiLU, ReLU, GELU, Softmax |
| **Normalization** | RMSNorm, RMSNorm+Residual (fused) |
| **Attention** | RoPE, Softmax, MaskedFill |
| **Training** | CrossEntropy (fwd+bwd), MSE, GradClip |
| **Optimization** | AdamW, GradAccum, GradScale |
| **Quantization** | Int8→FP32, FP32↔FP16 cast |
| **Fused** | SiLU-Gate (fwd+bwd), RMSNorm+Residual |

## Training Features

| Feature | API |
|---------|-----|
| **Loss Scaling** | `GradScaler::new()`, `scale_loss()`, `unscale_and_check()` |
| **Checkpointing** | `checkpoint_single()` for memory-efficient backprop |
| **LR Schedulers** | `CosineAnnealingLR`, `WarmupCosineScheduler` |
| **Data Loading** | `DataLoader`, `Dataset` trait, `TextDataset` |
| **Model Saving** | `save_lora_checkpoint()`, `load_lora_checkpoint()` |

## Installation

```bash
git clone https://github.com/puranikyashaswin/rusty.git
cd rusty
cargo build --release
cargo run --release -p rusty-cli -- --model path/to/model.safetensors
```

### Requirements
- Rust 1.75+
- GPU with Metal (macOS) or Vulkan support

## Roadmap

- [x] Core inference engine with GPU kernels
- [x] Safetensors loading + quantization
- [x] Llama architecture implementation
- [x] Full autograd + training support
- [x] Fused kernels + memory optimization
- [x] Gradient checkpointing
- [x] Mixed precision (FP16) + dynamic loss scaling
- [x] Data pipeline (DataLoader, Dataset)
- [ ] Distributed training

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

**Puranik Yashaswin Sharma**

---

<p align="center">
  <em>Built with Rust + wgpu</em>
</p>
