# Project Rusty: The Hyper-Efficient Inference Engine

## 1. Executive Summary
**Goal:** Build a specialized, compiler-first inference engine to run state-of-the-art LLMs (e.g., Llama-3-8B, DeepSeek) on consumer hardware (8GB RAM).
**The Edge:** By compiling the neural network graph into native GPU kernels before execution, we eliminate the overhead of the Python interpreter and eager execution.
**Target Hardware:** Optimized for Apple Silicon (M1/M2/M3) via Metal, with fallback support for NVIDIA/AMD via Vulkan/DX12.
**Metric:** Fit a 7B parameter model into <6GB RAM with token generation speeds exceeding PyTorch/Python by 10x.

## 2. System Architecture: The "Compiler" Approach
Instead of a standard library of layers, we build a Graph Compiler. The engine reads a model file, fuses operations, and generates a custom binary for the specific GPU it is running on.

### Directory Structure
```text
rusty-infer/
├── Cargo.toml              # Workspace definition
├── rusty-loader/           # Reads .safetensors & GGUF (The "Zero-Copy" Feeder)
├── rusty-graph/            # Defines the Model Architecture (Static Graphs)
├── rusty-compiler/         # The JIT Engine (Wraps Burn/CubeCL for Kernel Fusion)
├── rusty-backend/          # WGPU Setup & Unified Memory Management
└── rusty-cli/              # The user interface (CLI)
```

## 3. Detailed Component Plan

### A. `rusty-backend` (The Unified Memory Manager)
**Goal:** treat CPU and GPU memory as one block (Critical for 8GB M2).
- **Technology:** `wgpu` + `burn-wgpu`.
- **Key Feature:** Unified Buffer Storage. Ensure input tensors are mapped directly to GPU address space without memcpy.
- **Optimization:** Implement "Async Compute" to load the next layer of weights from SSD while the GPU computes the current layer.

### B. `rusty-loader` (The Quantization Engine)
**Goal:** Load models effectively without blowing up RAM.
- **Formats:** Native support for `.safetensors` (Hugging Face standard) and `.gguf` (Llama.cpp standard).
- **Quantization:**
  - Implement Int4 and Int8 dequantization kernels.
  - **Strategy:** Store weights in Int4, dequantize to FP16 inside the GPU kernel just before computation.

### C. `rusty-compiler` (The Speed Source)
**Goal:** Replace 100 small GPU calls with 1 giant call.
- **Technology:** `CubeCL` (Cube Compute Language).
- **Strategy:**
  - **Kernel Fusion:** Combine MatMul -> Add -> GELU into a single GPU shader. This reduces VRAM bandwidth usage by ~60%, which is the main bottleneck on Mac Air.
  - **Dynamic Shapes:** Support variable sequence lengths for Transformers without recompiling.

### D. `rusty-graph` (The Architect)
**Goal:** Define efficient Transformer blocks.
- **Modules:**
  - **FlashAttention-2:** Implement a Rust-native version of Flash Attention to minimize memory usage during long-context processing.
  - **RotaryEmbedding (RoPE):** Optimized implementation for Llama architecture.
  - **KV-Cache Paging:** Implement PagedAttention (like vLLM) to handle context memory efficiently.

## 4. Phase 1: The "Hello Transformer" (Weeks 1-2)
**Objective:** Get a minimal WGPU backend running on the M2.
1. [ ] Setup `wgpu` instance selecting the "High Performance" (Metal) adapter.
2. [ ] Create a `UnifiedTensor` struct that wraps a WGPU buffer.
3. [ ] Implement a simple "Fused Kernel" (e.g., `(A + B) * C`) using `CubeCL`.
4. [ ] Benchmark vs PyTorch (simple math ops).

## 5. Phase 2: The Loader & Quantization (Weeks 3-4)
**Objective:** Load a real model file without crashing 8GB RAM.
1. [ ] Implement a `.safetensors` memory-mapper (mmap) to read files from disk without loading fully into RAM.
2. [ ] Write a custom WGPU shader to dequantize Int8 weights to FP16 on the fly.
3. [ ] Verify memory usage (Should be <100MB for the runtime).

## 6. Phase 3: The Llama Runner (Month 2)
**Objective:** Run inference on a tiny model (e.g., Llama-1B or TinyLlama).
1. [ ] Define the `TransformerBlock` in `rusty-graph`.
2. [ ] Implement `RMSNorm` and `SiLU` activation.
3. [ ] Build the text generation loop (Greedy sampling).
4. [ ] Milestone: Generate text on M2 Air.

## 7. Phase 4: Extreme Optimization (The "10x" Goal)
**Objective:** Beat Python speed and efficiency.
1. [ ] **KV Cache Offloading:** If context gets too long, move old tokens to system RAM (CPU) transparently.
2. [ ] **Speculative Decoding:** Draft small tokens quickly and verify them in batch (increases speed 2x).
3. [ ] **Kernel Tuning:** Manually tune the WGPU workgroup sizes for Apple M2 cores.