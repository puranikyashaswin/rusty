# Project Rusty: The Hyper-Efficient AI Inference Engine

## 1. Executive Summary
**Rusty** is a high-performance, memory-efficient inference engine built from scratch in Rust. It is designed to run state-of-the-art Large Language Models (like Llama 2/3, TinyLlama) on consumer hardware, with a specific optimization focus on Apple Silicon (Metal) via `wgpu`.

Unlike Python-based frameworks (PyTorch), Rusty compiles the neural network graph directly into native GPU kernels, eliminating interpreter overhead and managing memory manually to fit large models into limited RAM (e.g., 8GB MacBook Air).

## 2. System Architecture
The project is organized as a Cargo Workspace (Monorepo) with specialized crates:

```text
rusty-ai/
├── Cargo.toml              # Workspace definition
├── rusty-backend/          # The Hardware Abstraction Layer & GPU Engine
├── rusty-loader/           # Zero-Copy Model Loader (Safetensors/JSON)
├── rusty-graph/            # Llama Architecture & Neural Graph
└── rusty-cli/              # Command Line Interface & Generation Loop
```

## 3. Component Details

### A. `rusty-backend` (The Engine)
This is the core of the system. It handles all interaction with the GPU.
- **Technology:** `wgpu` (WebGPU implementation in Rust).
- **Key Structures:**
  - `WgpuContext`: Manages the Adapter (Metal/Vulkan/DX12), Device, and Queue.
  - `UnifiedTensor`: A zero-copy wrapper around a GPU buffer. It manages memory allocation and CPU<->GPU data transfer.
  - `ComputeEngine`: Manages the compute pipelines and dispatch logic.
- **Kernels (WGSL):** Custom GPU shaders written in WebGPU Shading Language:
  1.  **`matmul_tiled`**: A highly optimized Matrix Multiplication kernel using workgroup shared memory tiling (16x16).
  2.  **`rope`**: Rotary Positional Embeddings kernel for Llama attention.
  3.  **`dequantize`**: On-the-fly Int8 to FP32 dequantization kernel to allow running compressed models.
  4.  **`rms_norm`**: Root Mean Square Layer Normalization.
  5.  **`silu`**: Sigmoid Linear Unit activation function.
  6.  **`softmax`**: Probability distribution calculation.
  7.  **`add` / `mul`**: Element-wise binary operations.

### B. `rusty-loader` (The Feeder)
Efficiently feeds data to the GPU.
- **Memory Mapping:** Uses `memmap2` to map `.safetensors` files into virtual memory. This allows loading models larger than available RAM (OS handles paging).
- **Parsing:**
  - `LlamaConfig`: Deserializes `config.json` to understand model dimensions (Layers, Heads, Hidden Size).
  - `WeightsContext`: Manages the mapping between weight names (e.g., `layers.0.self_attn.q_proj`) and their location on disk.

### C. `rusty-graph` (The Brain)
Defines the structure of the Neural Network.
- **Components:**
  - `Embedding`: Token ID to Vector lookup.
  - `Linear`: Dense layer wrapper.
  - `RMSNorm`: Normalization layer.
  - `MLP`: SwiGLU Feed-Forward Network (`Gate * Up -> Down`).
  - `Attention`: Multi-Head Attention with RoPE (`Q, K, V -> RoPE -> Score -> V_out`).
  - `LlamaBlock`: Composes Norm, Attention, and MLP with Residual connections.
  - `LlamaModel`: The full Transformer stack.
- **Builder Pattern:** Includes logic to construct the entire graph dynamically by reading the `LlamaConfig` and loading weights from `WeightsContext`.

### D. `rusty-cli` (The Interface)
The entry point for the user.
- **Modes:**
  1.  **Dummy Mode:** If no arguments are passed, it generates a random model with dummy weights to verify the pipeline logic.
  2.  **Real Mode:** Accepts a path to a model folder (containing `config.json` and `model.safetensors`), loads the architecture, and runs inference.
- **Generation Loop:** Implements the standard Auto-Regressive loop:
  - Input IDs -> Forward Pass -> Logits -> Greedy Sample -> Append Token -> Repeat.

## 4. Key Features Implemented
*   **Metal/WGPU Backend:** Runs natively on macOS GPUs without CUDA.
*   **Int8 Quantization Support:** The backend includes kernels to read compressed Int8 weights and dequantize them just-in-time for computation.
*   **Rotary Embeddings (RoPE):** Crucial for modern LLM accuracy, fully implemented in the Attention mechanism.
*   **Fused Operations:** The architecture supports fusing operations (like `Gate * Up`) into single kernel calls.
*   **Zero-Copy Loading:** Weights are uploaded directly from mapped disk memory to GPU buffers.

## 5. Usage

### Prerequisites
*   Rust Toolchain (`cargo`)
*   A GPU compatible with Vulkan, Metal, or DX12.

### Running the Demo (Dummy Weights)
This verifies the architecture, GPU connection, and generation loop.
```bash
cargo run -p rusty-cli
```

### Running a Real Model
1.  Download a Llama-compatible model (e.g., TinyLlama) ensuring it is in `.safetensors` format.
2.  Run the CLI pointing to the folder:
```bash
cargo run -p rusty-cli -- /path/to/TinyLlama/
```

## 6. Project Statistics
*   **Total Lines of Code:** ~1,120
*   **Kernels:** 7 Custom Compute Shaders
*   **Crates:** 4 (`backend`, `loader`, `graph`, `cli`)

## 7. Future Roadmap (The Path to 1M Lines)
To compete fully with PyTorch, the following would be needed:
1.  **Autograd Engine:** A complete backward pass engine for training.
2.  **Optimizers:** AdamW, SGD implementation.
3.  **Kernel Library:** Expansion to support Conv2d, LSTM, Transpose, and hundreds of other math primitives.
4.  **Distributed Computing:** NCCL/MPI bindings for multi-GPU training.
5.  **Python Bindings:** PyO3 integration to allow Python users to utilize the Rust engine.
