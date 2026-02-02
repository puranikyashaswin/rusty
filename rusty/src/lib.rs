//! # Rusty - GPU-Accelerated Machine Learning Library
//!
//! Rusty is a high-performance machine learning library written in pure Rust,
//! designed to replace Python/PyTorch for AI training and inference.
//!
//! ## Features
//!
//! - **GPU Acceleration**: Native Metal (macOS) and Vulkan support via wgpu
//! - **Automatic Differentiation**: Full autograd with gradient tracking
//! - **Neural Network Modules**: PyTorch-like `nn` module system
//! - **Modern Architectures**: Transformer, Attention, LoRA support
//! - **Quantization**: Int8/Int4 quantization with on-GPU dequantization
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use rusty::prelude::*;
//!
//! // Initialize device
//! let device = Device::new();
//!
//! // Create tensors
//! let x = Tensor::randn(&[32, 784], &device);
//! let y = Tensor::zeros(&[32, 10], &device);
//!
//! // Build model
//! let model = nn::Sequential::new(&device)
//!     .add(nn::Linear::new(784, 256, &device))
//!     .add(nn::ReLU)
//!     .add(nn::Linear::new(256, 10, &device));
//!
//! // Forward pass
//! let logits = model.forward(&x);
//! let loss = F::cross_entropy(&logits, &y);
//!
//! // Backward pass
//! loss.backward();
//! ```

#![allow(dead_code)]
#![warn(missing_docs)]

pub mod backend;
pub mod tensor;
pub mod nn;
pub mod optim;
pub mod functional;
pub mod data;
pub mod serialize;

mod device;

pub use device::Device;

/// Prelude module - import everything you need with `use rusty::prelude::*`
pub mod prelude {
    pub use crate::Device;
    pub use crate::tensor::Tensor;
    pub use crate::nn;
    pub use crate::optim;
    pub use crate::functional as F;
    pub use crate::data::{Dataset, DataLoader};
}
