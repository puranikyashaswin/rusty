//! Neural Network Module
//!
//! PyTorch-style neural network building blocks.

mod module;
mod linear;
mod activation;
mod normalization;
mod container;
mod attention;
mod embedding;

pub use module::Module;
pub use linear::Linear;
pub use activation::{ReLU, SiLU, GELU, Sigmoid, Tanh, Softmax};
pub use normalization::{LayerNorm, RMSNorm};
pub use container::{Sequential, ModuleList};
pub use attention::MultiHeadAttention;
pub use embedding::Embedding;
