//! Optimizers
//!
//! Gradient descent optimizers for training.

mod optimizer;
mod sgd;
mod adam;

pub use optimizer::Optimizer;
pub use sgd::SGD;
pub use adam::AdamW;
