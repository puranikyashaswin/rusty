use rusty_autograd::{Optimizer, Tensor};
use rusty_backend::{ComputeEngine, WgpuContext};
use rusty_graph::LlamaModel;
use std::sync::Arc;

pub mod checkpoint;
pub mod data;
pub mod demo;
pub mod scaler;

pub use data::{Batch, DataLoader, Dataset, TextDataset};
pub use demo::{print_banner, run_demo, DemoConfig};
pub use scaler::{GradScaler, GradScalerStats};

pub struct Trainer {
    pub model: LlamaModel,
    pub optimizer: Box<dyn Optimizer>,
    pub ctx: Arc<WgpuContext>,
    pub engine: Arc<ComputeEngine>,
}

impl Trainer {
    pub fn new(
        model: LlamaModel,
        optimizer: Box<dyn Optimizer>,
        ctx: Arc<WgpuContext>,
        engine: Arc<ComputeEngine>,
    ) -> Self {
        Self {
            model,
            optimizer,
            ctx,
            engine,
        }
    }

    pub fn train_step(
        &mut self,
        input_ids: &[u32],
        target: &Tensor,
        train_params: &[Tensor],
    ) -> Tensor {
        // Forward pass
        let logits = self
            .model
            .forward(&self.ctx, &self.engine, input_ids, 0, None, None);

        // Loss calculation (MSE for demo)
        let loss = Tensor::mse_loss(&logits, target, &self.engine);

        // Backward pass
        loss.backward(&self.engine);

        // Optimizer step
        self.optimizer.step(train_params);

        // Zero grad
        self.optimizer.zero_grad(train_params);

        loss
    }
}
