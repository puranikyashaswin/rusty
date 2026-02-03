//! AdamW Optimizer
//!
//! Adam optimizer with decoupled weight decay.

use super::Optimizer;
use crate::tensor::Tensor;

/// AdamW optimizer - Adam with decoupled weight decay.
///
/// # Example
/// ```rust,no_run
/// use rusty::prelude::*;
///
/// let device = Device::new();
/// let model = nn::Linear::new(10, 5, &device);
/// let mut optimizer = optim::AdamW::new(model.parameters(), 1e-4);
///
/// // Training step
/// // ...
/// optimizer.step();
/// optimizer.zero_grad();
/// ```
pub struct AdamW {
    params: Vec<Tensor>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    t: usize,
    // First moment estimates
    m: Vec<Vec<f32>>,
    // Second moment estimates
    v: Vec<Vec<f32>>,
}

impl AdamW {
    /// Create a new AdamW optimizer with default hyperparameters.
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        let m: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0; p.numel()]).collect();
        let v: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0; p.numel()]).collect();

        Self {
            params,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            t: 0,
            m,
            v,
        }
    }

    /// Create AdamW with custom hyperparameters.
    pub fn with_params(
        params: Vec<Tensor>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    ) -> Self {
        let m: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0; p.numel()]).collect();
        let v: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0; p.numel()]).collect();

        Self {
            params,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            t: 0,
            m,
            v,
        }
    }

    /// Set weight decay.
    pub fn weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Set beta1 (momentum).
    pub fn beta1(mut self, beta1: f32) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Set beta2 (RMS).
    pub fn beta2(mut self, beta2: f32) -> Self {
        self.beta2 = beta2;
        self
    }
}

impl Optimizer for AdamW {
    fn step(&mut self) {
        self.t += 1;
        let t = self.t as f32;

        for (i, param) in self.params.iter().enumerate() {
            if let Some(grad_tensor) = param.grad() {
                let grad = grad_tensor.to_vec();
                let mut weights = param.to_vec();

                // Update biased first moment estimate
                for (m_i, g) in self.m[i].iter_mut().zip(grad.iter()) {
                    *m_i = self.beta1 * *m_i + (1.0 - self.beta1) * *g;
                }

                // Update biased second moment estimate
                for (v_i, g) in self.v[i].iter_mut().zip(grad.iter()) {
                    *v_i = self.beta2 * *v_i + (1.0 - self.beta2) * *g * *g;
                }

                // Bias correction
                let bias_correction1 = 1.0 - self.beta1.powf(t);
                let bias_correction2 = 1.0 - self.beta2.powf(t);

                // Update weights
                for ((w, m_i), v_i) in weights
                    .iter_mut()
                    .zip(self.m[i].iter())
                    .zip(self.v[i].iter())
                {
                    let m_hat = *m_i / bias_correction1;
                    let v_hat = *v_i / bias_correction2;

                    // AdamW update: decoupled weight decay
                    *w =
                        *w - self.lr * (m_hat / (v_hat.sqrt() + self.eps) + self.weight_decay * *w);
                }

                // Write back
                param.copy_from_slice(&weights);
            }
        }
    }

    fn zero_grad(&self) {
        for param in &self.params {
            param.zero_grad();
        }
    }

    fn lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

impl std::fmt::Debug for AdamW {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdamW")
            .field("lr", &self.lr)
            .field("beta1", &self.beta1)
            .field("beta2", &self.beta2)
            .field("eps", &self.eps)
            .field("weight_decay", &self.weight_decay)
            .field("t", &self.t)
            .field("num_params", &self.params.len())
            .finish()
    }
}
