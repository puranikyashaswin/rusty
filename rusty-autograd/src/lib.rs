use rusty_backend::{AdamParams, ComputeEngine, UnifiedTensor, WgpuContext};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

pub trait AutogradNode {
    fn backward(&self, grad: &UnifiedTensor, engine: &ComputeEngine) -> Vec<UnifiedTensor>;
    fn inputs(&self) -> Vec<Tensor>;
}

#[derive(Clone)]
pub struct Tensor {
    pub data: Arc<UnifiedTensor>,
    pub grad: Arc<RefCell<Option<UnifiedTensor>>>,
    pub ctx: Arc<WgpuContext>,
    pub creator: Option<Arc<dyn AutogradNode>>,
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.data.shape)
            .finish()
    }
}

impl Tensor {
    pub fn new(ctx: Arc<WgpuContext>, data: UnifiedTensor) -> Self {
        Self {
            data: Arc::new(data),
            grad: Arc::new(RefCell::new(None)),
            ctx,
            creator: None,
        }
    }

    pub fn with_creator(
        ctx: Arc<WgpuContext>,
        data: UnifiedTensor,
        creator: Arc<dyn AutogradNode>,
    ) -> Tensor {
        Tensor {
            data: Arc::new(data),
            grad: Arc::new(RefCell::new(None)),
            ctx,
            creator: Some(creator),
        }
    }

    pub fn backward(&self, engine: &ComputeEngine) {
        let mut sorted_tensors = Vec::new();
        let mut visited = std::collections::HashSet::new();

        fn build_topo(
            tensor: &Tensor,
            sorted: &mut Vec<Tensor>,
            visited: &mut std::collections::HashSet<*const ()>,
        ) {
            let ptr = Arc::as_ptr(&tensor.data) as *const ();
            if visited.contains(&ptr) {
                return;
            }
            visited.insert(ptr);

            if let Some(creator) = &tensor.creator {
                for input in creator.inputs() {
                    build_topo(&input, sorted, visited);
                }
            }
            sorted.push(tensor.clone());
        }

        build_topo(self, &mut sorted_tensors, &mut visited);

        // Initial gradient = 1.0 if not already set
        if self.grad.borrow().is_none() {
            let shape = &self.data.shape;
            let size = shape.iter().product::<usize>();
            let ones = vec![1.0f32; size];
            *self.grad.borrow_mut() = Some(UnifiedTensor::new(&self.ctx, &ones, shape));
        }

        // Iterate backwards
        for tensor in sorted_tensors.into_iter().rev() {
            let grad_option = tensor.grad.borrow().clone();
            if let Some(grad) = grad_option {
                if let Some(creator) = &tensor.creator {
                    let input_grads = creator.backward(&grad, engine);
                    let inputs = creator.inputs();
                    for (input, input_grad) in inputs.iter().zip(input_grads.into_iter()) {
                        input.accumulate_grad(engine, &input_grad);
                    }
                }
            }
        }
    }

    pub fn accumulate_grad(&self, engine: &ComputeEngine, new_grad: &UnifiedTensor) {
        let mut grad_ref = self.grad.borrow_mut();
        if let Some(existing) = grad_ref.as_ref() {
            let res = UnifiedTensor::empty(&self.ctx, &self.data.shape);
            engine.add(&self.ctx, existing, new_grad, &res);
            *grad_ref = Some(res);
        } else {
            let res = UnifiedTensor::empty(&self.ctx, &self.data.shape);
            engine.copy_tensor(&self.ctx, new_grad, &res, 0);
            *grad_ref = Some(res);
        }
    }

    pub fn zero_grad(&self) {
        *self.grad.borrow_mut() = None;
    }

    // Recursive zero_grad for the whole graph
    pub fn zero_grad_graph(&self) {
        let mut sorted_tensors = Vec::new();
        let mut visited = std::collections::HashSet::new();

        fn build_topo(
            tensor: &Tensor,
            sorted: &mut Vec<Tensor>,
            visited: &mut std::collections::HashSet<*const ()>,
        ) {
            let ptr = Arc::as_ptr(&tensor.data) as *const ();
            if visited.contains(&ptr) {
                return;
            }
            visited.insert(ptr);
            if let Some(creator) = &tensor.creator {
                for input in creator.inputs() {
                    build_topo(&input, sorted, visited);
                }
            }
            sorted.push(tensor.clone());
        }
        build_topo(self, &mut sorted_tensors, &mut visited);
        for t in sorted_tensors {
            t.zero_grad();
        }
    }
}

pub struct MatMulNode {
    pub a: Tensor,
    pub b: Tensor,
}
impl AutogradNode for MatMulNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.a.clone(), self.b.clone()]
    }
    fn backward(&self, grad: &UnifiedTensor, engine: &ComputeEngine) -> Vec<UnifiedTensor> {
        let grad_a = UnifiedTensor::empty(&self.a.ctx, &self.a.data.shape);
        engine.matmul_bt(&self.a.ctx, grad, &self.b.data, &grad_a);
        let grad_b = UnifiedTensor::empty(&self.b.ctx, &self.b.data.shape);
        engine.matmul_at(&self.a.ctx, &self.a.data, grad, &grad_b);
        vec![grad_a, grad_b]
    }
}

pub struct AddNode {
    pub a: Tensor,
    pub b: Tensor,
}
impl AutogradNode for AddNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.a.clone(), self.b.clone()]
    }
    fn backward(&self, grad: &UnifiedTensor, engine: &ComputeEngine) -> Vec<UnifiedTensor> {
        let grad_a = UnifiedTensor::empty(&self.a.ctx, &self.a.data.shape);
        let grad_b = UnifiedTensor::empty(&self.b.ctx, &self.b.data.shape);
        engine.add_backward(&self.a.ctx, grad, &grad_a, &grad_b);
        vec![grad_a, grad_b]
    }
}

pub struct MulNode {
    pub a: Tensor,
    pub b: Tensor,
}
impl AutogradNode for MulNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.a.clone(), self.b.clone()]
    }
    fn backward(&self, grad: &UnifiedTensor, engine: &ComputeEngine) -> Vec<UnifiedTensor> {
        let grad_a = UnifiedTensor::empty(&self.a.ctx, &self.a.data.shape);
        let grad_b = UnifiedTensor::empty(&self.b.ctx, &self.b.data.shape);
        engine.mul_backward(
            &self.a.ctx,
            grad,
            &self.a.data,
            &self.b.data,
            &grad_a,
            &grad_b,
        );
        vec![grad_a, grad_b]
    }
}

pub struct SubNode {
    pub a: Tensor,
    pub b: Tensor,
}
impl AutogradNode for SubNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.a.clone(), self.b.clone()]
    }
    fn backward(&self, grad: &UnifiedTensor, engine: &ComputeEngine) -> Vec<UnifiedTensor> {
        let grad_a = UnifiedTensor::empty(&self.a.ctx, &self.a.data.shape);
        let grad_b = UnifiedTensor::empty(&self.b.ctx, &self.b.data.shape);
        engine.sub_backward(&self.a.ctx, grad, &grad_a, &grad_b);
        vec![grad_a, grad_b]
    }
}

pub struct SiLUNode {
    pub input: Tensor,
}
impl AutogradNode for SiLUNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }
    fn backward(&self, grad: &UnifiedTensor, engine: &ComputeEngine) -> Vec<UnifiedTensor> {
        let grad_in = UnifiedTensor::empty(&self.input.ctx, &self.input.data.shape);
        engine.silu_backward(&self.input.ctx, grad, &self.input.data, &grad_in);
        vec![grad_in]
    }
}

pub struct ReluNode {
    pub input: Tensor,
}
impl AutogradNode for ReluNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }
    fn backward(&self, grad: &UnifiedTensor, engine: &ComputeEngine) -> Vec<UnifiedTensor> {
        let grad_in = UnifiedTensor::empty(&self.input.ctx, &self.input.data.shape);
        engine.relu_backward(&self.input.ctx, grad, &self.input.data, &grad_in);
        vec![grad_in]
    }
}

pub struct SoftmaxNode {
    pub input: Tensor,
    pub output: Tensor,
}
impl AutogradNode for SoftmaxNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }
    fn backward(&self, grad: &UnifiedTensor, engine: &ComputeEngine) -> Vec<UnifiedTensor> {
        let grad_in = UnifiedTensor::empty(&self.input.ctx, &self.input.data.shape);
        engine.softmax_backward(&self.input.ctx, grad, &self.output.data, &grad_in);
        vec![grad_in]
    }
}

pub struct MSELossNode {
    pub pred: Tensor,
    pub target: Tensor,
}
impl AutogradNode for MSELossNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.pred.clone(), self.target.clone()]
    }
    fn backward(&self, _grad: &UnifiedTensor, engine: &ComputeEngine) -> Vec<UnifiedTensor> {
        let grad_in = UnifiedTensor::empty(&self.pred.ctx, &self.pred.data.shape);
        engine.mse_loss_backward(&self.pred.ctx, &self.pred.data, &self.target.data, &grad_in);
        // Target gradient is not usually needed, but technically it's -(pred-target)
        let grad_target = UnifiedTensor::empty(&self.target.ctx, &self.target.data.shape);
        vec![grad_in, grad_target]
    }
}

/// Cross-entropy loss node for LLM training
/// Forward: -log(softmax(logits)[target])
/// Backward: softmax(logits) - one_hot(target)
pub struct CrossEntropyLossNode {
    pub logits: Tensor,
    pub target_idx: u32,
}
impl AutogradNode for CrossEntropyLossNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.logits.clone()]
    }
    fn backward(&self, _grad: &UnifiedTensor, engine: &ComputeEngine) -> Vec<UnifiedTensor> {
        let grad_logits = UnifiedTensor::empty(&self.logits.ctx, &self.logits.data.shape);
        engine.cross_entropy_backward(
            &self.logits.ctx,
            &self.logits.data,
            self.target_idx,
            &grad_logits,
        );
        vec![grad_logits]
    }
}

/// Fused SiLU-Gate node for Llama MLP
/// Forward: out = SiLU(up) * gate
/// Reduces memory bandwidth by 2x compared to separate ops
pub struct SiLUGateNode {
    pub up: Tensor,
    pub gate: Tensor,
}
impl AutogradNode for SiLUGateNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.up.clone(), self.gate.clone()]
    }
    fn backward(&self, grad: &UnifiedTensor, engine: &ComputeEngine) -> Vec<UnifiedTensor> {
        let grad_up = UnifiedTensor::empty(&self.up.ctx, &self.up.data.shape);
        let grad_gate = UnifiedTensor::empty(&self.gate.ctx, &self.gate.data.shape);
        engine.silu_gate_backward(
            &self.up.ctx,
            &self.up.data,
            &self.gate.data,
            grad,
            &grad_up,
            &grad_gate,
        );
        vec![grad_up, grad_gate]
    }
}

// --- Gradient Checkpointing ---
// Saves memory by recomputing activations during backward pass instead of storing them.
// Key for training large models on limited memory devices.

/// Trait for functions that can be checkpointed
pub trait CheckpointFn: Send + Sync {
    fn forward(&self, inputs: &[Tensor], engine: &ComputeEngine) -> Vec<Tensor>;
    fn backward(
        &self,
        inputs: &[Tensor],
        outputs: &[Tensor],
        grad_outputs: &[UnifiedTensor],
        engine: &ComputeEngine,
    ) -> Vec<UnifiedTensor>;
}

/// Node that recomputes forward pass during backward to save memory
pub struct CheckpointedNode {
    pub inputs: Vec<Tensor>,
    pub checkpoint_fn: Arc<dyn CheckpointFn>,
    pub outputs: Vec<Tensor>, // Only shapes stored, data discarded after forward
}

impl AutogradNode for CheckpointedNode {
    fn inputs(&self) -> Vec<Tensor> {
        self.inputs.clone()
    }

    fn backward(&self, grad: &UnifiedTensor, engine: &ComputeEngine) -> Vec<UnifiedTensor> {
        // Recompute forward pass to get activations
        let recomputed_outputs = self.checkpoint_fn.forward(&self.inputs, engine);

        // Now compute backward with recomputed activations
        self.checkpoint_fn
            .backward(&self.inputs, &recomputed_outputs, &[grad.clone()], engine)
    }
}

/// Checkpoint a sequence of operations to save memory
/// During forward: computes normally but discards intermediate activations
/// During backward: recomputes activations from checkpoint
pub fn checkpoint<F>(inputs: Vec<Tensor>, forward_fn: F, engine: &ComputeEngine) -> Vec<Tensor>
where
    F: Fn(&[Tensor], &ComputeEngine) -> Vec<Tensor> + Send + Sync + 'static,
{
    // Create wrapper for the forward function
    struct FnWrapper<Func>(Func);

    impl<Func> CheckpointFn for FnWrapper<Func>
    where
        Func: Fn(&[Tensor], &ComputeEngine) -> Vec<Tensor> + Send + Sync,
    {
        fn forward(&self, inputs: &[Tensor], engine: &ComputeEngine) -> Vec<Tensor> {
            (self.0)(inputs, engine)
        }

        fn backward(
            &self,
            inputs: &[Tensor],
            outputs: &[Tensor],
            grad_outputs: &[UnifiedTensor],
            engine: &ComputeEngine,
        ) -> Vec<UnifiedTensor> {
            // Run backward on the recomputed outputs
            // Each output should have been created with autograd, so we can backprop through it
            let input_grads: Vec<UnifiedTensor> = vec![];
            for (i, (output, grad)) in outputs.iter().zip(grad_outputs.iter()).enumerate() {
                *output.grad.borrow_mut() = Some(grad.clone());
            }

            // Trigger backward on each output
            for output in outputs {
                if output.creator.is_some() {
                    // Get gradients for all inputs
                    let grad = output.grad.borrow().clone().unwrap();
                    if let Some(creator) = &output.creator {
                        return creator.backward(&grad, engine);
                    }
                }
            }

            // Return empty grads if no backward path
            inputs
                .iter()
                .map(|t| UnifiedTensor::zeros(&t.ctx, &t.data.shape))
                .collect()
        }
    }

    // Run forward pass
    let outputs = forward_fn(&inputs, engine);

    // Create checkpointed outputs that will recompute on backward
    let checkpoint_fn = Arc::new(FnWrapper(forward_fn));

    outputs
        .into_iter()
        .map(|output| {
            let node = Arc::new(CheckpointedNode {
                inputs: inputs.clone(),
                checkpoint_fn: checkpoint_fn.clone(),
                outputs: vec![output.clone()],
            });
            Tensor::with_creator(output.ctx.clone(), output.data.as_ref().clone(), node)
        })
        .collect()
}

/// Convenience function for checkpointing a single-output function
pub fn checkpoint_single<F>(input: Tensor, forward_fn: F, engine: &ComputeEngine) -> Tensor
where
    F: Fn(&Tensor, &ComputeEngine) -> Tensor + Send + Sync + 'static,
{
    let wrapper = move |inputs: &[Tensor], eng: &ComputeEngine| -> Vec<Tensor> {
        vec![forward_fn(&inputs[0], eng)]
    };
    checkpoint(vec![input], wrapper, engine).pop().unwrap()
}

/// Apply checkpointing to a sequence of functions (e.g., transformer blocks)
/// This is the main entry point for memory-efficient training of large models.
///
/// Usage:
/// ```rust
/// // Instead of:
/// let mut x = input;
/// for block in &blocks {
///     x = block.forward(&x, engine);
/// }
///
/// // Use checkpointed version:
/// let x = checkpoint_single(hidden, |h, eng| layer.forward(h, eng), engine);
/// ```
///
/// For sequential layers, apply checkpoint_single to each layer in a loop:
/// ```rust,ignore
/// let mut x = input;
/// for layer in &layers {
///     x = checkpoint_single(x, |h, eng| layer.forward(h, eng), engine);
/// }
/// ```

impl Tensor {
    pub fn matmul(a: &Tensor, b: &Tensor, engine: &ComputeEngine) -> Tensor {
        let out_shape = vec![a.data.shape[0], b.data.shape[1]];
        let out_data = UnifiedTensor::empty(&a.ctx, &out_shape);
        engine.matmul(&a.ctx, &a.data, &b.data, &out_data);
        let node = Arc::new(MatMulNode {
            a: a.clone(),
            b: b.clone(),
        });
        Tensor::with_creator(a.ctx.clone(), out_data, node)
    }
    pub fn add(a: &Tensor, b: &Tensor, engine: &ComputeEngine) -> Tensor {
        let out_data = UnifiedTensor::empty(&a.ctx, &a.data.shape);
        engine.add(&a.ctx, &a.data, &b.data, &out_data);
        let node = Arc::new(AddNode {
            a: a.clone(),
            b: b.clone(),
        });
        Tensor::with_creator(a.ctx.clone(), out_data, node)
    }
    pub fn mul(a: &Tensor, b: &Tensor, engine: &ComputeEngine) -> Tensor {
        let out_data = UnifiedTensor::empty(&a.ctx, &a.data.shape);
        engine.mul(&a.ctx, &a.data, &b.data, &out_data);
        let node = Arc::new(MulNode {
            a: a.clone(),
            b: b.clone(),
        });
        Tensor::with_creator(a.ctx.clone(), out_data, node)
    }
    pub fn sub(a: &Tensor, b: &Tensor, engine: &ComputeEngine) -> Tensor {
        let out_data = UnifiedTensor::empty(&a.ctx, &a.data.shape);
        engine.sub(&a.ctx, &a.data, &b.data, &out_data);
        let node = Arc::new(SubNode {
            a: a.clone(),
            b: b.clone(),
        });
        Tensor::with_creator(a.ctx.clone(), out_data, node)
    }
    pub fn silu(input: &Tensor, engine: &ComputeEngine) -> Tensor {
        let out_data = UnifiedTensor::empty(&input.ctx, &input.data.shape);
        engine.silu(&input.ctx, &input.data, &out_data);
        let node = Arc::new(SiLUNode {
            input: input.clone(),
        });
        Tensor::with_creator(input.ctx.clone(), out_data, node)
    }
    pub fn mse_loss(pred: &Tensor, target: &Tensor, engine: &ComputeEngine) -> Tensor {
        let out_data = UnifiedTensor::empty(&pred.ctx, &[1]);
        engine.mse_loss(&pred.ctx, &pred.data, &target.data, &out_data);
        let node = Arc::new(MSELossNode {
            pred: pred.clone(),
            target: target.clone(),
        });
        Tensor::with_creator(pred.ctx.clone(), out_data, node)
    }

    pub fn relu(input: &Tensor, engine: &ComputeEngine) -> Tensor {
        let out_data = UnifiedTensor::empty(&input.ctx, &input.data.shape);
        engine.relu(&input.ctx, &input.data, &out_data);
        let node = Arc::new(ReluNode {
            input: input.clone(),
        });
        Tensor::with_creator(input.ctx.clone(), out_data, node)
    }

    pub fn softmax(input: &Tensor, engine: &ComputeEngine) -> Tensor {
        let out_data = UnifiedTensor::empty(&input.ctx, &input.data.shape);
        engine.softmax(&input.ctx, &input.data, &out_data);
        let out_tensor = Tensor::new(input.ctx.clone(), out_data);
        let node = Arc::new(SoftmaxNode {
            input: input.clone(),
            output: out_tensor.clone(),
        });
        Tensor::with_creator(input.ctx.clone(), out_tensor.data.as_ref().clone(), node)
    }

    /// Cross-entropy loss for language model training
    /// Computes -log(softmax(logits)[target])
    pub fn cross_entropy_loss(logits: &Tensor, target: u32, engine: &ComputeEngine) -> Tensor {
        let out_data = UnifiedTensor::empty(&logits.ctx, &[1]);
        engine.cross_entropy_loss(&logits.ctx, &logits.data, target, &out_data);
        let node = Arc::new(CrossEntropyLossNode {
            logits: logits.clone(),
            target_idx: target,
        });
        Tensor::with_creator(logits.ctx.clone(), out_data, node)
    }

    /// Fused SiLU-Gate: out = SiLU(up) * gate
    /// Used in Llama MLP, reduces memory bandwidth by 2x
    pub fn silu_gate_fused(up: &Tensor, gate: &Tensor, engine: &ComputeEngine) -> Tensor {
        let out_data = UnifiedTensor::empty(&up.ctx, &up.data.shape);
        engine.silu_gate_fused(&up.ctx, &up.data, &gate.data, &out_data);
        let node = Arc::new(SiLUGateNode {
            up: up.clone(),
            gate: gate.clone(),
        });
        Tensor::with_creator(up.ctx.clone(), out_data, node)
    }
}

pub trait Optimizer: Send + Sync {
    fn step(&mut self, params: &[Tensor]);
    fn zero_grad(&mut self, params: &[Tensor]);
}

pub struct AdamW {
    pub engine: Arc<ComputeEngine>,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub t: f32,
    m_states: HashMap<usize, UnifiedTensor>,
    v_states: HashMap<usize, UnifiedTensor>,
}

impl AdamW {
    pub fn new(engine: Arc<ComputeEngine>, lr: f32) -> Self {
        Self {
            engine,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            t: 0.0,
            m_states: HashMap::new(),
            v_states: HashMap::new(),
        }
    }

    pub fn step(&mut self, params: &[Tensor]) {
        self.t += 1.0;
        let p = AdamParams {
            lr: self.lr,
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
            weight_decay: self.weight_decay,
            t: self.t,
            _pad: [0.0; 2],
        };

        for param in params {
            let grad_borrow = param.grad.borrow();
            if let Some(grad) = grad_borrow.as_ref() {
                let key = Arc::as_ptr(&param.data) as usize;

                let m = self
                    .m_states
                    .entry(key)
                    .or_insert_with(|| UnifiedTensor::zeros(&param.ctx, &param.data.shape));
                let v = self
                    .v_states
                    .entry(key)
                    .or_insert_with(|| UnifiedTensor::zeros(&param.ctx, &param.data.shape));

                self.engine
                    .adamw_update(&param.ctx, &param.data, grad, m, v, &p);
            }
        }
    }

    pub fn zero_grad(&mut self, params: &[Tensor]) {
        for param in params {
            *param.grad.borrow_mut() = None;
        }
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, params: &[Tensor]) {
        self.step(params);
    }
    fn zero_grad(&mut self, params: &[Tensor]) {
        self.zero_grad(params);
    }
}

// --- Learning Rate Schedulers ---

/// Trait for learning rate schedulers
pub trait LRScheduler {
    /// Get current learning rate
    fn get_lr(&self) -> f32;
    /// Advance the scheduler by one step
    fn step(&mut self);
    /// Get current step number
    fn current_step(&self) -> usize;
}

/// Cosine Annealing Learning Rate Scheduler
/// LR decays from initial_lr to min_lr following a cosine curve
pub struct CosineAnnealingLR {
    pub initial_lr: f32,
    pub min_lr: f32,
    pub total_steps: usize,
    pub current: usize,
}

impl CosineAnnealingLR {
    pub fn new(initial_lr: f32, min_lr: f32, total_steps: usize) -> Self {
        Self {
            initial_lr,
            min_lr,
            total_steps,
            current: 0,
        }
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn get_lr(&self) -> f32 {
        if self.current >= self.total_steps {
            return self.min_lr;
        }
        let progress = self.current as f32 / self.total_steps as f32;
        let cosine = (1.0 + (std::f32::consts::PI * progress).cos()) / 2.0;
        self.min_lr + (self.initial_lr - self.min_lr) * cosine
    }

    fn step(&mut self) {
        self.current += 1;
    }

    fn current_step(&self) -> usize {
        self.current
    }
}

/// Linear Warmup + Cosine Annealing Scheduler
/// Warmups linearly from 0 to peak_lr, then decays using cosine schedule
pub struct WarmupCosineScheduler {
    pub peak_lr: f32,
    pub min_lr: f32,
    pub warmup_steps: usize,
    pub total_steps: usize,
    pub current: usize,
}

impl WarmupCosineScheduler {
    pub fn new(peak_lr: f32, min_lr: f32, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            peak_lr,
            min_lr,
            warmup_steps,
            total_steps,
            current: 0,
        }
    }
}

impl LRScheduler for WarmupCosineScheduler {
    fn get_lr(&self) -> f32 {
        if self.current < self.warmup_steps {
            // Linear warmup
            let progress = self.current as f32 / self.warmup_steps as f32;
            self.peak_lr * progress
        } else if self.current >= self.total_steps {
            self.min_lr
        } else {
            // Cosine decay
            let decay_steps = self.total_steps - self.warmup_steps;
            let decay_current = self.current - self.warmup_steps;
            let progress = decay_current as f32 / decay_steps as f32;
            let cosine = (1.0 + (std::f32::consts::PI * progress).cos()) / 2.0;
            self.min_lr + (self.peak_lr - self.min_lr) * cosine
        }
    }

    fn step(&mut self) {
        self.current += 1;
    }

    fn current_step(&self) -> usize {
        self.current
    }
}
