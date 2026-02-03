//! Autograd - Automatic Differentiation Engine
//!
//! This module provides gradient computation through reverse-mode autodiff.

use std::sync::RwLock;
use wgpu::{Buffer, BufferUsages};

use crate::Device;
use super::Tensor;

/// Gradient storage cell (thread-safe).
pub struct GradCell {
    inner: RwLock<Option<Buffer>>,
}

impl GradCell {
    pub fn new() -> Self {
        Self { inner: RwLock::new(None) }
    }

    pub fn set(&self, device: &Device, data: &[f32], _shape: &[usize]) {
        let buffer = device.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Gradient Buffer"),
            contents: bytemuck::cast_slice(data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        });
        *self.inner.write().unwrap() = Some(buffer);
    }

    pub fn set_buffer(&self, buffer: Buffer) {
        *self.inner.write().unwrap() = Some(buffer);
    }

    pub fn get(&self) -> Option<Buffer> {
        self.inner.read().unwrap().as_ref().map(|b| {
            // Note: This is a simplification - in production we'd need proper cloning
            // For now we just return the buffer reference
            // This is a limitation - ideally we'd Arc<Buffer>
            unsafe { std::ptr::read(b as *const Buffer) }
        })
    }

    pub fn clear(&self) {
        *self.inner.write().unwrap() = None;
    }

    pub fn accumulate(&self, device: &Device, engine: &crate::backend::ComputeEngine, new_grad: &Buffer, shape: &[usize]) {
        let size: usize = shape.iter().product();
        let mut inner = self.inner.write().unwrap();
        
        if let Some(ref existing) = *inner {
            // Create output buffer for accumulated gradient
            let out_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Accumulated Gradient"),
                size: (size * 4) as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Dispatch add kernel
            let pipeline = engine.get_pipeline("add").unwrap();
            let bind_group = engine.create_binary_bind_group(&device.device, pipeline, existing, new_grad, &out_buffer);
            engine.dispatch(&device.device, &device.queue, "add", &bind_group, (crate::backend::ComputeEngine::workgroups_1d(size as u32), 1, 1));

            *inner = Some(out_buffer);
        } else {
            // Copy new_grad to this cell
            let out_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Gradient Copy"),
                size: (size * 4) as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let mut encoder = device.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Grad Copy Encoder"),
            });
            encoder.copy_buffer_to_buffer(new_grad, 0, &out_buffer, 0, (size * 4) as u64);
            device.queue.submit(std::iter::once(encoder.finish()));

            *inner = Some(out_buffer);
        }
    }
}


use wgpu::util::DeviceExt;

/// Trait for autograd computation nodes.
pub trait AutogradNode: Send + Sync {
    /// Get input tensors to this node.
    fn inputs(&self) -> Vec<Tensor>;
    
    /// Compute gradients for inputs given output gradient.
    fn backward(&self, grad_output: &Tensor, device: &Device);
}

// ============================================================================
// BINARY OPERATION NODES
// ============================================================================

/// Addition backward node.
pub struct AddNode {
    pub a: Tensor,
    pub b: Tensor,
}

impl AutogradNode for AddNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.a.clone(), self.b.clone()]
    }

    fn backward(&self, grad_output: &Tensor, device: &Device) {
        // grad_a = grad_out, grad_b = grad_out
        if self.a.requires_grad {
            self.a.grad.accumulate(device, device.engine(), &grad_output.buffer, &self.a.shape);
        }
        if self.b.requires_grad {
            self.b.grad.accumulate(device, device.engine(), &grad_output.buffer, &self.b.shape);
        }
    }
}

/// Subtraction backward node.
pub struct SubNode {
    pub a: Tensor,
    pub b: Tensor,
}

impl AutogradNode for SubNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.a.clone(), self.b.clone()]
    }

    fn backward(&self, grad_output: &Tensor, device: &Device) {
        let engine = device.engine();
        let size = self.a.numel();

        if self.a.requires_grad {
            self.a.grad.accumulate(device, engine, &grad_output.buffer, &self.a.shape);
        }
        if self.b.requires_grad {
            // grad_b = -grad_out
            let neg_grad = device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Neg Grad"),
                size: (size * 4) as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Use mul_scalar with -1.0
            let neg_one = device.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Neg One"),
                contents: bytemuck::cast_slice(&[-1.0f32]),
                usage: BufferUsages::UNIFORM,
            });

            let pipeline = engine.get_pipeline("mul_scalar").unwrap();
            let layout = pipeline.get_bind_group_layout(0);
            let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: grad_output.buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: neg_grad.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: neg_one.as_entire_binding() },
                ],
            });
            engine.dispatch(&device.device, &device.queue, "mul_scalar", &bind_group, 
                (crate::backend::ComputeEngine::workgroups_1d(size as u32), 1, 1));

            self.b.grad.accumulate(device, engine, &neg_grad, &self.b.shape);
        }
    }
}

/// Multiplication backward node.
pub struct MulNode {
    pub a: Tensor,
    pub b: Tensor,
}

impl AutogradNode for MulNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.a.clone(), self.b.clone()]
    }

    fn backward(&self, grad_output: &Tensor, device: &Device) {
        let engine = device.engine();
        let size = self.a.numel();

        // grad_a = grad_out * b
        if self.a.requires_grad {
            let grad_a = device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Grad A"),
                size: (size * 4) as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let pipeline = engine.get_pipeline("mul").unwrap();
            let bind_group = engine.create_binary_bind_group(&device.device, pipeline, &grad_output.buffer, &self.b.buffer, &grad_a);
            engine.dispatch(&device.device, &device.queue, "mul", &bind_group, 
                (crate::backend::ComputeEngine::workgroups_1d(size as u32), 1, 1));
            self.a.grad.accumulate(device, engine, &grad_a, &self.a.shape);
        }

        // grad_b = grad_out * a
        if self.b.requires_grad {
            let grad_b = device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Grad B"),
                size: (size * 4) as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let pipeline = engine.get_pipeline("mul").unwrap();
            let bind_group = engine.create_binary_bind_group(&device.device, pipeline, &grad_output.buffer, &self.a.buffer, &grad_b);
            engine.dispatch(&device.device, &device.queue, "mul", &bind_group, 
                (crate::backend::ComputeEngine::workgroups_1d(size as u32), 1, 1));
            self.b.grad.accumulate(device, engine, &grad_b, &self.b.shape);
        }
    }
}

/// Matrix multiplication backward node.
pub struct MatMulNode {
    pub a: Tensor, // [M, K]
    pub b: Tensor, // [K, N]
}

impl AutogradNode for MatMulNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.a.clone(), self.b.clone()]
    }

    fn backward(&self, grad_output: &Tensor, device: &Device) {
        let engine = device.engine();
        let m = self.a.shape[0];
        let k = self.a.shape[1];
        let n = self.b.shape[1];

        // grad_a = grad_out @ b.T  [M, N] @ [N, K] = [M, K]
        if self.a.requires_grad {
            let grad_a = device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Grad A MatMul"),
                size: (m * k * 4) as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let params = device.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("MatMul BT Params"),
                contents: bytemuck::cast_slice(&[m as u32, n as u32, k as u32, 0u32]),
                usage: BufferUsages::UNIFORM,
            });

            let pipeline = engine.get_pipeline("matmul_bt").unwrap();
            let bind_group = engine.create_matmul_bind_group(&device.device, pipeline, &grad_output.buffer, &self.b.buffer, &grad_a, &params);
            let (wx, wy) = crate::backend::ComputeEngine::workgroups_2d(m as u32, k as u32);
            engine.dispatch(&device.device, &device.queue, "matmul_bt", &bind_group, (wx, wy, 1));
            
            self.a.grad.accumulate(device, engine, &grad_a, &self.a.shape);
        }

        // grad_b = a.T @ grad_out  [K, M] @ [M, N] = [K, N]
        if self.b.requires_grad {
            let grad_b = device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Grad B MatMul"),
                size: (k * n * 4) as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let params = device.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("MatMul AT Params"),
                contents: bytemuck::cast_slice(&[k as u32, m as u32, n as u32, 0u32]),
                usage: BufferUsages::UNIFORM,
            });

            let pipeline = engine.get_pipeline("matmul_at").unwrap();
            let bind_group = engine.create_matmul_bind_group(&device.device, pipeline, &self.a.buffer, &grad_output.buffer, &grad_b, &params);
            let (wx, wy) = crate::backend::ComputeEngine::workgroups_2d(k as u32, n as u32);
            engine.dispatch(&device.device, &device.queue, "matmul_at", &bind_group, (wx, wy, 1));
            
            self.b.grad.accumulate(device, engine, &grad_b, &self.b.shape);
        }
    }
}

// ============================================================================
// ACTIVATION NODES
// ============================================================================

/// ReLU backward node.
pub struct ReLUNode {
    pub input: Tensor,
}

impl AutogradNode for ReLUNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }

    fn backward(&self, grad_output: &Tensor, device: &Device) {
        if !self.input.requires_grad { return; }
        
        let engine = device.engine();
        let size = self.input.numel();

        let grad_in = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ReLU Grad"),
            size: (size * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline = engine.get_pipeline("relu_backward").unwrap();
        let layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.input.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: grad_in.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: grad_output.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.input.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: grad_in.as_entire_binding() },
            ],
        });
        engine.dispatch(&device.device, &device.queue, "relu_backward", &bind_group, 
            (crate::backend::ComputeEngine::workgroups_1d(size as u32), 1, 1));
        
        self.input.grad.accumulate(device, engine, &grad_in, &self.input.shape);
    }
}

/// SiLU backward node.
pub struct SiLUNode {
    pub input: Tensor,
}

impl AutogradNode for SiLUNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }

    fn backward(&self, grad_output: &Tensor, device: &Device) {
        if !self.input.requires_grad { return; }
        
        let engine = device.engine();
        let size = self.input.numel();

        let grad_in = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SiLU Grad"),
            size: (size * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline = engine.get_pipeline("silu_backward").unwrap();
        let layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.input.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: grad_in.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: grad_output.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.input.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: grad_in.as_entire_binding() },
            ],
        });
        engine.dispatch(&device.device, &device.queue, "silu_backward", &bind_group, 
            (crate::backend::ComputeEngine::workgroups_1d(size as u32), 1, 1));
        
        self.input.grad.accumulate(device, engine, &grad_in, &self.input.shape);
    }
}

/// GELU backward node.
pub struct GELUNode {
    pub input: Tensor,
}

impl AutogradNode for GELUNode {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }

    fn backward(&self, grad_output: &Tensor, device: &Device) {
        if !self.input.requires_grad { return; }
        
        let engine = device.engine();
        let size = self.input.numel();

        let grad_in = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GELU Grad"),
            size: (size * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline = engine.get_pipeline("gelu_backward").unwrap();
        let layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.input.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: grad_in.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: grad_output.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.input.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: grad_in.as_entire_binding() },
            ],
        });
        engine.dispatch(&device.device, &device.queue, "gelu_backward", &bind_group, 
            (crate::backend::ComputeEngine::workgroups_1d(size as u32), 1, 1));
        
        self.input.grad.accumulate(device, engine, &grad_in, &self.input.shape);
    }
}
