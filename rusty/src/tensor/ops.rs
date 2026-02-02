//! Tensor Operations
//!
//! Math operations on tensors with automatic differentiation.

use std::sync::Arc;
use wgpu::BufferUsages;
use wgpu::util::DeviceExt;

use super::Tensor;
use super::autograd::*;
use crate::backend::ComputeEngine;

impl Tensor {
    // ========================================================================
    // BINARY OPERATIONS
    // ========================================================================

    /// Element-wise addition: self + other
    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Shapes must match for add");
        
        let device = &self.device;
        let engine = device.engine();
        let size = self.numel();

        let out_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Add Output"),
            size: (size * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline = engine.get_pipeline("add").unwrap();
        let bind_group = engine.create_binary_bind_group(&device.device, pipeline, &self.buffer, &other.buffer, &out_buffer);
        engine.dispatch(&device.device, &device.queue, "add", &bind_group, 
            (ComputeEngine::workgroups_1d(size as u32), 1, 1));

        let requires_grad = self.requires_grad || other.requires_grad;
        let creator: Option<Arc<dyn AutogradNode>> = if requires_grad {
            Some(Arc::new(AddNode { a: self.clone(), b: other.clone() }))
        } else {
            None
        };

        Tensor {
            buffer: Arc::new(out_buffer),
            shape: self.shape.clone(),
            device: device.clone(),
            grad: Arc::new(GradCell::new()),
            requires_grad,
            creator,
        }
    }

    /// Element-wise subtraction: self - other
    pub fn sub(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Shapes must match for sub");
        
        let device = &self.device;
        let engine = device.engine();
        let size = self.numel();

        let out_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sub Output"),
            size: (size * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline = engine.get_pipeline("sub").unwrap();
        let bind_group = engine.create_binary_bind_group(&device.device, pipeline, &self.buffer, &other.buffer, &out_buffer);
        engine.dispatch(&device.device, &device.queue, "sub", &bind_group, 
            (ComputeEngine::workgroups_1d(size as u32), 1, 1));

        let requires_grad = self.requires_grad || other.requires_grad;
        let creator: Option<Arc<dyn AutogradNode>> = if requires_grad {
            Some(Arc::new(SubNode { a: self.clone(), b: other.clone() }))
        } else {
            None
        };

        Tensor {
            buffer: Arc::new(out_buffer),
            shape: self.shape.clone(),
            device: device.clone(),
            grad: Arc::new(GradCell::new()),
            requires_grad,
            creator,
        }
    }

    /// Element-wise multiplication: self * other
    pub fn mul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Shapes must match for mul");
        
        let device = &self.device;
        let engine = device.engine();
        let size = self.numel();

        let out_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Mul Output"),
            size: (size * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline = engine.get_pipeline("mul").unwrap();
        let bind_group = engine.create_binary_bind_group(&device.device, pipeline, &self.buffer, &other.buffer, &out_buffer);
        engine.dispatch(&device.device, &device.queue, "mul", &bind_group, 
            (ComputeEngine::workgroups_1d(size as u32), 1, 1));

        let requires_grad = self.requires_grad || other.requires_grad;
        let creator: Option<Arc<dyn AutogradNode>> = if requires_grad {
            Some(Arc::new(MulNode { a: self.clone(), b: other.clone() }))
        } else {
            None
        };

        Tensor {
            buffer: Arc::new(out_buffer),
            shape: self.shape.clone(),
            device: device.clone(),
            grad: Arc::new(GradCell::new()),
            requires_grad,
            creator,
        }
    }

    /// Matrix multiplication: self @ other
    /// self: [M, K], other: [K, N] -> output: [M, N]
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.ndim(), 2, "matmul requires 2D tensors");
        assert_eq!(other.ndim(), 2, "matmul requires 2D tensors");
        assert_eq!(self.shape[1], other.shape[0], "Inner dimensions must match for matmul");

        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];

        let device = &self.device;
        let engine = device.engine();

        let out_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MatMul Output"),
            size: (m * n * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create params buffer [M, K, N, padding]
        let params = device.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MatMul Params"),
            contents: bytemuck::cast_slice(&[m as u32, k as u32, n as u32, 0u32]),
            usage: BufferUsages::UNIFORM,
        });

        let pipeline = engine.get_pipeline("matmul").unwrap();
        let bind_group = engine.create_matmul_bind_group(&device.device, pipeline, &self.buffer, &other.buffer, &out_buffer, &params);
        let (wx, wy) = ComputeEngine::workgroups_2d(m as u32, n as u32);
        engine.dispatch(&device.device, &device.queue, "matmul", &bind_group, (wx, wy, 1));

        let requires_grad = self.requires_grad || other.requires_grad;
        let creator: Option<Arc<dyn AutogradNode>> = if requires_grad {
            Some(Arc::new(MatMulNode { a: self.clone(), b: other.clone() }))
        } else {
            None
        };

        Tensor {
            buffer: Arc::new(out_buffer),
            shape: vec![m, n],
            device: device.clone(),
            grad: Arc::new(GradCell::new()),
            requires_grad,
            creator,
        }
    }

    // ========================================================================
    // ACTIVATION FUNCTIONS
    // ========================================================================

    /// ReLU activation: max(0, x)
    pub fn relu(&self) -> Tensor {
        let device = &self.device;
        let engine = device.engine();
        let size = self.numel();

        let out_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ReLU Output"),
            size: (size * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline = engine.get_pipeline("relu").unwrap();
        let bind_group = engine.create_unary_bind_group(&device.device, pipeline, &self.buffer, &out_buffer);
        engine.dispatch(&device.device, &device.queue, "relu", &bind_group, 
            (ComputeEngine::workgroups_1d(size as u32), 1, 1));

        let creator: Option<Arc<dyn AutogradNode>> = if self.requires_grad {
            Some(Arc::new(ReLUNode { input: self.clone() }))
        } else {
            None
        };

        Tensor {
            buffer: Arc::new(out_buffer),
            shape: self.shape.clone(),
            device: device.clone(),
            grad: Arc::new(GradCell::new()),
            requires_grad: self.requires_grad,
            creator,
        }
    }

    /// SiLU activation: x * sigmoid(x)
    pub fn silu(&self) -> Tensor {
        let device = &self.device;
        let engine = device.engine();
        let size = self.numel();

        let out_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SiLU Output"),
            size: (size * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline = engine.get_pipeline("silu").unwrap();
        let bind_group = engine.create_unary_bind_group(&device.device, pipeline, &self.buffer, &out_buffer);
        engine.dispatch(&device.device, &device.queue, "silu", &bind_group, 
            (ComputeEngine::workgroups_1d(size as u32), 1, 1));

        let creator: Option<Arc<dyn AutogradNode>> = if self.requires_grad {
            Some(Arc::new(SiLUNode { input: self.clone() }))
        } else {
            None
        };

        Tensor {
            buffer: Arc::new(out_buffer),
            shape: self.shape.clone(),
            device: device.clone(),
            grad: Arc::new(GradCell::new()),
            requires_grad: self.requires_grad,
            creator,
        }
    }

    /// GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    pub fn gelu(&self) -> Tensor {
        let device = &self.device;
        let engine = device.engine();
        let size = self.numel();

        let out_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GELU Output"),
            size: (size * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline = engine.get_pipeline("gelu").unwrap();
        let bind_group = engine.create_unary_bind_group(&device.device, pipeline, &self.buffer, &out_buffer);
        engine.dispatch(&device.device, &device.queue, "gelu", &bind_group, 
            (ComputeEngine::workgroups_1d(size as u32), 1, 1));

        let creator: Option<Arc<dyn AutogradNode>> = if self.requires_grad {
            Some(Arc::new(GELUNode { input: self.clone() }))
        } else {
            None
        };

        Tensor {
            buffer: Arc::new(out_buffer),
            shape: self.shape.clone(),
            device: device.clone(),
            grad: Arc::new(GradCell::new()),
            requires_grad: self.requires_grad,
            creator,
        }
    }

    /// Sigmoid activation: 1 / (1 + exp(-x))
    pub fn sigmoid(&self) -> Tensor {
        let device = &self.device;
        let engine = device.engine();
        let size = self.numel();

        let out_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sigmoid Output"),
            size: (size * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline = engine.get_pipeline("sigmoid").unwrap();
        let bind_group = engine.create_unary_bind_group(&device.device, pipeline, &self.buffer, &out_buffer);
        engine.dispatch(&device.device, &device.queue, "sigmoid", &bind_group, 
            (ComputeEngine::workgroups_1d(size as u32), 1, 1));

        Tensor {
            buffer: Arc::new(out_buffer),
            shape: self.shape.clone(),
            device: device.clone(),
            grad: Arc::new(GradCell::new()),
            requires_grad: self.requires_grad,
            creator: None, // TODO: Add SigmoidNode
        }
    }

    /// Tanh activation
    pub fn tanh(&self) -> Tensor {
        let device = &self.device;
        let engine = device.engine();
        let size = self.numel();

        let out_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tanh Output"),
            size: (size * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline = engine.get_pipeline("tanh_act").unwrap();
        let bind_group = engine.create_unary_bind_group(&device.device, pipeline, &self.buffer, &out_buffer);
        engine.dispatch(&device.device, &device.queue, "tanh_act", &bind_group, 
            (ComputeEngine::workgroups_1d(size as u32), 1, 1));

        Tensor {
            buffer: Arc::new(out_buffer),
            shape: self.shape.clone(),
            device: device.clone(),
            grad: Arc::new(GradCell::new()),
            requires_grad: self.requires_grad,
            creator: None, // TODO: Add TanhNode
        }
    }

    /// Softmax: exp(x) / sum(exp(x))
    pub fn softmax(&self, _dim: i32) -> Tensor {
        let device = &self.device;
        let engine = device.engine();
        let size = self.numel();

        let out_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Softmax Output"),
            size: (size * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let size_buffer = device.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Softmax Size"),
            contents: bytemuck::cast_slice(&[size as u32]),
            usage: BufferUsages::UNIFORM,
        });

        let pipeline = engine.get_pipeline("softmax").unwrap();
        let layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: out_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: size_buffer.as_entire_binding() },
            ],
        });
        engine.dispatch(&device.device, &device.queue, "softmax", &bind_group, (1, 1, 1));

        Tensor {
            buffer: Arc::new(out_buffer),
            shape: self.shape.clone(),
            device: device.clone(),
            grad: Arc::new(GradCell::new()),
            requires_grad: self.requires_grad,
            creator: None, // TODO: Add SoftmaxNode
        }
    }

    // ========================================================================
    // SCALAR OPERATIONS
    // ========================================================================

    /// Add a scalar: self + scalar
    pub fn add_scalar(&self, scalar: f32) -> Tensor {
        let device = &self.device;
        let engine = device.engine();
        let size = self.numel();

        let out_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Add Scalar Output"),
            size: (size * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let scalar_buffer = device.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Scalar"),
            contents: bytemuck::cast_slice(&[scalar]),
            usage: BufferUsages::UNIFORM,
        });

        let pipeline = engine.get_pipeline("add_scalar").unwrap();
        let layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: out_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: scalar_buffer.as_entire_binding() },
            ],
        });
        engine.dispatch(&device.device, &device.queue, "add_scalar", &bind_group, 
            (ComputeEngine::workgroups_1d(size as u32), 1, 1));

        Tensor {
            buffer: Arc::new(out_buffer),
            shape: self.shape.clone(),
            device: device.clone(),
            grad: Arc::new(GradCell::new()),
            requires_grad: self.requires_grad,
            creator: None,
        }
    }

    /// Multiply by a scalar: self * scalar
    pub fn mul_scalar(&self, scalar: f32) -> Tensor {
        let device = &self.device;
        let engine = device.engine();
        let size = self.numel();

        let out_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Mul Scalar Output"),
            size: (size * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let scalar_buffer = device.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Scalar"),
            contents: bytemuck::cast_slice(&[scalar]),
            usage: BufferUsages::UNIFORM,
        });

        let pipeline = engine.get_pipeline("mul_scalar").unwrap();
        let layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: out_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: scalar_buffer.as_entire_binding() },
            ],
        });
        engine.dispatch(&device.device, &device.queue, "mul_scalar", &bind_group, 
            (ComputeEngine::workgroups_1d(size as u32), 1, 1));

        Tensor {
            buffer: Arc::new(out_buffer),
            shape: self.shape.clone(),
            device: device.clone(),
            grad: Arc::new(GradCell::new()),
            requires_grad: self.requires_grad,
            creator: None,
        }
    }

    /// Negate: -self
    pub fn neg(&self) -> Tensor {
        self.mul_scalar(-1.0)
    }

    // ========================================================================
    // REDUCTION OPERATIONS
    // ========================================================================

    /// Sum all elements.
    pub fn sum(&self) -> Tensor {
        // CPU fallback for now - can optimize with parallel reduction later
        let data = self.to_vec();
        let sum: f32 = data.iter().sum();
        Tensor::from_data(&[sum], &[1], &self.device)
    }

    /// Mean of all elements.
    pub fn mean(&self) -> Tensor {
        let data = self.to_vec();
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        Tensor::from_data(&[mean], &[1], &self.device)
    }

    /// Maximum value.
    pub fn max(&self) -> Tensor {
        let data = self.to_vec();
        let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        Tensor::from_data(&[max], &[1], &self.device)
    }

    /// Minimum value.
    pub fn min(&self) -> Tensor {
        let data = self.to_vec();
        let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        Tensor::from_data(&[min], &[1], &self.device)
    }

    /// Argmax - index of maximum value.
    pub fn argmax(&self) -> usize {
        let data = self.to_vec();
        data.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

// ============================================================================
// OPERATOR OVERLOADS
// ============================================================================

impl std::ops::Add for &Tensor {
    type Output = Tensor;
    fn add(self, other: &Tensor) -> Tensor {
        Tensor::add(self, other)
    }
}

impl std::ops::Sub for &Tensor {
    type Output = Tensor;
    fn sub(self, other: &Tensor) -> Tensor {
        Tensor::sub(self, other)
    }
}

impl std::ops::Mul for &Tensor {
    type Output = Tensor;
    fn mul(self, other: &Tensor) -> Tensor {
        Tensor::mul(self, other)
    }
}

impl std::ops::Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        self.mul_scalar(-1.0)
    }
}
