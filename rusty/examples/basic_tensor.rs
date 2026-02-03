//! Example: Basic GPU Tensor Operations
//!
//! This example demonstrates basic tensor operations on the GPU.
//!
//! Run with: cargo run --example basic_tensor --release

use rusty_backend::{WgpuContext, ComputeEngine, UnifiedTensor};

fn main() {
    println!("🦀 Rusty ML - Basic Tensor Example\n");

    // Initialize GPU
    let ctx = pollster::block_on(async { WgpuContext::new().await });
    let engine = ComputeEngine::new(&ctx);

    // Print GPU info
    let info = ctx.adapter.get_info();
    println!("🖥️  GPU: {} ({:?})", info.name, info.backend);

    // Create tensors
    println!("\n📦 Creating tensors...");
    
    let data_a: Vec<f32> = (0..1024).map(|i| i as f32 / 1024.0).collect();
    let data_b: Vec<f32> = (0..1024).map(|i| (1024 - i) as f32 / 1024.0).collect();

    let a = UnifiedTensor::new(&ctx, &data_a, &[32, 32]);
    let b = UnifiedTensor::new(&ctx, &data_b, &[32, 32]);
    let c = UnifiedTensor::empty(&ctx, &[32, 32]);

    println!("   ✓ Tensor A: {:?}", a.shape);
    println!("   ✓ Tensor B: {:?}", b.shape);

    // Matrix multiplication
    println!("\n🔢 Performing matrix multiplication...");
    engine.matmul(&ctx, &a, &b, &c);

    // Read back result
    let result = pollster::block_on(c.to_vec(&ctx));
    println!("   ✓ Result shape: {:?}", c.shape);
    println!("   ✓ First few values: {:?}", &result[..5]);
    println!("   ✓ Sum: {:.4}", result.iter().sum::<f32>());

    // Element-wise operations
    println!("\n⚡ Element-wise operations...");
    
    let d = UnifiedTensor::empty(&ctx, &[32, 32]);
    engine.add(&ctx, &a, &b, &d);
    let add_result = pollster::block_on(d.to_vec(&ctx));
    println!("   ✓ Add: first 3 values = {:?}", &add_result[..3]);

    let e = UnifiedTensor::empty(&ctx, &[32, 32]);
    engine.mul(&ctx, &a, &b, &e);
    let mul_result = pollster::block_on(e.to_vec(&ctx));
    println!("   ✓ Mul: first 3 values = {:?}", &mul_result[..3]);

    // Activation functions
    println!("\n🔥 Activation functions...");
    
    let f = UnifiedTensor::empty(&ctx, &[32, 32]);
    engine.silu(&ctx, &a, &f);
    let silu_result = pollster::block_on(f.to_vec(&ctx));
    println!("   ✓ SiLU: first 3 values = {:?}", &silu_result[..3]);

    println!("\n✅ All operations completed successfully!");
    println!("   Total GPU operations: 5");
}
