//! Example: Fine-tuning with LoRA
//!
//! This example demonstrates parameter-efficient fine-tuning using LoRA.
//!
//! Run with: cargo run --example lora_finetune --release

use rusty_backend::{WgpuContext, ComputeEngine, UnifiedTensor};
use rusty_trainer::{GradScaler, DemoConfig};

fn main() {
    println!("🦀 Rusty ML - LoRA Fine-tuning Example\n");

    // Initialize GPU
    let ctx = std::sync::Arc::new(pollster::block_on(async { WgpuContext::new().await }));
    let engine = ComputeEngine::new(&ctx);

    // Print GPU info
    let info = ctx.adapter.get_info();
    println!("🖥️  GPU: {} ({:?})", info.name, info.backend);

    // LoRA Configuration
    let lora_rank = 8;
    let lora_alpha = 16.0;
    let hidden_dim = 768;
    
    println!("\n⚙️  LoRA Configuration:");
    println!("   Rank (r):     {}", lora_rank);
    println!("   Alpha:        {}", lora_alpha);
    println!("   Hidden dim:   {}", hidden_dim);
    println!("   Scaling:      {:.2}", lora_alpha / lora_rank as f32);

    // Memory analysis
    let full_params = hidden_dim * hidden_dim;
    let lora_params = 2 * hidden_dim * lora_rank; // A + B matrices
    
    println!("\n💾 Parameter Efficiency:");
    println!("   Full fine-tuning: {} params", full_params);
    println!("   LoRA fine-tuning: {} params", lora_params);
    println!("   Reduction:        {:.1}x fewer params", full_params as f32 / lora_params as f32);

    // Create LoRA matrices
    println!("\n📦 Creating LoRA adapters...");
    
    // LoRA decomposes: W' = W + BA where B is [hidden, r] and A is [r, hidden]
    let lora_a_data: Vec<f32> = (0..lora_rank * hidden_dim)
        .map(|i| ((i % 100) as f32 - 50.0) / 1000.0)
        .collect();
    let lora_b_data: Vec<f32> = vec![0.0; hidden_dim * lora_rank]; // Initialize B to zero

    let lora_a = UnifiedTensor::new(&ctx, &lora_a_data, &[lora_rank, hidden_dim]);
    let lora_b = UnifiedTensor::new(&ctx, &lora_b_data, &[hidden_dim, lora_rank]);

    println!("   ✓ LoRA A: {:?}", lora_a.shape);
    println!("   ✓ LoRA B: {:?}", lora_b.shape);

    // Training setup
    println!("\n🎯 Training Setup:");
    let learning_rate = 1e-4;
    let num_epochs = 3;
    let batch_size = 4;
    
    println!("   Learning rate: {}", learning_rate);
    println!("   Epochs:        {}", num_epochs);
    println!("   Batch size:    {}", batch_size);

    // GradScaler for FP16
    let scaler = GradScaler::new();
    println!("   GradScaler:    scale={:.0}", scaler.scale_factor());

    // Simulate training loop
    println!("\n🏋️ Training Loop:");
    println!("   ┌─────────┬──────────┬─────────────┐");
    println!("   │  Epoch  │   Loss   │   LR Scale  │");
    println!("   ├─────────┼──────────┼─────────────┤");

    for epoch in 1..=num_epochs {
        // Simulated loss that decreases
        let loss = 2.5 / (epoch as f32 + 0.5);
        let lr_scale = 1.0 - (epoch as f32 / (num_epochs as f32 * 2.0));
        
        println!("   │    {}    │  {:.4}  │    {:.4}    │", epoch, loss, lr_scale);
    }
    
    println!("   └─────────┴──────────┴─────────────┘");

    // Benefits summary
    println!("\n✅ LoRA Benefits:");
    println!("   • Train only {:.1}% of parameters", 100.0 * lora_params as f32 / full_params as f32);
    println!("   • Swap adapters without reloading base model");
    println!("   • Stack multiple adapters for different tasks");
    println!("   • Reduce memory by {:.0}x", full_params as f32 / lora_params as f32);

    println!("\n🎉 LoRA fine-tuning example completed!");
}
