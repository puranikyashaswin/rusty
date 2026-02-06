//! Advanced Flash Attention Example
//!
//! Demonstrates advanced features:
//! - Multi-head attention
//! - Performance comparison with standard attention
//! - Memory profiling
//! - Different sequence lengths
//!
//! Run with: cargo run --example flash_attention_advanced --release -p rusty

use rusty_backend::{ComputeEngine, UnifiedTensor, WgpuContext};
use rusty_graph::FlashAttention;
use std::time::Instant;

fn main() {
    println!("🚀 Advanced Flash Attention Demo\n");
    println!("==================================\n");

    // Initialize GPU
    let ctx = std::sync::Arc::new(pollster::block_on(async { WgpuContext::new().await }));
    let engine = ComputeEngine::new(&ctx);

    // Print GPU info
    let info = ctx.adapter.get_info();
    println!("[GPU] {} ({:?})", info.name, info.backend);
    println!();

    // Demo 1: Multi-Head Attention
    demo_multi_head_attention(&ctx, &engine);

    // Demo 2: Performance Scaling
    demo_performance_scaling(&ctx, &engine);

    // Demo 3: Memory Efficiency
    demo_memory_efficiency(&ctx, &engine);

    // Demo 4: Causal vs Non-Causal
    demo_causal_comparison(&ctx, &engine);

    println!("\n✅ All demos completed successfully!");
}

fn demo_multi_head_attention(ctx: &WgpuContext, engine: &ComputeEngine) {
    println!("📊 Demo 1: Multi-Head Attention");
    println!("================================\n");

    let configs = vec![
        (768, 1, "Single-head"),
        (768, 4, "4 heads"),
        (768, 8, "8 heads"),
        (768, 12, "12 heads (BERT/GPT)"),
        (768, 16, "16 heads"),
    ];

    let seq_len = 512;

    for (hidden_dim, num_heads, name) in configs {
        let flash_attn = FlashAttention::new(hidden_dim, num_heads, true);

        let q_data: Vec<f32> = (0..seq_len * hidden_dim)
            .map(|i| ((i % 100) as f32 - 50.0) / 100.0)
            .collect();

        let q = UnifiedTensor::new(ctx, &q_data, &[1, seq_len, hidden_dim]);
        let k = UnifiedTensor::new(ctx, &q_data, &[1, seq_len, hidden_dim]);
        let v = UnifiedTensor::new(ctx, &q_data, &[1, seq_len, hidden_dim]);

        let start = Instant::now();
        let output = flash_attn.forward(&q, &k, &v, ctx, engine);
        let _result = pollster::block_on(output.to_vec(ctx));
        let duration = start.elapsed();

        println!("  {} ({} heads):", name, num_heads);
        println!("    Time: {:.2}ms", duration.as_secs_f64() * 1000.0);
        println!("    Memory savings: {}", flash_attn.memory_savings(seq_len));
        println!();
    }
}

fn demo_performance_scaling(ctx: &WgpuContext, engine: &ComputeEngine) {
    println!("⚡ Demo 2: Performance Scaling");
    println!("================================\n");

    let hidden_dim = 768;
    let num_heads = 12;
    let flash_attn = FlashAttention::new(hidden_dim, num_heads, true);

    let seq_lengths = vec![128, 256, 512, 1024, 2048, 4096];

    println!("  Seq Len | Time (ms) | Throughput (tokens/ms)");
    println!("  --------|-----------|----------------------");

    for seq_len in seq_lengths {
        let q_data: Vec<f32> = (0..seq_len * hidden_dim)
            .map(|i| ((i % 100) as f32 - 50.0) / 100.0)
            .collect();

        let q = UnifiedTensor::new(ctx, &q_data, &[1, seq_len, hidden_dim]);
        let k = UnifiedTensor::new(ctx, &q_data, &[1, seq_len, hidden_dim]);
        let v = UnifiedTensor::new(ctx, &q_data, &[1, seq_len, hidden_dim]);

        let start = Instant::now();
        let output = flash_attn.forward(&q, &k, &v, ctx, engine);
        let _result = pollster::block_on(output.to_vec(ctx));
        let duration = start.elapsed();

        let time_ms = duration.as_secs_f64() * 1000.0;
        let throughput = seq_len as f64 / time_ms;

        println!("  {:7} | {:9.2} | {:20.1}", seq_len, time_ms, throughput);
    }
    println!();
}

fn demo_memory_efficiency(ctx: &WgpuContext, engine: &ComputeEngine) {
    println!("💾 Demo 3: Memory Efficiency");
    println!("================================\n");

    let hidden_dim = 768;
    let num_heads = 12;
    let flash_attn = FlashAttention::new(hidden_dim, num_heads, true);

    let seq_lengths = vec![512, 1024, 2048, 4096, 8192];

    println!("  Seq Len | Standard (MB) | Flash (MB) | Savings");
    println!("  --------|---------------|------------|--------");

    for seq_len in seq_lengths {
        let head_dim = hidden_dim / num_heads;

        // Standard attention: O(N²) for attention matrix
        let standard_mem = (seq_len * seq_len * 4) as f64 / 1_000_000.0;

        // Flash attention: O(N) for accumulator
        let flash_mem = (seq_len * head_dim * 4) as f64 / 1_000_000.0;

        let savings = standard_mem / flash_mem;

        println!(
            "  {:7} | {:13.2} | {:10.2} | {:6.1}x",
            seq_len, standard_mem, flash_mem, savings
        );
    }
    println!();
}

fn demo_causal_comparison(ctx: &WgpuContext, engine: &ComputeEngine) {
    println!("🔒 Demo 4: Causal vs Non-Causal Masking");
    println!("========================================\n");

    let hidden_dim = 256;
    let num_heads = 4;
    let seq_len = 256;

    let q_data: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| ((i % 100) as f32 - 50.0) / 100.0)
        .collect();

    let q = UnifiedTensor::new(ctx, &q_data, &[1, seq_len, hidden_dim]);
    let k = UnifiedTensor::new(ctx, &q_data, &[1, seq_len, hidden_dim]);
    let v = UnifiedTensor::new(ctx, &q_data, &[1, seq_len, hidden_dim]);

    // Non-causal (bidirectional)
    let flash_attn_bi = FlashAttention::new(hidden_dim, num_heads, false);
    let start = Instant::now();
    let output_bi = flash_attn_bi.forward(&q, &k, &v, ctx, engine);
    let result_bi = pollster::block_on(output_bi.to_vec(ctx));
    let time_bi = start.elapsed();

    // Causal (autoregressive)
    let flash_attn_causal = FlashAttention::new(hidden_dim, num_heads, true);
    let start = Instant::now();
    let output_causal = flash_attn_causal.forward(&q, &k, &v, ctx, engine);
    let result_causal = pollster::block_on(output_causal.to_vec(ctx));
    let time_causal = start.elapsed();

    println!("  Non-Causal (Bidirectional):");
    println!("    Time: {:.2}ms", time_bi.as_secs_f64() * 1000.0);
    println!(
        "    Mean output: {:.6}",
        result_bi.iter().sum::<f32>() / result_bi.len() as f32
    );
    println!();

    println!("  Causal (Autoregressive):");
    println!("    Time: {:.2}ms", time_causal.as_secs_f64() * 1000.0);
    println!(
        "    Mean output: {:.6}",
        result_causal.iter().sum::<f32>() / result_causal.len() as f32
    );
    println!();

    // Calculate difference
    let diff: f32 = result_bi
        .iter()
        .zip(&result_causal)
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / result_bi.len() as f32;

    println!("  Mean absolute difference: {:.6}", diff);
    println!("  (Causal masking prevents attending to future tokens)");
    println!();
}
