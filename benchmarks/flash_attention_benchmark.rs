//! Flash Attention Benchmark Suite
//!
//! Comprehensive benchmarks for Flash Attention implementation.
//! Measures performance, memory usage, and correctness.
//!
//! Run with: cargo bench --bench flash_attention_benchmark

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use rusty_backend::{ComputeEngine, UnifiedTensor, WgpuContext};
use rusty_graph::FlashAttention;
use std::time::Duration;

/// Benchmark Flash Attention forward pass across different sequence lengths
fn bench_flash_attention_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("flash_attention_forward");
    group.measurement_time(Duration::from_secs(10));
    
    // Test different sequence lengths
    let seq_lengths = vec![128, 256, 512, 1024, 2048, 4096];
    let hidden_dim = 768;
    let num_heads = 12;
    
    for seq_len in seq_lengths {
        group.throughput(Throughput::Elements((seq_len * seq_len) as u64));
        
        group.bench_with_input(
            BenchmarkId::new("seq_len", seq_len),
            &seq_len,
            |b, &seq_len| {
                // Setup
                let ctx = std::sync::Arc::new(pollster::block_on(async {
                    WgpuContext::new().await
                }));
                let engine = ComputeEngine::new(&ctx);
                let flash_attn = FlashAttention::new(hidden_dim, num_heads, true);
                
                // Create input tensors
                let q_data: Vec<f32> = (0..seq_len * hidden_dim)
                    .map(|i| (i as f32 % 100.0 - 50.0) / 100.0)
                    .collect();
                let k_data = q_data.clone();
                let v_data = q_data.clone();
                
                let q = UnifiedTensor::new(&ctx, &q_data, &[1, seq_len, hidden_dim]);
                let k = UnifiedTensor::new(&ctx, &k_data, &[1, seq_len, hidden_dim]);
                let v = UnifiedTensor::new(&ctx, &v_data, &[1, seq_len, hidden_dim]);
                
                // Benchmark
                b.iter(|| {
                    let output = flash_attn.forward(&q, &k, &v, &ctx, &engine);
                    black_box(output);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark Flash Attention backward pass
fn bench_flash_attention_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("flash_attention_backward");
    group.measurement_time(Duration::from_secs(10));
    
    let seq_lengths = vec![128, 256, 512, 1024, 2048];
    let hidden_dim = 768;
    let num_heads = 12;
    
    for seq_len in seq_lengths {
        group.throughput(Throughput::Elements((seq_len * seq_len) as u64));
        
        group.bench_with_input(
            BenchmarkId::new("seq_len", seq_len),
            &seq_len,
            |b, &seq_len| {
                let ctx = std::sync::Arc::new(pollster::block_on(async {
                    WgpuContext::new().await
                }));
                let engine = ComputeEngine::new(&ctx);
                let flash_attn = FlashAttention::new(hidden_dim, num_heads, true);
                
                let q_data: Vec<f32> = (0..seq_len * hidden_dim)
                    .map(|i| (i as f32 % 100.0 - 50.0) / 100.0)
                    .collect();
                
                let q = UnifiedTensor::new(&ctx, &q_data, &[seq_len, hidden_dim / num_heads]);
                let k = UnifiedTensor::new(&ctx, &q_data, &[seq_len, hidden_dim / num_heads]);
                let v = UnifiedTensor::new(&ctx, &q_data, &[seq_len, hidden_dim / num_heads]);
                
                // Forward pass to get output
                let output = flash_attn.forward(&q, &k, &v, &ctx, &engine);
                let grad_output = UnifiedTensor::new(&ctx, &q_data, &[seq_len, hidden_dim / num_heads]);
                
                b.iter(|| {
                    let grads = flash_attn.backward(&q, &k, &v, &output, &grad_output, &ctx, &engine);
                    black_box(grads);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory usage scaling
fn bench_memory_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("flash_attention_memory");
    group.measurement_time(Duration::from_secs(5));
    
    let seq_lengths = vec![512, 1024, 2048, 4096, 8192];
    let hidden_dim = 768;
    let num_heads = 12;
    
    for seq_len in seq_lengths {
        group.bench_with_input(
            BenchmarkId::new("seq_len", seq_len),
            &seq_len,
            |b, &seq_len| {
                let ctx = std::sync::Arc::new(pollster::block_on(async {
                    WgpuContext::new().await
                }));
                let engine = ComputeEngine::new(&ctx);
                let flash_attn = FlashAttention::new(hidden_dim, num_heads, true);
                
                let q_data: Vec<f32> = (0..seq_len * hidden_dim)
                    .map(|i| (i as f32 % 100.0 - 50.0) / 100.0)
                    .collect();
                
                let q = UnifiedTensor::new(&ctx, &q_data, &[1, seq_len, hidden_dim]);
                let k = UnifiedTensor::new(&ctx, &q_data, &[1, seq_len, hidden_dim]);
                let v = UnifiedTensor::new(&ctx, &q_data, &[1, seq_len, hidden_dim]);
                
                b.iter(|| {
                    let output = flash_attn.forward(&q, &k, &v, &ctx, &engine);
                    // Force GPU sync to measure actual memory usage
                    let _result = pollster::block_on(output.to_vec(&ctx));
                    black_box(_result);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark different head counts
fn bench_multi_head_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("flash_attention_heads");
    group.measurement_time(Duration::from_secs(5));
    
    let seq_len = 1024;
    let hidden_dim = 768;
    let head_counts = vec![1, 4, 8, 12, 16];
    
    for num_heads in head_counts {
        group.bench_with_input(
            BenchmarkId::new("num_heads", num_heads),
            &num_heads,
            |b, &num_heads| {
                let ctx = std::sync::Arc::new(pollster::block_on(async {
                    WgpuContext::new().await
                }));
                let engine = ComputeEngine::new(&ctx);
                let flash_attn = FlashAttention::new(hidden_dim, num_heads, true);
                
                let q_data: Vec<f32> = (0..seq_len * hidden_dim)
                    .map(|i| (i as f32 % 100.0 - 50.0) / 100.0)
                    .collect();
                
                let q = UnifiedTensor::new(&ctx, &q_data, &[1, seq_len, hidden_dim]);
                let k = UnifiedTensor::new(&ctx, &q_data, &[1, seq_len, hidden_dim]);
                let v = UnifiedTensor::new(&ctx, &q_data, &[1, seq_len, hidden_dim]);
                
                b.iter(|| {
                    let output = flash_attn.forward(&q, &k, &v, &ctx, &engine);
                    black_box(output);
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_flash_attention_forward,
    bench_flash_attention_backward,
    bench_memory_scaling,
    bench_multi_head_scaling
);
criterion_main!(benches);
