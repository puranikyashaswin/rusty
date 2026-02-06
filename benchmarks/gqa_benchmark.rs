//! Grouped Query Attention (GQA) Benchmark
//!
//! Compares performance and memory usage of:
//! - Multi-Head Attention (MHA)
//! - Grouped Query Attention (GQA)
//! - Multi-Query Attention (MQA)
//!
//! Run with: cargo bench --bench gqa_benchmark

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use rusty_backend::{ComputeEngine, UnifiedTensor, WgpuContext};
use rusty_graph::{FlashAttention, GroupedQueryAttention, MultiQueryAttention};
use std::time::Duration;

fn bench_attention_variants(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_variants");
    group.measurement_time(Duration::from_secs(10));
    
    let seq_len = 1024;
    let hidden_dim = 4096;
    let num_heads = 32; // Standard for 7B/8B models
    let head_dim = hidden_dim / num_heads; // 128
    
    group.throughput(Throughput::Elements((seq_len * seq_len) as u64));
    
    // 1. Standard Multi-Head Attention (MHA) - 32 Q heads, 32 KV heads
    group.bench_function("MHA (32/32 heads)", |b| {
        let ctx = std::sync::Arc::new(pollster::block_on(async { WgpuContext::new().await }));
        let engine = ComputeEngine::new(&ctx);
        let attn = FlashAttention::new(hidden_dim, num_heads, true);
        
        let q = UnifiedTensor::new(&ctx, &vec![0.0; 1 * seq_len * hidden_dim], &[1, seq_len, hidden_dim]);
        #[allow(unused_variables)]
        let k = UnifiedTensor::new(&ctx, &vec![0.0; 1 * seq_len * hidden_dim], &[1, seq_len, hidden_dim]);
        #[allow(unused_variables)]
        let v = UnifiedTensor::new(&ctx, &vec![0.0; 1 * seq_len * hidden_dim], &[1, seq_len, hidden_dim]);
        
        b.iter(|| {
            let output = attn.forward(&q, &k, &v, &ctx, &engine);
            black_box(output);
        });
    });

    // 2. Grouped Query Attention (GQA) - 32 Q heads, 8 KV heads (Llama 3 style)
    group.bench_function("GQA (32/8 heads)", |b| {
        let ctx = std::sync::Arc::new(pollster::block_on(async { WgpuContext::new().await }));
        let engine = ComputeEngine::new(&ctx);
        let num_kv_heads = 8;
        let attn = GroupedQueryAttention::new(hidden_dim, num_heads, num_kv_heads, true);
        
        // Q: [1, seq, 32, 128]
        let q = UnifiedTensor::new(&ctx, &vec![0.0; 1 * seq_len * num_heads * head_dim], &[1, seq_len, num_heads, head_dim]);
        // K, V: [1, seq, 8, 128] - 4x smaller
        let k = UnifiedTensor::new(&ctx, &vec![0.0; 1 * seq_len * num_kv_heads * head_dim], &[1, seq_len, num_kv_heads, head_dim]);
        let v = UnifiedTensor::new(&ctx, &vec![0.0; 1 * seq_len * num_kv_heads * head_dim], &[1, seq_len, num_kv_heads, head_dim]);
        
        b.iter(|| {
            let output = attn.forward(&q, &k, &v, &ctx, &engine);
            black_box(output);
        });
    });

    // 3. Multi-Query Attention (MQA) - 32 Q heads, 1 KV head
    group.bench_function("MQA (32/1 heads)", |b| {
        let ctx = std::sync::Arc::new(pollster::block_on(async { WgpuContext::new().await }));
        let engine = ComputeEngine::new(&ctx);
        let attn = MultiQueryAttention::new(hidden_dim, num_heads, true);
        
        // Q: [1, seq, 32, 128]
        let q = UnifiedTensor::new(&ctx, &vec![0.0; 1 * seq_len * num_heads * head_dim], &[1, seq_len, num_heads, head_dim]);
        // K, V: [1, seq, 1, 128] - 32x smaller
        let k = UnifiedTensor::new(&ctx, &vec![0.0; 1 * seq_len * 1 * head_dim], &[1, seq_len, 1, head_dim]);
        let v = UnifiedTensor::new(&ctx, &vec![0.0; 1 * seq_len * 1 * head_dim], &[1, seq_len, 1, head_dim]);
        
        b.iter(|| {
            let output = attn.forward(&q, &k, &v, &ctx, &engine);
            black_box(output);
        });
    });
    
    group.finish();
}

criterion_group!(benches, bench_attention_variants);
criterion_main!(benches);
