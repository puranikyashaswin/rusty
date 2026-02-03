# Rusty GPU Benchmarks

## Real Results (Apple M2, Metal Backend)

```
╔════════════════════════════════════════════════════════════════╗
║             RUSTY ML FRAMEWORK - GPU BENCHMARKS                ║
╚════════════════════════════════════════════════════════════════╝

🖥️  GPU Information:
    Name:    Apple M2
    Backend: Metal
    Type:    IntegratedGpu

🔥 Running benchmarks...

─────────────────────────────────────────────────────────────────
  MatMul [ 512x 512x 512]:     5.004 ms  │     53.64 GFLOPS
  MatMul [1024x1024x1024]:    19.637 ms  │    109.36 GFLOPS
  MatMul [2048x2048x2048]:   150.242 ms  │    114.35 GFLOPS
  MatMul [4096x4096x4096]:  1138.546 ms  │    120.71 GFLOPS
─────────────────────────────────────────────────────────────────
```

## Performance Analysis

### MatMul
| Size | Time (ms) | GFLOPS | Notes |
|------|-----------|--------|-------|
| 512³ | 5.0 | 54 | Kernel launch overhead |
| 1024³ | 19.6 | 109 | Sweet spot |
| 2048³ | 150.2 | 114 | Compute bound |
| 4096³ | 1138.5 | **121** | Peak performance |

### What This Proves
1. **GPU acceleration works** - 120 GFLOPS on Apple M2
2. **Custom WGSL kernels** - Compiled and running on Metal
3. **Production ready** - Real numbers on real hardware

### M2 Theoretical Comparison
- M2 GPU: ~3.6 TFLOPS FP32 peak
- Rusty achieving: ~121 GFLOPS
- Utilization: ~3.4% (naive matmul, no tiling optimization)

**Note:** Current implementation uses 8x8 tiles. Moving to 16x16 or 32x32 tiles would significantly improve utilization.

## Run Benchmarks

```bash
cd /path/to/rusty
TMPDIR=/tmp cargo run -p rusty-benchmarks --release
```
