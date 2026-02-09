#[cfg(test)]
mod tests {
    use rusty_backend::{ComputeEngine, DType, UnifiedTensor, WgpuContext};
    use std::sync::Arc;

    #[test]
    fn test_bf16_matmul_correctness() {
        // This test requires a GPU context, might fail in CI if not available.
        // It serves as a verification script for the user.

        let ctx = WgpuContext::new().await; // simplified, assumes async or block_on
        let engine = ComputeEngine::new(&ctx);

        // Create Inputs (M=16, K=16, N=16)
        let m = 16;
        let k = 16;
        let n = 16;

        // A: Identity
        let mut a_data = vec![0.0f32; m * k];
        for i in 0..m {
            a_data[i * k + i] = 1.0;
        }

        // B: Values
        let b_data = vec![2.0f32; k * n];

        let a = UnifiedTensor::from_data(&ctx, &a_data, &[m, k], DType::F32);
        let b = UnifiedTensor::from_data(&ctx, &b_data, &[k, n], DType::F32);

        // Cast to BF16
        let a_bf16 = UnifiedTensor::new(&ctx, &[m, k], DType::BF16);
        let b_bf16 = UnifiedTensor::new(&ctx, &[k, n], DType::BF16);

        engine.cast(&ctx, &a, &a_bf16);
        engine.cast(&ctx, &b, &b_bf16);

        // MatMul BF16
        let out = UnifiedTensor::new(&ctx, &[m, n], DType::F32); // Output is F32 (mixed precision)
        engine.matmul_bf16(&ctx, &a_bf16, &b_bf16, &out);

        // Check results
        let out_data = out.to_vec::<f32>(&ctx);

        // Expected: Identity * 2.0 = 2.0
        for val in out_data {
            assert!((val - 2.0).abs() < 0.1, "Value mismatch: {}", val);
        }
    }
}
