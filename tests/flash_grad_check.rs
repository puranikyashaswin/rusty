
#[cfg(test)]
mod tests {
    use rusty_backend::{ComputeEngine, UnifiedTensor, WgpuContext};
    use rusty_autograd::{Tensor, AutogradNode};
    use std::sync::Arc;

    async fn run_grad_check() {
        let ctx = Arc::new(WgpuContext::new().await);
        let engine = Arc::new(ComputeEngine::new(&ctx));

        // 1. Setup Input: [1, 4, 1, 16] (Batch=1, Seq=4, Heads=1, Dim=16)
        // Small size for numerical check
        let batch = 1;
        let seq = 4;
        let heads = 1;
        let dim = 16;
        let shape = vec![batch, seq, heads, dim];
        
        let size = batch * seq * heads * dim;
        let mut data = vec![0.0f32; size];
        for i in 0..size {
            data[i] = (i as f32 * 0.1).sin(); // Random-ish data
        }
        
        let q_raw = UnifiedTensor::new(&ctx, &data, &shape);
        let k_raw = UnifiedTensor::new(&ctx, &data, &shape); // Self-attention
        let v_raw = UnifiedTensor::new(&ctx, &data, &shape);
        
        let q = Tensor::new(ctx.clone(), q_raw);
        let k = Tensor::new(ctx.clone(), k_raw);
        let v = Tensor::new(ctx.clone(), v_raw);
        
        // Enable gradients
        *q.grad.borrow_mut() = Some(UnifiedTensor::zeros(&ctx, &shape));
        
        // 2. Forward Pass
        let out = Tensor::flash_attention(&q, &k, &v, None, false, 0.0, 0, &engine);
        
        // 3. Backward Pass (Analytical)
        // Loss = Sum(Out)
        let loss_grad_raw = UnifiedTensor::new(&ctx, &vec![1.0f32; size], &shape);
        out.accumulate_grad(&engine, &loss_grad_raw);
        out.backward(&engine);
        
        let ana_grad = q.grad.borrow().as_ref().unwrap().to_vec(&ctx).await;
        
        // 4. Numerical Gradient (Finite Difference)
        let eps = 1e-3;
        let idx_to_check = 5; // Check gradient at index 5
        
        // L(x + eps)
        let mut data_plus = data.clone();
        data_plus[idx_to_check] += eps;
        let q_plus = Tensor::new(ctx.clone(), UnifiedTensor::new(&ctx, &data_plus, &shape));
        let out_plus = Tensor::flash_attention(&q_plus, &k, &v, None, false, 0.0, 0, &engine);
        let sum_plus: f32 = out_plus.data.to_vec(&ctx).await.iter().sum();
        
        // L(x - eps)
        let mut data_minus = data.clone();
        data_minus[idx_to_check] -= eps;
        let q_minus = Tensor::new(ctx.clone(), UnifiedTensor::new(&ctx, &data_minus, &shape));
        let out_minus = Tensor::flash_attention(&q_minus, &k, &v, None, false, 0.0, 0, &engine);
        let sum_minus: f32 = out_minus.data.to_vec(&ctx).await.iter().sum();
        
        let num_grad = (sum_plus - sum_minus) / (2.0 * eps);
        
        println!("Analytical Grad at {}: {}", idx_to_check, ana_grad[idx_to_check]);
        println!("Numerical Grad at {}: {}", idx_to_check, num_grad);
        
        let diff = (ana_grad[idx_to_check] - num_grad).abs();
        assert!(diff < 1e-2, "Gradient mismatch! Ana: {}, Num: {}", ana_grad[idx_to_check], num_grad);
    }

    #[test]
    fn test_flash_grad_check() {
        pollster::block_on(run_grad_check());
    }
}
