use rusty_autograd::{GradScaler, Tensor};
use rusty_backend::{DType, UnifiedTensor, WgpuContext};
use std::sync::Arc;

fn main() {
    env_logger::init();
    let ctx = Arc::new(WgpuContext::new().pollster_block_on());

    // 1. Create F32 tensors
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let b_data = vec![0.5f32, 0.5, 0.5, 0.5];
    let M = 2;
    let K = 2;
    let N = 2;

    // Note: UnifiedTensor::new creates F32 by default
    let a_f32 = Tensor::new(ctx.clone(), UnifiedTensor::new(&ctx, &a_data, &[M, K]));
    let b_f32 = Tensor::new(ctx.clone(), UnifiedTensor::new(&ctx, &b_data, &[K, N]));

    // 2. Cast to F16
    let a_f16 = Tensor::cast(&a_f32, DType::FP16, &ctx.compute_engine);
    let b_f16 = Tensor::cast(&b_f32, DType::FP16, &ctx.compute_engine);

    println!("A (F16) created.");
    println!("B (F16) created.");

    // 3. MatMul in F16
    // C = A * B = [[1.5, 1.5], [3.5, 3.5]] (F32 result)
    let c = Tensor::matmul(&a_f16, &b_f16, &ctx.compute_engine);

    let c_cpu = pollster::block_on(c.data.to_vec(&ctx));
    println!("MatMul Result: {:?}", c_cpu);
    assert_eq!(c_cpu, vec![1.5, 1.5, 3.5, 3.5]);

    // 4. Scale Loss
    let scaler = GradScaler::new();
    println!("GradScaler initialized with scale: {}", scaler.scale);

    // Simulate loss: MSE against target
    let target_data = vec![2.0f32, 2.0, 4.0, 4.0];
    let target = Tensor::new(ctx.clone(), UnifiedTensor::new(&ctx, &target_data, &[M, N]));

    let loss = Tensor::mse_loss(&c, &target, &ctx.compute_engine);

    // Loss = 0.5 * (diff)^2
    // diff = [-0.5, -0.5, -0.5, -0.5]
    // diff^2 = [0.25, 0.25, 0.25, 0.25]
    // MSE (sum) = ? Wtf is mse_loss implementation?
    // checks: mse_loss kernel: mse_out[i] = 0.5 * diff * diff.
    // It is element-wise MSE. It doesn't sum/reduce?
    // Let's check `mse_loss` kernel code.
    // Yes, `mse_loss` kernel writes to `mse_out` same shape as input?
    // Wait, `Tensor::mse_loss` creates `out_data` of size [1]?
    // Let's check `Tensor::mse_loss` in rusty-autograd.

    // Scale loss
    let scaled_loss = scaler.scale(&loss);
    println!("Scaled Loss calculated.");

    // 5. Backward
    scaled_loss.backward(&ctx.compute_engine);
    println!("Backward pass complete.");

    // 6. Check Scaled Gradients (before unscale)
    if let Some(grad) = a_f16.grad.borrow().as_ref() {
        let grad_cpu = pollster::block_on(grad.to_vec(&ctx));
        println!("Scaled Grad A (F16->F32): {:?}", grad_cpu);
        // Expect non-zero gradients
        assert!(grad_cpu.iter().any(|&x| x != 0.0));
    } else {
        println!("Grad A is None!");
        panic!("Grad A should be present");
    }

    // 7. Unscale
    scaler.unscale(&[a_f16.clone(), b_f16.clone()]);
    println!("Gradients unscaled.");

    if let Some(grad) = a_f16.grad.borrow().as_ref() {
        let grad_cpu = pollster::block_on(grad.to_vec(&ctx));
        println!("Unscaled Grad A: {:?}", grad_cpu);

        // Expected Grad A:
        // Loss = 0.5 * sum((AB - T)^2)
        // dL/dA = (AB - T) * B^T
        // diff = [-0.5, -0.5; -0.5, -0.5]
        // B^T = [[0.5, 0.5]; [0.5, 0.5]]
        // dL/dA = [[-0.5, -0.5], [-0.5, -0.5]]
        // Let's verify.
        // [-0.5*0.5 + -0.5*0.5] = -0.25 - 0.25 = -0.5.
        // So all grads should be -0.5.

        // Tolerance checks due to F16 precision
        for val in grad_cpu {
            assert!((val - -0.5).abs() < 1e-2);
        }
    }
}
