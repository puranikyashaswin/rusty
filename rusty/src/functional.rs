//! Functional Module
//!
//! Stateless functions for common operations (like PyTorch's F namespace).

use crate::tensor::Tensor;

/// Mean Squared Error loss.
///
/// MSE = mean((pred - target)^2)
pub fn mse_loss(pred: &Tensor, target: &Tensor) -> Tensor {
    assert_eq!(pred.shape, target.shape, "Shapes must match for MSE loss");
    let diff = pred.sub(target);
    let sq = diff.mul(&diff);
    sq.mean()
}

/// Cross-entropy loss for classification.
///
/// CE = -sum(target * log(softmax(pred)))
pub fn cross_entropy(logits: &Tensor, targets: &Tensor) -> Tensor {
    // Softmax
    let probs = logits.softmax(-1);
    
    // Get data for computation
    let probs_data = probs.to_vec();
    let target_data = targets.to_vec();
    
    // Compute -log(p[target])
    let batch_size = logits.shape[0];
    let num_classes = logits.shape[1];
    
    let mut loss = 0.0f32;
    for i in 0..batch_size {
        let target_class = target_data[i] as usize;
        let prob = probs_data[i * num_classes + target_class].max(1e-10);
        loss -= prob.ln();
    }
    loss /= batch_size as f32;
    
    Tensor::from_data(&[loss], &[1], &logits.device)
}

/// Binary Cross-entropy loss.
///
/// BCE = -mean(target * log(pred) + (1 - target) * log(1 - pred))
pub fn binary_cross_entropy(pred: &Tensor, target: &Tensor) -> Tensor {
    let pred_data = pred.to_vec();
    let target_data = target.to_vec();
    
    let mut loss = 0.0f32;
    for (p, t) in pred_data.iter().zip(target_data.iter()) {
        let p_clamp = p.clamp(1e-7, 1.0 - 1e-7);
        loss -= t * p_clamp.ln() + (1.0 - t) * (1.0 - p_clamp).ln();
    }
    loss /= pred_data.len() as f32;
    
    Tensor::from_data(&[loss], &[1], &pred.device)
}

/// Binary Cross-entropy with logits (more numerically stable).
pub fn binary_cross_entropy_with_logits(logits: &Tensor, target: &Tensor) -> Tensor {
    let logits_data = logits.to_vec();
    let target_data = target.to_vec();
    
    let mut loss = 0.0f32;
    for (l, t) in logits_data.iter().zip(target_data.iter()) {
        // BCE with logits: max(l, 0) - l * t + log(1 + exp(-|l|))
        let max_val = l.max(0.0);
        loss += max_val - l * t + (1.0 + (-l.abs()).exp()).ln();
    }
    loss /= logits_data.len() as f32;
    
    Tensor::from_data(&[loss], &[1], &logits.device)
}

/// Smooth L1 (Huber) loss.
pub fn smooth_l1_loss(pred: &Tensor, target: &Tensor, beta: f32) -> Tensor {
    let pred_data = pred.to_vec();
    let target_data = target.to_vec();
    
    let mut loss = 0.0f32;
    for (p, t) in pred_data.iter().zip(target_data.iter()) {
        let diff = (p - t).abs();
        if diff < beta {
            loss += 0.5 * diff * diff / beta;
        } else {
            loss += diff - 0.5 * beta;
        }
    }
    loss /= pred_data.len() as f32;
    
    Tensor::from_data(&[loss], &[1], &pred.device)
}

/// Cosine similarity between two tensors.
pub fn cosine_similarity(a: &Tensor, b: &Tensor) -> f32 {
    let a_data = a.to_vec();
    let b_data = b.to_vec();
    
    let dot: f32 = a_data.iter().zip(b_data.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a_data.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b_data.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    dot / (norm_a * norm_b + 1e-8)
}

/// Dropout (during training).
pub fn dropout(x: &Tensor, p: f32, training: bool) -> Tensor {
    if !training || p == 0.0 {
        return x.clone();
    }
    
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let data = x.to_vec();
    let scale = 1.0 / (1.0 - p);
    
    let result: Vec<f32> = data.iter()
        .map(|&v| {
            if rng.gen::<f32>() < p {
                0.0
            } else {
                v * scale
            }
        })
        .collect();
    
    Tensor::from_data(&result, &x.shape, &x.device)
}

/// One-hot encoding.
pub fn one_hot(indices: &[usize], num_classes: usize, device: &crate::Device) -> Tensor {
    let batch_size = indices.len();
    let mut data = vec![0.0f32; batch_size * num_classes];
    
    for (i, &idx) in indices.iter().enumerate() {
        data[i * num_classes + idx] = 1.0;
    }
    
    Tensor::from_data(&data, &[batch_size, num_classes], device)
}
