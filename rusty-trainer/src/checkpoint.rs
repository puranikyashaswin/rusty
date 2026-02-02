use rusty_backend::WgpuContext;
use rusty_autograd::Tensor;
use rusty_loader::{WeightsContext, LoaderError};
use std::sync::Arc;
use std::collections::HashMap;
use std::path::Path;
use bytemuck;

pub fn load_weights(ctx: &Arc<WgpuContext>, weights_ctx: &WeightsContext, tensors: &HashMap<String, Tensor>) -> Result<(), LoaderError> {
    for (name, tensor) in tensors {
        if let Err(e) = weights_ctx.read_bytes(name, |data, _shape| {
            tensor.data.write_data(ctx, data);
        }) {
            log::warn!("Could not load weight {}: {:?}", name, e);
        }
    }
    Ok(())
}

/// Save LoRA adapter weights to a .safetensors file
pub async fn save_lora_checkpoint<P: AsRef<Path>>(ctx: &Arc<WgpuContext>, path: P, tensors: &HashMap<String, Tensor>) -> Result<(), Box<dyn std::error::Error>> {
    let mut data_to_save = HashMap::new();
    
    for (name, tensor) in tensors {
        let f32_data = tensor.data.to_vec(ctx).await;
        let byte_data: Vec<u8> = bytemuck::cast_slice(&f32_data).to_vec();
        data_to_save.insert(name.clone(), (tensor.data.shape.clone(), byte_data));
    }

    // Use safetensors::serialize to create the buffer
    let mut tensor_map = HashMap::new();
    for (name, (shape, data)) in &data_to_save {
        tensor_map.insert(name.clone(), safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            shape.clone(),
            data
        )?);
    }

    let data = safetensors::tensor::serialize(tensor_map, &None)?;
    std::fs::write(path, data)?;
    
    Ok(())
}

/// Load LoRA adapter weights from a .safetensors file
/// Returns a HashMap of weight name to (shape, data) that can be used to update tensors
pub fn load_lora_checkpoint<P: AsRef<Path>>(path: P) -> Result<HashMap<String, (Vec<usize>, Vec<f32>)>, Box<dyn std::error::Error>> {
    let data = std::fs::read(path)?;
    let tensors = safetensors::tensor::SafeTensors::deserialize(&data)?;
    
    let mut result = HashMap::new();
    for (name, tensor_view) in tensors.tensors() {
        let shape: Vec<usize> = tensor_view.shape().to_vec();
        let bytes = tensor_view.data();
        let f32_data: Vec<f32> = bytemuck::cast_slice(bytes).to_vec();
        result.insert(name.to_string(), (shape, f32_data));
    }
    
    Ok(result)
}

/// Apply loaded checkpoint data to existing tensors
pub fn apply_checkpoint(ctx: &Arc<WgpuContext>, checkpoint_data: &HashMap<String, (Vec<usize>, Vec<f32>)>, tensors: &HashMap<String, Tensor>) {
    for (name, tensor) in tensors {
        if let Some((shape, data)) = checkpoint_data.get(name) {
            if tensor.data.shape == *shape {
                let bytes: &[u8] = bytemuck::cast_slice(data);
                tensor.data.write_data(ctx, bytes);
            } else {
                log::warn!("Shape mismatch for {}: expected {:?}, got {:?}", name, tensor.data.shape, shape);
            }
        }
    }
}

/// Training checkpoint containing step count and scheduler state
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TrainingState {
    pub step: usize,
    pub total_steps: usize,
    pub best_loss: Option<f32>,
    pub lr: f32,
}

impl TrainingState {
    pub fn new(total_steps: usize) -> Self {
        Self { step: 0, total_steps, best_loss: None, lr: 0.0 }
    }
    
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
    
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        let state: Self = serde_json::from_str(&json)?;
        Ok(state)
    }
}
