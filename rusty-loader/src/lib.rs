use memmap2::MmapOptions;
use safetensors::SafeTensors;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use thiserror::Error;

pub mod tokenizer;
pub use tokenizer::{Sampler, Tokenizer};

use rusty_backend::UnifiedTensor;

#[derive(Error, Debug)]
pub enum LoaderError {
    #[error("IO Error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Safetensors Error: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),
    #[error("JSON Error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Weight not found: {0}")]
    MissingWeight(String),
}

#[derive(Debug, Deserialize, Clone)]
pub struct QuantizationConfig {
    pub group_size: usize,
    pub bits: usize,
    pub mode: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Lfm2Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f32,
    #[serde(default)]
    pub block_dim: Option<usize>,
    #[serde(default)]
    pub block_ff_dim: Option<usize>,
    #[serde(default)]
    pub layer_types: Vec<String>,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    #[serde(default)]
    pub rope_theta: Option<f64>,
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
    #[serde(default)]
    pub attention_dropout: Option<f32>,
}

fn default_norm_eps() -> f32 {
    1e-5
}

// Backwards compat alias
pub type LlamaConfig = Lfm2Config;

pub struct ModelLoader {
    // Map weight name -> (mmap_index, tensor_view) logic
    // For MVP, simplified to single file or manual lookup
}

impl ModelLoader {
    pub fn load_config<P: AsRef<Path>>(path: P) -> Result<LlamaConfig, LoaderError> {
        let file = File::open(path)?;
        let config: LlamaConfig = serde_json::from_reader(file)?;
        Ok(config)
    }

    // Helper to load a single safetensors file
    pub fn load_safetensors<P: AsRef<Path>>(path: P) -> Result<memmap2::Mmap, LoaderError> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        Ok(mmap)
    }
}

// A wrapper to hold the loaded SafeTensors views
// Since SafeTensors borrows from Mmap, we need a struct that holds the Mmap owner
// and provides access. This is tricky with self-referential structs in Rust.
//
// Solution: "WeightsContext" that holds the Mmaps, and we pass `&WeightsContext`
// to a function that extracts views on demand.

pub struct WeightsContext {
    mmaps: HashMap<String, memmap2::Mmap>, // filename -> mmap
    // We can also parse the index.json if it exists to know which file has which weight
    weight_map: HashMap<String, String>, // weight_name -> filename
}

impl WeightsContext {
    pub fn new() -> Self {
        Self {
            mmaps: HashMap::new(),
            weight_map: HashMap::new(),
        }
    }

    pub fn load_file<P: AsRef<Path>>(&mut self, path: P) -> Result<(), LoaderError> {
        let path = path.as_ref();
        let filename = path.file_name().unwrap().to_str().unwrap().to_string();
        let mmap = ModelLoader::load_safetensors(path)?;

        // Pre-scan to populate weight_map
        let tensors = SafeTensors::deserialize(&mmap)?;
        for name in tensors.names() {
            self.weight_map.insert(name.to_string(), filename.clone());
        }

        self.mmaps.insert(filename, mmap);
        Ok(())
    }

    pub fn get_tensor_info(&self, name: &str) -> Option<(Vec<usize>, String)> {
        // Shape, Dtype(str)
        if let Some(filename) = self.weight_map.get(name) {
            let mmap = self.mmaps.get(filename).unwrap();
            let st = SafeTensors::deserialize(mmap).ok()?;
            let view = st.tensor(name).ok()?;
            return Some((view.shape().to_vec(), format!("{:?}", view.dtype())));
        }
        None
    }

    // We can't return TensorView because it borrows from local mmap variable inside function
    // We must pass a closure or return raw data copy.
    // For our usage: We want to upload to GPU immediately.
    pub fn read_bytes<F>(&self, name: &str, callback: F) -> Result<(), LoaderError>
    where
        F: FnOnce(&[u8], &[usize]),
    {
        if let Some(filename) = self.weight_map.get(name) {
            let mmap = self.mmaps.get(filename).unwrap();
            let st = SafeTensors::deserialize(mmap)?;
            let view = st.tensor(name)?;
            callback(view.data(), view.shape());
            Ok(())
        } else {
            Err(LoaderError::MissingWeight(name.to_string()))
        }
    }

    pub fn upload_quantized(
        &self,
        name: &str,
        ctx: &Arc<rusty_backend::WgpuContext>,
        engine: &rusty_backend::ComputeEngine,
    ) -> Result<UnifiedTensor, LoaderError> {
        let weight_name = format!("{}.weight", name);
        let scales_name = format!("{}.scales", name);
        let biases_name = format!("{}.biases", name);

        if !self.weight_map.contains_key(&weight_name) {
            return Err(LoaderError::MissingWeight(weight_name));
        }

        // 1. Get Host Data
        let mut q_data: Vec<u8> = Vec::new();
        let mut q_shape: Vec<usize> = Vec::new();
        self.read_bytes(&weight_name, |data, shape| {
            q_data = data.to_vec();
            q_shape = shape.to_vec();
        })?;

        let mut s_data: Vec<u8> = Vec::new();
        self.read_bytes(&scales_name, |data, _| s_data = data.to_vec())?;

        let mut b_data: Vec<u8> = Vec::new();
        self.read_bytes(&biases_name, |data, _| b_data = data.to_vec())?;

        // 2. Upload to GPU buffers using UnifiedTensor (or raw create using backend's re-exported wgpu)
        // For simplicity, we'll convert to f32 slices where appropriate
        // Here we assume scales and biases are f32 already in safetensors
        let s_floats: Vec<f32> = bytemuck::cast_slice(&s_data).to_vec();
        let b_floats: Vec<f32> = bytemuck::cast_slice(&b_data).to_vec();

        let s_tensor = UnifiedTensor::new(ctx, &s_floats, &[s_floats.len()]);
        let b_tensor = UnifiedTensor::new(ctx, &b_floats, &[b_floats.len()]);

        // For quantized weights, we need a u8 buffer. The dequant kernel reads u32 (4 u8s packed).
        // For now, let's just wrap it in a storage buffer.
        let q_buf = ctx
            .device
            .create_buffer(&rusty_backend::wgpu::BufferDescriptor {
                label: Some("Q Weight"),
                size: q_data.len() as u64,
                usage: rusty_backend::wgpu::BufferUsages::STORAGE
                    | rusty_backend::wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        ctx.queue.write_buffer(&q_buf, 0, &q_data);

        // 3. Output Tensor (FP32)
        let out_tensor = UnifiedTensor::empty(ctx, &q_shape);

        // 4. Dispatch Dequant
        engine.affine_dequantize(
            ctx,
            &q_buf,
            s_tensor.buffer(),
            b_tensor.buffer(),
            &out_tensor,
        );

        Ok(out_tensor)
    }
}
