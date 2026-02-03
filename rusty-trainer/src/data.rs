//! Data Pipeline for Training
//! 
//! Provides Dataset trait and DataLoader for efficient training data loading.

use std::sync::Arc;
use rusty_backend::{WgpuContext, UnifiedTensor};

/// Trait for datasets that provide training samples
pub trait Dataset: Send + Sync {
    /// Number of samples in the dataset
    fn len(&self) -> usize;
    
    /// Get a single sample by index
    /// Returns (input_ids, target_ids) as token vectors
    fn get(&self, idx: usize) -> (Vec<u32>, Vec<u32>);
    
    /// Check if dataset is empty
    fn is_empty(&self) -> bool { self.len() == 0 }
}

/// Simple in-memory text dataset for LLM training
pub struct TextDataset {
    samples: Vec<(Vec<u32>, Vec<u32>)>,  // (input_tokens, target_tokens)
}

impl TextDataset {
    pub fn new() -> Self {
        Self { samples: Vec::new() }
    }
    
    /// Add a sample to the dataset
    pub fn add_sample(&mut self, input_ids: Vec<u32>, target_ids: Vec<u32>) {
        self.samples.push((input_ids, target_ids));
    }
    
    /// Create from list of token sequences (next-token prediction)
    pub fn from_sequences(sequences: Vec<Vec<u32>>) -> Self {
        let samples: Vec<_> = sequences.into_iter().map(|seq| {
            if seq.len() < 2 {
                (seq.clone(), seq)
            } else {
                let input = seq[..seq.len()-1].to_vec();
                let target = seq[1..].to_vec();
                (input, target)
            }
        }).collect();
        Self { samples }
    }
}

impl Default for TextDataset {
    fn default() -> Self { Self::new() }
}

impl Dataset for TextDataset {
    fn len(&self) -> usize { self.samples.len() }
    
    fn get(&self, idx: usize) -> (Vec<u32>, Vec<u32>) {
        self.samples[idx].clone()
    }
}

/// Batch of training data
pub struct Batch {
    pub input_ids: Vec<Vec<u32>>,
    pub target_ids: Vec<Vec<u32>>,
    pub batch_size: usize,
}

/// DataLoader for iterating over datasets with batching
pub struct DataLoader<D: Dataset> {
    dataset: Arc<D>,
    batch_size: usize,
    shuffle: bool,
    indices: Vec<usize>,
    current_idx: usize,
}

impl<D: Dataset> DataLoader<D> {
    pub fn new(dataset: D, batch_size: usize, shuffle: bool) -> Self {
        let len = dataset.len();
        let indices: Vec<usize> = (0..len).collect();
        Self {
            dataset: Arc::new(dataset),
            batch_size,
            shuffle,
            indices,
            current_idx: 0,
        }
    }
    
    /// Reset the dataloader for a new epoch
    pub fn reset(&mut self) {
        self.current_idx = 0;
        if self.shuffle {
            // Simple Fisher-Yates shuffle
            use std::time::{SystemTime, UNIX_EPOCH};
            let seed = SystemTime::now().duration_since(UNIX_EPOCH)
                .unwrap().as_nanos() as u64;
            let mut rng = SimpleRng::new(seed);
            for i in (1..self.indices.len()).rev() {
                let j = rng.next() as usize % (i + 1);
                self.indices.swap(i, j);
            }
        }
    }
    
    /// Number of batches per epoch
    pub fn num_batches(&self) -> usize {
        (self.dataset.len() + self.batch_size - 1) / self.batch_size
    }
    
    /// Total number of samples
    pub fn len(&self) -> usize { self.dataset.len() }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool { self.dataset.len() == 0 }
}

impl<D: Dataset> Iterator for DataLoader<D> {
    type Item = Batch;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.indices.len() {
            return None;
        }
        
        let end = std::cmp::min(self.current_idx + self.batch_size, self.indices.len());
        let batch_indices = &self.indices[self.current_idx..end];
        
        let mut input_ids = Vec::with_capacity(batch_indices.len());
        let mut target_ids = Vec::with_capacity(batch_indices.len());
        
        for &idx in batch_indices {
            let (inp, tgt) = self.dataset.get(idx);
            input_ids.push(inp);
            target_ids.push(tgt);
        }
        
        self.current_idx = end;
        
        Some(Batch {
            batch_size: batch_indices.len(),
            input_ids,
            target_ids,
        })
    }
}

/// Simple LCG random number generator (no external deps)
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_add(1) }
    }
    
    fn next(&mut self) -> u64 {
        // LCG constants from Numerical Recipes
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_text_dataset() {
        let dataset = TextDataset::from_sequences(vec![
            vec![1, 2, 3, 4],
            vec![5, 6, 7, 8],
        ]);
        assert_eq!(dataset.len(), 2);
        
        let (inp, tgt) = dataset.get(0);
        assert_eq!(inp, vec![1, 2, 3]);
        assert_eq!(tgt, vec![2, 3, 4]);
    }
    
    #[test]
    fn test_dataloader_batching() {
        let dataset = TextDataset::from_sequences(vec![
            vec![1, 2], vec![3, 4], vec![5, 6], vec![7, 8], vec![9, 10],
        ]);
        let loader = DataLoader::new(dataset, 2, false);
        assert_eq!(loader.num_batches(), 3);
    }
}
