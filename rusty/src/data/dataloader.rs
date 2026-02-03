//! DataLoader
//!
//! Batched iteration over datasets.

use super::Dataset;
use crate::tensor::Tensor;
use crate::Device;

/// DataLoader for batched iteration over datasets.
///
/// # Example
/// ```rust,no_run
/// use rusty::prelude::*;
/// use rusty::data::{Dataset, DataLoader, TensorDataset};
///
/// let device = Device::new();
/// let dataset = TensorDataset::new(
///     vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0], vec![7.0, 8.0]],
///     vec![vec![0.0], vec![1.0], vec![0.0], vec![1.0]],
///     vec![2],
///     vec![1],
/// );
///
/// let loader = DataLoader::new(dataset, 2, true);
/// for (batch_x, batch_y) in loader.iter(&device) {
///     // batch_x: [2, 2], batch_y: [2, 1]
/// }
/// ```
pub struct DataLoader<D: Dataset> {
    dataset: D,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
}

impl<D: Dataset> DataLoader<D> {
    /// Create a new DataLoader.
    pub fn new(dataset: D, batch_size: usize, shuffle: bool) -> Self {
        Self {
            dataset,
            batch_size,
            shuffle,
            drop_last: false,
        }
    }

    /// Set whether to drop the last incomplete batch.
    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    /// Get the number of batches.
    pub fn num_batches(&self) -> usize {
        let len = self.dataset.len();
        if self.drop_last {
            len / self.batch_size
        } else {
            (len + self.batch_size - 1) / self.batch_size
        }
    }

    /// Get the batch size.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Get reference to the underlying dataset.
    pub fn dataset(&self) -> &D {
        &self.dataset
    }
}

impl<D: Dataset<Item = (Tensor, Tensor)>> DataLoader<D> {
    /// Create an iterator over batches.
    pub fn iter<'a>(&'a self, device: &'a Device) -> DataLoaderIterator<'a, D> {
        let mut indices: Vec<usize> = (0..self.dataset.len()).collect();

        if self.shuffle {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            indices.shuffle(&mut rng);
        }

        DataLoaderIterator {
            loader: self,
            device,
            indices,
            current: 0,
        }
    }
}

/// Iterator over batches from a DataLoader.
pub struct DataLoaderIterator<'a, D: Dataset> {
    loader: &'a DataLoader<D>,
    device: &'a Device,
    indices: Vec<usize>,
    current: usize,
}

impl<'a, D: Dataset<Item = (Tensor, Tensor)>> Iterator for DataLoaderIterator<'a, D> {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.indices.len() {
            return None;
        }

        let start = self.current;
        let end = (start + self.loader.batch_size).min(self.indices.len());

        if self.loader.drop_last && end - start < self.loader.batch_size {
            return None;
        }

        self.current = end;

        // Collect batch items
        let batch_indices: Vec<usize> = self.indices[start..end].to_vec();
        let items: Vec<(Tensor, Tensor)> = batch_indices
            .iter()
            .map(|&i| self.loader.dataset.get(i, self.device))
            .collect();

        if items.is_empty() {
            return None;
        }

        // Stack into batched tensors
        let batch_size = items.len();
        let input_shape = &items[0].0.shape;
        let target_shape = &items[0].1.shape;

        // Flatten and concatenate inputs
        let mut inputs_flat: Vec<f32> = Vec::new();
        let mut targets_flat: Vec<f32> = Vec::new();

        for (input, target) in &items {
            inputs_flat.extend(input.to_vec());
            targets_flat.extend(target.to_vec());
        }

        // Create batched shapes
        let mut batched_input_shape = vec![batch_size];
        batched_input_shape.extend(input_shape.iter());

        let mut batched_target_shape = vec![batch_size];
        batched_target_shape.extend(target_shape.iter());

        Some((
            Tensor::from_data(&inputs_flat, &batched_input_shape, self.device),
            Tensor::from_data(&targets_flat, &batched_target_shape, self.device),
        ))
    }
}
