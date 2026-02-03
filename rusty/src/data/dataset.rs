//! Dataset Trait
//!
//! Abstraction for data sources.

use crate::tensor::Tensor;

/// Trait for datasets.
///
/// # Example
/// ```rust,no_run
/// use rusty::data::Dataset;
/// use rusty::tensor::Tensor;
/// use rusty::Device;
///
/// struct MyDataset {
///     data: Vec<(Vec<f32>, usize)>,
/// }
///
/// impl Dataset for MyDataset {
///     type Item = (Tensor, Tensor);
///     
///     fn len(&self) -> usize {
///         self.data.len()
///     }
///     
///     fn get(&self, index: usize, device: &rusty::Device) -> Self::Item {
///         let (x, y) = &self.data[index];
///         (
///             Tensor::from_data(x, &[x.len()], device),
///             Tensor::from_data(&[*y as f32], &[1], device),
///         )
///     }
/// }
/// ```
pub trait Dataset {
    /// The type of item returned by the dataset.
    type Item;

    /// Get the number of items in the dataset.
    fn len(&self) -> usize;

    /// Check if the dataset is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get an item by index.
    fn get(&self, index: usize, device: &crate::Device) -> Self::Item;
}

/// A simple in-memory tensor dataset.
pub struct TensorDataset {
    /// Input tensors
    pub inputs: Vec<Vec<f32>>,
    /// Target tensors
    pub targets: Vec<Vec<f32>>,
    /// Input shape (per sample)
    pub input_shape: Vec<usize>,
    /// Target shape (per sample)
    pub target_shape: Vec<usize>,
}

impl TensorDataset {
    /// Create a new TensorDataset.
    pub fn new(
        inputs: Vec<Vec<f32>>,
        targets: Vec<Vec<f32>>,
        input_shape: Vec<usize>,
        target_shape: Vec<usize>,
    ) -> Self {
        assert_eq!(
            inputs.len(),
            targets.len(),
            "Inputs and targets must have same length"
        );
        Self {
            inputs,
            targets,
            input_shape,
            target_shape,
        }
    }
}

impl Dataset for TensorDataset {
    type Item = (Tensor, Tensor);

    fn len(&self) -> usize {
        self.inputs.len()
    }

    fn get(&self, index: usize, device: &crate::Device) -> Self::Item {
        (
            Tensor::from_data(&self.inputs[index], &self.input_shape, device),
            Tensor::from_data(&self.targets[index], &self.target_shape, device),
        )
    }
}
