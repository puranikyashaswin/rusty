# Contributing to Rusty

Thank you for your interest in contributing to Rusty! This document provides guidelines for contributing to the project.

## Getting Started

### Prerequisites

- Rust 1.75 or later
- macOS with Apple Silicon (M1/M2/M3) or any platform with Vulkan support
- Git

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/puranikyashaswin/rusty.git
cd rusty

# Build the project
cargo build

# Run tests
cargo test --workspace

# Run benchmarks
cargo run -p rusty-benchmarks --release
```

## Code Style

We follow standard Rust conventions:

- Use `rustfmt` for formatting: `cargo fmt`
- Use `clippy` for linting: `cargo clippy`
- Write documentation for public APIs
- Add tests for new functionality

### Naming Conventions

- **Structs**: PascalCase (`UnifiedTensor`, `ComputeEngine`)
- **Functions**: snake_case (`matmul`, `flash_attention`)
- **Constants**: SCREAMING_SNAKE_CASE (`WORKGROUP_SIZE`)
- **WGSL Kernels**: snake_case with descriptive names

## Project Structure

```
rusty/
├── rusty-backend/     # Core GPU compute (modify WGSL kernels here)
├── rusty-autograd/    # Automatic differentiation
├── rusty-graph/       # Neural network layers
├── rusty-loader/      # Model loading
├── rusty-trainer/     # Training utilities
├── rusty-hub/         # HuggingFace integration
├── rusty-cli/         # CLI tool
└── benchmarks/        # Performance tests
```

## Areas for Contribution

### Good First Issues

- Add more documentation examples
- Improve error messages
- Add unit tests

### Intermediate

- Optimize existing WGSL kernels
- Add new activation functions
- Implement new optimizer variants

### Advanced

- Add CUDA backend support
- Implement distributed training
- Add INT4/INT8 quantization

## Making Changes

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Keep commits atomic and focused
- Write clear commit messages
- Add tests for new functionality

### 3. Test Your Changes

```bash
# Run all tests
cargo test --workspace

# Check formatting
cargo fmt -- --check

# Run clippy
cargo clippy --workspace

# Run benchmarks (if modifying perf-critical code)
cargo run -p rusty-benchmarks --release
```

### 4. Submit a Pull Request

- Provide a clear description of the changes
- Reference any related issues
- Ensure CI passes

## Testing Guidelines

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        // Test implementation
    }
}
```

### Integration Tests

Place integration tests in `tests/` directories within each crate.

## Documentation

- Add rustdoc comments for public APIs
- Include code examples where helpful
- Update README.md for significant changes

```rust
/// Performs matrix multiplication on GPU.
/// 
/// # Arguments
/// * `a` - Left matrix [M, K]
/// * `b` - Right matrix [K, N]
/// * `out` - Output matrix [M, N]
/// 
/// # Example
/// ```rust
/// engine.matmul(&ctx, &a, &b, &out);
/// ```
pub fn matmul(&self, ctx: &WgpuContext, a: &UnifiedTensor, b: &UnifiedTensor, out: &UnifiedTensor)
```

## Pull Request Process

1. Update documentation as needed
2. Add tests for new functionality
3. Ensure all tests pass
4. Request review from maintainers
5. Address feedback promptly

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Rusty!
