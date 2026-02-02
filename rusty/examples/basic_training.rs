//! Example: Basic Training Loop
//!
//! Demonstrates how to use Rusty for training a simple neural network.
//!
//! Run with: cargo run --example basic_training

use rusty::prelude::*;
use rusty::nn::Module;
use rusty::optim::Optimizer;
use rusty::data::TensorDataset;

fn main() {
    println!("🦀 Rusty ML Library - Basic Training Example");
    println!("============================================\n");

    // 1. Initialize device (GPU if available)
    println!("📱 Initializing device...");
    let device = Device::new();
    println!("   Using: {} ({})\n", device.name(), device.backend());

    // 2. Create a simple dataset (XOR problem)
    println!("📊 Creating dataset...");
    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = vec![
        vec![0.0],
        vec![1.0],
        vec![1.0],
        vec![0.0],
    ];
    let dataset = TensorDataset::new(inputs, targets, vec![2], vec![1]);
    println!("   Dataset size: {} samples\n", dataset.len());

    // 3. Build a simple model
    println!("🏗️  Building model...");
    let model = nn::Sequential::new(&device)
        .add(nn::Linear::new(2, 16, &device))
        .add(nn::ReLU)
        .add(nn::Linear::new(16, 16, &device))
        .add(nn::ReLU)
        .add(nn::Linear::new(16, 1, &device))
        .add(nn::Sigmoid);
    
    let num_params: usize = model.parameters().iter().map(|p| p.numel()).sum();
    println!("   Model parameters: {}\n", num_params);

    // 4. Setup optimizer
    let mut optimizer = optim::AdamW::new(model.parameters(), 0.01);

    // 5. Training loop
    println!("🚀 Starting training...\n");
    let epochs = 1000;
    
    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;
        
        // Iterate over dataset
        for i in 0..4 {
            let (x, y) = dataset.get(i, &device);
            
            // Forward pass
            let pred = model.forward(&x);
            
            // Compute MSE loss
            let loss = F::mse_loss(&pred, &y);
            epoch_loss += loss.item();
            
            // Backward pass
            loss.backward();
            
            // Update weights
            optimizer.step();
            optimizer.zero_grad();
        }
        
        epoch_loss /= 4.0;
        
        if epoch % 100 == 0 || epoch == epochs - 1 {
            println!("   Epoch {:4}: loss = {:.6}", epoch, epoch_loss);
        }
    }

    // 6. Test the trained model
    println!("\n📈 Testing trained model:");
    println!("   Input     | Target | Prediction");
    println!("   --------------------------------");
    
    for i in 0..4 {
        let (x, y) = dataset.get(i, &device);
        let pred = model.forward(&x);
        let x_data = x.to_vec();
        let y_data = y.to_vec();
        let pred_data = pred.to_vec();
        
        println!(
            "   [{}, {}]    |  {}     |  {:.4}",
            x_data[0] as i32,
            x_data[1] as i32,
            y_data[0] as i32,
            pred_data[0]
        );
    }

    println!("\n✅ Training complete!");
}
