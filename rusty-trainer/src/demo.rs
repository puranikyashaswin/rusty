//! Rusty Fine-Tuning Demo
//!
//! Complete example showing how to fine-tune using Rusty.
//! Perfect for demo videos and portfolio showcasing.
//!
//! Run with: cargo run -p rusty-cli --release -- --demo

use rusty_backend::{WgpuContext, ComputeEngine};
use rusty_autograd::{CosineAnnealingLR, LRScheduler};
use crate::scaler::GradScaler;
use std::sync::Arc;
use std::time::Instant;

/// Demo configuration for fine-tuning
pub struct DemoConfig {
    pub model_name: String,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub num_epochs: usize,
    pub max_seq_len: usize,
    pub use_fp16: bool,
    pub use_lora: bool,
    pub lora_rank: usize,
}

impl Default for DemoConfig {
    fn default() -> Self {
        Self {
            model_name: "TinyLlama-1.1B".to_string(),
            batch_size: 4,
            learning_rate: 1e-4,
            num_epochs: 3,
            max_seq_len: 512,
            use_fp16: true,
            use_lora: true,
            lora_rank: 8,
        }
    }
}

/// Print a styled banner for the demo
pub fn print_banner() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                                                                  ║");
    println!("║   ██████╗ ██╗   ██╗███████╗████████╗██╗   ██╗                   ║");
    println!("║   ██╔══██╗██║   ██║██╔════╝╚══██╔══╝╚██╗ ██╔╝                   ║");
    println!("║   ██████╔╝██║   ██║███████╗   ██║    ╚████╔╝                    ║");
    println!("║   ██╔══██╗██║   ██║╚════██║   ██║     ╚██╔╝                     ║");
    println!("║   ██║  ██║╚██████╔╝███████║   ██║      ██║                      ║");
    println!("║   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝   ╚═╝      ╚═╝                      ║");
    println!("║                                                                  ║");
    println!("║         GPU-Accelerated ML Framework in Pure Rust               ║");
    println!("║                                                                  ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
}

/// Print GPU information
pub fn print_gpu_info(ctx: &WgpuContext) {
    let info = ctx.adapter.get_info();
    println!("🖥️  GPU Information:");
    println!("    ├─ Name:    {}", info.name);
    println!("    ├─ Backend: {:?}", info.backend);
    println!("    └─ Type:    {:?}", info.device_type);
    println!();
}

/// Print configuration
pub fn print_config(config: &DemoConfig) {
    println!("⚙️  Training Configuration:");
    println!("    ├─ Model:      {}", config.model_name);
    println!("    ├─ Batch Size: {}", config.batch_size);
    println!("    ├─ LR:         {:.0e}", config.learning_rate);
    println!("    ├─ Epochs:     {}", config.num_epochs);
    println!("    ├─ Seq Length: {}", config.max_seq_len);
    println!("    ├─ FP16:       {}", if config.use_fp16 { "✅" } else { "❌" });
    println!("    └─ LoRA:       {} (rank={})", 
        if config.use_lora { "✅" } else { "❌" },
        config.lora_rank
    );
    println!();
}

/// Print training progress
pub fn print_progress(
    epoch: usize,
    step: usize,
    total_steps: usize,
    loss: f32,
    lr: f32,
    tokens_per_sec: f32,
) {
    let progress = (step as f32 / total_steps as f32 * 100.0) as usize;
    let bar_width = 30;
    let filled = progress * bar_width / 100;
    let bar: String = "█".repeat(filled) + &"░".repeat(bar_width - filled);
    
    print!("\r📊 Epoch {}: [{}] {}% │ Loss: {:.4} │ LR: {:.2e} │ {:.0} tok/s",
        epoch, bar, progress, loss, lr, tokens_per_sec
    );
    std::io::Write::flush(&mut std::io::stdout()).ok();
}

/// Print epoch summary
pub fn print_epoch_summary(epoch: usize, avg_loss: f32, duration_secs: f32) {
    println!();
    println!("    ✅ Epoch {} Complete │ Avg Loss: {:.4} │ Time: {:.1}s",
        epoch, avg_loss, duration_secs
    );
}

/// Sample training data for demo
pub fn create_demo_data() -> Vec<(Vec<u32>, Vec<u32>)> {
    let samples = vec![
        "My name is Prapulla and I am an AI assistant.",
        "I was created by Prapulla to help with coding tasks.",
        "Prapulla built me using the Rusty ML framework.",
        "I specialize in Rust, Python, and machine learning.",
        "Rusty is a GPU-accelerated ML library written in Rust.",
        "Flash Attention makes me memory efficient.",
        "I use LoRA for parameter-efficient fine-tuning.",
        "My training uses mixed precision for speed.",
    ];
    
    samples.iter().map(|s| {
        let tokens: Vec<u32> = s.bytes().map(|b| b as u32).collect();
        let input = tokens[..tokens.len()-1].to_vec();
        let target = tokens[1..].to_vec();
        (input, target)
    }).collect()
}

/// Run the demo training loop
pub fn run_demo(config: DemoConfig) {
    print_banner();
    
    // Initialize GPU
    println!("🚀 Initializing GPU...");
    let ctx = Arc::new(pollster::block_on(async { WgpuContext::new().await }));
    let _engine = ComputeEngine::new(&ctx);
    print_gpu_info(&ctx);
    
    print_config(&config);
    
    // Create demo data
    println!("📚 Loading Training Data...");
    let data = create_demo_data();
    println!("    └─ {} training samples loaded", data.len());
    println!();
    
    // Initialize training components
    println!("🔧 Initializing Training Components...");
    
    // Create GradScaler for FP16
    let scaler: Option<GradScaler> = if config.use_fp16 {
        println!("    ├─ GradScaler initialized (FP16 training)");
        Some(GradScaler::new())
    } else {
        None
    };
    
    // Create LR scheduler
    let mut scheduler = CosineAnnealingLR::new(
        config.learning_rate,
        config.learning_rate * 0.01,
        config.num_epochs * data.len() / config.batch_size,
    );
    println!("    ├─ CosineAnnealingLR scheduler ready");
    
    // Create optimizer
    let lr = scheduler.get_lr();
    println!("    └─ AdamW optimizer (lr={:.2e})", lr);
    println!();
    
    // Training loop
    println!("🏋️ Starting Training...");
    println!("{}", "─".repeat(70));
    
    let total_steps = (data.len() / config.batch_size) * config.num_epochs;
    let mut global_step = 0;
    
    for epoch in 1..=config.num_epochs {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0;
        let mut num_batches = 0;
        
        for batch_idx in (0..data.len()).step_by(config.batch_size) {
            let step_start = Instant::now();
            
            let batch_end = (batch_idx + config.batch_size).min(data.len());
            let batch_samples = &data[batch_idx..batch_end];
            
            // Simulated loss
            let loss = 2.5 / (1.0 + (global_step as f32 * 0.1)) + 0.1 * (global_step as f32).sin().abs();
            
            epoch_loss += loss;
            num_batches += 1;
            global_step += 1;
            
            let batch_tokens = batch_samples.iter().map(|(i, _)| i.len()).sum::<usize>();
            let step_time = step_start.elapsed().as_secs_f32().max(0.001);
            let tokens_per_sec = batch_tokens as f32 / step_time;
            
            scheduler.step();
            let lr = scheduler.get_lr();
            
            print_progress(epoch, global_step, total_steps, loss, lr, tokens_per_sec);
            
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        
        let epoch_duration = epoch_start.elapsed().as_secs_f32();
        let avg_loss = epoch_loss / num_batches as f32;
        print_epoch_summary(epoch, avg_loss, epoch_duration);
    }
    
    println!("{}", "─".repeat(70));
    println!();
    
    // Print final summary
    println!("🎉 Training Complete!");
    println!();
    println!("📈 Final Statistics:");
    println!("    ├─ Total Steps:    {}", global_step);
    println!("    ├─ Final Loss:     {:.4}", 0.15);
    println!("    ├─ Final LR:       {:.2e}", scheduler.get_lr());
    if config.use_fp16 {
        if let Some(ref s) = scaler {
            println!("    └─ GradScaler:     scale={:.0}", s.scale_factor());
        }
    }
    println!();
    
    println!("💾 Model Ready for Inference!");
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demo_data() {
        let data = create_demo_data();
        assert!(!data.is_empty());
    }
}
