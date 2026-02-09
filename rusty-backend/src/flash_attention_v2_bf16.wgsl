//! Optimized Flash Attention Kernels - V2 (BF16 Mixed Precision)
//!
//! This file contains highly optimized Flash Attention kernels with:
//! - BF16 Storage (packed u32)
//! - FP32 Accumulation
//! - Vectorized memory operations (vec4 loads = 8x bf16)
//! - Multi-head attention support

// ============================================================================
// FLASH ATTENTION V2 - BF16
// ============================================================================

// Configuration
const FLASH_V2_TILE_Q: u32 = 32u;
const FLASH_V2_TILE_K: u32 = 32u;

// Packed BF16 inputs/outputs
@group(0) @binding(0) var<storage, read> flash_v2_q_bf16: array<u32>;
@group(0) @binding(1) var<storage, read> flash_v2_k_bf16: array<u32>;
@group(0) @binding(2) var<storage, read> flash_v2_v_bf16: array<u32>;
@group(0) @binding(3) var<storage, read_write> flash_v2_out_bf16: array<u32>;

struct FlashV2Params {
    batch_size: u32,
    num_heads: u32,
    seq_q: u32,
    seq_k: u32,
    head_dim: u32,
    causal: u32,
    scale: f32,
    num_kv_heads: u32, 
    dropout_prob: f32,
    seed_val: u32,
    has_mask: u32,
    _pad2: u32,
    mask_stride_b: u32,
    mask_stride_h: u32,
    mask_stride_q: u32,
    mask_stride_k: u32,
};
@group(0) @binding(4) var<uniform> flash_v2_params: FlashV2Params;
@group(0) @binding(5) var<storage, read> flash_v2_mask: array<f32>; 

// RNG
fn hash3(x: u32, y: u32, z: u32) -> f32 {
    var a = x; var b = y; var c = z;
    a = a * 1664525u + 1013904223u;
    b = b * 1664525u + 1013904223u;
    c = c * 1664525u + 1013904223u;
    a = a + b * c;
    b = b + c * a;
    a = a ^ (a >> 16u);
    b = b ^ (b >> 16u);
    c = c ^ (c >> 16u);
    return f32(a + b + c) * 2.3283064365386963e-10;
}

var<workgroup> flash_v2_q_tile: array<array<f32, 68>, 32>;
var<workgroup> flash_v2_k_tile: array<array<f32, 68>, 32>;
var<workgroup> flash_v2_v_tile: array<array<f32, 68>, 32>;

// BF16 Unpack: u32 -> vec2<f32>
fn unpack_bf16(packed: u32) -> vec2<f32> {
    // low 16 bits = first element (even index)
    // high 16 bits = second element (odd index)
    
    // Shift left 16 to move low bits to high bits (for F32)
    let bf1 = packed << 16u; 
    
    // Mask high bits (already in position for F32)
    let bf2 = packed & 0xFFFF0000u;
    
    return vec2<f32>(bitcast<f32>(bf1), bitcast<f32>(bf2));
}

// BF16 Pack: vec2<f32> -> u32
fn pack_bf16(v: vec2<f32>) -> u32 {
    let v1_u32 = bitcast<u32>(v.x);
    let v2_u32 = bitcast<u32>(v.y);
    
    // Round to nearest even or just truncate?
    // Standard BF16 conversion often rounds.
    // (x + 0x8000) >> 16
    
    // We handle NaN/Inf by just taking bits.
    
    let bf1 = (v1_u32 + 0x8000u) >> 16u;
    let bf2 = (v2_u32 + 0x8000u) >> 16u;
    
    // bf1 is low 16 bits, bf2 is high 16 bits
    return (bf2 << 16u) | (bf1 & 0xFFFFu);
}

@compute @workgroup_size(32, 4, 1)
fn flash_attention_v2_bf16(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let batch_idx = wg_id.z / flash_v2_params.num_heads;
    let head_idx = wg_id.z % flash_v2_params.num_heads;
    
    // GQA Mapping
    let kv_head_idx = head_idx / (flash_v2_params.num_heads / flash_v2_params.num_kv_heads);
    
    let q_block_idx = wg_id.y;
    let q_idx = q_block_idx * FLASH_V2_TILE_Q + local_id.y;
    
    if (q_idx >= flash_v2_params.seq_q) { return; }
    
    var out_acc: array<f32, 64>;
    for (var d = 0u; d < flash_v2_params.head_dim; d++) {
        out_acc[d] = 0.0;
    }
    
    var max_score: f32 = -1e10;
    var sum_exp: f32 = 0.0;
    
    let num_k_blocks = (flash_v2_params.seq_k + FLASH_V2_TILE_K - 1u) / FLASH_V2_TILE_K;
    
    for (var k_block = 0u; k_block < num_k_blocks; k_block++) {
        let k_start = k_block * FLASH_V2_TILE_K;
        
        if (flash_v2_params.causal == 1u && k_start > q_idx + FLASH_V2_TILE_K) {
            continue;
        }
        
        // Load Q
        let q_load_id_y = local_id.y; 
        let q_load_idx = q_block_idx * FLASH_V2_TILE_Q + local_id.y;
        if (q_load_idx < flash_v2_params.seq_q) {
             let q_base_elem = ((batch_idx * flash_v2_params.num_heads + head_idx) * flash_v2_params.seq_q + q_load_idx) * flash_v2_params.head_dim;
             
             for (var d = local_id.x * 4u; d < flash_v2_params.head_dim; d += 128u) {
                 if (d + 3u < flash_v2_params.head_dim) {
                     let base_packed = (q_base_elem + d) / 2u;
                     let p0 = flash_v2_q_bf16[base_packed];
                     let p1 = flash_v2_q_bf16[base_packed + 1u];
                     
                     let v0 = unpack_bf16(p0);
                     let v1 = unpack_bf16(p1);
                     
                     flash_v2_q_tile[local_id.y][d] = v0.x;
                     flash_v2_q_tile[local_id.y][d+1u] = v0.y;
                     flash_v2_q_tile[local_id.y][d+2u] = v1.x;
                     flash_v2_q_tile[local_id.y][d+3u] = v1.y;
                 }
             }
        }
        
        // Load K
        let k_load_idx = k_start + local_id.y;
        if (k_load_idx < flash_v2_params.seq_k) {
             let k_base_elem = ((batch_idx * flash_v2_params.num_kv_heads + kv_head_idx) * flash_v2_params.seq_k + k_load_idx) * flash_v2_params.head_dim;
             
             for (var d = local_id.x * 4u; d < flash_v2_params.head_dim; d += 128u) {
                 if (d + 3u < flash_v2_params.head_dim) {
                     let base_packed = (k_base_elem + d) / 2u;
                     let p0 = flash_v2_k_bf16[base_packed];
                     let p1 = flash_v2_k_bf16[base_packed + 1u];
                     let v0 = unpack_bf16(p0);
                     let v1 = unpack_bf16(p1);
                     
                     flash_v2_k_tile[local_id.y][d] = v0.x;
                     flash_v2_k_tile[local_id.y][d+1u] = v0.y;
                     flash_v2_k_tile[local_id.y][d+2u] = v1.x;
                     flash_v2_k_tile[local_id.y][d+3u] = v1.y;
                 }
             }
        }
        
        // Load V
        if (k_load_idx < flash_v2_params.seq_k) {
             let v_base_elem = ((batch_idx * flash_v2_params.num_kv_heads + kv_head_idx) * flash_v2_params.seq_k + k_load_idx) * flash_v2_params.head_dim;
             
             for (var d = local_id.x * 4u; d < flash_v2_params.head_dim; d += 128u) {
                 if (d + 3u < flash_v2_params.head_dim) {
                     let base_packed = (v_base_elem + d) / 2u;
                     let p0 = flash_v2_v_bf16[base_packed];
                     let p1 = flash_v2_v_bf16[base_packed + 1u];
                     let v0 = unpack_bf16(p0);
                     let v1 = unpack_bf16(p1);
                     
                     flash_v2_v_tile[local_id.y][d] = v0.x;
                     flash_v2_v_tile[local_id.y][d+1u] = v0.y;
                     flash_v2_v_tile[local_id.y][d+2u] = v1.x;
                     flash_v2_v_tile[local_id.y][d+3u] = v1.y;
                 }
             }
        }
        
        workgroupBarrier();
        
        // Compute Attention (F32)
        for (var k_tile = 0u; k_tile < FLASH_V2_TILE_K; k_tile++) {
            let k_pos = k_start + k_tile;
            
            var score: f32 = 0.0;
            if (flash_v2_params.causal == 1u && k_pos > q_idx) {
                score = -1e10;
            } else if (k_pos < flash_v2_params.seq_k) {
                 var sum_vec = vec4<f32>(0.0);
                 for (var d = 0u; d + 3u < flash_v2_params.head_dim; d += 4u) {
                     let q_vec = vec4<f32>(
                         flash_v2_q_tile[local_id.y][d],
                         flash_v2_q_tile[local_id.y][d + 1u],
                         flash_v2_q_tile[local_id.y][d + 2u],
                         flash_v2_q_tile[local_id.y][d + 3u]
                     );
                     let k_vec = vec4<f32>(
                         flash_v2_k_tile[k_tile][d],
                         flash_v2_k_tile[k_tile][d + 1u],
                         flash_v2_k_tile[k_tile][d + 2u],
                         flash_v2_k_tile[k_tile][d + 3u]
                     );
                     sum_vec += q_vec * k_vec;
                 }
                 score = (sum_vec.x + sum_vec.y + sum_vec.z + sum_vec.w) * flash_v2_params.scale;
                 
                 if (flash_v2_params.has_mask == 1u) {
                    let mask_idx = 
                        batch_idx * flash_v2_params.mask_stride_b + 
                        head_idx * flash_v2_params.mask_stride_h + 
                        q_idx * flash_v2_params.mask_stride_q + 
                        k_pos * flash_v2_params.mask_stride_k;
                    score += flash_v2_mask[mask_idx];
                 }
            } else {
                score = -1e10;
            }
            
            let new_max = max(max_score, score);
            let old_scale = exp(max_score - new_max);
            let new_scale = exp(score - new_max);
            
            let rand_val = hash3(
                batch_idx * flash_v2_params.num_heads + head_idx,
                q_idx,
                k_pos + flash_v2_params.seed_val
            );
            let keep = step(flash_v2_params.dropout_prob, rand_val);
            let dropout_scale = 1.0 / (1.0 - flash_v2_params.dropout_prob + 1e-6);
            let mask_scale = keep * dropout_scale;
            
            for (var d = 0u; d < flash_v2_params.head_dim; d++) {
                out_acc[d] *= old_scale;
            }
            sum_exp = sum_exp * old_scale + new_scale;
            max_score = new_max;
            
            if (k_pos < flash_v2_params.seq_k && !(flash_v2_params.causal == 1u && k_pos > q_idx)) {
                let weighted_scale = new_scale * mask_scale;
                for (var d = 0u; d < flash_v2_params.head_dim; d++) {
                    out_acc[d] += weighted_scale * flash_v2_v_tile[k_tile][d];
                }
            }
        }
        workgroupBarrier();
    }
    
    // Write Output (F32 -> BF16)
    if (q_idx < flash_v2_params.seq_q) {
        let norm = 1.0 / max(sum_exp, 1e-10);
        let out_base_elem = ((batch_idx * flash_v2_params.num_heads + head_idx) * flash_v2_params.seq_q + q_idx) * flash_v2_params.head_dim;
        
        for (var d = 0u; d + 3u < flash_v2_params.head_dim; d += 4u) {
            let v0 = out_acc[d] * norm;
            let v1 = out_acc[d+1u] * norm;
            let v2 = out_acc[d+2u] * norm;
            let v3 = out_acc[d+3u] * norm;
            
            let p0 = pack_bf16(vec2<f32>(v0, v1));
            let p1 = pack_bf16(vec2<f32>(v2, v3));
            
            let base_packed = (out_base_elem + d) / 2u;
            flash_v2_out_bf16[base_packed] = p0;
            flash_v2_out_bf16[base_packed + 1u] = p1;
        }
    }
}
