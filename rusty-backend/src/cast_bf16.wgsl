// BF16 Casting Kernels

@group(0) @binding(0) var<storage, read> input_f32: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_bf16: array<u32>; // packed 2xbf16

@group(0) @binding(0) var<storage, read> input_bf16: array<u32>; // packed 2xbf16
@group(0) @binding(1) var<storage, read_write> output_f32: array<f32>;

// Reuse bindings 0 and 1, but we need separate pipelines/shaders or reuse same source?
// We generally use bindings 0 and 1 for input/output in casting.
// The layout is compatible (Buffer, Buffer).

@compute @workgroup_size(64)
fn cast_f32_to_bf16(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let idx_f32 = i * 2u;
    
    if (idx_f32 >= arrayLength(&input_f32)) {
        return;
    }

    let v1 = input_f32[idx_f32];
    let v2 = if (idx_f32 + 1u < arrayLength(&input_f32)) { input_f32[idx_f32 + 1u] } else { 0.0 };

    // F32 to BF16 conversion
    // 1. Bitcast to u32
    // 2. Rounding (add 0x8000) - optional, but good for accuracy
    // 3. Shift right 16
    
    let u1 = bitcast<u32>(v1);
    let u2 = bitcast<u32>(v2);
    
    // Check for NaN? (Exponent all 1s, mantissa non-zero). 
    // Usually simple shifting preserves NaNs as BF16 NaNs (mostly).
    
    // Fast rounding:
    // Only round if not NaN. 
    // Actually, simplest is truncation for performance, or add 0x8000.
    // Let's add 0x8000 before shifting.
    // Note: If NaN, adding might change payload, but still NaN.
    
    let bf1_u32 = (u1 + 0x00008000u) >> 16u;
    let bf2_u32 = (u2 + 0x00008000u) >> 16u;
    
    // Pack 2x16 bits into u32
    // output[i] = (bf2 << 16) | bf1;  (Little Endian packing: first element loop order usually low bits? 
    // Wait, bytemuck/Rust slice: [bf1, bf2]. 
    // u32 from LE bytes: [byte0, byte1, byte2, byte3]
    // bf1 is [byte0, byte1]. bf2 is [byte2, byte3].
    // So u32 = (bf2 << 16) | bf1.
    
    let packed = (bf2_u32 << 16u) | (bf1_u32 & 0xFFFFu);
    
    output_bf16[i] = packed;
}

@compute @workgroup_size(64)
fn cast_bf16_to_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= arrayLength(&input_bf16)) {
        return;
    }

    let packed = input_bf16[i];
    
    // Unpack (Little Endian assumption matched with packing)
    let bf1 = packed & 0xFFFFu;
    let bf2 = packed >> 16u;
    
    // BF16 to F32: Shift left 16.
    let u1 = bf1 << 16u;
    let u2 = bf2 << 16u;
    
    let f1 = bitcast<f32>(u1);
    let f2 = bitcast<f32>(u2);
    
    let idx_out = i * 2u;
    if (idx_out < arrayLength(&output_f32)) {
        output_f32[idx_out] = f1;
    }
    if (idx_out + 1u < arrayLength(&output_f32)) {
        output_f32[idx_out + 1u] = f2;
    }
}
