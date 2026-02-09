// WGSL Mixed Precision Casting Kernels

// Shared bindings for both kernels (reused layout)
@group(0) @binding(0) var<storage, read> input_source: array<u32>; // interpreted as f32/u32 based on kernel
@group(0) @binding(1) var<storage, read_write> output_dest: array<u32>; // interpreted as u32/f32 based on kernel

// We use u32 for the array type in the binding definition to be generic, 
// and bitcast inside if needed, or alias. 
// Actually, WGSL allows aliasing via multiple var definitions if they aren't used in same entry point.
// But to be safe and simple, let's explicit defines for each.

@group(0) @binding(0) var<storage, read> input_f32: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_f16_packed: array<u32>;

@compute @workgroup_size(64)
fn cast_f32_to_f16(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let idx_f32 = i * 2u;
    
    if (idx_f32 >= arrayLength(&input_f32)) {
        return;
    }

    let v1 = input_f32[idx_f32];
    let v2 = if (idx_f32 + 1u < arrayLength(&input_f32)) { input_f32[idx_f32 + 1u] } else { 0.0 };

    let packed = pack2x16float(vec2<f32>(v1, v2));
    output_f16_packed[i] = packed;
}

// Separate vars for the reverse cast, but reusing binding indices 0 and 1
@group(0) @binding(0) var<storage, read> input_f16_packed: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_f32: array<f32>;

@compute @workgroup_size(64)
fn cast_f16_to_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= arrayLength(&input_f16_packed)) {
        return;
    }

    let packed = input_f16_packed[i];
    let unpacked = unpack2x16float(packed);

    let idx_out = i * 2u;
    if (idx_out < arrayLength(&output_f32)) {
        output_f32[idx_out] = unpacked.x;
    }
    if (idx_out + 1u < arrayLength(&output_f32)) {
        output_f32[idx_out + 1u] = unpacked.y;
    }
}
