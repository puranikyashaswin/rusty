// --- BF16 Matrix Multiplication Kernels ---

// Bindings
@group(0) @binding(0) var<storage, read> mm_a_bf16: array<u32>; // Packed bf16
@group(0) @binding(1) var<storage, read> mm_b_bf16: array<u32>; // Packed bf16
@group(0) @binding(2) var<storage, read_write> mm_out: array<f32>; // Output in f32 (Mixed Precision)
@group(0) @binding(3) var<uniform> mm_params: vec4<u32>; // M, K, N, Padding

var<workgroup> tile_a: array<array<f32, 16>, 16>;
var<workgroup> tile_b: array<array<f32, 16>, 16>;

fn get_bf16_val(arr: ptr<storage, array<u32>, read>, idx: u32, limit: u32) -> f32 {
    if (idx >= limit) { return 0.0; }
    let packed_idx = idx / 2u;
    let packed = (*arr)[packed_idx];
    
    var val_u32: u32;
    if ((idx % 2u) == 0u) {
        // Lower 16 bits
        val_u32 = (packed & 0xFFFFu) << 16u;
    } else {
        // Upper 16 bits (already in position 16-31 of packed)
        val_u32 = packed & 0xFFFF0000u;
    }
    return bitcast<f32>(val_u32);
}

@compute @workgroup_size(16, 16, 1)
fn matmul_bf16_tiled(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;
    let m = mm_params.x;
    let k = mm_params.y;
    let n = mm_params.z;

    var acc = 0.0;
    let num_tiles = (k + 15u) / 16u;

    for (var t = 0u; t < num_tiles; t = t + 1u) {
        let r = row;
        let c = t * 16u + local_id.x;
        
        // Load tile A
        // A is [M, K], index = r * K + c
        let idx_a = r * k + c;
        if (r < m && c < k) {
            tile_a[local_id.y][local_id.x] = get_bf16_val(&mm_a_bf16, idx_a, m * k);
        } else {
            tile_a[local_id.y][local_id.x] = 0.0;
        }

        let r_b = t * 16u + local_id.y;
        let c_b = col;
        
        // Load tile B
        // B is [K, N], index = r_b * N + c_b
        let idx_b = r_b * n + c_b;
        if (r_b < k && c_b < n) {
            tile_b[local_id.y][local_id.x] = get_bf16_val(&mm_b_bf16, idx_b, k * n);
        } else {
            tile_b[local_id.y][local_id.x] = 0.0;
        }

        workgroupBarrier();

        for (var i = 0u; i < 16u; i = i + 1u) {
            acc = acc + tile_a[local_id.y][i] * tile_b[i][local_id.x];
        }

        workgroupBarrier();
    }

    if (row < m && col < n) {
        mm_out[row * n + col] = acc;
    }
}
